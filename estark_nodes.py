import os
import shutil
import subprocess
import importlib
import inspect
import queue
import sys
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch

import comfy.utils
import folder_paths
import comfy.model_management as mm
import nodes

VIDEO_EXTS = ('.mp4', '.mkv', '.avi', '.mov')
FINAL_SUFFIX = "_ESTARK_Final.mp4"
MIN_FREE_DISK = 10 * 1024**3  # stop before the disk fills up
PNG_PARAMS = [cv2.IMWRITE_PNG_COMPRESSION, 3]
DEFAULT_OUT = os.path.join(folder_paths.get_output_directory(), "ESTARK")


def _guard(folder, min_free=MIN_FREE_DISK):
    mm.throw_exception_if_processing_interrupted()
    free = shutil.disk_usage(folder).free
    if free < min_free:
        raise RuntimeError(f"[ESTARK] Stopped: only {free / 1024**3:.1f} GB free on disk. Free some space and run again.")


def _prefetch(iterable, size):
    """Read ahead on a background thread so the GPU never waits for disk."""
    q = queue.Queue(maxsize=size)
    stop = threading.Event()
    done = object()

    def run():
        try:
            for item in iterable:
                if stop.is_set():
                    return
                q.put(item)
            q.put(done)
        except BaseException as e:
            q.put(e)

    t = threading.Thread(target=run, daemon=True)
    t.start()
    try:
        while True:
            item = q.get()
            if item is done:
                return
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        stop.set()
        while t.is_alive():
            try:
                q.get_nowait()
            except queue.Empty:
                t.join(0.05)


class _Writer:
    """Writes PNGs on several threads; at most max_pending frames wait in RAM."""
    def __init__(self, max_pending=64):
        self.pool = ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4))
        self.pending = deque()
        self.max_pending = max_pending

    def put(self, path, rgb):
        self.pending.append(self.pool.submit(_write_png, path, rgb))
        while len(self.pending) > self.max_pending:
            self.pending.popleft().result()

    def close(self):
        try:
            while self.pending:
                self.pending.popleft().result()
        finally:
            self.pool.shutdown(wait=True)


def _write_png(path, rgb):
    if not cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), PNG_PARAMS):
        raise IOError(f"[ESTARK] Could not write {path}")


def _read_rgb(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"[ESTARK] Could not read {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _video_frames(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"[ESTARK] Cannot open video: {path}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                return
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def _to_uint8(t):
    return t.float().clamp_(0, 1).mul_(255).round_().to(torch.uint8).cpu().numpy()


# ---------------------------------------------------------
# Node 1: ESTARK Hybrid Loader
# ---------------------------------------------------------
class ESTARK_HybridLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_path": ("STRING", {"default": folder_paths.get_input_directory(), "multiline": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("original_video_path",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "load_input"
    CATEGORY = "ESTARK/Studio"

    def load_input(self, input_path):
        if os.path.isfile(input_path) and input_path.lower().endswith(VIDEO_EXTS):
            video_paths = [input_path]
        elif os.path.isdir(input_path):
            # skip our own finished videos, which the muxer writes next to the frame folders
            video_paths = [os.path.join(input_path, f) for f in sorted(os.listdir(input_path))
                           if f.lower().endswith(VIDEO_EXTS) and not f.endswith(FINAL_SUFFIX)]
        else:
            raise RuntimeError(f"[ESTARK Loader] Not a video file or folder: {input_path}")
        if not video_paths:
            raise RuntimeError(f"[ESTARK Loader] No videos found in: {input_path}")
        print(f"[ESTARK Loader] {len(video_paths)} video(s)")
        return (video_paths,)


# ---------------------------------------------------------
# Node 2: ESTARK Dynamic Batch Upscaler
# ---------------------------------------------------------
class ESTARK_DynamicBatchUpscaler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {"forceInput": True}),
                "upscale_model": ("UPSCALE_MODEL",),
                "output_folder": ("STRING", {"default": DEFAULT_OUT}),
                "keep_vram_free_percent": ("INT", {"default": 5, "min": 0, "max": 90, "step": 5}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("frames_folder_path",)
    FUNCTION = "process_video_dynamic"
    CATEGORY = "ESTARK/Studio"
    OUTPUT_NODE = True

    def process_video_dynamic(self, video_path, upscale_model, output_folder, keep_vram_free_percent):
        if isinstance(video_path, list):
            video_path = video_path[0]
        save_dir = os.path.join(output_folder, os.path.splitext(os.path.basename(video_path))[0] + "_dynamic_upscaled")
        os.makedirs(save_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        mm.unload_all_models()
        mm.load_models_gpu([upscale_model.patcher], force_full_load=True)
        device = upscale_model.patcher.load_device
        dtype = (torch.float16 if upscale_model.supports_half else
                 torch.bfloat16 if upscale_model.supports_bfloat16 else torch.float32)
        budget = mm.get_free_memory(device) * (100 - keep_vram_free_percent) / 100.0
        print(f"[ESTARK Upscaler] {total} frames, {dtype}, VRAM budget {budget / 1024**3:.1f} GB")

        batch, measured, tile = 2, False, None

        def run(chunk):
            x = torch.from_numpy(np.stack(chunk)).to(device).permute(0, 3, 1, 2).float().div_(255)
            with torch.no_grad(), torch.autocast(device.type, dtype=dtype, enabled=dtype != torch.float32):
                if tile is None:
                    y = upscale_model(x)
                else:
                    y = comfy.utils.tiled_scale(x, upscale_model, tile_x=tile, tile_y=tile, overlap=32,
                                                upscale_amount=upscale_model.scale, output_device=device)
            return _to_uint8(y.permute(0, 2, 3, 1))

        pbar = comfy.utils.ProgressBar(max(total, 1))
        writer = _Writer()
        frames = _prefetch(_video_frames(video_path), 64)
        pending, idx, exhausted = [], 0, False
        try:
            while True:
                _guard(save_dir)
                while len(pending) < batch and not exhausted:
                    f = next(frames, None)
                    if f is None:
                        exhausted = True
                    else:
                        pending.append(f)
                if not pending:
                    break
                chunk = pending[:batch]
                if not measured:
                    torch.cuda.reset_peak_memory_stats(device)
                    base = torch.cuda.memory_allocated(device)
                try:
                    out = run(chunk)
                except Exception as e:
                    mm.raise_non_oom(e)
                    mm.soft_empty_cache(True)
                    measured = True
                    if batch > 1:
                        batch = max(1, batch // 2)
                    elif tile is None:
                        tile = 512
                    elif tile > 64:
                        tile //= 2
                    else:
                        raise
                    print(f"[ESTARK Upscaler] OOM -> batch {batch}, tile {tile or 'off'}")
                    continue
                if not measured:
                    per_frame = max(torch.cuda.max_memory_allocated(device) - base, 1) / len(chunk)
                    batch = max(1, min(16, int(budget * 0.85 / per_frame)))
                    measured = True
                    print(f"[ESTARK Upscaler] {per_frame / 1024**3:.2f} GB per frame -> batch {batch}")
                for i, img in enumerate(out):
                    writer.put(os.path.join(save_dir, f"frame_{idx + i:06d}.png"), img)
                idx += len(chunk)
                del pending[:len(chunk)]
                pbar.update(len(chunk))
        finally:
            frames.close()
            writer.close()
        print(f"[ESTARK Upscaler] Done: {idx} frames -> {save_dir}")
        return (save_dir,)


# ---------------------------------------------------------
# Node 3: ESTARK Dynamic Batch RIFE
# ---------------------------------------------------------
class ESTARK_DynamicBatchRIFE:
    CHUNK = 16  # frames sent to RIFE per call

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames_folder_path": ("STRING", {"forceInput": True}),
                "ckpt_name": (["rife49.pth", "rife47.pth", "rife46.pth"],),
                "multiplier": ("INT", {"default": 2, "min": 2, "max": 8, "step": 1}),
                "keep_vram_free_percent": ("INT", {"default": 5, "min": 0, "max": 90, "step": 5}),
                "delete_input_upscale_folder": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("rife_frames_folder", "fps_multiplier")
    FUNCTION = "process_rife_dynamic"
    CATEGORY = "ESTARK/Studio"
    OUTPUT_NODE = True

    def process_rife_dynamic(self, frames_folder_path, ckpt_name, multiplier, keep_vram_free_percent, delete_input_upscale_folder):
        if isinstance(frames_folder_path, list):
            frames_folder_path = frames_folder_path[0]
        rife_class = nodes.NODE_CLASS_MAPPINGS.get("RIFE VFI")
        if rife_class is None:
            raise RuntimeError("[ESTARK RIFE] ComfyUI-Frame-Interpolation is not installed.")
        rife = rife_class()
        method = getattr(rife, rife.FUNCTION)
        params = inspect.signature(method).parameters

        src = frames_folder_path.rstrip("/")
        save_dir = src + f"_RIFE_{multiplier}x"
        os.makedirs(save_dir, exist_ok=True)
        names = sorted(f for f in os.listdir(src) if f.lower().endswith(('.png', '.jpg')))
        if len(names) < 2:
            return (save_dir, multiplier)

        mm.unload_all_models()
        mm.soft_empty_cache(True)
        device = mm.get_torch_device()
        budget = mm.get_free_memory(device) * (100 - keep_vram_free_percent) / 100.0

        reader = _prefetch((_read_rgb(os.path.join(src, n)) for n in names), 48)
        prev = next(reader)
        h, w = prev.shape[:2]
        # RIFE estimates flow at a lower scale on 4K-class frames (recommended by its authors; also saves VRAM)
        kwargs = {"ckpt_name": ckpt_name, "clear_cache_after_n_frames": 10, "multiplier": multiplier,
                  "fast_mode": True, "ensemble": True, "scale_factor": 0.5 if max(h, w) >= 3000 else 1.0,
                  "dtype": "float32", "batch_size": 1}
        measured = "batch_size" not in params
        print(f"[ESTARK RIFE] {len(names)} frames {w}x{h}, x{multiplier}, scale {kwargs['scale_factor']}")

        pbar = comfy.utils.ProgressBar(len(names) - 1)
        writer = _Writer()
        saved, consumed, exhausted = 0, 0, False
        try:
            while not exhausted:
                _guard(save_dir)
                group = [prev]
                while len(group) < self.CHUNK + 1:
                    f = next(reader, None)
                    if f is None:
                        exhausted = True
                        break
                    group.append(f)
                if len(group) == 1:
                    break
                kwargs["frames"] = torch.from_numpy(np.stack(group)).float().div_(255)
                while True:
                    if not measured:
                        torch.cuda.reset_peak_memory_stats(device)
                        base = torch.cuda.memory_allocated(device)
                    try:
                        with torch.no_grad():
                            result = method(**{k: v for k, v in kwargs.items() if k in params})[0]
                        break
                    except Exception as e:
                        mm.raise_non_oom(e)
                        mm.soft_empty_cache(True)
                        measured = True
                        if kwargs["batch_size"] > 1:
                            kwargs["batch_size"] //= 2
                        elif kwargs["scale_factor"] > 0.25:
                            kwargs["scale_factor"] /= 2
                        else:
                            raise
                        print(f"[ESTARK RIFE] OOM -> batch {kwargs['batch_size']}, scale {kwargs['scale_factor']}")
                if not measured:
                    per_pair = max(torch.cuda.max_memory_allocated(device) - base, 1)
                    kwargs["batch_size"] = max(1, min(8, int(budget * 0.8 / per_pair)))
                    measured = True
                    print(f"[ESTARK RIFE] {per_pair / 1024**3:.2f} GB per pair -> batch {kwargs['batch_size']}")

                for img in _to_uint8(result[:-1]):
                    writer.put(os.path.join(save_dir, f"frame_{saved:06d}.png"), img)
                    saved += 1
                if delete_input_upscale_folder:
                    for n in names[consumed:consumed + len(group) - 1]:
                        os.remove(os.path.join(src, n))
                consumed += len(group) - 1
                prev = group[-1]
                pbar.update(len(group) - 1)
            writer.put(os.path.join(save_dir, f"frame_{saved:06d}.png"), prev)
            saved += 1
        finally:
            reader.close()
            writer.close()

        if delete_input_upscale_folder:
            shutil.rmtree(src, ignore_errors=True)
        print(f"[ESTARK RIFE] Done: {saved} frames -> {save_dir}")
        return (save_dir, multiplier)


# ---------------------------------------------------------
# Node 4: ESTARK Video Muxer
# ---------------------------------------------------------
class ESTARK_VideoMuxer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "rife_frames_folder": ("STRING", {"forceInput": True}),
                "original_video_path": ("STRING", {"forceInput": True}),
                "fps_multiplier": ("INT", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("final_mp4_path",)
    FUNCTION = "mux_video"
    CATEGORY = "ESTARK/Studio"
    OUTPUT_NODE = True

    def mux_video(self, rife_frames_folder, original_video_path, fps_multiplier):
        if isinstance(rife_frames_folder, list):
            rife_frames_folder = rife_frames_folder[0]
        if isinstance(original_video_path, list):
            original_video_path = original_video_path[0]
        if isinstance(fps_multiplier, list):
            fps_multiplier = fps_multiplier[0]

        cap = cv2.VideoCapture(original_video_path)
        base_fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if base_fps <= 0:
            base_fps = 30.0
        target_fps = base_fps * fps_multiplier

        base_name = os.path.splitext(os.path.basename(original_video_path))[0]
        final_video_path = os.path.join(os.path.dirname(rife_frames_folder.rstrip("/")), base_name + FINAL_SUFFIX)
        print(f"[ESTARK Muxer] {base_fps} fps x{fps_multiplier} = {target_fps} fps -> {final_video_path}")

        cmd = ["ffmpeg", "-y", "-framerate", str(target_fps),
               "-i", os.path.join(rife_frames_folder, "frame_%06d.png"),
               "-i", original_video_path,
               "-map", "0:v:0", "-map", "1:a:0?",
               "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
               "-c:a", "aac", "-b:a", "192k", "-shortest",
               final_video_path]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError("[ESTARK Muxer] ffmpeg failed:\n" + r.stderr[-1500:])
        return (final_video_path,)


# ---------------------------------------------------------
# Node 5: ESTARK Video Upscale (Stream) - video -> upscale -> RIFE -> mp4 in one pass,
# frames stay on the GPU until they are piped to ffmpeg; nothing is written to disk.
# ---------------------------------------------------------
def _load_rife(ckpt_name, device):
    rife_class = nodes.NODE_CLASS_MAPPINGS.get("RIFE VFI")
    if rife_class is None:
        raise RuntimeError("[ESTARK] ComfyUI-Frame-Interpolation is not installed.")
    mod = sys.modules[rife_class.__module__]
    IFNet = importlib.import_module(mod.__name__ + ".rife_arch").IFNet
    arch = mod.CKPT_NAME_VER_DICT[ckpt_name]
    model = IFNet(arch_ver=arch)
    model.load_state_dict(torch.load(mod.load_file_from_github_release(mod.MODEL_TYPE, ckpt_name), weights_only=False))
    return model.eval().to(device), arch


class ESTARK_VideoUpscaleStream:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {"forceInput": True}),
                "upscale_model": ("UPSCALE_MODEL",),
                "rife_ckpt": (["rife49.pth", "rife47.pth", "rife46.pth"],),
                "multiplier": ("INT", {"default": 2, "min": 1, "max": 8, "tooltip": "Frame multiplier. 1 = upscale only."}),
                "output_folder": ("STRING", {"default": DEFAULT_OUT}),
                "keep_vram_free_percent": ("INT", {"default": 5, "min": 0, "max": 90, "step": 5}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("final_mp4_path",)
    FUNCTION = "run"
    CATEGORY = "ESTARK/Studio"
    OUTPUT_NODE = True

    def run(self, video_path, upscale_model, rife_ckpt, multiplier, output_folder, keep_vram_free_percent):
        if isinstance(video_path, list):
            video_path = video_path[0]
        os.makedirs(output_folder, exist_ok=True)
        out_path = os.path.join(output_folder, os.path.splitext(os.path.basename(video_path))[0] + FINAL_SUFFIX)
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        mm.unload_all_models()
        mm.load_models_gpu([upscale_model.patcher], force_full_load=True)
        device = upscale_model.patcher.load_device
        up_dtype = (torch.float16 if upscale_model.supports_half else
                    torch.bfloat16 if upscale_model.supports_bfloat16 else torch.float32)
        rife, arch = _load_rife(rife_ckpt, device) if multiplier > 1 else (None, None)
        steps = [k / multiplier for k in range(1, multiplier)]
        budget = mm.get_free_memory(device) * (100 - keep_vram_free_percent) / 100.0
        up_share = 0.5 if rife is not None else 0.85  # leave room for RIFE

        st = {"up_batch": 2, "up_measured": False, "tile": None, "rife_batch": 1, "rife_measured": rife is None, "scale_list": None}

        def upscale(chunk):
            x = torch.from_numpy(np.stack(chunk)).to(device).permute(0, 3, 1, 2).float().div_(255)
            with torch.no_grad(), torch.autocast(device.type, dtype=up_dtype, enabled=up_dtype != torch.float32):
                if st["tile"] is None:
                    y = upscale_model(x)
                else:
                    y = comfy.utils.tiled_scale(x, upscale_model, tile_x=st["tile"], tile_y=st["tile"], overlap=32,
                                                upscale_amount=upscale_model.scale, output_device=device)
            return y.float().clamp_(0, 1)

        def interpolate(a, b):
            """Middle frames for every pair (a[i], b[i]), pair-major, as uint8 on the GPU."""
            tasks = [(i, t) for i in range(a.shape[0]) for t in steps]
            mids, pos = [], 0
            while pos < len(tasks):
                sub = tasks[pos:pos + st["rife_batch"]]
                idx = torch.tensor([i for i, _ in sub], device=device)
                ts = torch.tensor([t for _, t in sub], device=device, dtype=a.dtype).view(-1, 1, 1, 1)
                if not st["rife_measured"]:
                    torch.cuda.reset_peak_memory_stats(device)
                    base = torch.cuda.memory_allocated(device)
                try:
                    with torch.no_grad():
                        mid = rife(a[idx], b[idx], ts, st["scale_list"], True, arch != "4.26")
                except Exception as e:
                    mm.raise_non_oom(e)
                    mm.soft_empty_cache(True)
                    st["rife_measured"] = True
                    if st["rife_batch"] > 1:
                        st["rife_batch"] //= 2
                    elif st["scale_list"][-1] < 4:
                        st["scale_list"] = [v * 2 for v in st["scale_list"]]
                    else:
                        raise
                    print(f"[ESTARK Stream] RIFE OOM -> batch {st['rife_batch']}, scale {1 / st['scale_list'][-1]}")
                    continue
                if not st["rife_measured"]:
                    per = max(torch.cuda.max_memory_allocated(device) - base, 1) / len(sub)
                    st["rife_batch"] = max(1, min(16, int(mm.get_free_memory(device) * 0.8 / per)))
                    st["rife_measured"] = True
                    print(f"[ESTARK Stream] RIFE {per / 1024**3:.2f} GB per frame -> batch {st['rife_batch']}")
                mids.append(mid.clamp_(0, 1).mul_(255).round_().to(torch.uint8))
                pos += len(sub)
            return torch.cat(mids) if mids else None

        pbar = comfy.utils.ProgressBar(max(total, 1))
        frames = _prefetch(_video_frames(video_path), 32)
        pipe = ThreadPoolExecutor(max_workers=1)  # one worker keeps frame order
        pending = deque()
        proc, prev, pending_in, exhausted, n_in, n_out = None, None, [], False, 0, 0
        try:
            while True:
                _guard(output_folder, 2 * 1024**3)
                while len(pending_in) < st["up_batch"] and not exhausted:
                    f = next(frames, None)
                    if f is None:
                        exhausted = True
                    else:
                        pending_in.append(f)
                if not pending_in:
                    break
                chunk = pending_in[:st["up_batch"]]
                if not st["up_measured"]:
                    torch.cuda.reset_peak_memory_stats(device)
                    base = torch.cuda.memory_allocated(device)
                try:
                    y = upscale(chunk)
                except Exception as e:
                    mm.raise_non_oom(e)
                    mm.soft_empty_cache(True)
                    st["up_measured"] = True
                    if st["up_batch"] > 1:
                        st["up_batch"] = max(1, st["up_batch"] // 2)
                    elif st["tile"] is None:
                        st["tile"] = 512
                    elif st["tile"] > 64:
                        st["tile"] //= 2
                    else:
                        raise
                    print(f"[ESTARK Stream] upscale OOM -> batch {st['up_batch']}, tile {st['tile'] or 'off'}")
                    continue
                if not st["up_measured"]:
                    per = max(torch.cuda.max_memory_allocated(device) - base, 1) / len(chunk)
                    st["up_batch"] = max(1, min(16, int(budget * up_share / per)))
                    st["up_measured"] = True
                    print(f"[ESTARK Stream] upscale {per / 1024**3:.2f} GB per frame -> batch {st['up_batch']}")
                del pending_in[:len(chunk)]

                h, w = y.shape[2], y.shape[3]
                if proc is None:
                    if rife is not None:
                        sf = 0.5 if max(h, w) >= 3000 else 1.0  # RIFE's own advice for 4K-class frames
                        st["scale_list"] = [v / sf for v in ([16, 8, 4, 2, 1] if arch == "4.26" else [8, 4, 2, 1])]
                    cmd = ["ffmpeg", "-y", "-loglevel", "error",
                           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps * multiplier), "-i", "-",
                           "-i", video_path, "-map", "0:v:0", "-map", "1:a:0?",
                           "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                           "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
                           "-c:a", "aac", "-b:a", "192k", "-shortest", out_path]
                    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
                    print(f"[ESTARK Stream] {total} frames -> {w}x{h} @ {fps * multiplier:g} fps, upscale {up_dtype}")

                y8 = y.mul(255).round_().to(torch.uint8)
                if rife is None:
                    seq = y8
                else:
                    a = torch.cat([prev, y[:-1]]) if prev is not None else y[:-1]
                    b = y if prev is not None else y[1:]
                    mids = interpolate(a, b) if a.shape[0] else None
                    parts = [] if prev is not None else [y8[:1]]
                    off = 0 if prev is not None else 1
                    m1 = len(steps)
                    for i in range(b.shape[0]):
                        parts.append(mids[i * m1:(i + 1) * m1])
                        parts.append(y8[off + i:off + i + 1])
                    seq = torch.cat(parts)
                    prev = y[-1:].clone()
                del y
                data = seq.permute(0, 2, 3, 1).contiguous().cpu().numpy()
                n_out += data.shape[0]
                pending.append(pipe.submit(proc.stdin.write, data.tobytes()))
                while len(pending) > 3:
                    pending.popleft().result()
                n_in += len(chunk)
                pbar.update(len(chunk))
            while pending:
                pending.popleft().result()
            proc.stdin.close()
            err = proc.stderr.read().decode(errors="replace")
            if proc.wait() != 0:
                raise RuntimeError("[ESTARK Stream] ffmpeg failed:\n" + err[-1500:])
        except BaseException:
            if proc is not None:
                proc.kill()
                proc.wait()
            if os.path.exists(out_path):
                os.remove(out_path)
            raise
        finally:
            frames.close()
            pipe.shutdown(wait=True)
            mm.soft_empty_cache(True)
        print(f"[ESTARK Stream] Done: {n_in} -> {n_out} frames -> {out_path}")
        return (out_path,)


NODE_CLASS_MAPPINGS = {
    "ESTARK_HybridLoader": ESTARK_HybridLoader,
    "ESTARK_DynamicBatchUpscaler": ESTARK_DynamicBatchUpscaler,
    "ESTARK_DynamicBatchRIFE": ESTARK_DynamicBatchRIFE,
    "ESTARK_VideoMuxer": ESTARK_VideoMuxer,
    "ESTARK_VideoUpscaleStream": ESTARK_VideoUpscaleStream,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ESTARK_HybridLoader": "ESTARK Hybrid Loader",
    "ESTARK_DynamicBatchUpscaler": "ESTARK Dynamic Upscaler (Smart)",
    "ESTARK_DynamicBatchRIFE": "ESTARK Dynamic RIFE + Cleanup",
    "ESTARK_VideoMuxer": "ESTARK Video Muxer",
    "ESTARK_VideoUpscaleStream": "ESTARK Video Upscale (Stream)",
}
