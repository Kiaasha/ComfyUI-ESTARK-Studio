import os
import shutil
import subprocess
import cv2
import torch
import numpy as np
from PIL import Image
import gc
import comfy.utils
import comfy.model_management
import inspect
import nodes

# ---------------------------------------------------------
# Universal Adapter Hack
# ---------------------------------------------------------
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False
ANY_TYPE = AnyType("*")

# ---------------------------------------------------------
# Node 1: ESTARK Hybrid Loader
# ---------------------------------------------------------
class ESTARK_HybridLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_path": ("STRING", {"default": "/home/kiavash/Desktop/New Folder", "multiline": False}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("original_video_path",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "load_input"
    CATEGORY = "ESTARK/Studio"

    def load_input(self, input_path):
        valid_extensions = ('.mp4', '.mkv', '.avi', '.mov')
        video_paths = []
        
        # If the user provided a single file path
        if os.path.isfile(input_path) and input_path.lower().endswith(valid_extensions):
            video_paths.append(input_path)
            print(f"[ESTARK Loader] Loaded single video: {input_path}")
            
        # If the user provided a directory path
        elif os.path.isdir(input_path):
            files = sorted(os.listdir(input_path))
            for file in files:
                if file.lower().endswith(valid_extensions):
                    video_paths.append(os.path.join(input_path, file))
            print(f"[ESTARK Loader] Found {len(video_paths)} videos in folder: {input_path}")
        else:
            print("[ESTARK Loader ERROR] Invalid path or unsupported file format.")
            
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
                "output_folder": ("STRING", {"default": "/home/kiavash/Desktop/Upscaled/"}),
                "keep_vram_free_percent": ("INT", {"default": 5, "min": 0, "max": 90, "step": 5}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("frames_folder_path",)
    FUNCTION = "process_video_dynamic"
    CATEGORY = "ESTARK/Studio"
    OUTPUT_NODE = True 

    def process_video_dynamic(self, video_path, upscale_model, output_folder, keep_vram_free_percent):
        if isinstance(video_path, list): video_path = video_path[0]
        device = comfy.model_management.get_torch_device()
        upscale_model.to(device)

        vid_name = os.path.splitext(os.path.basename(video_path))[0]
        save_dir = os.path.join(output_folder, vid_name + "_dynamic_upscaled")
        os.makedirs(save_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        all_frames = []
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            all_frames.append(torch.from_numpy(frame_rgb).float() / 255.0)
        cap.release()

        total_frames = len(all_frames)
        h, w = all_frames[0].shape[:2]
        
        total_vram = torch.cuda.get_device_properties(device).total_memory
        allowed_vram = total_vram * ((100 - keep_vram_free_percent) / 100.0)
        
        # Completely safe start (Warm-up phase)
        current_batch_size = 2 
        is_warmup = True
        
        print(f"\n[ESTARK Progressive UPSCALER]")
        print(f"  -> Target Free VRAM: {keep_vram_free_percent}% (Budget: {allowed_vram / (1024**3):.2f} GB)")
        print(f"  -> Starting Warm-up Phase with Batch Size: {current_batch_size}\n")

        frame_idx = 0

        while frame_idx < total_frames:
            end_idx = min(frame_idx + current_batch_size, total_frames)
            batch_tensors = torch.stack(all_frames[frame_idx:end_idx]).permute(0, 3, 1, 2).to(device)

            if is_warmup:
                torch.cuda.reset_peak_memory_stats(device)

            try:
                with torch.no_grad():
                    upscaled = upscale_model(batch_tensors)
                
                # Record peak memory for optimal gear shifting
                if is_warmup:
                    peak_mem = torch.cuda.max_memory_allocated(device)
                    # Calculate actual system capacity with a 20% safety margin (0.8) to prevent fragmentation OOM
                    multiplier = allowed_vram / peak_mem
                    optimal_batch = int((current_batch_size * multiplier) * 0.8)
                    
                    next_batch_size = max(1, min(32, optimal_batch))
                    print(f"\n[WARM-UP COMPLETE] Real VRAM used for 2 frames: {peak_mem / (1024**3):.2f} GB")
                    print(f"[ESTARK GEAR SHIFT] Jumping to Optimal Batch Size: {next_batch_size}\n")
                    
                    current_batch_size = next_batch_size
                    is_warmup = False

                out_imgs = (upscaled.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255.0).astype(np.uint8)
                for i in range(len(out_imgs)):
                    Image.fromarray(out_imgs[i]).save(os.path.join(save_dir, f"frame_{frame_idx + i:06d}.png"))
                
                print(f"Processed batch {frame_idx} to {end_idx-1} [Batch Size: {end_idx - frame_idx}]")
                frame_idx = end_idx
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    gc.collect()
                    # Exit warm-up mode in case of an unexpected error
                    is_warmup = False 
                    if current_batch_size > 1:
                        current_batch_size -= 1
                        print(f"\n[ESTARK WARNING] OOM Hit! Downgrading batch to {current_batch_size}...\n")
                    else:
                        raise e
                else:
                    raise e
            finally:
                del batch_tensors
                if 'upscaled' in locals(): del upscaled
                if 'out_imgs' in locals(): del out_imgs
                torch.cuda.empty_cache()

        upscale_model.to(comfy.model_management.unet_offload_device())
        return (save_dir,)

# ---------------------------------------------------------
# Node 3: ESTARK Dynamic Batch RIFE
# ---------------------------------------------------------
class ESTARK_DynamicBatchRIFE:
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
    
    # Dual output: frames directory and fps multiplier
    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("rife_frames_folder", "fps_multiplier")
    FUNCTION = "process_rife_dynamic"
    CATEGORY = "ESTARK/Studio"
    OUTPUT_NODE = True

    def process_rife_dynamic(self, frames_folder_path, ckpt_name, multiplier, keep_vram_free_percent, delete_input_upscale_folder):
        if isinstance(frames_folder_path, list): frames_folder_path = frames_folder_path[0]
        rife_class = nodes.NODE_CLASS_MAPPINGS.get("RIFE VFI")
        if not rife_class: raise Exception("ComfyUI-Frame-Interpolation not loaded!")
        rife_instance = rife_class()
        method = getattr(rife_instance, rife_instance.FUNCTION)
        sig = inspect.signature(method)

        parent_dir = os.path.dirname(frames_folder_path)
        base_name = os.path.basename(frames_folder_path)
        save_dir = os.path.join(parent_dir, base_name + f"_RIFE_{multiplier}x")
        os.makedirs(save_dir, exist_ok=True)

        frames = sorted([f for f in os.listdir(frames_folder_path) if f.lower().endswith(('.png', '.jpg'))])
        total_frames = len(frames)
        if total_frames < 2: return (save_dir, multiplier)

        device = comfy.model_management.get_torch_device()
        total_vram = torch.cuda.get_device_properties(device).total_memory
        allowed_vram = total_vram * ((100 - keep_vram_free_percent) / 100.0)
        
        current_batch_size = 2
        is_warmup = True
        
        print(f"\n[ESTARK Progressive RIFE]")
        frame_idx = 0
        saved_count = 0

        while frame_idx < total_frames - 1:
            end_idx = min(frame_idx + current_batch_size + 1, total_frames)
            batch_list = []
            for i in range(frame_idx, end_idx):
                img = Image.open(os.path.join(frames_folder_path, frames[i])).convert("RGB")
                batch_list.append(torch.from_numpy(np.array(img)).float() / 255.0)
            
            batch_tensors = torch.stack(batch_list)
            kwargs = {"ckpt_name": ckpt_name, "frames": batch_tensors, "clear_cache_after_n_frames": 1, "multiplier": multiplier, "fast_mode": True, "ensemble": True, "scale_factor": 1.0}
            
            if is_warmup:
                torch.cuda.reset_peak_memory_stats(device)

            try:
                with torch.no_grad():
                    result = method(**{k: v for k, v in kwargs.items() if k in sig.parameters})[0]
                
                if is_warmup:
                    peak_mem = torch.cuda.max_memory_allocated(device)
                    multiplier_val = allowed_vram / peak_mem
                    optimal_batch = int((current_batch_size * multiplier_val) * 0.9)
                    next_batch_size = max(1, min(32, optimal_batch))
                    current_batch_size = next_batch_size
                    is_warmup = False

                frames_to_save = result[:-1]
                res_cpu = (frames_to_save.cpu().numpy() * 255.0).astype(np.uint8)
                for j in range(len(res_cpu)):
                    Image.fromarray(res_cpu[j]).save(os.path.join(save_dir, f"frame_{saved_count:06d}.png"))
                    saved_count += 1
                
                intervals_processed = end_idx - 1 - frame_idx
                frame_idx += intervals_processed

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    gc.collect()
                    is_warmup = False
                    if current_batch_size > 1:
                        current_batch_size -= 1
                    else:
                        raise e
                else:
                    raise e
            finally:
                del batch_list, batch_tensors
                if 'result' in locals(): del result
                if 'frames_to_save' in locals(): del frames_to_save
                if 'res_cpu' in locals(): del res_cpu
                torch.cuda.empty_cache()

        Image.open(os.path.join(frames_folder_path, frames[-1])).save(os.path.join(save_dir, f"frame_{saved_count:06d}.png"))
        print(f"[ESTARK RIFE] Completed. Frames saved to {save_dir}")

        # --- Smart Cleanup: Delete the intermediate upscale folder ---
        if delete_input_upscale_folder and os.path.exists(frames_folder_path):
            try:
                shutil.rmtree(frames_folder_path)
                print(f"[ESTARK RIFE] Cleanup: Deleted intermediate upscale folder successfully.")
            except Exception as e:
                print(f"[ESTARK RIFE WARNING] Could not delete folder: {e}")

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
        if isinstance(rife_frames_folder, list): rife_frames_folder = rife_frames_folder[0]
        if isinstance(original_video_path, list): original_video_path = original_video_path[0]
        if isinstance(fps_multiplier, list): fps_multiplier = fps_multiplier[0]

        # 1. Read the original video's base framerate using OpenCV
        cap = cv2.VideoCapture(original_video_path)
        base_fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        if base_fps <= 0:
            base_fps = 30.0 # Safe fallback
            
        target_fps = base_fps * fps_multiplier
        
        # 2. Determine the path and name of the final video
        parent_dir = os.path.dirname(rife_frames_folder)
        base_video_name = os.path.splitext(os.path.basename(original_video_path))[0]
        final_video_path = os.path.join(parent_dir, f"{base_video_name}_ESTARK_Final.mp4")

        # 3. FFmpeg magic command to multiplex audio, video, and framerate
        print(f"\n[ESTARK Muxer] Compiling Final Video...")
        print(f"  -> Base FPS: {base_fps} | Multiplier: {fps_multiplier}x | Target FPS: {target_fps}")
        
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(target_fps),
            "-i", os.path.join(rife_frames_folder, "frame_%06d.png"),
            "-i", original_video_path,
            "-map", "0:v:0",       # Extract video stream from the first input (image sequence)
            "-map", "1:a:0?",      # Extract audio stream from the second input (original video), if exists
            "-c:v", "libx264",     # Standard codec for compatibility
            "-crf", "18",          # High quality (Visually lossless)
            "-pix_fmt", "yuv420p", # Ensures color compatibility across all media players
            "-c:a", "aac",         # Audio codec
            "-b:a", "192k",        # Audio bitrate
            "-shortest",           # Stop encoding when the shortest stream ends
            final_video_path
        ]

        try:
            # Execute in the background without cluttering the output console
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            print(f"[ESTARK Muxer] Success! Final video saved to: {final_video_path}\n")
        except Exception as e:
            print(f"[ESTARK Muxer ERROR] FFmpeg compilation failed: {e}")

        return (final_video_path,)

# ---------------------------------------------------------
# Registrations
# ---------------------------------------------------------
NODE_CLASS_MAPPINGS = {
    "ESTARK_HybridLoader": ESTARK_HybridLoader,
    "ESTARK_DynamicBatchUpscaler": ESTARK_DynamicBatchUpscaler,
    "ESTARK_DynamicBatchRIFE": ESTARK_DynamicBatchRIFE,
    "ESTARK_VideoMuxer": ESTARK_VideoMuxer
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ESTARK_HybridLoader": "ESTARK Hybrid Loader",
    "ESTARK_DynamicBatchUpscaler": "ESTARK Dynamic Upscaler (Smart)",
    "ESTARK_DynamicBatchRIFE": "ESTARK Dynamic RIFE + Cleanup",
    "ESTARK_VideoMuxer": "ESTARK Video Muxer"
}
