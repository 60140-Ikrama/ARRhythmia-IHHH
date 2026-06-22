import os
import argparse
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

def generate_ultrasound_frame(t: int, rx: float, ry: float, angle: float, noise_level: float = 20.0) -> np.ndarray:
    """
    Generates a single simulated ultrasound frame of a 4-Chamber view (LA, RA, LV, RV).
    Features an ultrasound scan sector, bright myocardial walls, dark blood-filled cavities,
    and realistic speckle noise.
    """
    h, w = 112, 112
    frame = np.zeros((h, w), dtype=np.uint8)
    
    if cv2 is None:
        # Absolute fallback if CV2 is missing
        return frame
        
    # 1. Draw scan sector (cone shape)
    center = (56, -10)
    cv2.ellipse(frame, center, (110, 110), 90, -35, 35, 30, -1)
    
    # Add dim gray background tissue noise
    tissue_mask = (frame == 30)
    background_noise = np.random.normal(50, 15, (h, w)).astype(np.uint8)
    frame[tissue_mask] = background_noise[tissue_mask]
    
    # 2. Draw Multi-Chamber structures (LV, RV, LA, RA)
    # Centers:
    # LV: (42, 65)
    # RV: (70, 60)
    # LA: (46, 38)
    # RA: (66, 36)
    
    chambers = [
        # (cx, cy, rx, ry, rot_angle, blood_val)
        (42, 65, rx, ry, angle, 15), # LV
        (70, 60, rx * 0.85, ry * 0.9, -angle, 15), # RV
        (46, 38, rx * 0.8, ry * 0.7, 5.0, 18), # LA
        (66, 36, rx * 0.75, ry * 0.65, -5.0, 18) # RA
    ]
    
    # Draw myocardial walls first (bright shells around cavities)
    for cx, cy, crx, cry, rot, _ in chambers:
        cv2.ellipse(frame, (int(cx), int(cy)), (int(crx + 3.0), int(cry + 4.0)), int(rot), 0, 360, 160, -1)
        
    # Draw blood-filled cavities (dark central ellipses)
    for cx, cy, crx, cry, rot, b_val in chambers:
        cavity_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(cavity_mask, (int(cx), int(cy)), (int(crx), int(cry)), int(rot), 0, 360, 255, -1)
        
        # Add fine cavity blood texture
        blood_noise = np.random.normal(b_val, 6, (h, w)).astype(np.uint8)
        cavity_indices = (cavity_mask == 255) & (frame > 0)
        frame[cavity_indices] = blood_noise[cavity_indices]
        
    # 3. Apply Gaussian blur to simulate ultrasound resolution degradation
    frame = cv2.GaussianBlur(frame, (5, 5), 1.2)
    
    # 4. Add post-processed speckle noise
    speckle = np.random.normal(0, noise_level, (h, w)).astype(np.float32)
    final_frame = np.clip(frame.astype(np.float32) + speckle, 0, 255).astype(np.uint8)
    
    return cv2.cvtColor(final_frame, cv2.COLOR_GRAY2BGR)

def generate_synthetic_video(rhythm: str, output_path: str, T: int = 150, fps: int = 50):
    """Generates an .avi video containing a simulated left ventricle contracting under a given rhythm."""
    if cv2 is None:
        raise ImportError("OpenCV (cv2) must be installed to generate videos.")
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    writer = cv2.VideoWriter(output_path, fourcc, float(fps), (112, 112))
    
    print(f"Generating synthetic 4-chamber {rhythm.upper()} video ({T} frames, {fps} FPS) -> {output_path}...")
    
    bpm = 75
    if rhythm == "bradycardia":
        bpm = 48
    elif rhythm == "tachycardia":
        bpm = 125
        
    for t in range(T):
        time_sec = t / float(fps)
        
        if rhythm == "normal":
            phase = 2 * np.pi * (bpm / 60.0) * time_sec
            rx = 18.0 + 6.0 * np.cos(phase)
            ry = 32.0 + 8.0 * np.cos(phase)
        elif rhythm == "bradycardia":
            phase = 2 * np.pi * (bpm / 60.0) * time_sec
            rx = 19.0 + 6.0 * np.cos(phase)
            ry = 33.0 + 8.0 * np.cos(phase)
        elif rhythm == "tachycardia":
            phase = 2 * np.pi * (bpm / 60.0) * time_sec
            rx = 16.0 + 4.5 * np.cos(phase)
            ry = 29.0 + 6.0 * np.cos(phase)
        elif rhythm == "afib":
            phase = 0.0
            for k in range(t + 1):
                inst_bpm = 100.0 + 35.0 * np.sin(k / 5.0) + 15.0 * np.cos(k / 13.0)
                phase += 2 * np.pi * (inst_bpm / 60.0) * (1.0 / float(fps))
            amp_mod = 1.0 + 0.3 * np.cos(t / 7.0)
            rx = 17.0 + (5.0 * amp_mod) * np.cos(phase)
            ry = 30.0 + (7.0 * amp_mod) * np.cos(phase)
        elif rhythm == "pvc":
            cycle_frame = t % 120
            if cycle_frame < 40:
                phase = 2 * np.pi * (72 / 60.0) * (cycle_frame / float(fps))
                vol_amp = 7.0
            elif cycle_frame < 70:
                pvc_t = (cycle_frame - 40)
                phase = 2 * np.pi * (110 / 60.0) * (pvc_t / float(fps)) + np.pi/2
                vol_amp = 3.5
            else:
                phase = np.pi
                vol_amp = 1.0
                
            rx = 18.0 + vol_amp * np.cos(phase)
            ry = 32.0 + (vol_amp * 1.3) * np.cos(phase)
        else:
            raise ValueError(f"Unknown rhythm type: {rhythm}")
            
        angle = 15.0 + 4.0 * np.sin(t / 8.0)
        img = generate_ultrasound_frame(t, rx, ry, angle)
        writer.write(img)
        
    writer.release()
    print("Video generation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic 4-chamber ultrasound videos.")
    parser.add_argument("--rhythm", type=str, default="normal", choices=["normal", "afib", "pvc", "bradycardia", "tachycardia"],
                        help="Rhythm to simulate (default: normal)")
    parser.add_argument("--output", type=str, default="", help="Output video path")
    parser.add_argument("--frames", type=int, default=150, help="Number of frames to generate")
    parser.add_argument("--fps", type=int, default=50, help="Video frame rate")
    
    args = parser.parse_args()
    
    if cv2 is None:
        print("Error: OpenCV (cv2) is not installed in the current environment.")
        exit(1)
        
    out = args.output
    if not out:
        out = f"data/synthetic_{args.rhythm}.avi"
        
    generate_synthetic_video(args.rhythm, out, args.frames, args.fps)
