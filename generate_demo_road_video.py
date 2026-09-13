import os
import cv2
import numpy as np

def generate_synthetic_road_video(output_path="input/sample_road_driving.mp4", num_frames=150, width=960, height=540, fps=30):
    """
    Generates a synthetic realistic highway driving video sequence complete with 
    curved/straight lane lines, dashed center dividers, asphalt texture, and vehicle motion.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        output_path = output_path.replace(".mp4", ".avi")
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
    print(f"[+] Generating synthetic highway driving video: '{output_path}' ({num_frames} frames)...")
    
    # Horizon line Y coordinate
    horizon_y = int(height * 0.55)
    
    # Base perspective lane points at bottom of screen
    base_left_x = int(width * 0.18)
    base_right_x = int(width * 0.82)
    top_left_x = int(width * 0.44)
    top_right_x = int(width * 0.56)
    
    for f in range(num_frames):
        # Create asphalt road background
        frame = np.ones((height, width, 3), dtype=np.uint8) * 45
        
        # Draw sky above horizon
        frame[0:horizon_y, :] = (200, 160, 100) # Sunset/dusk sky color
        
        # Add subtle road texture noise
        road_noise = np.random.randint(-8, 8, (height - horizon_y, width, 3), dtype=np.int16)
        frame[horizon_y:, :] = np.clip(frame[horizon_y:, :].astype(np.int16) + road_noise, 0, 255).astype(np.uint8)
        
        # Vehicle steering motion drift (simulates dynamic highway curve)
        drift = int(25 * np.sin(f / 20.0))
        
        cur_left_bot = base_left_x + drift
        cur_right_bot = base_right_x + drift
        cur_left_top = top_left_x + int(drift * 0.4)
        cur_right_top = top_right_x + int(drift * 0.4)
        
        # Draw Solid Yellow Left Shoulder Line
        cv2.line(frame, (cur_left_bot - 30, height), (cur_left_top - 15, horizon_y), (0, 215, 255), 8, lineType=cv2.LINE_AA)
        
        # Draw Solid White Right Shoulder Line
        cv2.line(frame, (cur_right_bot + 30, height), (cur_right_top + 15, horizon_y), (240, 240, 240), 8, lineType=cv2.LINE_AA)
        
        # Draw Dashed White Center Lane Lines
        num_dashes = 7
        dash_offset = (f * 12) % 100
        for d in range(num_dashes):
            t1 = (d * 0.14 + dash_offset * 0.0014) % 1.0
            t2 = min(t1 + 0.07, 1.0)
            
            y1 = int(horizon_y + t1 * (height - horizon_y))
            y2 = int(horizon_y + t2 * (height - horizon_y))
            
            x1_left = int(cur_left_top + t1 * (cur_left_bot - cur_left_top))
            x2_left = int(cur_left_top + t2 * (cur_left_bot - cur_left_top))
            
            x1_right = int(cur_right_top + t1 * (cur_right_bot - cur_right_top))
            x2_right = int(cur_right_top + t2 * (cur_right_bot - cur_right_top))
            
            # Left Center Dashed Line
            cv2.line(frame, (x1_left, y1), (x2_left, y2), (255, 255, 255), max(2, int(t2 * 6)), lineType=cv2.LINE_AA)
            # Right Center Dashed Line
            cv2.line(frame, (x1_right, y1), (x2_right, y2), (255, 255, 255), max(2, int(t2 * 6)), lineType=cv2.LINE_AA)
            
        # Draw Distance Overlay HUD text
        cv2.putText(frame, f"SYNTHETIC HIGHWAY TEST CAMERA | FRAME: {f+1:03d}/{num_frames}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
        out.write(frame)
        
    out.release()
    print(f"[OK] Synthetic highway video saved to '{output_path}'")
    return output_path

if __name__ == "__main__":
    generate_synthetic_road_video()
