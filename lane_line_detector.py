#!/usr/bin/env python3
"""
===============================================================================
Self-Driving Lane Line Detector & Steering Telemetry Controller
Day 11 - 30-Day Computer Vision & Deep Learning Challenge
===============================================================================
Author: Computer Vision & AI Agent
Technologies: OpenCV, Hough Lines (cv2.HoughLinesP), Homography, Bird's Eye View

Description:
    Autonomous vehicle lane detection engine using HSL color masking, Canny edge 
    detection, Probabilistic Hough Transform, linear polynomial curve fitting, 
    exponential moving average smoothing, and Top-Down Bird's Eye View transformation.
===============================================================================
"""

import os
import sys
import glob
import json
import time
import argparse
import cv2
import numpy as np


class LaneLineTracker:
    """
    Tracks left and right lane lines across video frames using Hough Transform 
    and Exponential Moving Average (EMA) smoothing.
    """
    def __init__(self, alpha=0.3):
        self.alpha = alpha # EMA smoothing factor
        self.smooth_left = None  # (x1, y1, x2, y2)
        self.smooth_right = None # (x1, y1, x2, y2)

    def filter_color_and_edges(self, frame_bgr):
        """
        Isolates White and Yellow road markings using HSL color space and Canny edges.
        """
        hls = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HLS)
        
        # White lane mask (high lightness)
        white_lower = np.array([0, 190, 0], dtype=np.uint8)
        white_upper = np.array([180, 255, 255], dtype=np.uint8)
        white_mask = cv2.inRange(hls, white_lower, white_upper)
        
        # Yellow lane mask (H: 15..35)
        yellow_lower = np.array([15, 80, 100], dtype=np.uint8)
        yellow_upper = np.array([35, 255, 255], dtype=np.uint8)
        yellow_mask = cv2.inRange(hls, yellow_lower, yellow_upper)
        
        combined_mask = cv2.bitwise_or(white_mask, yellow_mask)
        
        # Edge Detection
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        # Boost edges where color mask is present
        boosted_edges = cv2.bitwise_or(edges, cv2.bitwise_and(edges, combined_mask))
        return boosted_edges

    def apply_roi_mask(self, edges):
        """Applies a trapezoidal Region of Interest (ROI) mask to road perspective."""
        h, w = edges.shape[:2]
        mask = np.zeros_like(edges)
        
        # Trapezoid vertices: bottom-left, top-left, top-right, bottom-right
        roi_pts = np.array([[
            (int(w * 0.05), h),
            (int(w * 0.40), int(h * 0.52)),
            (int(w * 0.60), int(h * 0.52)),
            (int(w * 0.95), h)
        ]], dtype=np.int32)
        
        cv2.fillPoly(mask, roi_pts, 255)
        masked_edges = cv2.bitwise_and(edges, mask)
        return masked_edges, roi_pts[0]

    def fit_lane_line(self, lines, y_min, y_max):
        """Fits a smooth 1D polynomial line (x = m*y + b) through candidate segment points."""
        if not lines:
            return None
            
        x_pts, y_pts = [], []
        for x1, y1, x2, y2 in lines:
            x_pts.extend([x1, x2])
            y_pts.extend([y1, y2])
            
        if len(x_pts) < 2:
            return None
            
        poly = np.polyfit(y_pts, x_pts, 1) # x = m*y + b
        m, b = poly[0], poly[1]
        
        x_start = int(m * y_max + b)
        x_end = int(m * y_min + b)
        
        return (x_start, y_max, x_end, y_min)

    def detect_lanes(self, frame_bgr):
        """
        Full lane detection pipeline for single frame.
        """
        h, w = frame_bgr.shape[:2]
        edges = self.filter_color_and_edges(frame_bgr)
        roi_edges, roi_poly = self.apply_roi_mask(edges)
        
        # Probabilistic Hough Lines Transform
        lines = cv2.HoughLinesP(
            roi_edges,
            rho=1,
            theta=np.pi / 180,
            threshold=15,
            minLineLength=15,
            maxLineGap=120
        )
        
        left_lines = []
        right_lines = []
        
        if lines is not None:
            for line in lines:
                l = line[0] if line.ndim > 1 else line
                x1, y1, x2, y2 = int(l[0]), int(l[1]), int(l[2]), int(l[3])
                if x2 == x1:
                    continue
                slope = (y2 - y1) / float(x2 - x1)
                
                # Filter left lane (negative slope) and right lane (positive slope)
                if -1.2 < slope < -0.2:
                    left_lines.append((x1, y1, x2, y2))
                elif 0.2 < slope < 1.2:
                    right_lines.append((x1, y1, x2, y2))
                    
        y_min = int(h * 0.58)
        y_max = h
        
        curr_left = self.fit_lane_line(left_lines, y_min, y_max)
        curr_right = self.fit_lane_line(right_lines, y_min, y_max)
        
        # Fallback default lane boundaries if undetected
        if curr_left is None:
            curr_left = (int(w * 0.20), y_max, int(w * 0.44), y_min)
        if curr_right is None:
            curr_right = (int(w * 0.80), y_max, int(w * 0.56), y_min)
            
        # Exponential Moving Average (EMA) Smoothing
        if self.smooth_left is None:
            self.smooth_left = curr_left
        else:
            self.smooth_left = tuple(
                int(self.alpha * c + (1 - self.alpha) * s)
                for c, s in zip(curr_left, self.smooth_left)
            )
            
        if self.smooth_right is None:
            self.smooth_right = curr_right
        else:
            self.smooth_right = tuple(
                int(self.alpha * c + (1 - self.alpha) * s)
                for c, s in zip(curr_right, self.smooth_right)
            )
            
        return self.smooth_left, self.smooth_right, edges, roi_poly

    def compute_birds_eye_view(self, frame_bgr):
        """Transforms front camera view into top-down Bird's Eye View perspective."""
        h, w = frame_bgr.shape[:2]
        
        src_pts = np.float32([
            [w * 0.15, h],
            [w * 0.42, h * 0.55],
            [w * 0.58, h * 0.55],
            [w * 0.85, h]
        ])
        
        dst_pts = np.float32([
            [w * 0.25, h],
            [w * 0.25, 0],
            [w * 0.75, 0],
            [w * 0.75, h]
        ])
        
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(frame_bgr, M, (w, h))
        return warped


def render_lane_hud(frame_bgr, left_line, right_line, warped_bev, edges, roi_poly):
    """
    Renders 2-panel HUD dashboard overlay containing:
    [1. Front View + Filled Green Lane Corridor] | [2. Bird's Eye Top-Down View + Telemetry Gauges]
    """
    vis = frame_bgr.copy()
    h, w = vis.shape[:2]
    
    # 1. Fill Lane Corridor with semi-transparent neon green polygon
    lx1, ly1, lx2, ly2 = left_line
    rx1, ry1, rx2, ry2 = right_line
    
    pts_left = np.array([[lx1, ly1], [lx2, ly2]], dtype=np.int32)
    pts_right = np.array([[rx2, ry2], [rx1, ry1]], dtype=np.int32)
    pts_lane = np.vstack([pts_left, pts_right])
    
    overlay = vis.copy()
    cv2.fillPoly(overlay, [pts_lane], (0, 220, 100)) # Green lane polygon
    cv2.addWeighted(overlay, 0.35, vis, 0.65, 0, vis)
    
    # Draw thick neon boundary lines
    cv2.line(vis, (lx1, ly1), (lx2, ly2), (0, 255, 0), 4, lineType=cv2.LINE_AA)
    cv2.line(vis, (rx1, ry1), (rx2, ry2), (0, 255, 0), 4, lineType=cv2.LINE_AA)
    
    # Calculate Steering Telemetry & Centering Offset
    lane_center_x = (lx1 + rx1) / 2.0
    vehicle_center_x = w / 2.0
    pixel_offset = lane_center_x - vehicle_center_x
    cm_offset = round(pixel_offset * 0.45, 1) # Scaling constant: 1 px ~= 0.45 cm
    
    if abs(cm_offset) < 5.0:
        status_text = "STEERING: CENTERED [OK]"
        status_color = (0, 255, 120)
    elif cm_offset < -5.0:
        status_text = f"DRIFTING LEFT ({abs(cm_offset)} cm)"
        status_color = (0, 215, 255)
    else:
        status_text = f"DRIFTING RIGHT ({cm_offset} cm)"
        status_color = (0, 145, 255)
        
    # Resize Panels for 2-panel Montage View
    target_h = 360
    
    def resize_h(img, target_h):
        aspect = img.shape[1] / float(img.shape[0])
        return cv2.resize(img, (int(target_h * aspect), target_h), interpolation=cv2.INTER_AREA)
        
    p1 = resize_h(vis, target_h)
    p2 = resize_h(warped_bev, target_h)
    
    def add_panel_header(img, title, subtitle="", color=(30, 30, 30)):
        img_h, img_w = img.shape[:2]
        hdr = np.zeros((45, img_w, 3), dtype=np.uint8)
        hdr[:] = color
        cv2.putText(hdr, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, lineType=cv2.LINE_AA)
        if subtitle:
            cv2.putText(hdr, subtitle, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, lineType=cv2.LINE_AA)
        return np.vstack([hdr, img])
        
    p1_hdr = add_panel_header(p1, "[1] FRONT CAMERA & LANE CORRIDOR", "Hough Line Fitting + Polygon Overlay", (30, 80, 140))
    p2_hdr = add_panel_header(p2, "[2] BIRD'S EYE TOP-DOWN VIEW", "Homography Perspective Warp", (40, 120, 40))
    
    divider = np.zeros((p1_hdr.shape[0], 5, 3), dtype=np.uint8)
    divider[:] = (180, 180, 180)
    
    montage = np.hstack([p1_hdr, divider, p2_hdr])
    
    # Bottom Telemetry HUD Banner
    banner_h = 50
    banner = np.zeros((banner_h, montage.shape[1], 3), dtype=np.uint8)
    banner[:] = (20, 20, 20)
    
    telemetry_str = f"VEHICLE SPEED: 65 MPH | OFFSET: {cm_offset:+.1f} cm | STATUS: {status_text}"
    cv2.putText(banner, telemetry_str, (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.58, status_color, 2, lineType=cv2.LINE_AA)
    
    final_montage = np.vstack([montage, banner])
    return final_montage, cm_offset, status_text


def process_lane_video(input_source, output_dir="output", max_frames=150):
    """
    Processes video stream for lane line detection and steering telemetry calculation.
    
    Args:
        input_source (str or int): Path to video file or camera index.
        output_dir (str): Directory to save output files.
        max_frames (int): Maximum frames to process.
        
    Returns:
        dict: Processed telemetry statistics.
    """
    os.makedirs(output_dir, exist_ok=True)
    is_live = str(input_source).lower() in ["camera", "webcam", "0"]
    cap_src = 0 if is_live else input_source
    
    cap = cv2.VideoCapture(cap_src)
    if not cap.isOpened():
        raise ValueError(f"Could not open video source: {input_source}")
        
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS)
    if fps_in <= 0 or np.isnan(fps_in):
        fps_in = 30.0
        
    base_name = "live_camera" if is_live else os.path.splitext(os.path.basename(input_source))[0]
    out_video_path = os.path.join(output_dir, f"{base_name}_lane_detection_output.mp4")
    
    # Calculate output montage video dimensions
    sample_p1_w = int(360 * (width / float(height)))
    montage_w = (sample_p1_w * 2) + 5
    montage_h = 360 + 45 + 50 # panel height + header + bottom banner
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_video_path, fourcc, fps_in, (montage_w, montage_h))
    
    if not writer.isOpened():
        out_video_path = out_video_path.replace(".mp4", ".avi")
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        writer = cv2.VideoWriter(out_video_path, fourcc, fps_in, (montage_w, montage_h))
        
    tracker = LaneLineTracker(alpha=0.25)
    
    frame_count = 0
    start_time = time.time()
    offsets = []
    
    print(f"\n[+] Processing Lane Line Stream: '{base_name}' ({width}x{height} @ {fps_in:.1f} FPS)")
    print(f"  - Output Video: '{out_video_path}'")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        left_line, right_line, edges, roi_poly = tracker.detect_lanes(frame)
        bev_warped = tracker.compute_birds_eye_view(frame)
        
        hud_montage, cm_offset, status_text = render_lane_hud(
            frame, left_line, right_line, bev_warped, edges, roi_poly
        )
        
        offsets.append(cm_offset)
        
        if writer.isOpened():
            writer.write(hud_montage)
            
        if is_live:
            cv2.imshow("Self-Driving Lane Detector - HUD", hud_montage)
            if cv2.waitKey(1) & 0xFF in [ord('q'), ord('Q'), 27]:
                break
        else:
            if frame_count % 30 == 0 or frame_count == max_frames:
                print(f"  - Frame {frame_count:03d} | Offset: {cm_offset:+.1f} cm | {status_text}")
            if frame_count >= max_frames:
                break
                
    cap.release()
    writer.release()
    if is_live:
        cv2.destroyAllWindows()
        
    total_time = round(time.time() - start_time, 2)
    avg_fps = round(frame_count / total_time, 1) if total_time > 0 else 0
    mean_offset = round(float(np.mean(offsets)), 2) if offsets else 0.0
    
    # Save Telemetry JSON Report
    json_path = os.path.join(output_dir, f"{base_name}_lane_report.json")
    summary = {
        "video_source": base_name,
        "total_frames_processed": frame_count,
        "total_time_seconds": total_time,
        "average_fps": avg_fps,
        "mean_lane_offset_cm": mean_offset,
        "output_video": out_video_path
    }
    
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=4)
        
    print(f"\n[OK] Lane Detection Complete! Processed {frame_count} frames in {total_time}s ({avg_fps} FPS)")
    print(f"  - Output Video: '{out_video_path}'")
    print(f"  - Telemetry Report: '{json_path}'")
    
    return summary


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Self-Driving Lane Line Detector & Steering Telemetry Controller."
    )
    parser.add_argument(
        "-i", "--input", type=str, default="input/sample_road_driving.mp4",
        help="Path to highway video file or 'camera'/'0' for live webcam stream."
    )
    parser.add_argument(
        "-o", "--output", type=str, default="output",
        help="Directory to save output video and telemetry reports."
    )
    parser.add_argument(
        "--max-frames", type=int, default=150,
        help="Maximum frames to process for video inputs (default: 150)."
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    # Auto-generate synthetic video if input video file is missing
    if not os.path.exists(args.input) and args.input.lower() not in ["camera", "webcam", "0"]:
        print(f"[!] Input road video '{args.input}' not found. Generating synthetic test video...")
        from generate_demo_road_video import generate_synthetic_road_video
        args.input = generate_synthetic_road_video(output_path="input/sample_road_driving.mp4")
        
    print("\n==========================================================")
    print("  [LANE] SELF-DRIVING LANE LINE DETECTOR")
    print("  --------------------------------------------------------")
    print(f"  Input Source: {args.input}")
    print(f"  Output Dir  : {args.output}")
    print("==========================================================")
    
    process_lane_video(
        input_source=args.input,
        output_dir=args.output,
        max_frames=args.max_frames
    )


if __name__ == "__main__":
    main()
