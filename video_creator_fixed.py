import cv2
import subprocess
import os
import shutil
import tempfile
import numpy as np
from pathlib import Path
import time

def check_ffmpeg_available():
    """Check if FFmpeg is available"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✅ FFmpeg available")
            return True
        return False
    except:
        print("⚠️ FFmpeg not available")
        return False

def create_processed_video_opencv(input_path, output_path, detection_result, process_func=None):
    """Create processed video using OpenCV with detection overlay"""
    print(f"🎬 Creating processed video with OpenCV...")
    print(f"   Input: {input_path}")
    print(f"   Output: {output_path}")
    print(f"   Detection Score: {detection_result:.1f}%")
    
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Open input video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"❌ Cannot open input video: {input_path}")
            return False
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Ensure even dimensions for codec compatibility
        width = width if width % 2 == 0 else width - 1
        height = height if height % 2 == 0 else height - 1
        
        print(f"📹 Video properties: {width}x{height}, {fps}fps, {total_frames} frames")
        
        # Try different codecs in order of compatibility
        codecs_to_try = [
            ('mp4v', cv2.VideoWriter_fourcc(*'mp4v')),
            ('MJPG', cv2.VideoWriter_fourcc(*'MJPG')),
            ('XVID', cv2.VideoWriter_fourcc(*'XVID')),
        ]
        
        out = None
        successful_codec = None
        
        for codec_name, fourcc in codecs_to_try:
            try:
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                if out.isOpened():
                    successful_codec = codec_name
                    print(f"✅ Using {codec_name} codec")
                    break
                else:
                    if out:
                        out.release()
                    out = None
            except Exception as e:
                print(f"⚠️ {codec_name} codec failed: {e}")
                if out:
                    out.release()
                out = None
        
        if out is None or not out.isOpened():
            print("❌ All codecs failed")
            cap.release()
            return False
        
        # Determine overlay style based on detection result
        if detection_result > 70:
            overlay_color = (0, 0, 255)  # Red
            risk_text = "HIGH RISK"
            overlay_alpha = 0.8
        elif detection_result > 40:
            overlay_color = (0, 165, 255)  # Orange
            risk_text = "MEDIUM RISK"
            overlay_alpha = 0.7
        else:
            overlay_color = (0, 255, 0)  # Green
            risk_text = "LOW RISK"
            overlay_alpha = 0.6
        
        # Process frames
        frame_count = 0
        processed_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize frame if needed
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))
            
            # Apply custom processing if provided
            if process_func:
                try:
                    frame = process_func(frame, frame_count, detection_result)
                except Exception as e:
                    print(f"⚠️ Custom processing failed for frame {frame_count}: {e}")
            else:
                # Apply default overlay
                frame = add_detection_overlay(frame, detection_result, risk_text, overlay_color, overlay_alpha)
            
            # Write frame
            success = out.write(frame)
            if success:
                processed_count += 1
            
            frame_count += 1
            
            # Progress update
            if frame_count % 100 == 0:
                progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                print(f"📈 Processing: {progress:.1f}% ({frame_count}/{total_frames})")
        
        # Cleanup
        cap.release()
        out.release()
        
        # Verify output
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            if file_size > 1000:  # At least 1KB
                print(f"✅ Video created successfully: {file_size:,} bytes")
                print(f"📊 Processed {processed_count}/{frame_count} frames")
                return True
            else:
                print(f"❌ Output file too small: {file_size} bytes")
                return False
        else:
            print("❌ Output file not created")
            return False
            
    except Exception as e:
        print(f"❌ OpenCV video creation error: {e}")
        if 'cap' in locals():
            cap.release()
        if 'out' in locals() and out:
            out.release()
        return False

def add_detection_overlay(frame, detection_score, risk_text, color, alpha):
    """Add detection overlay to frame"""
    try:
        overlay = frame.copy()
        h, w = frame.shape[:2]
        
        # Create semi-transparent overlay bar
        overlay_height = 80
        cv2.rectangle(overlay, (0, 0), (w, overlay_height), (0, 0, 0), -1)
        
        # Add risk indicator
        risk_width = int((detection_score / 100) * (w - 40))
        cv2.rectangle(overlay, (20, 20), (20 + risk_width, 40), color, -1)
        
        # Add text
        main_text = f"{risk_text}: {detection_score:.1f}%"
        cv2.putText(overlay, main_text, (25, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Timestamp
        timestamp_text = f"SurakshaNetra AI Analysis"
        cv2.putText(overlay, timestamp_text, (w - 300, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Blend overlay
        result = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)
        return result
        
    except Exception as e:
        print(f"⚠️ Overlay error: {e}")
        return frame

def create_processed_video_ffmpeg(input_path, output_path, detection_result):
    """Create processed video using FFmpeg with overlay"""
    print(f"🎬 Creating processed video with FFmpeg...")
    
    try:
        if not check_ffmpeg_available():
            return False
        
        # Prepare overlay text
        if detection_result > 70:
            color = "red"
            risk_text = "HIGH_RISK"
        elif detection_result > 40:
            color = "orange"
            risk_text = "MEDIUM_RISK"
        else:
            color = "green"
            risk_text = "LOW_RISK"
        
        # Simple text overlay without special characters
        overlay_text = f"{risk_text}_{detection_result:.0f}percent"
        
        # FFmpeg command with simple text overlay
        cmd = [
            'ffmpeg', '-y',
            '-i', input_path,
            '-vf', f"drawtext=text='{overlay_text}':fontcolor={color}:fontsize=24:x=10:y=30:box=1:boxcolor=black@0.5",
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '23',
            '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',
            '-c:a', 'copy',
            output_path
        ]
        
        print(f"🔧 Running FFmpeg...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
                print(f"✅ FFmpeg success: {os.path.getsize(output_path):,} bytes")
                return True
        
        print(f"❌ FFmpeg failed: {result.stderr}")
        return False
        
    except Exception as e:
        print(f"❌ FFmpeg error: {e}")
        return False

def copy_with_verification(input_path, output_path):
    """Copy video with verification"""
    try:
        print(f"📁 Copying video as fallback...")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Copy file
        shutil.copy2(input_path, output_path)
        
        # Verify copy
        if os.path.exists(output_path):
            input_size = os.path.getsize(input_path)
            output_size = os.path.getsize(output_path)
            
            if output_size >= input_size * 0.9:  # Allow 10% variance
                print(f"✅ Copy successful: {output_size:,} bytes")
                return True
            else:
                print(f"❌ Copy size mismatch: {input_size} -> {output_size}")
                return False
        else:
            print("❌ Copy failed - file not created")
            return False
            
    except Exception as e:
        print(f"❌ Copy error: {e}")
        return False

def create_processed_video_reliable(input_path, output_path, detection_score, process_func=None):
    """Reliable video creation with multiple fallbacks"""
    print(f"\n🚀 Creating reliable processed video...")
    print(f"   Input: {input_path} ({os.path.getsize(input_path):,} bytes)")
    print(f"   Output: {output_path}")
    print(f"   Score: {detection_score:.1f}%")
    
    # Ensure input exists
    if not os.path.exists(input_path):
        print(f"❌ Input file doesn't exist: {input_path}")
        return False
    
    if os.path.getsize(input_path) < 1000:
        print(f"❌ Input file too small: {os.path.getsize(input_path)} bytes")
        return False
    
    # Strategy 1: OpenCV with overlay (most reliable)
    print("\n📹 Trying OpenCV video creation...")
    if create_processed_video_opencv(input_path, output_path, detection_score, process_func):
        print("✅ OpenCV video creation successful!")
        return True
    else:
        print("⚠️ OpenCV failed, trying FFmpeg...")
    
    # Strategy 2: FFmpeg with overlay
    print("\n📹 Trying FFmpeg video creation...")
    if create_processed_video_ffmpeg(input_path, output_path, detection_score):
        print("✅ FFmpeg video creation successful!")
        return True
    else:
        print("⚠️ FFmpeg failed, copying original...")
    
    # Strategy 3: Copy original as fallback
    print("\n📁 Copying original video...")
    if copy_with_verification(input_path, output_path):
        print("✅ Video copy successful!")
        return True
    else:
        print("❌ All video creation strategies failed!")
        return False