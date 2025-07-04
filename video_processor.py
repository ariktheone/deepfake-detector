import subprocess
import os
import cv2
import json
import tempfile
from pathlib import Path
import shlex

def check_ffmpeg_available():
    """Check if FFmpeg is available on the system"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True, timeout=5)
        return result.returncode == 0
    except Exception:
        return False

def create_processed_video_ffmpeg(input_path, output_path, detection_result):
    """Create processed video using FFmpeg with detection overlay"""
    try:
        if not check_ffmpeg_available():
            print("❌ FFmpeg not available, copying original video")
            import shutil
            shutil.copy2(input_path, output_path)
            return True
        
        # Determine overlay color and text based on detection result
        if detection_result > 60:
            color = "red"
            risk_text = "HIGH RISK"
        elif detection_result > 30:
            color = "orange" 
            risk_text = "MEDIUM RISK"
        else:
            color = "green"
            risk_text = "LOW RISK"
        
        # Create simple text without special characters
        overlay_text = f"{risk_text} {detection_result}%"
        
        # Escape text properly for FFmpeg
        escaped_text = overlay_text.replace("'", "\\'").replace('"', '\\"').replace(':', '\\:')
        
        # Use a more robust approach with simpler text filter
        cmd = [
            'ffmpeg',
            '-i', input_path,
            '-vf', f"drawtext=text='{escaped_text}':fontcolor={color}:fontsize=28:x=10:y=50:box=1:boxcolor=black@0.5:boxborderw=5",
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            '-c:a', 'copy',  # Copy audio without re-encoding
            '-movflags', '+faststart',
            '-y',  # Overwrite output file
            output_path
        ]
        
        print(f"🎬 Processing video with FFmpeg...")
        print(f"📝 Adding overlay: {overlay_text}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
                print(f"✅ FFmpeg processing successful: {output_path}")
                return True
            else:
                print(f"❌ FFmpeg output file issue")
                return False
        else:
            print(f"❌ FFmpeg error: {result.stderr}")
            # Try a simpler approach without text overlay
            return create_simple_ffmpeg_copy(input_path, output_path)
            
    except subprocess.TimeoutExpired:
        print("❌ FFmpeg processing timed out")
        return False
    except Exception as e:
        print(f"❌ FFmpeg processing failed: {e}")
        return False

def create_simple_ffmpeg_copy(input_path, output_path):
    """Create a simple web-compatible copy using FFmpeg without overlay"""
    try:
        cmd = [
            'ffmpeg',
            '-i', input_path,
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '23',
            '-c:a', 'aac',
            '-movflags', '+faststart',
            '-y',
            output_path
        ]
        
        print(f"🔄 Creating simple web-compatible copy...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
                print(f"✅ Simple FFmpeg copy successful: {output_path}")
                return True
        
        print(f"❌ Simple FFmpeg copy failed: {result.stderr}")
        return False
        
    except Exception as e:
        print(f"❌ Simple FFmpeg copy error: {e}")
        return False

def create_simple_copy(input_path, output_path):
    """Simple fallback: copy original video"""
    try:
        import shutil
        shutil.copy2(input_path, output_path)
        print(f"📁 File copied successfully: {output_path}")
        return os.path.exists(output_path) and os.path.getsize(output_path) > 0
    except Exception as e:
        print(f"❌ File copy failed: {e}")
        return False

def ensure_video_compatibility(video_path):
    """Ensure video is web-compatible using FFmpeg"""
    if not check_ffmpeg_available():
        print("⚠️ FFmpeg not available, cannot check compatibility")
        return True
    
    try:
        # Check current codec
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=codec_name',
            '-of', 'csv=p=0',
            video_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            codec = result.stdout.strip()
            if codec in ['h264', 'avc']:
                print(f"✅ Video already has compatible codec: {codec}")
                return True
        
        # Convert to web-compatible format if needed
        temp_path = video_path + '_temp.mp4'
        convert_cmd = [
            'ffmpeg',
            '-i', video_path,
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '23',
            '-c:a', 'aac',
            '-movflags', '+faststart',
            '-y',
            temp_path
        ]
        
        print(f"🔧 Converting to web-compatible format...")
        result = subprocess.run(convert_cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0 and os.path.exists(temp_path):
            # Replace original with converted version
            os.replace(temp_path, video_path)
            print(f"✅ Video converted to web-compatible format")
            return True
        else:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            print(f"❌ Video conversion failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Video compatibility check failed: {e}")
        return False

def add_detection_overlay_to_image(image_path, detection_result):
    """Add detection result overlay to a static image (fallback method)"""
    try:
        import cv2
        import numpy as np
        
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            return False
        
        # Determine color based on detection result
        if detection_result > 60:
            color = (0, 0, 255)  # Red in BGR
            risk_text = "HIGH RISK"
        elif detection_result > 30:
            color = (0, 165, 255)  # Orange in BGR
            risk_text = "MEDIUM RISK"
        else:
            color = (0, 255, 0)  # Green in BGR
            risk_text = "LOW RISK"
        
        # Add text overlay
        text = f"{risk_text}: {detection_result}%"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        thickness = 2
        
        # Get text size for background rectangle
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Add semi-transparent background
        overlay = img.copy()
        cv2.rectangle(overlay, (10, 30), (text_width + 20, text_height + 50), (0, 0, 0), -1)
        img = cv2.addWeighted(overlay, 0.7, img, 0.3, 0)
        
        # Add text
        cv2.putText(img, text, (15, 30 + text_height), font, font_scale, color, thickness)
        
        # Save image
        cv2.imwrite(image_path, img)
        return True
        
    except Exception as e:
        print(f"❌ Error adding overlay to image: {e}")
        return False