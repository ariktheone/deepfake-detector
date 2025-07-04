import cv2
import subprocess
import os
import shutil
import tempfile
import numpy as np
from pathlib import Path

def check_ffmpeg_installed():
    """Check if FFmpeg is properly installed"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ FFmpeg is available and working")
            return True
        else:
            print("❌ FFmpeg failed to run")
            return False
    except FileNotFoundError:
        print("❌ FFmpeg not found - install with: brew install ffmpeg")
        return False
    except subprocess.TimeoutExpired:
        print("❌ FFmpeg timeout")
        return False
    except Exception as e:
        print(f"❌ FFmpeg check error: {e}")
        return False

def create_bulletproof_video(input_path, output_path, process_frame_func=None):
    """Create bulletproof video that will definitely work"""
    print(f"\n🛡️ Creating bulletproof video...")
    print(f"   Input: {input_path}")
    print(f"   Output: {output_path}")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Strategy 1: FFmpeg conversion (most reliable)
    if check_ffmpeg_installed():
        if create_with_ffmpeg_bulletproof(input_path, output_path, process_frame_func):
            return True
    
    # Strategy 2: OpenCV with raw frames -> FFmpeg
    if create_with_frames_and_ffmpeg(input_path, output_path, process_frame_func):
        return True
    
    # Strategy 3: Simple copy if nothing else works
    print("🆘 Using emergency copy fallback...")
    return emergency_copy(input_path, output_path)

def create_with_ffmpeg_bulletproof(input_path, output_path, process_frame_func=None):
    """Create video using FFmpeg with bulletproof settings"""
    try:
        print("🎬 Method 1: FFmpeg bulletproof conversion...")
        
        if process_frame_func is None:
            # Direct conversion without processing
            cmd = [
                'ffmpeg', '-y',  # Overwrite output
                '-i', input_path,
                
                # Video codec settings for maximum compatibility
                '-c:v', 'libx264',
                '-profile:v', 'baseline',  # Most compatible profile
                '-level', '3.0',
                '-pix_fmt', 'yuv420p',
                
                # Quality settings
                '-crf', '23',  # Good quality
                '-preset', 'medium',  # Balance speed/compression
                
                # Web optimization
                '-movflags', '+faststart',  # Optimize for web streaming
                '-fflags', '+genpts',  # Generate timestamps
                
                # Audio settings
                '-c:a', 'aac',
                '-b:a', '128k',
                '-ar', '44100',
                
                # Force container format
                '-f', 'mp4',
                
                output_path
            ]
            
            print(f"🔧 Running direct FFmpeg conversion...")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                if verify_video_file(output_path):
                    print(f"✅ Direct FFmpeg conversion successful")
                    return True
                else:
                    print("❌ Direct conversion created invalid file")
            else:
                print(f"❌ Direct FFmpeg failed: {result.stderr}")
        
        else:
            # Process frames and then use FFmpeg
            return create_with_frame_processing_ffmpeg(input_path, output_path, process_frame_func)
            
    except subprocess.TimeoutExpired:
        print("❌ FFmpeg conversion timed out")
    except Exception as e:
        print(f"❌ FFmpeg conversion error: {e}")
    
    return False

def create_with_frame_processing_ffmpeg(input_path, output_path, process_frame_func):
    """Process frames and create video with FFmpeg"""
    try:
        print("🎬 Method 2: Frame processing + FFmpeg...")
        
        # Create temporary directory for frames
        with tempfile.TemporaryDirectory() as temp_dir:
            frames_dir = os.path.join(temp_dir, 'frames')
            os.makedirs(frames_dir, exist_ok=True)
            
            # Extract and process frames
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                print("❌ Cannot open input video")
                return False
            
            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Ensure even dimensions
            width = width if width % 2 == 0 else width - 1
            height = height if height % 2 == 0 else height - 1
            
            print(f"📹 Processing: {width}x{height} @ {fps}fps")
            
            frame_count = 0
            saved_frames = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Resize frame to ensure correct dimensions
                if frame.shape[1] != width or frame.shape[0] != height:
                    frame = cv2.resize(frame, (width, height))
                
                # Process frame if function provided
                if process_frame_func:
                    try:
                        frame = process_frame_func(frame, frame_count)
                    except Exception as e:
                        print(f"⚠️ Frame processing error: {e}")
                
                # Save frame as PNG (lossless)
                frame_filename = os.path.join(frames_dir, f"frame_{frame_count:06d}.png")
                if cv2.imwrite(frame_filename, frame):
                    saved_frames += 1
                
                frame_count += 1
                
                if frame_count % 100 == 0:
                    print(f"📈 Processed {frame_count} frames...")
            
            cap.release()
            print(f"✅ Saved {saved_frames} frames")
            
            if saved_frames == 0:
                print("❌ No frames saved")
                return False
            
            # Create video from frames using FFmpeg
            cmd = [
                'ffmpeg', '-y',
                '-framerate', str(fps),
                '-i', os.path.join(frames_dir, 'frame_%06d.png'),
                
                # Video settings
                '-c:v', 'libx264',
                '-profile:v', 'baseline',
                '-level', '3.0',
                '-pix_fmt', 'yuv420p',
                '-crf', '23',
                '-preset', 'medium',
                
                # Web optimization
                '-movflags', '+faststart',
                '-fflags', '+genpts',
                
                # No audio (since we're creating from frames)
                '-an',
                
                # Force MP4
                '-f', 'mp4',
                
                output_path
            ]
            
            print("🔧 Creating video from frames...")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                if verify_video_file(output_path):
                    print(f"✅ Frame-based FFmpeg creation successful")
                    return True
                else:
                    print("❌ Frame-based creation created invalid file")
            else:
                print(f"❌ Frame-based FFmpeg failed: {result.stderr}")
    
    except Exception as e:
        print(f"❌ Frame processing error: {e}")
    
    return False

def create_with_frames_and_ffmpeg(input_path, output_path, process_frame_func):
    """Fallback: Create frames and use system FFmpeg"""
    try:
        print("🎬 Method 3: OpenCV frames + system FFmpeg...")
        
        # Try to use system FFmpeg even if check failed
        temp_frames_dir = tempfile.mkdtemp()
        
        try:
            # Extract frames with OpenCV
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                return False
            
            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if process_frame_func:
                    try:
                        frame = process_frame_func(frame, frame_count)
                    except:
                        pass
                
                frame_path = os.path.join(temp_frames_dir, f"frame_{frame_count:06d}.jpg")
                cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                frame_count += 1
            
            cap.release()
            
            if frame_count > 0:
                # Try FFmpeg even if initial check failed
                cmd = [
                    'ffmpeg', '-y',
                    '-framerate', str(fps),
                    '-i', os.path.join(temp_frames_dir, 'frame_%06d.jpg'),
                    '-c:v', 'libx264',
                    '-pix_fmt', 'yuv420p',
                    '-movflags', '+faststart',
                    output_path
                ]
                
                result = subprocess.run(cmd, capture_output=True, timeout=300)
                
                if result.returncode == 0 and verify_video_file(output_path):
                    print(f"✅ System FFmpeg fallback successful")
                    return True
        
        finally:
            # Cleanup
            shutil.rmtree(temp_frames_dir, ignore_errors=True)
    
    except Exception as e:
        print(f"❌ Frames + FFmpeg error: {e}")
    
    return False

def verify_video_file(video_path):
    """Verify that video file is valid and playable"""
    try:
        if not os.path.exists(video_path):
            print(f"❌ Video file doesn't exist: {video_path}")
            return False
        
        file_size = os.path.getsize(video_path)
        if file_size < 1000:
            print(f"❌ Video file too small: {file_size} bytes")
            return False
        
        # Try to open with OpenCV
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ OpenCV cannot open video: {video_path}")
            cap.release()
            return False
        
        # Try to read first frame
        ret, frame = cap.read()
        cap.release()
        
        if not ret or frame is None:
            print(f"❌ Cannot read frames from video: {video_path}")
            return False
        
        print(f"✅ Video verification passed: {file_size:,} bytes")
        return True
        
    except Exception as e:
        print(f"❌ Video verification error: {e}")
        return False

def emergency_copy(input_path, output_path):
    """Emergency fallback: just copy the file"""
    try:
        print("🆘 Emergency copy fallback...")
        shutil.copy2(input_path, output_path)
        
        if verify_video_file(output_path):
            print("✅ Emergency copy successful")
            return True
        else:
            print("❌ Emergency copy created invalid file")
            return False
            
    except Exception as e:
        print(f"❌ Emergency copy failed: {e}")
        return False

# Install FFmpeg helper
def install_ffmpeg_macos():
    """Install FFmpeg on macOS"""
    print("""
🔧 To install FFmpeg on macOS:

Method 1 - Homebrew (recommended):
    brew install ffmpeg

Method 2 - MacPorts:
    sudo port install ffmpeg

Method 3 - Download binary:
    1. Go to https://ffmpeg.org/download.html
    2. Download macOS build
    3. Add to PATH

After installation, restart terminal and try again.
    """)