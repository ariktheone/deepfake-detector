import cv2
import subprocess
import os
import shutil
import tempfile
from pathlib import Path

def check_ffmpeg_available():
    """Check if FFmpeg is available on the system"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✅ FFmpeg is available")
            return True
        return False
    except Exception:
        print("❌ FFmpeg not found")
        return False

def get_video_info(video_path):
    """Get video information"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        cap.release()
        
        return {
            'fps': fps,
            'width': width,
            'height': height,
            'frame_count': frame_count
        }
    except Exception as e:
        print(f"❌ Error getting video info: {e}")
        return None

def create_web_compatible_video_ffmpeg(input_path, output_path):
    """Create web-compatible video using FFmpeg with maximum compatibility"""
    try:
        if not check_ffmpeg_available():
            print("⚠️ FFmpeg not available")
            return False
        
        print(f"🎬 Creating web-compatible video with FFmpeg...")
        
        # Get input video info
        video_info = get_video_info(input_path)
        if not video_info:
            print("❌ Cannot read input video")
            return False
        
        # Ensure even dimensions (required for H.264)
        width = video_info['width'] if video_info['width'] % 2 == 0 else video_info['width'] - 1
        height = video_info['height'] if video_info['height'] % 2 == 0 : video_info['height'] - 1
        
        print(f"📹 Input: {video_info['width']}x{video_info['height']} -> Output: {width}x{height}")
        
        # FFmpeg command for maximum web compatibility
        cmd = [
            'ffmpeg', '-y',  # Overwrite output
            '-i', input_path,  # Input file
            
            # Video encoding settings
            '-c:v', 'libx264',  # H.264 codec
            '-preset', 'medium',  # Balance speed/quality
            '-crf', '23',  # Constant rate factor (quality)
            '-profile:v', 'high',  # H.264 profile
            '-level:v', '4.0',  # H.264 level
            '-pix_fmt', 'yuv420p',  # Pixel format for compatibility
            
            # Audio settings
            '-c:a', 'aac',  # AAC audio codec
            '-b:a', '128k',  # Audio bitrate
            '-ar', '44100',  # Audio sample rate
            
            # Video filters for scaling and compatibility
            '-vf', f'scale={width}:{height}:flags=lanczos',
            
            # Web optimization
            '-movflags', '+faststart',  # Move metadata to beginning
            '-fflags', '+genpts',  # Generate presentation timestamps
            
            # Output settings
            '-f', 'mp4',  # Force MP4 container
            output_path
        ]
        
        print(f"🔧 Running FFmpeg conversion...")
        print(f"   Command: ffmpeg -i {input_path} [options] {output_path}")
        
        # Run FFmpeg with timeout
        result = subprocess.run(cmd, 
                              capture_output=True, 
                              text=True, 
                              timeout=600)  # 10 minutes max
        
        if result.returncode == 0:
            # Verify output
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                if file_size > 1000:  # At least 1KB
                    print(f"✅ FFmpeg success: {file_size:,} bytes")
                    
                    # Verify video is readable
                    if verify_video_playable(output_path):
                        print("✅ Video verification passed")
                        return True
                    else:
                        print("❌ Video verification failed")
                        return False
                else:
                    print(f"❌ Output file too small: {file_size} bytes")
                    return False
            else:
                print("❌ Output file not created")
                return False
        else:
            print(f"❌ FFmpeg failed:")
            print(f"   Return code: {result.returncode}")
            print(f"   Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ FFmpeg timed out")
        return False
    except Exception as e:
        print(f"❌ FFmpeg error: {e}")
        return False

def verify_video_playable(video_path):
    """Verify that video is playable"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("❌ Video cannot be opened by OpenCV")
            return False
        
        # Try to read first frame
        ret, frame = cap.read()
        cap.release()
        
        if ret and frame is not None:
            print("✅ Video readable by OpenCV")
            return True
        else:
            print("❌ Cannot read video frames")
            return False
            
    except Exception as e:
        print(f"❌ Video verification error: {e}")
        return False

def create_opencv_web_video(input_path, output_path):
    """Reliable OpenCV video creation as fallback"""
    try:
        print(f"🎬 Creating video with OpenCV (fallback mode)...")
        
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print("❌ Cannot open input video")
            return False
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Ensure even dimensions
        width = width if width % 2 == 0 else width - 1
        height = height if height % 2 == 0 else height - 1
        
        print(f"📹 Processing: {width}x{height}, {fps}fps, {total_frames} frames")
        
        # Try different codecs in order of preference
        codec_options = [
            ('MJPG', cv2.VideoWriter_fourcc(*'MJPG')),  # Most compatible
            ('XVID', cv2.VideoWriter_fourcc(*'XVID')),  # Good compatibility
            ('MP4V', cv2.VideoWriter_fourcc(*'MP4V')),  # Fallback
        ]
        
        for codec_name, fourcc in codec_options:
            try:
                # Create temporary output first
                temp_output = output_path + f'_temp_{codec_name.lower()}.avi'
                
                out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
                
                if not out.isOpened():
                    print(f"⚠️ {codec_name} codec failed to initialize")
                    continue
                
                print(f"✅ Using {codec_name} codec")
                
                # Process frames
                frame_count = 0
                success_count = 0
                
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Resize if needed
                    if frame.shape[1] != width or frame.shape[0] != height:
                        frame = cv2.resize(frame, (width, height))
                    
                    # Write frame
                    out.write(frame)
                    success_count += 1
                    frame_count += 1
                    
                    # Progress update
                    if frame_count % 100 == 0:
                        progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                        print(f"📈 Progress: {progress:.1f}%")
                
                out.release()
                
                # Check if successful
                if os.path.exists(temp_output) and os.path.getsize(temp_output) > 1000:
                    print(f"✅ Successfully created with {codec_name}: {success_count} frames")
                    
                    # Move to final location
                    if output_path.endswith('.mp4'):
                        # Convert to MP4 if possible
                        if create_web_compatible_video_ffmpeg(temp_output, output_path):
                            os.remove(temp_output)
                            cap.release()
                            return True
                        else:
                            # Just rename to mp4 extension
                            shutil.move(temp_output, output_path)
                            cap.release()
                            return True
                    else:
                        shutil.move(temp_output, output_path)
                        cap.release()
                        return True
                else:
                    print(f"❌ {codec_name} failed to create valid output")
                    if os.path.exists(temp_output):
                        os.remove(temp_output)
                    continue
                    
            except Exception as e:
                print(f"❌ {codec_name} codec error: {e}")
                if 'out' in locals():
                    out.release()
                continue
        
        cap.release()
        print("❌ All OpenCV codecs failed")
        return False
        
    except Exception as e:
        print(f"❌ OpenCV video creation error: {e}")
        return False

def copy_video_with_validation(input_path, output_path):
    """Copy video with validation"""
    try:
        print(f"📁 Copying video: {input_path} -> {output_path}")
        
        # Check input exists and is readable
        if not os.path.exists(input_path):
            print("❌ Input file does not exist")
            return False
        
        input_size = os.path.getsize(input_path)
        if input_size < 1000:
            print(f"❌ Input file too small: {input_size} bytes")
            return False
        
        # Copy file
        shutil.copy2(input_path, output_path)
        
        # Verify copy
        if os.path.exists(output_path):
            output_size = os.path.getsize(output_path)
            if output_size == input_size:
                print(f"✅ Copy successful: {output_size:,} bytes")
                return True
            else:
                print(f"❌ Copy size mismatch: {input_size} -> {output_size}")
                return False
        else:
            print("❌ Copy failed - output not created")
            return False
            
    except Exception as e:
        print(f"❌ Copy error: {e}")
        return False

def create_reliable_web_video(input_path, output_path):
    """Main function: Create reliable web-compatible video"""
    try:
        print(f"\n🚀 Creating reliable web video...")
        print(f"   Input: {input_path}")
        print(f"   Output: {output_path}")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Strategy 1: Try FFmpeg (best quality and compatibility)
        if check_ffmpeg_available():
            print("\n📹 Attempting FFmpeg conversion...")
            if create_web_compatible_video_ffmpeg(input_path, output_path):
                print("✅ FFmpeg conversion successful!")
                return True
            else:
                print("⚠️ FFmpeg conversion failed, trying OpenCV...")
        
        # Strategy 2: OpenCV with multiple codec fallbacks
        print("\n📹 Attempting OpenCV conversion...")
        if create_opencv_web_video(input_path, output_path):
            print("✅ OpenCV conversion successful!")
            return True
        else:
            print("⚠️ OpenCV conversion failed, copying original...")
        
        # Strategy 3: Simple copy as last resort
        print("\n📁 Copying original file...")
        if copy_video_with_validation(input_path, output_path):
            print("✅ File copy successful!")
            return True
        else:
            print("❌ All strategies failed!")
            return False
            
    except Exception as e:
        print(f"💥 Fatal error in video creation: {e}")
        return False

# Install FFmpeg helper function
def install_ffmpeg_instructions():
    """Provide FFmpeg installation instructions"""
    instructions = """
🔧 To install FFmpeg for better video compatibility:

Mac (using Homebrew):
    brew install ffmpeg

Mac (using MacPorts):
    sudo port install ffmpeg

Linux (Ubuntu/Debian):
    sudo apt update && sudo apt install ffmpeg

Linux (CentOS/RHEL):
    sudo yum install ffmpeg

Windows:
    Download from: https://ffmpeg.org/download.html
    """
    return instructions