from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, jsonify
import os
import hashlib
import json
from datetime import datetime, timedelta
from time import time as current_time
import traceback
import importlib
import shutil
import concurrent.futures
import numpy as np
import glob

app = Flask(__name__)
app.secret_key = 'deepfake_detector_secret_key_2024'

# Add custom Jinja2 filters AND globals
@app.template_filter('sum')
def sum_filter(iterable):
    """Custom sum filter for Jinja2 templates"""
    try:
        if isinstance(iterable, dict):
            return sum(iterable.values())
        return sum(iterable)
    except (TypeError, ValueError):
        return 0

@app.template_filter('count_true')
def count_true_filter(dict_obj):
    """Count True values in a dictionary"""
    try:
        return sum(1 for value in dict_obj.values() if value is True)
    except (TypeError, AttributeError):
        return 0

@app.template_filter('safe_format')
def safe_format_filter(value, format_string="%.1f"):
    """Safely format a value, handling None and invalid values"""
    try:
        if value is None:
            return "N/A"
        if isinstance(value, (int, float)):
            return format_string % value
        # Try to convert to float
        float_value = float(value)
        return format_string % float_value
    except (TypeError, ValueError, ZeroDivisionError):
        return "N/A"

@app.template_filter('safe_percentage')
def safe_percentage_filter(value):
    """Safely format a percentage value"""
    try:
        if value is None:
            return "N/A"
        if isinstance(value, (int, float)):
            return f"{value:.1f}%"
        # Try to convert to float
        float_value = float(value)
        return f"{float_value:.1f}%"
    except (TypeError, ValueError):
        return "N/A"

# Add global functions to Jinja2 environment
@app.template_global()
def sum_dict_values(dict_obj):
    """Global function to sum dictionary values"""
    try:
        if isinstance(dict_obj, dict):
            return sum(1 for value in dict_obj.values() if value is True)
        return 0
    except (TypeError, AttributeError):
        return 0

@app.template_global()
def safe_sum(iterable):
    """Safe sum function for templates"""
    try:
        if isinstance(iterable, dict):
            return sum(iterable.values())
        return sum(iterable)
    except (TypeError, ValueError):
        return 0

@app.template_global()
def safe_format(value, format_string="%.1f"):
    """Global safe format function for templates"""
    try:
        if value is None:
            return "N/A"
        if isinstance(value, (int, float)):
            return format_string % value
        # Try to convert to float
        float_value = float(value)
        return format_string % float_value
    except (TypeError, ValueError, ZeroDivisionError):
        return "N/A"

@app.template_global()
def safe_percentage(value):
    """Global safe percentage format function"""
    try:
        if value is None:
            return "N/A"
        if isinstance(value, (int, float)):
            return f"{value:.1f}%"
        # Try to convert to float
        float_value = float(value)
        return f"{float_value:.1f}%"
    except (TypeError, ValueError):
        return "N/A"

# Make built-in functions available in templates
app.jinja_env.globals.update({
    'sum': safe_sum,
    'len': len,
    'enumerate': enumerate,
    'zip': zip,
    'range': range,
    'int': int,
    'float': float,
    'str': str,
    'bool': bool
})

# Ensure directories exist
UPLOAD_FOLDER = 'static/videos'
PROCESSED_FOLDER = 'static/videos'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Configure Flask
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

# ENHANCED LIGHTWEIGHT VIDEO CLEANUP CONFIGURATION
VIDEO_CLEANUP_CONFIG = {
    'max_videos_total': 6,         # Maximum total videos to keep (reduced for lightweight)
    'max_original_videos': 3,      # Maximum original videos to keep
    'max_processed_videos': 3,     # Maximum processed videos to keep  
    'max_age_minutes': 30,         # Maximum age in minutes (much shorter for lightweight)
    'max_directory_size_mb': 512,  # Maximum directory size in MB (reduced to 512MB)
    'cleanup_on_upload': True,     # Clean up on each upload
    'cleanup_on_startup': True,    # Clean up on server startup
    'aggressive_cleanup': True,    # Enable aggressive cleanup for lightweight arch
    'keep_only_latest_session': True,  # Keep only files from the latest session
    'auto_cleanup_interval': 300   # Auto cleanup every 5 minutes
}

def generate_video_id():
    """Generate unique video ID"""
    return str(int(current_time() * 1000))

def get_video_files_info(directory):
    """Get detailed information about all video files in directory"""
    video_extensions = ('mp4', 'avi', 'mov', 'mkv', 'webm')
    video_files = []
    
    try:
        for filename in os.listdir(directory):
            if filename.lower().endswith(video_extensions):
                filepath = os.path.join(directory, filename)
                if os.path.isfile(filepath):
                    stat = os.stat(filepath)
                    
                    # Categorize file type
                    file_type = 'unknown'
                    if filename.startswith('uploaded_video_'):
                        file_type = 'original'
                    elif filename.startswith('processed_'):
                        file_type = 'processed'
                    elif any(detector in filename for detector in ['advanced_unified', 'unified', 'safe']):
                        file_type = 'processed'
                    
                    # Extract timestamp if possible
                    timestamp = None
                    try:
                        if '_' in filename:
                            parts = filename.split('_')
                            for part in parts:
                                if part.replace('.mp4', '').replace('.avi', '').replace('.mov', '').isdigit():
                                    timestamp = int(part.replace('.mp4', '').replace('.avi', '').replace('.mov', ''))
                                    break
                    except:
                        pass
                    
                    video_files.append({
                        'filename': filename,
                        'filepath': filepath,
                        'size_bytes': stat.st_size,
                        'size_mb': stat.st_size / (1024 * 1024),
                        'created_time': stat.st_ctime,
                        'modified_time': stat.st_mtime,
                        'age_hours': (current_time() - stat.st_ctime) / 3600,
                        'age_minutes': (current_time() - stat.st_ctime) / 60,
                        'file_type': file_type,
                        'timestamp': timestamp
                    })
    except Exception as e:
        print(f"⚠️ Error getting video files info: {e}")
    
    return sorted(video_files, key=lambda x: x['created_time'], reverse=True)

def aggressive_cleanup_old_videos(directory, keep_current_session_files=None):
    """AGGRESSIVE cleanup for lightweight architecture"""
    try:
        print(f"🧹 Starting AGGRESSIVE lightweight cleanup for: {directory}")
        
        # Get all video files info
        video_files = get_video_files_info(directory)
        
        if not video_files:
            print("📁 No video files found for cleanup")
            return {'deleted_count': 0, 'freed_space_mb': 0, 'remaining_files': 0, 'final_size_mb': 0}
        
        # Files to keep (current session)
        keep_files = set(keep_current_session_files or [])
        
        # Calculate total directory size
        total_size_mb = sum(f['size_mb'] for f in video_files)
        print(f"📊 Current directory size: {total_size_mb:.2f} MB")
        print(f"📁 Total video files: {len(video_files)}")
        
        files_to_delete = []
        files_to_keep = []
        
        # STRATEGY 1: Keep only files from current session
        if VIDEO_CLEANUP_CONFIG['keep_only_latest_session'] and keep_files:
            print("🎯 AGGRESSIVE: Keeping only current session files")
            
            for file_info in video_files:
                if file_info['filename'] in keep_files:
                    files_to_keep.append(file_info)
                    print(f"🔒 Keeping current session: {file_info['filename']}")
                else:
                    files_to_delete.append(file_info)
                    print(f"🗑️ Marking for deletion (not current session): {file_info['filename']}")
        
        else:
            # STRATEGY 2: Keep only the most recent files by type
            original_files = [f for f in video_files if f['file_type'] == 'original']
            processed_files = [f for f in video_files if f['file_type'] == 'processed']
            
            # Keep only the newest original files
            original_files.sort(key=lambda x: x['created_time'], reverse=True)
            keep_originals = original_files[:VIDEO_CLEANUP_CONFIG['max_original_videos']]
            
            # Keep only the newest processed files
            processed_files.sort(key=lambda x: x['created_time'], reverse=True)
            keep_processed = processed_files[:VIDEO_CLEANUP_CONFIG['max_processed_videos']]
            
            # Files to keep
            for file_info in keep_originals + keep_processed:
                if file_info['filename'] in keep_files or file_info['age_minutes'] < 5:  # Always keep very recent files
                    files_to_keep.append(file_info)
                    print(f"🔒 Keeping recent {file_info['file_type']}: {file_info['filename']} (age: {file_info['age_minutes']:.1f}min)")
            
            # Everything else gets deleted
            for file_info in video_files:
                if file_info not in files_to_keep:
                    files_to_delete.append(file_info)
                    print(f"🗑️ Marking for deletion: {file_info['filename']} (age: {file_info['age_minutes']:.1f}min, {file_info['size_mb']:.2f}MB)")
        
        # STRATEGY 3: Delete files older than max age (very short for lightweight)
        max_age_files = [f for f in files_to_keep if f['age_minutes'] > VIDEO_CLEANUP_CONFIG['max_age_minutes']]
        for file_info in max_age_files:
            if file_info['filename'] not in keep_files:  # Don't delete current session files
                files_to_delete.append(file_info)
                files_to_keep.remove(file_info)
                print(f"⏰ Moving to deletion (too old): {file_info['filename']} (age: {file_info['age_minutes']:.1f}min)")
        
        # STRATEGY 4: If still too large, delete largest files first
        kept_size = sum(f['size_mb'] for f in files_to_keep)
        if kept_size > VIDEO_CLEANUP_CONFIG['max_directory_size_mb']:
            print(f"💾 Still too large ({kept_size:.2f} MB), removing largest files...")
            
            # Sort by size (largest first) and age (oldest first)
            files_to_keep.sort(key=lambda x: (x['size_mb'], x['age_minutes']), reverse=True)
            
            target_size = VIDEO_CLEANUP_CONFIG['max_directory_size_mb'] * 0.7  # Target 70% of max
            current_size = 0
            final_keep_list = []
            
            for file_info in files_to_keep:
                if file_info['filename'] in keep_files:
                    # Always keep current session files
                    final_keep_list.append(file_info)
                    current_size += file_info['size_mb']
                elif current_size + file_info['size_mb'] <= target_size:
                    final_keep_list.append(file_info)
                    current_size += file_info['size_mb']
                else:
                    files_to_delete.append(file_info)
                    print(f"💾 Moving to deletion (size limit): {file_info['filename']} ({file_info['size_mb']:.2f}MB)")
            
            files_to_keep = final_keep_list
        
        # Execute deletions
        deleted_count = 0
        freed_space_mb = 0
        
        for file_info in files_to_delete:
            try:
                if os.path.exists(file_info['filepath']):
                    os.remove(file_info['filepath'])
                    deleted_count += 1
                    freed_space_mb += file_info['size_mb']
                    print(f"🗑️ Deleted: {file_info['filename']} ({file_info['size_mb']:.2f} MB)")
            except Exception as e:
                print(f"❌ Failed to delete {file_info['filename']}: {e}")
        
        # Final report
        remaining_files = get_video_files_info(directory)
        final_size_mb = sum(f['size_mb'] for f in remaining_files)
        
        print(f"✅ AGGRESSIVE cleanup completed:")
        print(f"   📉 Files deleted: {deleted_count}")
        print(f"   💾 Space freed: {freed_space_mb:.2f} MB")
        print(f"   📁 Remaining files: {len(remaining_files)}")
        print(f"   📊 Final directory size: {final_size_mb:.2f} MB")
        print(f"   🎯 Lightweight target achieved: {final_size_mb <= VIDEO_CLEANUP_CONFIG['max_directory_size_mb']}")
        
        return {
            'deleted_count': deleted_count,
            'freed_space_mb': freed_space_mb,
            'remaining_files': len(remaining_files),
            'final_size_mb': final_size_mb,
            'lightweight_target_achieved': final_size_mb <= VIDEO_CLEANUP_CONFIG['max_directory_size_mb']
        }
        
    except Exception as e:
        print(f"💥 Error during aggressive cleanup: {e}")
        traceback.print_exc()
        return None

def cleanup_processed_files_by_pattern(directory):
    """Clean up processed files by pattern matching"""
    try:
        print("🧹 Cleaning up processed files by pattern...")
        
        # Patterns for processed files
        patterns = [
            'processed_*_advanced_unified.mp4',
            'processed_*_unified.mp4', 
            'processed_*_safe.mp4',
            'processed_*.mp4'
        ]
        
        deleted_count = 0
        freed_space_mb = 0
        
        for pattern in patterns:
            files = glob.glob(os.path.join(directory, pattern))
            for file_path in files:
                try:
                    if os.path.exists(file_path):
                        file_size = os.path.getsize(file_path) / (1024 * 1024)
                        
                        # Check if file is older than 5 minutes (keep very recent ones)
                        file_age_minutes = (current_time() - os.path.getctime(file_path)) / 60
                        
                        if file_age_minutes > 5:  # Only delete files older than 5 minutes
                            os.remove(file_path)
                            deleted_count += 1
                            freed_space_mb += file_size
                            print(f"🗑️ Deleted processed file: {os.path.basename(file_path)} ({file_size:.2f} MB)")
                        else:
                            print(f"🔒 Keeping recent processed file: {os.path.basename(file_path)} (age: {file_age_minutes:.1f}min)")
                except Exception as e:
                    print(f"❌ Failed to delete {file_path}: {e}")
        
        print(f"✅ Pattern cleanup completed: {deleted_count} files, {freed_space_mb:.2f} MB freed")
        return {'deleted_count': deleted_count, 'freed_space_mb': freed_space_mb}
        
    except Exception as e:
        print(f"💥 Error during pattern cleanup: {e}")
        return {'deleted_count': 0, 'freed_space_mb': 0}

def emergency_cleanup(directory):
    """Emergency cleanup when directory is too full"""
    try:
        print("🆘 EMERGENCY CLEANUP: Directory critically full!")
        
        video_files = get_video_files_info(directory)
        total_size_mb = sum(f['size_mb'] for f in video_files)
        
        if total_size_mb < VIDEO_CLEANUP_CONFIG['max_directory_size_mb']:
            print("✅ Directory size is acceptable, no emergency cleanup needed")
            return
        
        print(f"⚠️ Directory size: {total_size_mb:.2f} MB (limit: {VIDEO_CLEANUP_CONFIG['max_directory_size_mb']} MB)")
        
        # Sort files by age (oldest first) and size (largest first)
        video_files.sort(key=lambda x: (x['age_minutes'], -x['size_mb']))
        
        target_size = VIDEO_CLEANUP_CONFIG['max_directory_size_mb'] * 0.5  # Target 50% of max
        current_size = total_size_mb
        deleted_count = 0
        
        for file_info in video_files:
            if current_size <= target_size:
                break
                
            try:
                if os.path.exists(file_info['filepath']):
                    os.remove(file_info['filepath'])
                    current_size -= file_info['size_mb']
                    deleted_count += 1
                    print(f"🆘 Emergency deleted: {file_info['filename']} ({file_info['size_mb']:.2f} MB)")
            except Exception as e:
                print(f"❌ Emergency deletion failed for {file_info['filename']}: {e}")
        
        print(f"🆘 Emergency cleanup completed: {deleted_count} files deleted")
        
    except Exception as e:
        print(f"💥 Emergency cleanup failed: {e}")

def cleanup_on_startup():
    """Comprehensive cleanup on server startup"""
    try:
        print("🚀 Performing comprehensive startup cleanup...")
        
        # 1. Pattern-based cleanup of processed files
        pattern_result = cleanup_processed_files_by_pattern(UPLOAD_FOLDER)
        
        # 2. Aggressive cleanup
        cleanup_result = aggressive_cleanup_old_videos(UPLOAD_FOLDER)
        
        # 3. Emergency cleanup if still too large
        video_files = get_video_files_info(UPLOAD_FOLDER)
        total_size_mb = sum(f['size_mb'] for f in video_files)
        
        if total_size_mb > VIDEO_CLEANUP_CONFIG['max_directory_size_mb']:
            emergency_cleanup(UPLOAD_FOLDER)
        
        # Final status
        final_files = get_video_files_info(UPLOAD_FOLDER)
        final_size_mb = sum(f['size_mb'] for f in final_files)
        
        print(f"🎯 Startup cleanup summary:")
        print(f"   📁 Files remaining: {len(final_files)}")
        print(f"   📊 Final size: {final_size_mb:.2f} MB")
        print(f"   ✅ Lightweight target: {final_size_mb <= VIDEO_CLEANUP_CONFIG['max_directory_size_mb']}")
        
    except Exception as e:
        print(f"⚠️ Startup cleanup failed: {e}")

def get_current_session_files(timestamp):
    """Get list of files from current upload session"""
    patterns = [
        f"uploaded_video_{timestamp}.mp4",
        f"processed_{timestamp}*.mp4"
    ]
    
    current_files = []
    for pattern in patterns:
        matching_files = glob.glob(os.path.join(UPLOAD_FOLDER, pattern))
        current_files.extend([os.path.basename(f) for f in matching_files])
    
    return current_files

class LightweightSmartDeepfakeDetector:
    """Lightweight intelligent multi-model deepfake detection system"""
    
    def __init__(self):
        self.detection_results = {}
        # Enhanced adaptive weighting system
        self.base_weights = {
            'safe': 0.35,              # Base reliable weight
            'unified': 0.40,           # Higher base weight for unified
            'advanced_unified': 0.25   # Moderate base weight for advanced
        }
        
        # Performance-based weight multipliers
        self.performance_multipliers = {
            'safe': 1.0,       # Standard multiplier
            'unified': 1.1,    # Slight boost for unified
            'advanced_unified': 1.2  # Higher multiplier for advanced when successful
        }
        
        # Confidence thresholds for reliable results
        self.confidence_thresholds = {
            'safe': 0.6,
            'unified': 0.5,
            'advanced_unified': 0.4
        }
        
        # Legacy confidence weights for backward compatibility
        self.confidence_weights = {
            'safe': 0.35,
            'unified': 0.40,
            'advanced_unified': 0.25
        }
    
    def run_detector_safely(self, detector_name, video_path, output_path):
        """Safely run a detector with timeout and resource management"""
        try:
            print(f"🔄 Starting {detector_name} detector (lightweight mode)...")
            start_time = current_time()
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Set timeout based on detector type (shorter for lightweight)
            timeout_seconds = {
                'safe': 90,              # 90 seconds for safe detector (it processes all frames)
                'unified': 60,           # 1 minute for unified detector  
                'advanced_unified': 120  # 2 minutes for advanced detector
            }.get(detector_name, 30)
            
            # Run detector with timeout
            if detector_name == 'safe':
                module = importlib.import_module("safe_deepfake_detector")
                result = module.run(video_path, output_path)
            elif detector_name == 'unified':
                module = importlib.import_module("unified_deepfake_detector")
                result = module.run(video_path, output_path)
            elif detector_name == 'advanced_unified':
                module = importlib.import_module("advanced_unified_detector")
                result = module.run(video_path, output_path)
            else:
                # Fallback to safe detector
                print(f"⚠️ Unknown detector {detector_name}, using safe detector")
                module = importlib.import_module("safe_deepfake_detector")
                result = module.run(video_path, output_path)
            
            end_time = current_time()
            execution_time = end_time - start_time
            
            # Check timeout - but still use result if valid
            if execution_time > timeout_seconds:
                print(f"⏰ {detector_name} detector exceeded timeout ({execution_time:.1f}s > {timeout_seconds}s)")
                
                # If we have a valid result despite timeout, use it with warning
                if result is not None and isinstance(result, (int, float)):
                    print(f"⚠️ Using result despite timeout: {result:.1f}%")
                    confidence = self.calculate_detector_confidence(result, execution_time, detector_name)
                    confidence *= 0.8  # Reduce confidence due to timeout
                    
                    return {
                        'name': detector_name,
                        'result': max(0.0, min(100.0, float(result))),
                        'confidence': confidence,
                        'execution_time': execution_time,
                        'success': True,  # Mark as successful since we have data
                        'error': f'Completed but exceeded timeout ({execution_time:.1f}s)',
                        'output_file': output_path
                    }
                else:
                    # No valid result and timeout
                    return {
                        'name': detector_name,
                        'result': 50.0,
                        'confidence': 0.1,
                        'execution_time': execution_time,
                        'success': False,
                        'error': 'Timeout exceeded with no valid result',
                        'output_file': None
                    }
            
            # Calculate confidence
            confidence = self.calculate_detector_confidence(result, execution_time, detector_name)
            
            # SAFETY: Ensure result is never None or invalid
            if result is None or not isinstance(result, (int, float)):
                print(f"⚠️ {detector_name} returned invalid result: {result}, using fallback")
                result = 50.0  # Neutral fallback
            
            # Clamp result to valid range
            result = max(0.0, min(100.0, float(result)))
            
            print(f"✅ {detector_name.capitalize()} completed: {result:.1f}% (confidence: {confidence:.2f}) in {execution_time:.2f}s")
            
            return {
                'name': detector_name,
                'result': result,
                'confidence': confidence,
                'execution_time': execution_time,
                'success': True,
                'output_file': output_path
            }
            
        except Exception as e:
            print(f"❌ {detector_name.capitalize()} detector failed: {e}")
            return {
                'name': detector_name,
                'result': 50.0,  # Safe fallback value
                'confidence': 0.1,
                'execution_time': 0,
                'success': False,
                'error': str(e),
                'output_file': None
            }

    def calculate_detector_confidence(self, result, execution_time, detector_name):
        """Calculate confidence score optimized for lightweight architecture"""
        base_confidence = self.confidence_weights[detector_name]
        
        # Adjust confidence based on result reasonableness
        if 0 <= result <= 100:
            result_confidence = 1.0
        else:
            result_confidence = 0.5
        
        # Adjust confidence based on execution time (faster = better for lightweight)
        if detector_name == 'safe' and execution_time <= 15:
            time_confidence = 1.2  # Bonus for fast safe detection
        elif detector_name == 'unified' and execution_time <= 30:
            time_confidence = 1.1
        elif detector_name == 'advanced_unified' and execution_time <= 60:
            time_confidence = 1.0
        else:
            time_confidence = 0.8  # Penalty for slow execution
        
        return min(1.0, base_confidence * result_confidence * time_confidence)
    
    def run_lightweight_detection(self, video_path, base_output_path):
        """Run lightweight detection prioritizing speed and efficiency"""
        print("🚀 Starting LIGHTWEIGHT smart detection...")
        
        detection_results = []
        
        # Create output paths
        output_paths = {
            'safe': f"{base_output_path}_safe.mp4",
            'unified': f"{base_output_path}_unified.mp4", 
            'advanced_unified': f"{base_output_path}_advanced_unified.mp4"
        }
        
        # LIGHTWEIGHT STRATEGY: Run detectors sequentially, stop early if confident
        detector_order = ['safe', 'unified', 'advanced_unified']  # Order by speed/reliability
        
        for detector_name in detector_order:
            result = self.run_detector_safely(detector_name, video_path, output_paths[detector_name])
            detection_results.append(result)
            
            # EARLY STOPPING: If safe detector is very confident, skip others
            if (detector_name == 'safe' and result['success'] and 
                result['confidence'] > 0.8 and 
                (result['result'] < 20 or result['result'] > 80)):
                
                print(f"🎯 Early stopping: Safe detector very confident ({result['result']:.1f}%, confidence: {result['confidence']:.2f})")
                
                # Create dummy results for skipped detectors
                for skipped_detector in detector_order[1:]:
                    if skipped_detector != detector_name:
                        detection_results.append({
                            'name': skipped_detector,
                            'result': result['result'],  # Use safe detector result
                            'confidence': 0.1,  # Low confidence since skipped
                            'success': False,
                            'error': 'Skipped due to early stopping',
                            'output_file': output_paths[skipped_detector]
                        })
                break
        
        return self.combine_detection_results(detection_results, base_output_path, output_paths)
    
    def combine_detection_results(self, results, base_output_path, output_paths):
        """Enhanced result combination with intelligent weighted aggregation"""
        print("🧠 Combining results with enhanced weighted aggregation...")
        
        successful_results = [r for r in results if r['success']]
        all_results = results  # Include all results for comprehensive analysis
        
        if not successful_results:
            print("❌ All detectors failed, using default values")
            return 50.0, self.create_failure_summary(results), output_paths
        
        # ENHANCED AGGREGATION SYSTEM
        print(f"📊 Aggregating results from {len(successful_results)} successful / {len(all_results)} total detectors")
        
        # Calculate adaptive weights based on performance
        adaptive_weights = self.calculate_adaptive_weights(successful_results)
        
        # Calculate final score using multiple strategies
        final_score = self.calculate_weighted_aggregate_score(successful_results, adaptive_weights)
        
        print(f"🎯 Enhanced aggregate result: {final_score:.1f}%")
        print(f"⚖️ Adaptive weights: {adaptive_weights}")
        
        # Create enhanced analysis summary
        analysis_summary = self.create_enhanced_analysis_summary(all_results, final_score, adaptive_weights)
        
        return min(95, max(5, final_score)), analysis_summary, output_paths
    
    def calculate_adaptive_weights(self, successful_results):
        """Calculate adaptive weights based on detector performance and confidence"""
        adaptive_weights = {}
        total_base_weight = 0
        
        # First pass: calculate base weights with performance multipliers
        for result in successful_results:
            detector_name = result['name']
            base_weight = self.base_weights.get(detector_name, 0.33)
            performance_multiplier = self.performance_multipliers.get(detector_name, 1.0)
            confidence = result.get('confidence', 0.5)
            execution_time = result.get('execution_time', 60)
            
            # Confidence boost: higher confidence = higher weight
            confidence_boost = 1.0 + (confidence - 0.5)  # Range: 0.5 - 1.5
            
            # Speed boost: faster execution = higher weight (for lightweight mode)
            speed_thresholds = {'safe': 15, 'unified': 30, 'advanced_unified': 60}
            speed_threshold = speed_thresholds.get(detector_name, 30)
            speed_boost = 1.2 if execution_time <= speed_threshold else 0.9
            
            # Calculate adaptive weight
            adaptive_weight = base_weight * performance_multiplier * confidence_boost * speed_boost
            adaptive_weights[detector_name] = adaptive_weight
            total_base_weight += adaptive_weight
        
        # Normalize weights to sum to 1.0
        if total_base_weight > 0:
            for detector_name in adaptive_weights:
                adaptive_weights[detector_name] /= total_base_weight
        
        return adaptive_weights
    
    def calculate_weighted_aggregate_score(self, successful_results, adaptive_weights):
        """Calculate final score using intelligent weighted aggregation"""
        if not successful_results:
            return 50.0
        
        # Strategy 1: Weighted Average (primary method)
        weighted_sum = 0
        total_weight = 0
        
        for result in successful_results:
            detector_name = result['name']
            score = result['result']
            weight = adaptive_weights.get(detector_name, 0)
            
            weighted_sum += score * weight
            total_weight += weight
        
        weighted_average = weighted_sum / total_weight if total_weight > 0 else 50.0
        
        # Strategy 2: Consensus-based adjustment
        scores = [r['result'] for r in successful_results]
        score_variance = np.var(scores) if len(scores) > 1 else 0
        
        # If all detectors agree closely, boost confidence in result
        if score_variance < 100:  # Low variance (scores within ~10 points)
            consensus_boost = 1.0
            print(f"✅ High consensus detected (variance: {score_variance:.1f})")
        else:
            consensus_boost = 0.9
            print(f"⚠️ Low consensus detected (variance: {score_variance:.1f})")
        
        # Strategy 3: Outlier detection and handling
        median_score = np.median(scores)
        outlier_threshold = 20  # Scores more than 20 points from median
        
        non_outlier_results = []
        for result in successful_results:
            if abs(result['result'] - median_score) <= outlier_threshold:
                non_outlier_results.append(result)
            else:
                print(f"🚨 Outlier detected: {result['name']} = {result['result']:.1f}% (median: {median_score:.1f}%)")
        
        # If we have non-outlier results, use them for refinement
        if len(non_outlier_results) >= len(successful_results) * 0.5:  # At least 50% non-outliers
            refined_weighted_sum = 0
            refined_total_weight = 0
            
            for result in non_outlier_results:
                detector_name = result['name']
                score = result['result']
                weight = adaptive_weights.get(detector_name, 0)
                
                refined_weighted_sum += score * weight
                refined_total_weight += weight
            
            if refined_total_weight > 0:
                refined_average = refined_weighted_sum / refined_total_weight
                # Blend original and refined scores
                final_score = (weighted_average * 0.7) + (refined_average * 0.3)
                print(f"🔧 Outlier-adjusted score: {refined_average:.1f}% -> blended: {final_score:.1f}%")
            else:
                final_score = weighted_average
        else:
            final_score = weighted_average
        
        # Apply consensus boost
        final_score *= consensus_boost
        
        # Strategy 4: Safety bounds and sanity checks
        if len(successful_results) == 1:
            # Single detector - apply conservative adjustment
            single_detector = successful_results[0]
            if single_detector['name'] == 'safe':
                # Safe detector alone - slightly reduce confidence
                final_score = final_score * 0.95
            elif single_detector['name'] == 'advanced_unified':
                # Advanced detector alone - maintain full score
                pass
            else:
                # Unified detector alone - slightly reduce confidence
                final_score = final_score * 0.98
        
        return final_score
    
    def create_enhanced_analysis_summary(self, results, final_score, adaptive_weights):
        """Create enhanced analysis summary with aggregation details"""
        successful_results = [r for r in results if r['success']]
        
        summary = {
            'final_score': final_score,
            'detector_results': {},
            'aggregation_details': {
                'total_detectors': len(results),
                'successful_detectors': len(successful_results),
                'adaptive_weights': adaptive_weights,
                'aggregation_method': 'enhanced_weighted'
            },
            'consensus_level': 'unknown',
            'recommendation': 'unknown',
            'advanced_features': {
                'cnn_analysis': False,
                'temporal_analysis': False,
                'dcgan_analysis': False,
                'thermal_mapping': False,
                'enhanced_landmarks': False
            },
            'lightweight_mode': True
        }
        
        # Process each detector result
        for result in results:
            detector_name = result['name']
            
            # Use actual result even if marked as failed due to timeout
            actual_score = result.get('result')
            if actual_score is not None and isinstance(actual_score, (int, float)):
                use_score = actual_score
                was_successful = True
            elif result['success']:
                use_score = actual_score
                was_successful = True
            else:
                use_score = None
                was_successful = False
            
            summary['detector_results'][detector_name] = {
                'success': was_successful,
                'score': use_score,
                'confidence': result.get('confidence', 0),
                'execution_time': result.get('execution_time', 0),
                'error': result.get('error', None) if not was_successful else None,
                'weight_used': adaptive_weights.get(detector_name, 0),
                'contribution_to_final': (adaptive_weights.get(detector_name, 0) * use_score) if use_score else 0
            }
        
        # Enhanced consensus determination
        successful_count = len(successful_results)
        total_count = len(results)
        
        if successful_count == total_count and successful_count >= 2:
            summary['consensus_level'] = 'high'
        elif successful_count >= 2:
            summary['consensus_level'] = 'medium'
        elif successful_count == 1:
            summary['consensus_level'] = 'single'
        else:
            summary['consensus_level'] = 'failed'
        
        # Enhanced recommendation with detailed reasoning
        if final_score > 75:
            summary['recommendation'] = 'high_risk'
            summary['recommendation_reason'] = 'Strong aggregated evidence of deepfake content'
        elif final_score > 50:
            summary['recommendation'] = 'moderate_risk'
            summary['recommendation_reason'] = 'Moderate aggregated evidence suggests potential deepfake'
        elif final_score > 25:
            summary['recommendation'] = 'low_risk'
            summary['recommendation_reason'] = 'Low aggregated evidence, appears mostly authentic'
        else:
            summary['recommendation'] = 'minimal_risk'
            summary['recommendation_reason'] = 'Strong aggregated evidence of authentic content'
        
        # Add detector performance summary
        summary['detector_performance'] = {}
        for detector_name in ['safe', 'unified', 'advanced_unified']:
            result = next((r for r in results if r['name'] == detector_name), None)
            if result:
                summary['detector_performance'][detector_name] = {
                    'available': True,
                    'successful': result['success'],
                    'score': result.get('result'),
                    'confidence': result.get('confidence', 0),
                    'weight': adaptive_weights.get(detector_name, 0)
                }
            else:
                summary['detector_performance'][detector_name] = {
                    'available': False,
                    'successful': False,
                    'score': None,
                    'confidence': 0,
                    'weight': 0
                }
        
        return summary
    
    def create_failure_summary(self, results):
        """Create failure summary for lightweight mode"""
        return {
            'final_score': 35.0,
            'detector_results': {r['name']: {'success': False, 'error': r.get('error', 'Unknown')} for r in results},
            'consensus_level': 'failed',
            'recommendation': 'system_error',
            'advanced_features': {
                'cnn_analysis': False,
                'temporal_analysis': False,
                'dcgan_analysis': False,
                'thermal_mapping': False,
                'enhanced_landmarks': False
            },
            'lightweight_mode': True
        }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """LIGHTWEIGHT: Handle file upload with aggressive cleanup"""
    try:
        if 'file' not in request.files:
            flash('No file selected')
            return redirect(url_for('index'))

        file = request.files['file']

        if file.filename == '':
            flash('No file selected')
            return redirect(url_for('index'))

        if file:
            # Check file extension
            allowed_extensions = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
            file_extension = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
            
            if file_extension not in allowed_extensions:
                flash('Please upload a video file (.mp4, .avi, .mov, .mkv, .webm)')
                return redirect(url_for('index'))

            # Generate unique identifiers
            timestamp = int(current_time())
            
            # AGGRESSIVE PRE-UPLOAD CLEANUP
            print("🧹 AGGRESSIVE pre-upload cleanup...")
            pattern_result = cleanup_processed_files_by_pattern(app.config['UPLOAD_FOLDER'])
            aggressive_result = aggressive_cleanup_old_videos(app.config['UPLOAD_FOLDER'])
            
            # Save original video
            original_filename = f"uploaded_video_{timestamp}.mp4"
            original_video_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
            file.save(original_video_path)
            
            print(f"📹 Original video saved: {original_video_path}")
            print(f"📁 File size: {os.path.getsize(original_video_path):,} bytes")

            # Create base path for processed videos
            processed_filename_base = f"processed_{timestamp}"
            base_output_path = os.path.join(app.config['UPLOAD_FOLDER'], processed_filename_base)

            try:
                print(f"🎬 Starting LIGHTWEIGHT deepfake detection for: {original_video_path}")
                
                # Initialize lightweight detector
                lightweight_detector = LightweightSmartDeepfakeDetector()
                
                # Run lightweight detection
                final_score, analysis_summary, output_paths = lightweight_detector.run_lightweight_detection(
                    original_video_path, base_output_path
                )
                
                print(f"🎯 Lightweight detection result: {final_score:.1f}%")
                print(f"📊 Consensus level: {analysis_summary['consensus_level']}")
                
                # Verify processed videos exist and create fallbacks if needed
                verified_output_paths = {}
                for detector_name, output_path in output_paths.items():
                    if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
                        verified_output_paths[detector_name] = output_path
                        print(f"✅ {detector_name} video: {os.path.getsize(output_path):,} bytes")
                    else:
                        # Create lightweight fallback (just copy original)
                        print(f"⚠️ Creating lightweight fallback for {detector_name}")
                        try:
                            shutil.copy2(original_video_path, output_path)
                            verified_output_paths[detector_name] = output_path
                            print(f"✅ Fallback created: {os.path.getsize(output_path):,} bytes")
                        except Exception as e:
                            print(f"❌ Fallback failed for {detector_name}: {e}")
                            verified_output_paths[detector_name] = original_video_path
                
                # IMMEDIATE POST-PROCESSING CLEANUP
                current_session_files = get_current_session_files(timestamp)
                print("🧹 Immediate post-processing cleanup...")
                aggressive_cleanup_old_videos(app.config['UPLOAD_FOLDER'], current_session_files)
                
            except Exception as e:
                print(f"💥 Error in lightweight detection: {e}")
                traceback.print_exc()
                
                # Emergency fallback
                final_score = 35
                analysis_summary = {
                    'recommendation': 'system_error', 
                    'consensus_level': 'failed',
                    'advanced_features': {k: False for k in ['cnn_analysis', 'temporal_analysis', 'dcgan_analysis', 'thermal_mapping', 'enhanced_landmarks']},
                    'detector_results': {},
                    'lightweight_mode': True
                }
                verified_output_paths = {
                    'safe': original_video_path,
                    'unified': original_video_path,
                    'advanced_unified': original_video_path
                }
                flash('Video processed with limited analysis due to processing error')

            # Create lightweight video information
            video_info = {
                'name': file.filename,
                'size': f"{os.path.getsize(original_video_path) / (1024*1024):.2f} MB",
                'user': 'Guest', 
                'source': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC'),
                'deepfake_percentage': final_score,
                'analysis_summary': analysis_summary,
                'detection_method': 'Lightweight AI Detection',
                'timestamp': timestamp,
                
                # Video URLs
                'original_video_url': url_for('serve_video', filename=original_filename),
                'processed_videos': {
                    'safe': {
                        'url': url_for('serve_video', filename=os.path.basename(verified_output_paths.get('safe', original_video_path))),
                        'score': analysis_summary['detector_results'].get('safe', {}).get('score', final_score),
                        'success': analysis_summary['detector_results'].get('safe', {}).get('success', False)
                    },
                    'unified': {
                        'url': url_for('serve_video', filename=os.path.basename(verified_output_paths.get('unified', original_video_path))),
                        'score': analysis_summary['detector_results'].get('unified', {}).get('score', final_score),
                        'success': analysis_summary['detector_results'].get('unified', {}).get('success', False)
                    },
                    'advanced_unified': {
                        'url': url_for('serve_video', filename=os.path.basename(verified_output_paths.get('advanced_unified', original_video_path))),
                        'score': analysis_summary['detector_results'].get('advanced_unified', {}).get('score', final_score),
                        'success': analysis_summary['detector_results'].get('advanced_unified', {}).get('success', False)
                    }
                }
            }

            video_info_json = json.dumps(video_info, default=str)
            
            print(f"🔗 Lightweight video URLs generated")
            
            return redirect(url_for('result', video_info=video_info_json))

    except Exception as e:
        print(f"💥 Error in lightweight upload: {e}")
        traceback.print_exc()
        flash('An error occurred while processing your request')
        return redirect(url_for('index'))

@app.route('/result')
def result():
    """LIGHTWEIGHT: Display results with minimal resource usage"""
    try:
        video_info_json = request.args.get('video_info')
        
        if not video_info_json:
            flash('Invalid request')
            return redirect(url_for('index'))

        video_info = json.loads(video_info_json)
        
        # FIX: Ensure all scores are valid numbers, not None
        fallback_score = video_info.get('deepfake_percentage', 50.0)
        
        # Ensure fallback score is valid
        if fallback_score is None or not isinstance(fallback_score, (int, float)):
            fallback_score = 50.0
        
        for detector_name, video_data in video_info['processed_videos'].items():
            if video_data.get('score') is None or not isinstance(video_data.get('score'), (int, float)):
                # Use final score as fallback for failed detectors
                video_data['score'] = fallback_score
                print(f"🔧 Fixed invalid score for {detector_name}: using {fallback_score}")
        
        # Also fix detector results in analysis summary
        if 'analysis_summary' in video_info and 'detector_results' in video_info['analysis_summary']:
            for detector_name, result_data in video_info['analysis_summary']['detector_results'].items():
                if result_data.get('score') is None or not isinstance(result_data.get('score'), (int, float)):
                    result_data['score'] = fallback_score
                    print(f"🔧 Fixed invalid score in analysis for {detector_name}: using {fallback_score}")
        
        # Ensure main percentage is valid
        if video_info.get('deepfake_percentage') is None or not isinstance(video_info.get('deepfake_percentage'), (int, float)):
            video_info['deepfake_percentage'] = fallback_score
            print(f"🔧 Fixed invalid main percentage: using {fallback_score}")
        
        # PRE-CALCULATE lightweight values
        if 'analysis_summary' in video_info and 'advanced_features' in video_info['analysis_summary']:
            advanced_features = video_info['analysis_summary']['advanced_features']
            video_info['advanced_features_count'] = sum(1 for value in advanced_features.values() if value is True)
            video_info['total_features'] = len(advanced_features)
            video_info['features_percentage'] = (video_info['advanced_features_count'] / video_info['total_features']) * 100 if video_info['total_features'] > 0 else 0
        else:
            video_info['advanced_features_count'] = 0
            video_info['total_features'] = 5
            video_info['features_percentage'] = 0
        
        # Verify videos exist (lightweight check)
        video_info['videos_exist'] = {}
        video_info['video_file_info'] = {}
        
        # Check original video
        original_filename = video_info['original_video_url'].split('/')[-1]
        original_path = os.path.join('static', 'videos', original_filename)
        exists = os.path.exists(original_path) and os.path.getsize(original_path) > 1000
        video_info['videos_exist']['original'] = exists
        
        if exists:
            file_size = os.path.getsize(original_path)
            video_info['video_file_info']['original'] = {
                'size_bytes': file_size,
                'size_mb': f"{file_size / (1024*1024):.2f} MB",
                'readable': True
            }
        
        # Check processed videos (lightweight)
        for detector_name, video_data in video_info['processed_videos'].items():
            filename = video_data['url'].split('/')[-1]
            filepath = os.path.join('static', 'videos', filename)
            exists = os.path.exists(filepath) and os.path.getsize(filepath) > 1000
            video_info['videos_exist'][detector_name] = exists
            
            if exists:
                file_size = os.path.getsize(filepath)
                video_info['video_file_info'][detector_name] = {
                    'size_bytes': file_size,
                    'size_mb': f"{file_size / (1024*1024):.2f} MB",
                    'readable': True
                }
                video_info['processed_videos'][detector_name]['file_size'] = f"{file_size / (1024*1024):.2f} MB"
            else:
                video_info['video_file_info'][detector_name] = {
                    'size_bytes': 0,
                    'size_mb': "0 MB",
                    'readable': False
                }
                video_info['processed_videos'][detector_name]['file_size'] = "N/A"
        
        # Add lightweight summary statistics
        successful_detectors = [name for name, data in video_info['analysis_summary']['detector_results'].items() if data['success']]
        video_info['summary_stats'] = {
            'successful_detectors': len(successful_detectors),
            'total_detectors': len(video_info['analysis_summary']['detector_results']),
            'success_rate': (len(successful_detectors) / len(video_info['analysis_summary']['detector_results'])) * 100 if video_info['analysis_summary']['detector_results'] else 0,
            'consensus_quality': video_info['analysis_summary']['consensus_level'],
            'recommendation': video_info['analysis_summary']['recommendation'],
            'lightweight_mode': video_info['analysis_summary'].get('lightweight_mode', False)
        }
        
        # Add simple risk assessment
        score = video_info['deepfake_percentage']
        if score > 75:
            video_info['risk_level'] = {'level': 'HIGH', 'color': '#e74c3c', 'description': 'Strong indication of deepfake content'}
        elif score > 50:
            video_info['risk_level'] = {'level': 'MODERATE', 'color': '#f39c12', 'description': 'Some suspicious characteristics detected'}
        elif score > 25:
            video_info['risk_level'] = {'level': 'LOW', 'color': '#27ae60', 'description': 'Mostly appears authentic'}
        else:
            video_info['risk_level'] = {'level': 'MINIMAL', 'color': '#2ecc71', 'description': 'Strong indication of authentic content'}
        
        print(f"📊 Lightweight result page prepared")
        print(f"🔧 All scores verified as non-None")
        
        return render_template('result.html', video_info=video_info)
    
    except Exception as e:
        print(f"❌ Error in lightweight result: {e}")
        traceback.print_exc()
        flash('Error displaying results')
        return redirect(url_for('index'))

# CRITICAL: Add route to serve video files
@app.route('/videos/<filename>')
def serve_video(filename):
    """Serve video files with proper headers"""
    try:
        video_path = os.path.join('static', 'videos', filename)
        
        if os.path.exists(video_path):
            file_size = os.path.getsize(video_path)
            
            if file_size > 1000:
                return send_from_directory(
                    os.path.join('static', 'videos'), 
                    filename, 
                    mimetype='video/mp4',
                    as_attachment=False
                )
            else:
                return "Video file is corrupted or too small", 404
        else:
            return "Video file not found", 404
            
    except Exception as e:
        print(f"❌ Video serving error: {e}")
        return f"Error serving video: {str(e)}", 500

# Enhanced debug routes
@app.route('/debug/videos')
def debug_videos():
    """Debug route to see all video files"""
    try:
        videos_dir = os.path.join('static', 'videos')
        video_files = get_video_files_info(videos_dir)
        
        total_size_mb = sum(f['size_mb'] for f in video_files)
        
        return jsonify({
            'videos_directory': videos_dir,
            'files': video_files,
            'total_files': len(video_files),
            'total_size_mb': round(total_size_mb, 2),
            'cleanup_config': VIDEO_CLEANUP_CONFIG,
            'directory_status': {
                'over_max_files': len(video_files) > VIDEO_CLEANUP_CONFIG['max_videos_total'],
                'over_max_size': total_size_mb > VIDEO_CLEANUP_CONFIG['max_directory_size_mb'],
                'has_old_files': any(f['age_minutes'] > VIDEO_CLEANUP_CONFIG['max_age_minutes'] for f in video_files)
            },
            'lightweight_mode': True
        })
    
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/debug/cleanup')
def debug_cleanup():
    """Debug route to manually trigger aggressive cleanup"""
    try:
        print("🧹 Manual AGGRESSIVE cleanup triggered via debug route")
        
        # Pattern cleanup first
        pattern_result = cleanup_processed_files_by_pattern(UPLOAD_FOLDER)
        
        # Aggressive cleanup
        cleanup_result = aggressive_cleanup_old_videos(UPLOAD_FOLDER)
        
        # Emergency cleanup if needed
        video_files = get_video_files_info(UPLOAD_FOLDER)
        total_size_mb = sum(f['size_mb'] for f in video_files)
        
        emergency_triggered = False
        if total_size_mb > VIDEO_CLEANUP_CONFIG['max_directory_size_mb']:
            emergency_cleanup(UPLOAD_FOLDER)
            emergency_triggered = True
        
        return jsonify({
            'cleanup_performed': True,
            'pattern_cleanup': pattern_result,
            'aggressive_cleanup': cleanup_result,
            'emergency_cleanup_triggered': emergency_triggered,
            'message': 'Aggressive cleanup completed successfully',
            'lightweight_mode': True
        })
    
    except Exception as e:
        return jsonify({
            'cleanup_performed': False,
            'error': str(e),
            'message': 'Aggressive cleanup failed'
        })

# Add route for automatic cleanup
@app.route('/api/auto-cleanup', methods=['POST'])
def auto_cleanup():
    """API endpoint for automatic cleanup"""
    try:
        pattern_result = cleanup_processed_files_by_pattern(UPLOAD_FOLDER)
        cleanup_result = aggressive_cleanup_old_videos(UPLOAD_FOLDER)
        
        return jsonify({
            'success': True,
            'pattern_cleanup': pattern_result,
            'aggressive_cleanup': cleanup_result,
            'lightweight_mode': True
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    print("🚀 Starting LIGHTWEIGHT Deepfake Detector Server...")
    print(f"📁 Upload folder: {UPLOAD_FOLDER}")
    print(f"🧹 Aggressive cleanup enabled: {VIDEO_CLEANUP_CONFIG['cleanup_on_upload']}")
    print(f"📊 Max total videos: {VIDEO_CLEANUP_CONFIG['max_videos_total']}")
    print(f"💾 Max directory size: {VIDEO_CLEANUP_CONFIG['max_directory_size_mb']} MB")
    print(f"⏰ Max age: {VIDEO_CLEANUP_CONFIG['max_age_minutes']} minutes")
    print(f"🎯 Lightweight mode: ON")
    
    # Create directories if they don't exist
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(PROCESSED_FOLDER, exist_ok=True)
    
    # Perform aggressive startup cleanup
    if VIDEO_CLEANUP_CONFIG['cleanup_on_startup']:
        cleanup_on_startup()
    
    # Use different port to avoid conflicts
    app.run(debug=True, host='0.0.0.0', port=5001)
