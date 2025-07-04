from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, jsonify
import os
import hashlib
import json
from datetime import datetime
from time import time as current_time
import traceback
import importlib
import shutil
import concurrent.futures
import numpy as np

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

def generate_video_id():
    """Generate unique video ID"""
    return str(int(current_time() * 1000))

class AdvancedSmartDeepfakeDetector:
    """Advanced intelligent multi-model deepfake detection system"""
    
    def __init__(self):
        self.detection_results = {}
        # Updated confidence weights - prioritize advanced unified detector
        self.confidence_weights = {
            'advanced_unified': 0.60,   # Highest weight for advanced unified detector
            'unified': 0.25,            # Medium weight for unified detector
            'safe': 0.15               # Lower weight for safe detector
        }
    
    def run_detector_safely(self, detector_name, video_path, output_path):
        """Safely run a detector and return results"""
        try:
            print(f"🔄 Starting {detector_name} detector...")
            start_time = current_time()
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            if detector_name == 'advanced_unified':
                # Use the new advanced unified detector
                module = importlib.import_module("advanced_unified_detector")
                result = module.run(video_path, output_path)
            elif detector_name == 'unified':
                # Use the unified detector
                module = importlib.import_module("unified_deepfake_detector")
                result = module.run(video_path, output_path)
            elif detector_name == 'safe':
                # Use the safe detector
                module = importlib.import_module("safe_deepfake_detector")
                result = module.run(video_path, output_path)
            else:
                # Fallback to safe detector
                print(f"⚠️ Unknown detector {detector_name}, using safe detector")
                module = importlib.import_module("safe_deepfake_detector")
                result = module.run(video_path, output_path)
            
            end_time = current_time()
            execution_time = end_time - start_time
            
            # Calculate confidence
            confidence = self.calculate_detector_confidence(result, execution_time, detector_name)
            
            print(f"✅ {detector_name.capitalize()} detector completed: {result}% (confidence: {confidence:.2f}) in {execution_time:.2f}s")
            
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
                'result': 50.0,
                'confidence': 0.1,
                'execution_time': 0,
                'success': False,
                'error': str(e),
                'output_file': None
            }

    def calculate_detector_confidence(self, result, execution_time, detector_name):
        """Calculate confidence score for detector result"""
        base_confidence = self.confidence_weights[detector_name]
        
        # Adjust confidence based on result reasonableness
        if 0 <= result <= 100:
            result_confidence = 1.0
        else:
            result_confidence = 0.5
        
        # Adjust confidence based on execution time
        if detector_name == 'advanced_unified' and 30 <= execution_time <= 300:
            time_confidence = 1.0
        elif detector_name == 'unified' and 15 <= execution_time <= 180:
            time_confidence = 1.0
        elif detector_name == 'safe' and 5 <= execution_time <= 60:
            time_confidence = 1.0
        else:
            time_confidence = 0.7
        
        return base_confidence * result_confidence * time_confidence
    
    def run_parallel_detection(self, video_path, base_output_path):
        """Run selected detectors in parallel with proper file management"""
        print("🚀 Starting advanced smart parallel detection...")
        
        detection_results = []
        
        # Create output paths for each detector
        output_paths = {
            'advanced_unified': f"{base_output_path}_advanced_unified.mp4",
            'unified': f"{base_output_path}_unified.mp4", 
            'safe': f"{base_output_path}_safe.mp4"
        }
        
        # Use advanced detectors with fallbacks
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(self.run_detector_safely, 'advanced_unified', video_path, output_paths['advanced_unified']): 'advanced_unified',
                executor.submit(self.run_detector_safely, 'unified', video_path, output_paths['unified']): 'unified',
                executor.submit(self.run_detector_safely, 'safe', video_path, output_paths['safe']): 'safe'
            }
            
            # Collect results with longer timeout for advanced models
            for future in concurrent.futures.as_completed(futures, timeout=360):  # 6 minutes
                try:
                    result = future.result(timeout=60)  # 1 minute per result
                    detection_results.append(result)
                except concurrent.futures.TimeoutError:
                    detector_name = futures[future]
                    print(f"⏰ {detector_name.capitalize()} detector timed out")
                    detection_results.append({
                        'name': detector_name,
                        'result': 45.0,
                        'confidence': 0.1,
                        'success': False,
                        'error': 'Timeout',
                        'output_file': output_paths[detector_name]
                    })
                except Exception as e:
                    detector_name = futures[future]
                    print(f"💥 {detector_name.capitalize()} detector exception: {e}")
                    detection_results.append({
                        'name': detector_name,
                        'result': 45.0,
                        'confidence': 0.1,
                        'success': False,
                        'error': str(e),
                        'output_file': output_paths[detector_name]
                    })
        
        return self.combine_detection_results(detection_results, base_output_path, output_paths)
    
    def combine_detection_results(self, results, base_output_path, output_paths):
        """Intelligently combine results from multiple detectors"""
        print("🧠 Combining detection results using advanced intelligent weighting...")
        
        successful_results = [r for r in results if r['success']]
        
        if not successful_results:
            print("❌ All detectors failed, using default values")
            return 35.0, self.create_failure_summary(results), output_paths
        
        # Calculate weighted average with emphasis on advanced models
        total_weighted_score = 0
        total_confidence = 0
        
        for result in successful_results:
            weight = result['confidence']
            
            # Bonus weight for advanced detector
            if result['name'] == 'advanced_unified':
                weight *= 1.5  # 50% bonus for advanced unified
            
            total_weighted_score += result['result'] * weight
            total_confidence += weight
        
        if total_confidence > 0:
            final_score = total_weighted_score / total_confidence
        else:
            final_score = np.mean([r['result'] for r in successful_results])
        
        # Apply consensus adjustment
        scores = [r['result'] for r in successful_results]
        if len(scores) >= 2:
            score_std = np.std(scores)
            if score_std > 25:
                # If high variance, trust the advanced model more
                advanced_results = [r for r in successful_results if r['name'] == 'advanced_unified']
                if advanced_results:
                    final_score = 0.7 * advanced_results[0]['result'] + 0.3 * np.median(scores)
                    print(f"⚠️ High variance detected ({score_std:.1f}), emphasizing advanced model")
                else:
                    final_score = np.median(scores)
                    print(f"⚠️ High variance detected ({score_std:.1f}), using median: {final_score:.1f}")
        
        # Create detailed analysis summary
        analysis_summary = self.create_advanced_analysis_summary(results, final_score)
        
        print(f"🎯 Final advanced combined result: {final_score:.1f}%")
        print(f"📊 Based on {len(successful_results)}/{len(results)} successful detections")
        
        return min(95, max(5, final_score)), analysis_summary, output_paths
    
    def create_advanced_analysis_summary(self, results, final_score):
        """Create detailed advanced analysis summary"""
        summary = {
            'final_score': final_score,
            'detector_results': {},
            'consensus_level': 'unknown',
            'recommendation': 'unknown',
            'advanced_features': {
                'cnn_analysis': False,
                'temporal_analysis': False,
                'dcgan_analysis': False,
                'thermal_mapping': False,
                'enhanced_landmarks': False
            }
        }
        
        successful_count = 0
        successful_scores = []
        has_advanced = False
        
        for result in results:
            detector_name = result['name']
            summary['detector_results'][detector_name] = {
                'success': result['success'],
                'score': result['result'] if result['success'] else None,
                'confidence': result.get('confidence', 0),
                'execution_time': result.get('execution_time', 0),
                'error': result.get('error', None) if not result['success'] else None
            }
            
            if result['success']:
                successful_count += 1
                successful_scores.append(result['result'])
                
                # Check for advanced features
                if detector_name == 'advanced_unified':
                    has_advanced = True
                    summary['advanced_features'] = {
                        'cnn_analysis': True,
                        'temporal_analysis': True,
                        'dcgan_analysis': True,
                        'thermal_mapping': True,
                        'enhanced_landmarks': True
                    }
        
        # Determine consensus level
        if successful_count >= 2:
            score_variance = np.var(successful_scores) if len(successful_scores) > 1 else 0
            if has_advanced and score_variance < 100:
                summary['consensus_level'] = 'high'
            elif score_variance < 200:
                summary['consensus_level'] = 'medium'
            else:
                summary['consensus_level'] = 'low'
        elif successful_count == 1:
            if has_advanced:
                summary['consensus_level'] = 'single_advanced'
            else:
                summary['consensus_level'] = 'single'
        else:
            summary['consensus_level'] = 'failed'
        
        # Generate advanced recommendation
        if final_score > 75:
            summary['recommendation'] = 'high_risk'
        elif final_score > 50:
            summary['recommendation'] = 'moderate_risk'
        elif final_score > 25:
            summary['recommendation'] = 'low_risk'
        else:
            summary['recommendation'] = 'authentic'
        
        return summary
    
    def create_failure_summary(self, results):
        """Create summary when all detectors fail"""
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
            }
        }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """FIXED: Handle file upload with proper video management"""
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
                print(f"🎬 Starting advanced intelligent deepfake detection for: {original_video_path}")
                
                # Initialize advanced smart detector
                smart_detector = AdvancedSmartDeepfakeDetector()
                
                # Run parallel detection with advanced models
                final_score, analysis_summary, output_paths = smart_detector.run_parallel_detection(
                    original_video_path, base_output_path
                )
                
                print(f"🎯 Advanced smart detection result: {final_score:.1f}%")
                print(f"📊 Consensus level: {analysis_summary['consensus_level']}")
                print(f"🔬 Advanced features used: {sum(analysis_summary['advanced_features'].values())}/5")
                
                # Verify processed videos exist and create fallbacks if needed
                verified_output_paths = {}
                for detector_name, output_path in output_paths.items():
                    if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
                        verified_output_paths[detector_name] = output_path
                        print(f"✅ {detector_name} video: {os.path.getsize(output_path):,} bytes")
                    else:
                        # Create fallback by copying original
                        print(f"⚠️ Creating fallback for {detector_name}")
                        try:
                            shutil.copy2(original_video_path, output_path)
                            verified_output_paths[detector_name] = output_path
                            print(f"✅ Fallback created: {os.path.getsize(output_path):,} bytes")
                        except Exception as e:
                            print(f"❌ Fallback failed for {detector_name}: {e}")
                            verified_output_paths[detector_name] = original_video_path
                
            except Exception as e:
                print(f"💥 Error in advanced deepfake detection: {e}")
                traceback.print_exc()
                
                # Emergency fallback
                final_score = 35
                analysis_summary = {
                    'recommendation': 'system_error', 
                    'consensus_level': 'failed',
                    'advanced_features': {k: False for k in ['cnn_analysis', 'temporal_analysis', 'dcgan_analysis', 'thermal_mapping', 'enhanced_landmarks']},
                    'detector_results': {}
                }
                verified_output_paths = {
                    'advanced_unified': original_video_path,
                    'unified': original_video_path,
                    'safe': original_video_path
                }
                flash('Video processed with limited analysis due to processing error')

            # Create comprehensive video information
            video_info = {
                'name': file.filename,
                'size': f"{os.path.getsize(original_video_path) / (1024*1024):.2f} MB",
                'user': 'Guest', 
                'source': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC'),
                'deepfake_percentage': final_score,
                'analysis_summary': analysis_summary,
                'detection_method': 'Advanced Multi-Model AI Detection',
                
                # Video URLs
                'original_video_url': url_for('serve_video', filename=original_filename),
                'processed_videos': {
                    'advanced_unified': {
                        'url': url_for('serve_video', filename=os.path.basename(verified_output_paths.get('advanced_unified', original_video_path))),
                        'score': analysis_summary['detector_results'].get('advanced_unified', {}).get('score', final_score),
                        'success': analysis_summary['detector_results'].get('advanced_unified', {}).get('success', False)
                    },
                    'unified': {
                        'url': url_for('serve_video', filename=os.path.basename(verified_output_paths.get('unified', original_video_path))),
                        'score': analysis_summary['detector_results'].get('unified', {}).get('score', final_score),
                        'success': analysis_summary['detector_results'].get('unified', {}).get('success', False)
                    },
                    'safe': {
                        'url': url_for('serve_video', filename=os.path.basename(verified_output_paths.get('safe', original_video_path))),
                        'score': analysis_summary['detector_results'].get('safe', {}).get('score', final_score),
                        'success': analysis_summary['detector_results'].get('safe', {}).get('success', False)
                    }
                }
            }

            video_info_json = json.dumps(video_info, default=str)
            
            print(f"🔗 Video URLs generated:")
            print(f"   Original: {video_info['original_video_url']}")
            for detector, info in video_info['processed_videos'].items():
                print(f"   {detector}: {info['url']} (score: {info['score']:.1f}%)")

            return redirect(url_for('result', video_info=video_info_json))

    except Exception as e:
        print(f"💥 Error in upload_file: {e}")
        traceback.print_exc()
        flash('An error occurred while processing your request')
        return redirect(url_for('index'))

@app.route('/result')
def result():
    """ENHANCED: Display results with pre-calculated values"""
    try:
        video_info_json = request.args.get('video_info')
        
        if not video_info_json:
            flash('Invalid request')
            return redirect(url_for('index'))

        video_info = json.loads(video_info_json)
        
        # PRE-CALCULATE values to avoid template issues
        if 'analysis_summary' in video_info and 'advanced_features' in video_info['analysis_summary']:
            advanced_features = video_info['analysis_summary']['advanced_features']
            
            # Count active features
            video_info['advanced_features_count'] = sum(1 for value in advanced_features.values() if value is True)
            video_info['total_features'] = len(advanced_features)
            
            # Calculate feature percentage
            if video_info['total_features'] > 0:
                video_info['features_percentage'] = (video_info['advanced_features_count'] / video_info['total_features']) * 100
            else:
                video_info['features_percentage'] = 0
        else:
            video_info['advanced_features_count'] = 0
            video_info['total_features'] = 5
            video_info['features_percentage'] = 0
        
        # Verify videos exist and add file info
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
        
        # Check processed videos
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
                
                # Add file size to processed video info
                video_info['processed_videos'][detector_name]['file_size'] = f"{file_size / (1024*1024):.2f} MB"
            else:
                video_info['video_file_info'][detector_name] = {
                    'size_bytes': 0,
                    'size_mb': "0 MB",
                    'readable': False
                }
                video_info['processed_videos'][detector_name]['file_size'] = "N/A"
        
        # Add summary statistics
        successful_detectors = [name for name, data in video_info['analysis_summary']['detector_results'].items() if data['success']]
        video_info['summary_stats'] = {
            'successful_detectors': len(successful_detectors),
            'total_detectors': len(video_info['analysis_summary']['detector_results']),
            'success_rate': (len(successful_detectors) / len(video_info['analysis_summary']['detector_results'])) * 100 if video_info['analysis_summary']['detector_results'] else 0,
            'consensus_quality': video_info['analysis_summary']['consensus_level'],
            'recommendation': video_info['analysis_summary']['recommendation']
        }
        
        # Add risk assessment
        score = video_info['deepfake_percentage']
        if score > 75:
            video_info['risk_level'] = {'level': 'HIGH', 'color': '#e74c3c', 'description': 'Strong indication of deepfake content'}
        elif score > 50:
            video_info['risk_level'] = {'level': 'MODERATE', 'color': '#f39c12', 'description': 'Some suspicious characteristics detected'}
        elif score > 25:
            video_info['risk_level'] = {'level': 'LOW', 'color': '#27ae60', 'description': 'Mostly appears authentic'}
        else:
            video_info['risk_level'] = {'level': 'MINIMAL', 'color': '#2ecc71', 'description': 'Strong indication of authentic content'}
        
        print(f"📊 Result page - Enhanced data prepared:")
        print(f"   Videos exist: {video_info['videos_exist']}")
        print(f"   Advanced features: {video_info['advanced_features_count']}/{video_info['total_features']}")
        print(f"   Success rate: {video_info['summary_stats']['success_rate']:.1f}%")
        print(f"   Risk level: {video_info['risk_level']['level']}")
        
        return render_template('result.html', video_info=video_info)
    
    except Exception as e:
        print(f"❌ Error in result: {e}")
        traceback.print_exc()
        flash('Error displaying results')
        return redirect(url_for('index'))

# CRITICAL: Add route to serve video files
@app.route('/videos/<filename>')
def serve_video(filename):
    """Serve video files with proper headers"""
    try:
        video_path = os.path.join('static', 'videos', filename)
        
        print(f"🎬 Serving video: {filename}")
        print(f"📁 Full path: {video_path}")
        print(f"📊 Exists: {os.path.exists(video_path)}")
        
        if os.path.exists(video_path):
            file_size = os.path.getsize(video_path)
            print(f"📏 File size: {file_size:,} bytes")
            
            if file_size > 1000:
                return send_from_directory(
                    os.path.join('static', 'videos'), 
                    filename, 
                    mimetype='video/mp4',
                    as_attachment=False
                )
            else:
                print(f"❌ File too small: {file_size} bytes")
                return "Video file is corrupted or too small", 404
        else:
            print(f"❌ File not found: {video_path}")
            return "Video file not found", 404
            
    except Exception as e:
        print(f"❌ Video serving error: {e}")
        return f"Error serving video: {str(e)}", 500

# Add debug route
@app.route('/debug/videos')
def debug_videos():
    """Debug route to see all video files"""
    try:
        videos_dir = os.path.join('static', 'videos')
        files = []
        
        if os.path.exists(videos_dir):
            for filename in os.listdir(videos_dir):
                if filename.endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
                    filepath = os.path.join(videos_dir, filename)
                    file_size = os.path.getsize(filepath)
                    files.append({
                        'filename': filename,
                        'size': file_size,
                        'url': f'/videos/{filename}',
                        'readable': file_size > 1000
                    })
        
        return jsonify({
            'videos_directory': videos_dir,
            'files': files,
            'total_files': len(files)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    print("🚀 Starting Deepfake Detector Server...")
    print(f"📁 Upload folder: {UPLOAD_FOLDER}")
    print(f"📁 Processed folder: {PROCESSED_FOLDER}")
    
    # Create directories if they don't exist
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(PROCESSED_FOLDER, exist_ok=True)
    
    # Use different port to avoid conflicts
    app.run(debug=True, host='0.0.0.0', port=5001)
