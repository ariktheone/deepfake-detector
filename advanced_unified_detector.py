"""
Advanced Unified Deepfake Detector
Combines multiple detection techniques for comprehensive analysis
"""

import cv2
import numpy as np
import os
from time import time as current_time
import traceback

# Safe dlib import
DLIB_AVAILABLE = False
try:
    import dlib
    DLIB_AVAILABLE = True
    print("✅ Dlib available for advanced detector")
except ImportError:
    print("⚠️ Dlib not available for advanced detector - using fallback mode")
    # Create dummy dlib functions
    class DlibDummy:
        def get_frontal_face_detector(self):
            return None
        def shape_predictor(self, path):
            return None
    dlib = DlibDummy()

def run(video_path, output_path):
    """
    Run advanced unified deepfake detection
    
    Args:
        video_path (str): Path to input video
        output_path (str): Path to save processed video
        
    Returns:
        float: Deepfake probability percentage (0-100)
    """
    try:
        print(f"🧠 Starting advanced unified detection for: {video_path}")
        start_time = current_time()
        
        # Initialize detection components
        if DLIB_AVAILABLE:
            face_detector = dlib.get_frontal_face_detector()
            
            # Check if shape predictor exists
            predictor_path = "models/shape_predictor_68_face_landmarks.dat"
            if os.path.exists(predictor_path):
                predictor = dlib.shape_predictor(predictor_path)
                landmarks_available = True
            else:
                predictor = None
                landmarks_available = False
                print("⚠️ Landmark predictor not found, using basic detection")
        else:
            # Fallback to OpenCV for face detection
            print("⚠️ Using OpenCV fallback for face detection")
            face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            predictor = None
            landmarks_available = False
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"❌ Cannot open video: {video_path}")
            return 50.0
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📹 Video info: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Detection variables
        deepfake_indicators = []
        face_consistency_scores = []
        temporal_inconsistencies = []
        edge_artifacts = []
        
        frame_count = 0
        processed_frames = 0
        face_frames = 0
        
        # Process frames (sample every 5th frame for performance)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Process every 5th frame to improve performance
            if frame_count % 5 != 0:
                out.write(frame)
                continue
                
            processed_frames += 1
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces (handle both dlib and OpenCV)
            if DLIB_AVAILABLE:
                faces = face_detector(gray)
                detected_faces = [(face.left(), face.top(), face.width(), face.height()) for face in faces]
            else:
                # Use OpenCV cascade classifier
                faces = face_detector.detectMultiScale(gray, 1.1, 4)
                detected_faces = [(x, y, w, h) for (x, y, w, h) in faces]
            
            if len(detected_faces) > 0:
                face_frames += 1
                
                for x, y, w, h in detected_faces:
                    # Extract face region
                    face_roi = frame[y:y+h, x:x+w]
                    
                    if face_roi.size > 0:
                        # Analysis 1: Facial landmark consistency
                        if landmarks_available and predictor and DLIB_AVAILABLE:
                            # Convert back to dlib rectangle for landmark detection
                            dlib_rect = dlib.rectangle(x, y, x+w, y+h)
                            landmarks = predictor(gray, dlib_rect)
                            landmark_score = analyze_landmark_consistency(landmarks)
                            face_consistency_scores.append(landmark_score)
                        
                        # Analysis 2: Edge artifact detection
                        edge_score = detect_edge_artifacts(face_roi)
                        edge_artifacts.append(edge_score)
                        
                        # Analysis 3: Color inconsistency
                        color_score = analyze_color_inconsistency(face_roi)
                        deepfake_indicators.append(color_score)
                        
                        # Analysis 4: Temporal consistency (if we have previous frame)
                        if processed_frames > 1:
                            temporal_score = analyze_temporal_consistency(face_roi)
                            temporal_inconsistencies.append(temporal_score)
                        
                        # Visualize detection on frame
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(frame, f"Advanced Analysis", 
                                  (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Write processed frame
            out.write(frame)
            
            # Limit processing time for performance
            if current_time() - start_time > 90:  # 90 second timeout
                print("⏰ Advanced detector timeout reached")
                break
        
        # Release resources
        cap.release()
        out.release()
        
        # Calculate final score
        final_score = calculate_advanced_score(
            deepfake_indicators,
            face_consistency_scores,
            temporal_inconsistencies,
            edge_artifacts,
            face_frames,
            processed_frames
        )
        
        execution_time = current_time() - start_time
        print(f"🧠 Advanced detection completed: {final_score:.1f}% in {execution_time:.2f}s")
        print(f"📊 Processed {processed_frames} frames, {face_frames} with faces")
        
        return final_score
        
    except Exception as e:
        print(f"❌ Advanced detector error: {e}")
        traceback.print_exc()
        return 45.0  # Return neutral score on error

def analyze_landmark_consistency(landmarks):
    """Analyze facial landmark consistency for deepfake detection"""
    try:
        # Extract landmark points
        points = []
        for i in range(68):
            x = landmarks.part(i).x
            y = landmarks.part(i).y
            points.append([x, y])
        
        points = np.array(points)
        
        # Calculate landmark ratios and symmetry
        # Eye aspect ratios
        left_eye_ratio = calculate_eye_aspect_ratio(points[36:42])
        right_eye_ratio = calculate_eye_aspect_ratio(points[42:48])
        
        # Mouth aspect ratio
        mouth_ratio = calculate_mouth_aspect_ratio(points[48:68])
        
        # Facial symmetry
        symmetry_score = calculate_facial_symmetry(points)
        
        # Combine scores (higher score = more suspicious)
        consistency_score = abs(left_eye_ratio - right_eye_ratio) * 50 + \
                          (1 - symmetry_score) * 30 + \
                          min(mouth_ratio * 20, 20)
        
        return min(consistency_score, 100)
        
    except Exception as e:
        print(f"⚠️ Landmark analysis error: {e}")
        return 25.0

def calculate_eye_aspect_ratio(eye_points):
    """Calculate eye aspect ratio"""
    try:
        # Vertical distances
        A = np.linalg.norm(eye_points[1] - eye_points[5])
        B = np.linalg.norm(eye_points[2] - eye_points[4])
        # Horizontal distance
        C = np.linalg.norm(eye_points[0] - eye_points[3])
        
        # Eye aspect ratio
        ear = (A + B) / (2.0 * C)
        return ear
    except:
        return 0.25

def calculate_mouth_aspect_ratio(mouth_points):
    """Calculate mouth aspect ratio"""
    try:
        # Vertical distances
        A = np.linalg.norm(mouth_points[2] - mouth_points[10])
        B = np.linalg.norm(mouth_points[4] - mouth_points[8])
        # Horizontal distance
        C = np.linalg.norm(mouth_points[0] - mouth_points[6])
        
        # Mouth aspect ratio
        mar = (A + B) / (2.0 * C)
        return mar
    except:
        return 0.15

def calculate_facial_symmetry(points):
    """Calculate facial symmetry score"""
    try:
        # Get face center
        center_x = np.mean(points[:, 0])
        
        # Calculate symmetry by comparing left and right sides
        left_points = points[points[:, 0] < center_x]
        right_points = points[points[:, 0] >= center_x]
        
        if len(left_points) == 0 or len(right_points) == 0:
            return 0.5
        
        # Mirror right points and compare with left
        mirrored_right = right_points.copy()
        mirrored_right[:, 0] = 2 * center_x - mirrored_right[:, 0]
        
        # Calculate average distance (simplified symmetry measure)
        if len(left_points) == len(mirrored_right):
            distances = np.linalg.norm(left_points - mirrored_right, axis=1)
            symmetry = 1.0 / (1.0 + np.mean(distances) / 10.0)
        else:
            symmetry = 0.5
        
        return symmetry
    except:
        return 0.5

def detect_edge_artifacts(face_roi):
    """Detect edge artifacts common in deepfakes"""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Calculate edge density
        edge_density = np.sum(edges > 0) / edges.size
        
        # Apply Laplacian for blur detection
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        blur_variance = laplacian.var()
        
        # Combine metrics (higher = more suspicious)
        artifact_score = edge_density * 100 + max(0, (100 - blur_variance) / 2)
        
        return min(artifact_score, 100)
        
    except Exception as e:
        print(f"⚠️ Edge artifact detection error: {e}")
        return 30.0

def analyze_color_inconsistency(face_roi):
    """Analyze color inconsistencies in face region"""
    try:
        # Convert to different color spaces
        hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(face_roi, cv2.COLOR_BGR2LAB)
        
        # Calculate color variance
        hsv_var = np.var(hsv.reshape(-1, 3), axis=0)
        lab_var = np.var(lab.reshape(-1, 3), axis=0)
        
        # Skin tone consistency check
        skin_consistency = analyze_skin_tone_consistency(face_roi)
        
        # Combine color metrics
        color_score = np.mean(hsv_var) / 10 + np.mean(lab_var) / 20 + skin_consistency
        
        return min(color_score, 100)
        
    except Exception as e:
        print(f"⚠️ Color analysis error: {e}")
        return 25.0

def analyze_skin_tone_consistency(face_roi):
    """Analyze skin tone consistency"""
    try:
        # Convert to YCrCb for skin detection
        ycrcb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2YCrCb)
        
        # Define skin color range in YCrCb
        lower_skin = np.array([0, 133, 77])
        upper_skin = np.array([255, 173, 127])
        
        # Create skin mask
        skin_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
        
        # Calculate skin pixel ratio
        skin_ratio = np.sum(skin_mask > 0) / skin_mask.size
        
        # Lower skin ratio might indicate synthetic content
        if skin_ratio < 0.3:
            return 40.0
        elif skin_ratio < 0.5:
            return 20.0
        else:
            return 10.0
            
    except:
        return 15.0

def analyze_temporal_consistency(current_frame):
    """Analyze temporal consistency between frames"""
    try:
        # This is a simplified version - in practice, you'd compare with previous frames
        # For now, we'll use optical flow-like analysis
        
        gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate local variance to detect temporal artifacts
        kernel = np.ones((5, 5), np.float32) / 25
        smoothed = cv2.filter2D(gray, -1, kernel)
        variance = np.var(gray - smoothed)
        
        # Higher variance might indicate temporal inconsistencies
        temporal_score = min(variance / 10, 50)
        
        return temporal_score
        
    except Exception as e:
        print(f"⚠️ Temporal analysis error: {e}")
        return 20.0

def calculate_advanced_score(deepfake_indicators, face_consistency_scores, 
                           temporal_inconsistencies, edge_artifacts, 
                           face_frames, processed_frames):
    """Calculate final advanced detection score"""
    try:
        # Base score
        base_score = 30.0
        
        # Factor 1: Face detection ratio
        if processed_frames > 0:
            face_ratio = face_frames / processed_frames
            if face_ratio < 0.3:
                base_score += 20  # Low face detection might indicate synthetic content
        
        # Factor 2: Average deepfake indicators
        if deepfake_indicators:
            avg_deepfake_score = np.mean(deepfake_indicators)
            base_score += avg_deepfake_score * 0.3
        
        # Factor 3: Face consistency
        if face_consistency_scores:
            avg_consistency = np.mean(face_consistency_scores)
            base_score += avg_consistency * 0.25
        
        # Factor 4: Temporal inconsistencies
        if temporal_inconsistencies:
            avg_temporal = np.mean(temporal_inconsistencies)
            base_score += avg_temporal * 0.2
        
        # Factor 5: Edge artifacts
        if edge_artifacts:
            avg_edge = np.mean(edge_artifacts)
            base_score += avg_edge * 0.15
        
        # Normalize and clamp
        final_score = max(5, min(95, base_score))
        
        return final_score
        
    except Exception as e:
        print(f"⚠️ Score calculation error: {e}")
        return 45.0

if __name__ == "__main__":
    # Test the detector
    test_video = "data/FAKE/fake4.mp4"
    output_video = "static/videos/test_advanced.mp4"
    
    if os.path.exists(test_video):
        score = run(test_video, output_video)
        print(f"🧠 Advanced detection result: {score:.1f}%")
    else:
        print("❌ Test video not found")
