"""Face Recognition with OpenCV and ONNX"""

import os
import cv2
import numpy as np
from typing import List, Tuple, Optional, Any

try:
    import frappe
    FRAPPE_AVAILABLE = True
except ImportError:
    FRAPPE_AVAILABLE = False
    class DummyFrappe:
        @staticmethod
        def log_error(msg):
            print(f"LOG: {msg}")
    frappe = DummyFrappe()

try:
    from PIL import Image
    import hashlib
    import cv2
    import numpy as np
    FACE_RECOGNITION_AVAILABLE = True
except ImportError as e:
    FACE_RECOGNITION_AVAILABLE = False
    try:
        frappe.log_error(f"Face recognition dependencies not available: {e}")
    except:
        print(f"Face recognition dependencies not available: {e}")

try:
    from .onnx_face_recognition import get_onnx_face_recognition
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


class SimpleFaceRecognition:
    
    def __init__(self, production_mode=True, use_onnx=False):
        if not FACE_RECOGNITION_AVAILABLE:
            self.initialized = False
            return
            
        try:
            # ONNX disabled for stability - using OpenCV only
            self.onnx_recognizer = None
            
            # Initialize OpenCV face detection (fallback)
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
            
            # Production-optimized feature extractors
            if production_mode:
                # Reduced features for faster processing in production
                self.feature_extractor = cv2.ORB_create(nfeatures=500)
                self.sift_extractor = cv2.SIFT_create(nfeatures=250) if hasattr(cv2, 'SIFT_create') else None
            else:
                # Higher quality for development/testing
                self.feature_extractor = cv2.ORB_create(nfeatures=1000)
                self.sift_extractor = cv2.SIFT_create(nfeatures=500) if hasattr(cv2, 'SIFT_create') else None
            
            self.production_mode = production_mode
            self.use_onnx = use_onnx
            self.initialized = True
            
        except Exception as e:
            try:
                frappe.log_error(f"Failed to initialize face recognition: {e}")
            except:
                print(f"Failed to initialize face recognition: {e}")
            self.initialized = False
    
    def _preprocess_image_for_detection(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image to improve face detection"""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Apply histogram equalization
        equalized = cv2.equalizeHist(gray)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(equalized, (3, 3), 0)
        
        return blurred

    def face_locations(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect face locations in image
        Returns list of (top, right, bottom, left) tuples
        """
        if not self.initialized:
            return []
            
        try:
            # OpenCV face detection
            # Convert to grayscale for detection
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            face_locations = []
            
            # Try multiple detection strategies
            
            # Strategy 1: Standard detection
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # Strategy 2: More sensitive detection if no faces found
            if len(faces) == 0:
                faces = self.face_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=1.05, 
                    minNeighbors=3, 
                    minSize=(20, 20),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
            
            # Strategy 3: Try with preprocessed image
            if len(faces) == 0:
                preprocessed = self._preprocess_image_for_detection(image)
                faces = self.face_cascade.detectMultiScale(
                    preprocessed, 
                    scaleFactor=1.1, 
                    minNeighbors=4, 
                    minSize=(25, 25)
                )
            
            # Strategy 4: Try profile detection
            if len(faces) == 0:
                profile_faces = self.profile_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=1.1, 
                    minNeighbors=5, 
                    minSize=(30, 30)
                )
                faces = profile_faces
            
            # Convert to face_recognition format: (top, right, bottom, left)
            for (x, y, w, h) in faces:
                top = y
                right = x + w
                bottom = y + h
                left = x
                face_locations.append((top, right, bottom, left))
                
            return face_locations
            
        except Exception as e:
            try:
                frappe.log_error(f"Face detection error: {e}")
            except:
                print(f"Face detection error: {e}")
            return []
    
    def _extract_face_features(self, face_image: np.ndarray) -> np.ndarray:
        """
        Extract facial features using OpenCV methods
        Returns a combined feature vector
        """
        try:
            # Resize to standard size
            face_resized = cv2.resize(face_image, (128, 128))
            
            # Convert to grayscale
            gray_face = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY) if len(face_resized.shape) == 3 else face_resized
            
            # Method 1: Histogram features (64 features)
            hist_features = self._extract_histogram_features(gray_face)
            
            # Method 2: LBP features (64 features)
            lbp_features = self._extract_lbp_features(gray_face)
            
            # Method 3: HOG features (64 features)
            hog_features = self._extract_hog_features(gray_face)
            
            # Method 4: ORB features (128 features)
            orb_features = self._extract_orb_features(gray_face)
            
            # Method 5: SIFT features (128 features) if available
            sift_features = self._extract_sift_features(gray_face)
            
            # Method 6: Geometric features (64 features)
            geo_features = self._extract_geometric_features(gray_face)
            
            # Use only histogram features for simplicity and consistency (64 features)
            # Normalize the feature vector
            if np.linalg.norm(hist_features) > 0:
                hist_features = hist_features / np.linalg.norm(hist_features)
            
            return hist_features.astype(np.float32)
            
        except Exception as e:
            try:
                frappe.log_error(f"Feature extraction error: {e}")
            except:
                print(f"Feature extraction error: {e}")
            return np.zeros(64, dtype=np.float32)
    
    def _extract_histogram_features(self, gray_face: np.ndarray) -> np.ndarray:
        """Extract histogram features from face"""
        hist = cv2.calcHist([gray_face], [0], None, [256], [0, 256])
        hist = hist.flatten()
        # Reduce to 64 features using interpolation
        hist_reduced = np.interp(np.linspace(0, len(hist)-1, 64), np.arange(len(hist)), hist)
        return hist_reduced.astype(np.float32)
    
    def _extract_lbp_features(self, gray_face: np.ndarray) -> np.ndarray:
        """Extract Local Binary Pattern features"""
        # Simple LBP implementation
        lbp = np.zeros_like(gray_face, dtype=np.uint8)
        
        for i in range(1, gray_face.shape[0] - 1):
            for j in range(1, gray_face.shape[1] - 1):
                center = gray_face[i, j]
                code = 0
                code |= (gray_face[i-1, j-1] >= center) << 7
                code |= (gray_face[i-1, j] >= center) << 6
                code |= (gray_face[i-1, j+1] >= center) << 5
                code |= (gray_face[i, j+1] >= center) << 4
                code |= (gray_face[i+1, j+1] >= center) << 3
                code |= (gray_face[i+1, j] >= center) << 2
                code |= (gray_face[i+1, j-1] >= center) << 1
                code |= (gray_face[i, j-1] >= center) << 0
                lbp[i, j] = code
        
        # Get histogram of LBP
        hist = cv2.calcHist([lbp], [0], None, [256], [0, 256])
        hist = hist.flatten()
        # Reduce to 64 features
        hist_reduced = np.interp(np.linspace(0, len(hist)-1, 64), np.arange(len(hist)), hist)
        return hist_reduced.astype(np.float32)
    
    def _extract_hog_features(self, gray_face: np.ndarray) -> np.ndarray:
        """Extract HOG features"""
        # Calculate gradients
        grad_x = cv2.Sobel(gray_face, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_face, cv2.CV_32F, 0, 1, ksize=3)
        
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        angle = np.arctan2(grad_y, grad_x)
        
        # Create histogram of gradients
        hist, _ = np.histogram(angle.flatten(), bins=64, weights=magnitude.flatten())
        return hist.astype(np.float32)
    
    def _extract_orb_features(self, gray_face: np.ndarray) -> np.ndarray:
        """Extract ORB keypoint features"""
        try:
            keypoints, descriptors = self.feature_extractor.detectAndCompute(gray_face, None)
            
            if descriptors is not None and len(descriptors) > 0:
                # Aggregate descriptors by taking mean
                feature_vector = np.mean(descriptors, axis=0)
                # Ensure 128 features
                if len(feature_vector) < 128:
                    feature_vector = np.pad(feature_vector, (0, 128 - len(feature_vector)))
                else:
                    feature_vector = feature_vector[:128]
                return feature_vector.astype(np.float32)
            else:
                return np.zeros(128, dtype=np.float32)
        except:
            return np.zeros(128, dtype=np.float32)
    
    def _extract_sift_features(self, gray_face: np.ndarray) -> np.ndarray:
        """Extract SIFT features if available"""
        try:
            if self.sift_extractor is not None:
                keypoints, descriptors = self.sift_extractor.detectAndCompute(gray_face, None)
                
                if descriptors is not None and len(descriptors) > 0:
                    # Aggregate descriptors by taking mean
                    feature_vector = np.mean(descriptors, axis=0)
                    # Ensure 128 features
                    if len(feature_vector) < 128:
                        feature_vector = np.pad(feature_vector, (0, 128 - len(feature_vector)))
                    else:
                        feature_vector = feature_vector[:128]
                    return feature_vector.astype(np.float32)
            
            return np.zeros(128, dtype=np.float32)
        except:
            return np.zeros(128, dtype=np.float32)
    
    def _extract_geometric_features(self, gray_face: np.ndarray) -> np.ndarray:
        """Extract geometric features from face"""
        try:
            # Eye detection for geometric features
            eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            eyes = eye_cascade.detectMultiScale(gray_face, 1.1, 5)
            
            # Face measurements
            h, w = gray_face.shape
            features = []
            
            # Basic geometric measurements
            features.extend([w, h, w/h])  # Width, height, aspect ratio
            
            # Eye features
            if len(eyes) >= 2:
                # Distance between eyes
                eye1, eye2 = eyes[0], eyes[1]
                eye_distance = np.sqrt((eye1[0] - eye2[0])**2 + (eye1[1] - eye2[1])**2)
                features.extend([eye_distance, eye1[2], eye2[2]])  # Distance, eye widths
            else:
                features.extend([0, 0, 0])
            
            # Intensity statistics in different regions
            # Divide face into 8x8 grid and get mean intensity
            for i in range(8):
                for j in range(8):
                    y1, y2 = i * h // 8, (i + 1) * h // 8
                    x1, x2 = j * w // 8, (j + 1) * w // 8
                    region_mean = np.mean(gray_face[y1:y2, x1:x2])
                    features.append(region_mean)
            
            # Pad or truncate to 64 features
            while len(features) < 64:
                features.append(0.0)
            features = features[:64]
            
            return np.array(features, dtype=np.float32)
            
        except:
            return np.zeros(64, dtype=np.float32)
    
    def face_encodings(self, image: np.ndarray, known_face_locations: List[Tuple[int, int, int, int]] = None) -> List[np.ndarray]:
        """
        Generate face embeddings using OpenCV feature extraction
        Returns list of 64-dimensional face embeddings
        """
        if not self.initialized:
            return []
            
        try:
            # If no face locations provided, detect them
            if known_face_locations is None:
                known_face_locations = self.face_locations(image)
            
            if not known_face_locations:
                return []
                
            encodings = []
            
            for (top, right, bottom, left) in known_face_locations:
                # Extract face region
                face_image = image[top:bottom, left:right]
                
                if face_image.size == 0:
                    continue
                
                # Extract features using OpenCV methods
                features = self._extract_face_features(face_image)
                encodings.append(features)
                
            return encodings
            
        except Exception as e:
            try:
                frappe.log_error(f"Face encoding error: {e}")
            except:
                print(f"Face encoding error: {e}")
            return []
    
    def compare_faces(self, known_face_encodings: List[np.ndarray], face_encoding_to_check: np.ndarray, tolerance: float = 0.6) -> List[bool]:
        """
        Compare face encodings using cosine similarity
        Returns list of boolean matches
        """
        if not known_face_encodings:
            return []
            
        try:
            distances = self.face_distance(known_face_encodings, face_encoding_to_check)
            # Convert cosine distance to similarity threshold
            threshold = 1.0 - tolerance
            return [distance <= threshold for distance in distances]
            
        except Exception as e:
            try:
                frappe.log_error(f"Face comparison error: {e}")
            except:
                print(f"Face comparison error: {e}")
            return [False] * len(known_face_encodings)
    
    def face_distance(self, face_encodings: List[np.ndarray], face_to_compare: np.ndarray) -> List[float]:
        """
        Calculate cosine distance between face encodings
        Returns list of distances (lower = more similar)
        """
        if not face_encodings:
            return []
            
        try:
            distances = []
            
            for encoding in face_encodings:
                # Calculate cosine distance
                dot_product = np.dot(encoding, face_to_compare)
                norm_a = np.linalg.norm(encoding)
                norm_b = np.linalg.norm(face_to_compare)
                
                if norm_a == 0 or norm_b == 0:
                    distance = 1.0  # Maximum distance
                else:
                    cosine_similarity = dot_product / (norm_a * norm_b)
                    distance = 1.0 - cosine_similarity
                    
                distances.append(distance)
                
            return distances
            
        except Exception as e:
            try:
                frappe.log_error(f"Face distance calculation error: {e}")
            except:
                print(f"Face distance calculation error: {e}")
            return [1.0] * len(face_encodings)
    
    def validate_face_quality(self, image: np.ndarray, face_location: Tuple[int, int, int, int], lenient_mode: bool = False, strict_accuracy: bool = True) -> dict:
        """
        Validate face image quality for better recognition accuracy
        Returns quality metrics and recommendations
        """
        try:
            top, right, bottom, left = face_location
            face_image = image[top:bottom, left:right]
            
            if face_image.size == 0:
                return {"valid": False, "reason": "Empty face region"}
            
            # Convert to grayscale for analysis
            if len(face_image.shape) == 3:
                gray_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            else:
                gray_face = face_image
            
            quality_issues = []
            
            # Enhanced quality checks for strict accuracy mode
            if strict_accuracy:
                # STRICT ACCURACY MODE - Higher standards
                min_face_size = 80 if not lenient_mode else 60
                dark_threshold = 35 if not lenient_mode else 25
                bright_threshold = 210 if not lenient_mode else 225
                contrast_threshold = 20 if not lenient_mode else 15
                blur_threshold = 80 if not lenient_mode else 50
            else:
                # Standard mode thresholds
                min_face_size = 50
                dark_threshold = 20 if lenient_mode else 30
                bright_threshold = 230 if lenient_mode else 220
                contrast_threshold = 10 if lenient_mode else 15
                blur_threshold = 30 if lenient_mode else 50
            
            # 1. Check face size
            height, width = gray_face.shape
            if height < min_face_size or width < min_face_size:
                quality_issues.append(f"Face too small ({width}x{height}, need ≥{min_face_size}x{min_face_size}) - move closer to camera")
            
            # 2. Check brightness
            mean_brightness = np.mean(gray_face)
            if mean_brightness < dark_threshold:
                quality_issues.append(f"Too dark (brightness: {mean_brightness:.1f}/{dark_threshold}) - improve lighting")
            elif mean_brightness > bright_threshold:
                quality_issues.append(f"Too bright (brightness: {mean_brightness:.1f}/{bright_threshold}) - reduce lighting")
            
            # 3. Check contrast
            contrast = np.std(gray_face)
            if contrast < contrast_threshold:
                quality_issues.append(f"Low contrast ({contrast:.1f}/{contrast_threshold}) - improve lighting conditions")
            
            # 4. Check blur (using Laplacian variance)
            laplacian_var = cv2.Laplacian(gray_face, cv2.CV_64F).var()
            if laplacian_var < blur_threshold:
                quality_issues.append(f"Image too blurry (sharpness: {laplacian_var:.1f}/{blur_threshold}) - hold steady and ensure good focus")
            
            # 5. STRICT ACCURACY: Additional advanced quality checks
            if strict_accuracy:
                # Check for over/under exposure
                hist = cv2.calcHist([gray_face], [0], None, [256], [0, 256])
                hist = hist.flatten()
                
                # Check if too many pixels are at extremes (over/under exposed)
                dark_pixels = np.sum(hist[:20]) / gray_face.size  # Very dark pixels
                bright_pixels = np.sum(hist[235:]) / gray_face.size  # Very bright pixels
                
                if dark_pixels > 0.15:  # More than 15% very dark pixels
                    quality_issues.append(f"Underexposed image ({dark_pixels:.1%} very dark pixels)")
                if bright_pixels > 0.10:  # More than 10% very bright pixels
                    quality_issues.append(f"Overexposed image ({bright_pixels:.1%} very bright pixels)")
                
                # Check for proper face positioning (eyes should be visible)
                try:
                    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
                    eyes = eye_cascade.detectMultiScale(gray_face, 1.1, 5)
                    if len(eyes) < 2:
                        quality_issues.append("Eyes not clearly visible - ensure face is front-facing")
                except:
                    pass
                
                # Check image noise levels
                noise_level = np.std(cv2.GaussianBlur(gray_face, (5, 5), 0) - gray_face)
                if noise_level > 8.0:
                    quality_issues.append(f"High image noise level ({noise_level:.1f}) - improve camera quality or lighting")
            
            # 5. Check if face is roughly centered and not cut off
            face_width = right - left
            face_height = bottom - top
            image_height, image_width = image.shape[:2]
            
            # Face should be at least 10% of image width/height for good quality
            if face_width < 0.10 * image_width or face_height < 0.10 * image_height:
                quality_issues.append("Face too small in frame - move closer")
            
            # Face shouldn't be cut off at edges
            edge_margin = 10
            if left < edge_margin or top < edge_margin or right > (image_width - edge_margin) or bottom > (image_height - edge_margin):
                quality_issues.append("Face partially cut off - center yourself in frame")
            
            result = {
                "valid": len(quality_issues) == 0,
                "quality_score": max(0, 100 - len(quality_issues) * 20),
                "issues": quality_issues,
                "metrics": {
                    "brightness": mean_brightness,
                    "contrast": contrast,
                    "sharpness": laplacian_var,
                    "face_size": (width, height)
                }
            }
            
            if lenient_mode and not result["valid"]:
                result["lenient_mode"] = True
                result["note"] = "Using relaxed quality standards for employee enrollment"
            
            return result
            
        except Exception as e:
            try:
                frappe.log_error(f"Face quality validation error: {e}")
            except:
                print(f"Face quality validation error: {e}")
            return {"valid": False, "reason": "Quality validation failed"}
    
    def get_best_face_from_multiple(self, image: np.ndarray, face_locations: List[Tuple[int, int, int, int]]) -> Tuple[int, int, int, int]:
        """
        Select the best quality face from multiple detected faces
        Returns the face location with highest quality score
        """
        if not face_locations:
            return None
            
        if len(face_locations) == 1:
            return face_locations[0]
        
        best_face = None
        best_score = -1
        
        for face_location in face_locations:
            quality = self.validate_face_quality(image, face_location)
            if quality["valid"] and quality["quality_score"] > best_score:
                best_score = quality["quality_score"]
                best_face = face_location
        
        # If no valid faces, return the largest one
        if best_face is None:
            largest_area = 0
            for face_location in face_locations:
                top, right, bottom, left = face_location
                area = (right - left) * (bottom - top)
                if area > largest_area:
                    largest_area = area
                    best_face = face_location
        
        return best_face


# Global instance
_face_recognition_instance = None

def get_face_recognition():
    """Get or create global face recognition instance"""
    global _face_recognition_instance
    if _face_recognition_instance is None:
        _face_recognition_instance = SimpleFaceRecognition()
    return _face_recognition_instance


# Compatibility functions that mimic face_recognition library API
def face_locations(img_np):
    """Detect face locations - compatibility function"""
    fr = get_face_recognition()
    return fr.face_locations(img_np)


def face_encodings(img_np, face_locations=None):
    """Generate face encodings - compatibility function"""
    fr = get_face_recognition()
    return fr.face_encodings(img_np, face_locations)


def compare_faces(known_encodings, face_encoding, tolerance=0.6):
    """Compare face encodings - compatibility function"""
    fr = get_face_recognition()
    return fr.compare_faces(known_encodings, face_encoding, tolerance)


def face_distance(known_encodings, face_encoding):
    """Calculate face distances - compatibility function"""
    fr = get_face_recognition()
    return fr.face_distance(known_encodings, face_encoding)


def validate_face_quality(image, face_location, lenient_mode=False, strict_accuracy=True):
    """Validate face quality - compatibility function"""
    fr = get_face_recognition()
    return fr.validate_face_quality(image, face_location, lenient_mode, strict_accuracy)


def get_best_face_from_multiple(image, face_locations):
    """Get best quality face from multiple faces - compatibility function"""
    fr = get_face_recognition()
    return fr.get_best_face_from_multiple(image, face_locations)