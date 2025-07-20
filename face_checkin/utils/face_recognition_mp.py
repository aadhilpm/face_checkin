"""
MediaPipe + ArcFace Face Recognition Implementation
Replaces the face-recognition library with MediaPipe for detection and ArcFace for embeddings
"""

import os
import cv2
import numpy as np
from typing import List, Tuple, Optional, Any
import frappe

try:
    import mediapipe as mp
    import onnxruntime as ort
    from PIL import Image
    FACE_RECOGNITION_AVAILABLE = True
except ImportError as e:
    FACE_RECOGNITION_AVAILABLE = False
    try:
        frappe.log_error(f"Face recognition dependencies not available: {e}")
    except:
        print(f"Face recognition dependencies not available: {e}")
    
    # Create dummy modules to prevent crashes
    class DummyModule:
        pass
    mp = DummyModule()
    ort = DummyModule()


class MediaPipeFaceRecognition:
    """MediaPipe + ArcFace face recognition implementation"""
    
    def __init__(self):
        if not FACE_RECOGNITION_AVAILABLE:
            self.initialized = False
            return
            
        try:
            # Initialize MediaPipe Face Detection with multiple configurations
            self.mp_face_detection = mp.solutions.face_detection
            
            # Primary detector - more sensitive for enrollment
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=1,  # 1 for full range (better for varied distances)
                min_detection_confidence=0.3  # Lower threshold for better detection
            )
            
            # Backup detector for difficult cases
            self.face_detection_backup = self.mp_face_detection.FaceDetection(
                model_selection=0,  # 0 for short range (better for close faces)
                min_detection_confidence=0.2  # Even lower threshold
            )
            
            # Initialize ArcFace ONNX model
            model_path = None
            
            # Try Frappe path first (if available)
            try:
                model_path = frappe.get_app_path('face_checkin', 'models', 'arcface_r100.onnx')
                if not os.path.exists(model_path):
                    model_path = None  # Force fallback
            except:
                pass  # Use fallback
            
            # Fallback: Try multiple possible locations
            if not model_path:
                current_dir = os.path.dirname(os.path.abspath(__file__))
                possible_paths = [
                    # From face_checkin/utils/ to app root models/
                    os.path.join(current_dir, '..', '..', 'models', 'arcface_r100.onnx'),
                    # Absolute path based on current working directory
                    os.path.join(os.getcwd(), 'models', 'arcface_r100.onnx'),
                    # From face_checkin/utils/ to face_checkin/models/
                    os.path.join(current_dir, '..', 'models', 'arcface_r100.onnx'),
                    os.path.join(os.getcwd(), 'face_checkin', 'models', 'arcface_r100.onnx')
                ]
                
                for path in possible_paths:
                    normalized_path = os.path.normpath(path)
                    if os.path.exists(normalized_path):
                        model_path = normalized_path
                        break
                
                if not model_path:
                    raise Exception(f"ArcFace model not found in any of these locations: {[os.path.normpath(p) for p in possible_paths]}")
            
            if not os.path.exists(model_path):
                raise Exception(f"ArcFace model not found at {model_path}")
                
            self.arcface_session = ort.InferenceSession(model_path)
            self.input_name = self.arcface_session.get_inputs()[0].name
            self.output_name = self.arcface_session.get_outputs()[0].name
            
            self.initialized = True
            
        except Exception as e:
            try:
                frappe.log_error(f"Failed to initialize face recognition: {e}")
            except:
                # Frappe not available, print to console instead
                print(f"Failed to initialize face recognition: {e}")
            self.initialized = False
    
    def _preprocess_image_for_detection(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image to improve face detection
        """
        # Convert to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image
            
        # Enhance contrast and brightness for better detection
        # Convert to LAB color space
        lab = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB)
        l_channel, a, b = cv2.split(lab)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l_channel = clahe.apply(l_channel)
        
        # Merge back and convert to RGB
        enhanced = cv2.merge((l_channel, a, b))
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        
        return enhanced

    def face_locations(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect face locations in image with enhanced detection
        Returns list of (top, right, bottom, left) tuples
        """
        if not self.initialized:
            return []
            
        try:
            # Try multiple detection strategies
            face_locations = []
            
            # Strategy 1: Original image with primary detector
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 else image
            results = self.face_detection.process(rgb_image)
            
            if results.detections:
                face_locations = self._extract_face_locations(results.detections, image.shape[:2])
            
            # Strategy 2: If no faces found, try with enhanced image
            if not face_locations:
                enhanced_image = self._preprocess_image_for_detection(image)
                results = self.face_detection.process(enhanced_image)
                if results.detections:
                    face_locations = self._extract_face_locations(results.detections, image.shape[:2])
            
            # Strategy 3: If still no faces, try backup detector
            if not face_locations:
                results = self.face_detection_backup.process(rgb_image)
                if results.detections:
                    face_locations = self._extract_face_locations(results.detections, image.shape[:2])
            
            # Strategy 4: Backup detector with enhanced image
            if not face_locations:
                enhanced_image = self._preprocess_image_for_detection(image)
                results = self.face_detection_backup.process(enhanced_image)
                if results.detections:
                    face_locations = self._extract_face_locations(results.detections, image.shape[:2])
                
            return face_locations
            
        except Exception as e:
            try:
                frappe.log_error(f"Face detection error: {e}")
            except:
                print(f"Face detection error: {e}")
            return []
    
    def _extract_face_locations(self, detections, image_shape) -> List[Tuple[int, int, int, int]]:
        """
        Extract face location coordinates from MediaPipe detections
        """
        face_locations = []
        h, w = image_shape
        
        for detection in detections:
            bbox = detection.location_data.relative_bounding_box
            
            # Convert relative coordinates to absolute
            left = int(bbox.xmin * w)
            top = int(bbox.ymin * h)
            right = int((bbox.xmin + bbox.width) * w)
            bottom = int((bbox.ymin + bbox.height) * h)
            
            # Ensure coordinates are within image bounds
            left = max(0, left)
            top = max(0, top)
            right = min(w, right)
            bottom = min(h, bottom)
            
            # Only add if the face region is reasonable size
            if (right - left) > 30 and (bottom - top) > 30:
                # Return in face_recognition format: (top, right, bottom, left)
                face_locations.append((top, right, bottom, left))
                
        return face_locations
    
    def _preprocess_face(self, face_image: np.ndarray) -> np.ndarray:
        """
        Preprocess face image for ArcFace model
        Expected input: 112x112 RGB image, normalized to [-1, 1]
        """
        # Resize to 112x112
        face_resized = cv2.resize(face_image, (112, 112))
        
        # Convert BGR to RGB if needed
        if len(face_resized.shape) == 3 and face_resized.shape[2] == 3:
            face_resized = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [-1, 1]
        face_normalized = (face_resized.astype(np.float32) / 127.5) - 1.0
        
        # Add batch dimension and transpose to NCHW format
        face_batch = np.transpose(face_normalized, (2, 0, 1))
        face_batch = np.expand_dims(face_batch, axis=0)
        
        return face_batch
    
    def face_encodings(self, image: np.ndarray, known_face_locations: List[Tuple[int, int, int, int]] = None) -> List[np.ndarray]:
        """
        Generate face embeddings using ArcFace model
        Returns list of 512-dimensional face embeddings
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
                
                # Preprocess face for ArcFace
                preprocessed_face = self._preprocess_face(face_image)
                
                # Run inference
                embedding = self.arcface_session.run(
                    [self.output_name], 
                    {self.input_name: preprocessed_face}
                )[0]
                
                # Normalize embedding
                embedding = embedding / np.linalg.norm(embedding)
                
                encodings.append(embedding.flatten())
                
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
            # Lower cosine distance = higher similarity
            # tolerance of 0.6 corresponds to cosine distance of ~0.4
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


# Global instance
_face_recognition_instance = None

def get_face_recognition():
    """Get or create global face recognition instance"""
    global _face_recognition_instance
    if _face_recognition_instance is None:
        _face_recognition_instance = MediaPipeFaceRecognition()
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