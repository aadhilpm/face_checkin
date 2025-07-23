"""ONNX Face Recognition for Frappe Cloud"""

import os
import cv2
import numpy as np
import urllib.request
import hashlib
from typing import List, Tuple, Optional, Any

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("Warning: onnxruntime not available. Install with: pip install onnxruntime")
    # Create dummy ort module to prevent errors
    class DummyORT:
        class SessionOptions:
            def __init__(self):
                self.intra_op_num_threads = 2
                self.inter_op_num_threads = 2
                self.graph_optimization_level = None
                self.execution_mode = None
        class GraphOptimizationLevel:
            ORT_ENABLE_ALL = "ORT_ENABLE_ALL"
        class ExecutionMode:
            ORT_SEQUENTIAL = "ORT_SEQUENTIAL"
        class InferenceSession:
            def __init__(self, *args, **kwargs):
                pass
    ort = DummyORT()

try:
    import frappe
    FRAPPE_AVAILABLE = True
except ImportError:
    FRAPPE_AVAILABLE = False


class ONNXFaceRecognition:
    
    def __init__(self):
        self.face_detector = None
        self.face_recognizer = None
        self.models_dir = self._get_models_dir()
        self.model_urls = {
            'face_detection': {
                'url': 'https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx',
                'filename': 'face_detection_yunet.onnx',
                'size': 2.8
            },
            'face_recognition': {
                'url': 'https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx',
                'filename': 'face_recognition_sface.onnx',
                'size': 4.2
            }
        }
        
        if ONNX_AVAILABLE:
            self._initialize_models()
    
    def _get_models_dir(self) -> str:
        if FRAPPE_AVAILABLE:
            return os.path.join(frappe.get_site_path(), 'private', 'files', 'onnx_models')
        else:
            return os.path.join(os.path.dirname(__file__), '..', 'models')
    
    def _ensure_models_dir(self):
        os.makedirs(self.models_dir, exist_ok=True)
    
    def _download_model(self, model_key: str) -> str:
        model_info = self.model_urls[model_key]
        model_path = os.path.join(self.models_dir, model_info['filename'])
        
        if os.path.exists(model_path):
            return model_path
        
        self._ensure_models_dir()
        
        try:
            print(f"Downloading {model_info['filename']} ({model_info['size']}MB)...")
            urllib.request.urlretrieve(model_info['url'], model_path)
            print(f"Downloaded {model_info['filename']} successfully")
            return model_path
        except Exception as e:
            if FRAPPE_AVAILABLE:
                frappe.log_error(f"Failed to download ONNX model {model_key}: {str(e)}")
            print(f"Error downloading model {model_key}: {str(e)}")
            return None
    
    def _create_cpu_session(self, model_path: str) -> ort.InferenceSession:
        if not model_path or not os.path.exists(model_path):
            return None
        
        try:
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = 2
            sess_options.inter_op_num_threads = 2
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            providers = ['CPUExecutionProvider']
            
            session = ort.InferenceSession(
                model_path, 
                sess_options=sess_options,
                providers=providers
            )
            
            return session
            
        except Exception as e:
            if FRAPPE_AVAILABLE:
                frappe.log_error(f"Failed to create ONNX session: {str(e)}")
            print(f"Error creating ONNX session: {str(e)}")
            return None
    
    def _initialize_models(self):
        try:
            detection_path = self._download_model('face_detection')
            if detection_path:
                self.face_detector = self._create_cpu_session(detection_path)
            
            recognition_path = self._download_model('face_recognition')
            if recognition_path:
                self.face_recognizer = self._create_cpu_session(recognition_path)
                
            print("ONNX models initialized successfully")
            
        except Exception as e:
            if FRAPPE_AVAILABLE:
                frappe.log_error(f"Failed to initialize ONNX models: {str(e)}")
            print(f"Error initializing ONNX models: {str(e)}")
    
    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        if not self.face_detector:
            return []
        
        try:
            height, width = image.shape[:2]
            
            blob = cv2.dnn.blobFromImage(image, 1.0, (320, 240), (0, 0, 0), swapRB=True)
            input_name = self.face_detector.get_inputs()[0].name
            output = self.face_detector.run(None, {input_name: blob})
            
            faces = []
            detections = output[0][0]
            
            for detection in detections:
                confidence = detection[14]
                if confidence > 0.7:
                    x1 = int(detection[0] * width)
                    y1 = int(detection[1] * height)
                    x2 = int(detection[2] * width)
                    y2 = int(detection[3] * height)
                    
                    faces.append((x1, y1, x2 - x1, y2 - y1))
            
            return faces
            
        except Exception as e:
            if FRAPPE_AVAILABLE:
                frappe.log_error(f"ONNX face detection error: {str(e)}")
            return []
    
    def get_face_embedding(self, image: np.ndarray, face_box: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """
        Extract face embedding using ONNX SFace model
        """
        if not self.face_recognizer:
            return None
        
        try:
            x, y, w, h = face_box
            
            # Extract and align face
            face_img = image[y:y+h, x:x+w]
            if face_img.size == 0:
                return None
            
            # Resize to model input size (112x112 for SFace)
            face_img = cv2.resize(face_img, (112, 112))
            
            # Normalize and prepare input
            face_img = face_img.astype(np.float32) / 255.0
            face_img = np.transpose(face_img, (2, 0, 1))  # HWC to CHW
            face_img = np.expand_dims(face_img, axis=0)   # Add batch dimension
            
            # Run inference
            input_name = self.face_recognizer.get_inputs()[0].name
            embedding = self.face_recognizer.run(None, {input_name: face_img})[0]
            
            # Normalize embedding
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding.flatten()
            
        except Exception as e:
            if FRAPPE_AVAILABLE:
                frappe.log_error(f"ONNX face embedding error: {str(e)}")
            return None
    
    def compare_embeddings(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compare two face embeddings using cosine similarity
        Returns similarity score (0-1, higher = more similar)
        """
        try:
            # Cosine similarity
            similarity = np.dot(embedding1, embedding2) / (
                np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
            )
            return float(similarity)
        except:
            return 0.0
    
    def is_available(self) -> bool:
        """Check if ONNX models are available and loaded"""
        return ONNX_AVAILABLE and self.face_detector is not None and self.face_recognizer is not None


# Singleton instance
_onnx_face_recognition = None

def get_onnx_face_recognition():
    """Get singleton ONNX face recognition instance"""
    global _onnx_face_recognition
    if _onnx_face_recognition is None:
        _onnx_face_recognition = ONNXFaceRecognition()
    return _onnx_face_recognition