#!/usr/bin/env python3
"""
Test script for Face Checkin API functions
This script tests the API functions to ensure they work correctly
"""

import sys
import os

# Add the app path to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'face_checkin'))

def test_imports():
    """Test if all necessary imports work"""
    try:
        from face_checkin.api.face_api import check_system_status, get_detailed_status
        print("✓ API imports successful")
        return True
    except ImportError as e:
        print(f"✗ API import failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error during import: {e}")
        return False

def test_face_recognition_imports():
    """Test face recognition utility imports"""
    try:
        from face_checkin.utils.face_recognition_simple import SimpleFaceRecognition, FACE_RECOGNITION_AVAILABLE
        print(f"✓ Face recognition imports successful (Available: {FACE_RECOGNITION_AVAILABLE})")
        return True
    except ImportError as e:
        print(f"✗ Face recognition import failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error during face recognition import: {e}")
        return False

def test_onnx_imports():
    """Test ONNX face recognition imports"""
    try:
        from face_checkin.utils.onnx_face_recognition import get_onnx_face_recognition
        onnx_fr = get_onnx_face_recognition()
        print(f"✓ ONNX imports successful (Available: {onnx_fr.is_available() if onnx_fr else False})")
        return True
    except ImportError as e:
        print(f"✗ ONNX import failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error during ONNX import: {e}")
        return False

def test_core_dependencies():
    """Test core dependencies"""
    dependencies = [
        ('numpy', 'numpy'),
        ('opencv-python', 'cv2'),
        ('Pillow', 'PIL.Image'),
        ('onnxruntime', 'onnxruntime')
    ]
    
    results = {}
    for dep_name, import_name in dependencies:
        try:
            __import__(import_name)
            print(f"✓ {dep_name} available")
            results[dep_name] = True
        except ImportError:
            print(f"✗ {dep_name} not available")
            results[dep_name] = False
    
    return results

if __name__ == "__main__":
    print("Face Checkin API Test")
    print("=" * 40)
    
    # Test core dependencies first
    print("\n1. Testing core dependencies:")
    deps = test_core_dependencies()
    
    print("\n2. Testing API imports:")
    api_ok = test_imports()
    
    print("\n3. Testing face recognition imports:")
    face_ok = test_face_recognition_imports()
    
    print("\n4. Testing ONNX imports:")
    onnx_ok = test_onnx_imports()
    
    print("\n" + "=" * 40)
    print("Summary:")
    print(f"Core dependencies: {sum(deps.values())}/{len(deps)} available")
    print(f"API imports: {'✓' if api_ok else '✗'}")
    print(f"Face recognition: {'✓' if face_ok else '✗'}")
    print(f"ONNX support: {'✓' if onnx_ok else '✗'}")
    
    if all([api_ok, face_ok]):
        print("\n✓ All critical components are working")
        sys.exit(0)
    else:
        print("\n✗ Some components have issues")
        sys.exit(1)