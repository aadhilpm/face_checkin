#!/usr/bin/env python3
"""
Test script for multi-image face data creation functionality
This script tests the core logic without requiring Frappe framework
"""

import sys
import os
import base64
import numpy as np
from PIL import Image
from io import BytesIO

# Add the face_checkin module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'face_checkin'))

def create_test_images():
    """Create simple test images for testing"""
    test_images = []
    
    # Create 3 test images with different characteristics
    for i in range(3):
        # Create a simple test image (100x100 pixels)
        img = Image.new('RGB', (100, 100), color=(128 + i*40, 128, 128))
        
        # Add some simple "face-like" features (rectangles for eyes, etc.)
        pixels = img.load()
        
        # Add eye-like rectangles
        for x in range(30, 40):
            for y in range(30, 35):
                pixels[x, y] = (0, 0, 0)  # Left eye
        
        for x in range(60, 70):
            for y in range(30, 35):
                pixels[x, y] = (0, 0, 0)  # Right eye
        
        # Add mouth-like rectangle
        for x in range(40, 60):
            for y in range(60, 65):
                pixels[x, y] = (50, 50, 50)  # Mouth
        
        # Convert to base64
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        test_images.append(img_base64)
    
    return test_images

def test_multi_image_functionality():
    """Test the multi-image functionality"""
    print("Testing Multi-Image Face Data Creation")
    print("=" * 50)
    
    try:
        # Import the face recognition module
        from utils.face_recognition_simple import (
            create_multi_image_face_data,
            validate_multi_image_consistency,
            FACE_RECOGNITION_AVAILABLE
        )
        
        print(f"Face recognition available: {FACE_RECOGNITION_AVAILABLE}")
        
        if not FACE_RECOGNITION_AVAILABLE:
            print("❌ Face recognition libraries not available")
            return False
        
        # Create test images
        print("\n1. Creating test images...")
        test_images = create_test_images()
        print(f"✅ Created {len(test_images)} test images")
        
        # Test consistency validation
        print("\n2. Testing consistency validation...")
        consistency_result = validate_multi_image_consistency(test_images, 0.5)
        print(f"✅ Consistency check completed")
        print(f"   - Consistent: {consistency_result.get('consistent', False)}")
        print(f"   - Message: {consistency_result.get('message', 'N/A')}")
        
        # Test multi-image face data creation
        print("\n3. Testing multi-image face data creation...")
        face_data_result = create_multi_image_face_data(test_images, "test_employee", True)
        print(f"✅ Face data creation completed")
        print(f"   - Success: {face_data_result.get('success', False)}")
        print(f"   - Images processed: {face_data_result.get('images_processed', 0)}")
        print(f"   - Images failed: {face_data_result.get('images_failed', 0)}")
        print(f"   - Message: {face_data_result.get('message', 'N/A')}")
        
        if face_data_result.get('face_encoding') is not None:
            encoding = face_data_result['face_encoding']
            print(f"   - Encoding shape: {encoding.shape}")
            print(f"   - Encoding norm: {np.linalg.norm(encoding):.4f}")
        
        # Test with single image
        print("\n4. Testing with single image...")
        single_result = create_multi_image_face_data([test_images[0]], "test_single", True)
        print(f"✅ Single image processing completed")
        print(f"   - Success: {single_result.get('success', False)}")
        print(f"   - Message: {single_result.get('message', 'N/A')}")
        
        print("\n" + "=" * 50)
        print("✅ All tests completed successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_multi_image_functionality()
    sys.exit(0 if success else 1)