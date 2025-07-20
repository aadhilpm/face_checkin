#!/usr/bin/env python3
"""
Setup script for Face Check-in System on Frappe Cloud
This script handles Docker-specific configuration and dependencies
"""

import os
import sys
import subprocess

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'face_recognition',
        'dlib', 
        'numpy',
        'opencv-python',
        'Pillow',
        'scikit-image'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✓ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} is missing")
    
    return missing_packages

def install_dependencies():
    """Install missing dependencies"""
    print("Installing face recognition dependencies for Frappe Cloud...")
    
    # For Frappe Cloud Docker environment
    commands = [
        # System dependencies (if needed)
        "apt-get update",
        "apt-get install -y build-essential cmake libopenblas-dev liblapack-dev",
        
        # Python packages
        "pip install --upgrade pip",
        "pip install cmake",
        "pip install dlib==19.24.0",
        "pip install face-recognition==1.3.0",
        "pip install opencv-python==4.8.1.78",
        "pip install Pillow==10.0.0",
        "pip install scikit-image==0.21.0",
        "pip install imutils==0.5.4"
    ]
    
    for cmd in commands:
        try:
            print(f"Running: {cmd}")
            subprocess.run(cmd.split(), check=True, capture_output=True, text=True)
            print(f"✓ {cmd} completed successfully")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to run: {cmd}")
            print(f"Error: {e.stderr}")
            continue
        except Exception as e:
            print(f"✗ Error with command: {cmd}")
            print(f"Error: {str(e)}")
            continue

def setup_directories():
    """Setup required directories for face embeddings storage"""
    print("Setting up directories for face recognition data...")
    
    try:
        import frappe
        frappe.init()
        
        # Primary embedding directory (in app)
        try:
            app_embedding_dir = frappe.get_app_path('face_checkin', 'face_store', 'embeddings')
            os.makedirs(app_embedding_dir, exist_ok=True)
            print(f"✓ Created app embedding directory: {app_embedding_dir}")
        except Exception as e:
            print(f"⚠ Could not create app embedding directory: {str(e)}")
        
        # Fallback embedding directory (in site files)
        site_embedding_dir = os.path.join(frappe.get_site_path(), 'private', 'files', 'face_embeddings')
        os.makedirs(site_embedding_dir, exist_ok=True)
        print(f"✓ Created fallback embedding directory: {site_embedding_dir}")
        
        # Set proper permissions
        os.chmod(site_embedding_dir, 0o755)
        
    except Exception as e:
        print(f"✗ Error setting up directories: {str(e)}")
        
        # Manual fallback
        fallback_dir = "/home/frappe/frappe-bench/sites/face_embeddings"
        os.makedirs(fallback_dir, exist_ok=True)
        print(f"✓ Created manual fallback directory: {fallback_dir}")

def create_requirements_for_cloud():
    """Create a requirements.txt file optimized for Frappe Cloud"""
    requirements_content = """# Face Recognition Dependencies for Frappe Cloud Docker
# Optimized versions for better compatibility

# Core dependencies with specific versions
cmake==3.27.0
dlib==19.24.0
face-recognition==1.3.0
numpy==1.24.3
opencv-python==4.8.1.78
Pillow==10.0.0

# Image processing
scikit-image==0.21.0
imutils==0.5.4

# Optional performance improvements
redis==4.6.0
"""
    
    with open('/tmp/face_checkin_requirements.txt', 'w') as f:
        f.write(requirements_content)
    
    print("✓ Created optimized requirements file at /tmp/face_checkin_requirements.txt")
    print("Please install these dependencies in your Frappe Cloud environment:")
    print("pip install -r /tmp/face_checkin_requirements.txt")

def verify_installation():
    """Verify that the face recognition system is working"""
    print("Verifying face recognition installation...")
    
    try:
        import face_recognition
        import cv2
        import numpy as np
        from PIL import Image
        
        print("✓ All face recognition modules imported successfully")
        
        # Test basic functionality
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        face_locations = face_recognition.face_locations(test_image)
        print("✓ Face detection test completed")
        
        return True
        
    except Exception as e:
        print(f"✗ Verification failed: {str(e)}")
        return False

def main():
    """Main setup function"""
    print("=== Face Check-in System Setup for Frappe Cloud ===")
    print()
    
    # Check current dependencies
    missing = check_dependencies()
    
    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print("Creating requirements file for Frappe Cloud...")
        create_requirements_for_cloud()
        
        response = input("\nWould you like to attempt automatic installation? (y/n): ")
        if response.lower() == 'y':
            install_dependencies()
        else:
            print("Please install the dependencies manually using:")
            print("pip install -r /tmp/face_checkin_requirements.txt")
            return
    
    # Setup directories
    setup_directories()
    
    # Verify installation
    if verify_installation():
        print("\n✓ Face Check-in System setup completed successfully!")
        print("\nNext steps:")
        print("1. Add employee images via HR > Employee")
        print("2. Visit /employee-images to create face recognition data")
        print("3. Test the system at /checkin")
    else:
        print("\n⚠ Setup completed but verification failed.")
        print("Please check the error messages above and ensure all dependencies are properly installed.")

if __name__ == "__main__":
    main()