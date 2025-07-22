# Face Check-in System

Employee attendance tracking using facial recognition for Frappe/ERPNext with real-time face recognition, project tracking, and geolocation support.

## Screenshots

### System Setup
![Setup Page](https://github.com/user-attachments/assets/b958a0c7-666c-4463-8469-87a4363ff5f0)

### Face Recognition Check-in
![Check-in Page](https://github.com/user-attachments/assets/9ff4aa01-0cef-4d75-8f7a-983f60958c58)

### Employee Management
![Employee Management](https://github.com/user-attachments/assets/075be19c-8b4d-43ed-9726-adfbd93a2882)

## Features

- Advanced face recognition using OpenCV with 512-dimensional embeddings
- Real-time face detection and recognition with live camera feed
- Project tracking - associate check-ins with specific ERPNext projects
- Geolocation support with GPS coordinate tracking via HR Settings
- Progressive Web App with offline support and service worker
- Auto check-in/check-out detection based on last entry
- Image quality validation for accurate recognition
- Persistent face data storage in site directory
- Bulk employee enrollment and management
- Configurable recognition tolerance settings
- Modern responsive UI design
- Comprehensive system diagnostics and troubleshooting

## Installation

### Prerequisites
- Frappe Framework v15+
- ERPNext with HRMS module
- Python 3.10+

### Install
```bash
bench get-app https://github.com/aadhilpm/face_checkin
bench --site your-site install-app face_checkin
pip install opencv-python Pillow numpy
```

### System Dependencies (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install -y libopencv-dev libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev
bench restart
```

## Data Storage

All employee check-in data is stored in the **Employee Checkin** doctype in ERPNext, maintaining full integration with the existing HRMS system.

## Usage

1. **Setup** - Visit `/setup` to check system status and dependencies
2. **Employee Enrollment** - Visit `/employee-images` to upload photos and manage face data
3. **Daily Check-ins** - Visit `/checkin` for face recognition check-in/out
4. **Geolocation** - Enable "Allow Geolocation Tracking" in ERPNext HR Settings (Optional)

## API Reference

### Main API Endpoints
```python
# Face recognition
face_api.upload_face(employee_id, image_base64=None)
face_api.recognize_and_checkin(image_base64, project=None, device_id=None, log_type=None, latitude=None, longitude=None)

# System management
face_api.check_system_status()
face_api.get_detailed_status()
face_api.get_projects()
face_api.get_geolocation_settings()

# Employee management
face_api.bulk_enroll_from_employee_images()
face_api.check_enrollment_status(employee_ids=None)
face_api.upload_employee_image(employee_id, image_base64, filename)
face_api.delete_face_data(employee_id)
face_api.get_checkin_status(employee_id=None, project=None)
```

## Configuration

### Site Config Options
Add to `site_config.json`:
```json
{
  "face_recognition_initial_tolerance": 0.7,
  "face_recognition_strict_tolerance": 0.55,
  "face_recognition_min_quality": 60
}
```

## Geolocation Integration

- Controlled via ERPNext HR Settings
- GPS coordinates recorded with check-ins when enabled
- Uses ERPNext's built-in distance validation
- Requires user permission for location access

## Requirements

- Frappe Framework v15+
- ERPNext with HRMS module
- Python 3.10+
- OpenCV 4.5.0+ (`opencv-python`)
- Pillow 8.3.0+
- NumPy 1.21.0+

## Troubleshooting

### Dependencies Not Found
```bash
pip install --upgrade opencv-python Pillow numpy
bench restart
```

### Face Recognition Issues
1. Check system status at `/setup`
2. Verify OpenCV installation
3. Ensure camera permissions granted
4. Re-enroll employees with better quality images

### Poor Recognition Accuracy
- Ensure good lighting during enrollment
- Adjust recognition tolerance in site config
- Check face image quality scores

## License

MIT License

## Support

- Issues: [GitHub Issues](https://github.com/aadhilpm/face_checkin/issues)

Built by Aadhil Badeel Technology