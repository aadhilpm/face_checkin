# Face Checkin

Employee attendance tracking using facial recognition for Frappe/ERPNext.

## Features

- **Facial Recognition**: OpenCV-based face recognition for employee check-in/out
- **Project Tracking**: Associate check-ins with specific projects  
- **Persistent Storage**: Face data survives app updates (stored in site directory)
- **PWA Support**: Works offline with service worker
- **Quality Validation**: Ensures good face image quality for accurate recognition

## Installation

```bash
# Install the app
bench get-app https://github.com/your-repo/face_checkin
bench --site your-site install-app face_checkin

# Install dependencies
pip install opencv-python Pillow numpy
```

## Usage

1. **Setup**: Visit `/setup` to upload employee photos and create face embeddings
2. **Check-in**: Visit `/checkin` for face recognition check-in/out
3. **Management**: View employee images at `/employee-images`

## API Endpoints

- `face_api.upload_face(employee_id, image_base64)` - Create face embedding
- `face_api.recognize_and_checkin(image_base64, project, device_id)` - Face recognition check-in
- `face_api.get_projects()` - Get available projects
- `face_api.get_checkin_status()` - Get recent check-ins

## Requirements

- Frappe Framework v13+
- ERPNext
- OpenCV Python
- PIL/Pillow
- NumPy

## License

MIT