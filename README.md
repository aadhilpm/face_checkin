# Face Check-in System

Employee attendance using facial recognition for Frappe/ERPNext.

## Features

- Face recognition with camera
- Project tracking 
- Geolocation support
- Progressive Web App
- Auto check-in/out detection

## Installation

```bash
bench get-app https://github.com/aadhilpm/face_checkin
bench --site your-site install-app face_checkin
pip install -r apps/face_checkin/requirements.txt
bench restart
```

## Usage

1. Visit `/setup` to check system
2. Visit `/employee-images` to upload photos  
3. Visit `/checkin` for daily attendance

## Requirements

- Frappe Framework v15+
- ERPNext with HRMS
- Python 3.10+

## License

MIT License