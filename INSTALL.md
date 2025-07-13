# Face Check-in System Installation Guide

## Prerequisites

### System Requirements
- **Operating System**: Ubuntu 18.04+, CentOS 7+, or macOS 10.14+
- **Python**: 3.7 or higher
- **Frappe/ERPNext**: Version 13.0 or higher
- **RAM**: Minimum 4GB (8GB+ recommended for face recognition)
- **Storage**: At least 2GB free space

### System Dependencies

#### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    python3-dev \
    python3-pip
```

#### CentOS/RHEL
```bash
sudo yum groupinstall "Development Tools"
sudo yum install -y \
    cmake \
    openblas-devel \
    lapack-devel \
    gtk3-devel \
    python3-devel
```

#### macOS
```bash
# Install Xcode command line tools
xcode-select --install

# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install cmake
```

## Installation Steps

### 1. Navigate to Your Bench Directory
```bash
cd /path/to/your/bench
```

### 2. Get the App
```bash
bench get-app https://github.com/your-repo/face_checkin
```

### 3. Install Python Dependencies
```bash
# Activate bench environment
source env/bin/activate

# Install face recognition dependencies
pip install -r apps/face_checkin/requirements.txt
```

### 4. Install App to Site
```bash
# Replace 'your-site' with your actual site name
bench --site your-site install-app face_checkin
```

### 5. Setup Custom Field (Required)
The app requires a custom field in Employee Checkin doctype. Run this in your site console:

```bash
bench --site your-site console
```

Then execute:
```python
import frappe

# Create custom field for project selection
custom_field = {
    "doctype": "Custom Field",
    "dt": "Employee Checkin",
    "fieldname": "custom_project",
    "label": "Project",
    "fieldtype": "Link",
    "options": "Project",
    "insert_after": "device_id",
    "allow_on_submit": 1
}

if not frappe.db.exists("Custom Field", {"dt": "Employee Checkin", "fieldname": "custom_project"}):
    doc = frappe.get_doc(custom_field)
    doc.insert()
    print("Custom field created successfully")
else:
    print("Custom field already exists")

frappe.db.commit()
```

### 6. Restart Services
```bash
bench restart
```

### 7. Verify Installation
Visit: `https://your-site/setup` to verify face recognition dependencies are properly installed.

## Post-Installation Setup

### 1. Configure Employee Images
- Go to **HR > Employee**
- Add photos for employees who will use face check-in
- Use clear, well-lit photos facing the camera

### 2. Create Face Recognition Data
- Visit `https://your-site/employee-images`
- Use "Bulk Enroll All" to process all employee images
- Or individually create face data for each employee

### 3. Test the System
- Visit `https://your-site/checkin`
- Select a project and test face recognition

## Troubleshooting

### Common Issues

#### 1. Face Recognition Import Error
```
ModuleNotFoundError: No module named 'face_recognition'
```
**Solution**: Ensure you've installed the requirements and are in the correct Python environment:
```bash
source env/bin/activate
pip install face-recognition
```

#### 2. dlib Compilation Error
**Solution**: Install system dependencies first, then:
```bash
pip install cmake
pip install dlib
pip install face-recognition
```

#### 3. Permission Errors
Ensure the bench user has write permissions to the face_checkin app directory:
```bash
sudo chown -R [bench-user]:[bench-user] apps/face_checkin/
```

#### 4. Camera Access Issues
- Ensure the site is served over HTTPS for camera access
- Check browser permissions for camera access

### Performance Optimization

#### 1. Enable Redis Caching
In your site's site_config.json:
```json
{
    "redis_cache": "redis://localhost:13000",
    "redis_queue": "redis://localhost:11000"
}
```

#### 2. Optimize Face Recognition
For production with many employees:
- Consider reducing face image resolution to 300x300px
- Implement face recognition queue for bulk processing
- Use SSD storage for faster face data access

## Security Considerations

### 1. HTTPS Required
Face recognition requires camera access, which browsers only allow over HTTPS.

### 2. Data Privacy
- Face recognition data is stored locally in the face_store directory
- Consider implementing data retention policies
- Ensure compliance with local privacy laws (GDPR, etc.)

### 3. Access Control
- Use Frappe's role-based permissions
- Limit access to employee image management
- Implement audit logs for face recognition access

## Maintenance

### Regular Tasks
1. **Backup face recognition data**: Include `apps/face_checkin/face_store/` in backups
2. **Monitor storage**: Face embeddings accumulate over time
3. **Update dependencies**: Regularly update face recognition libraries
4. **Performance monitoring**: Monitor API response times

### Updates
```bash
cd /path/to/your/bench
bench get-app face_checkin  # Pull latest changes
bench --site your-site migrate
bench restart
```

## Support

For issues and support:
1. Check the troubleshooting section above
2. Review system logs: `bench logs`
3. Check browser console for JavaScript errors
4. Verify face recognition system status at `/setup`