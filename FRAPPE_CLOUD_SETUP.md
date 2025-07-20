# Face Check-in System - Frappe Cloud Setup Guide

This guide is specifically for deploying the Face Check-in System on **Frappe Cloud** using Docker containers.

## Prerequisites

- Frappe Cloud account with custom app deployment enabled
- Admin access to your Frappe Cloud site
- Access to Frappe Cloud's container management (if available)

## Installation Steps

### 1. Deploy the App to Frappe Cloud

#### Option A: Via Frappe Cloud Dashboard
1. Log into your Frappe Cloud dashboard
2. Navigate to your site
3. Go to "Apps" section
4. Add custom app using this repository URL

#### Option B: Via Bench Commands (if you have access)
```bash
bench get-app https://github.com/your-repo/face_checkin
bench --site your-site install-app face_checkin
```

### 2. Install Dependencies

The app requires additional Python packages that may not be included in the standard Frappe Cloud container.


1. Go to your site's setup page: `https://your-site.frappe.cloud/setup`
2. Check the system status to verify dependencies are installed
3. The page will show:
   - Docker environment detection
   - Dependency versions
   - Available storage directories
   - Setup suggestions

### 5. Initial Setup

1. **Add Employee Images**:
   - Go to HR > Employee
   - Add clear, well-lit photos for employees
   - Ensure photos show face clearly without obstructions

2. **Create Face Recognition Data**:
   - Visit `https://your-site.frappe.cloud/employee-images`
   - Use "Bulk Enroll All" to process all employee images
   - Or process employees individually

3. **Test the System**:
   - Visit `https://your-site.frappe.cloud/checkin`
   - Test face recognition with enrolled employees

## Troubleshooting for Frappe Cloud

### Issue: Dependencies Not Available

**Symptoms:**
- Error: "Face recognition dependencies not installed"
- Import errors for face_recognition, dlib, etc.

**Solutions:**

### Issue: Face Data Not Storing

**Symptoms:**
- "0 enrolled" showing in dashboard
- Face recognition fails with "No registered faces"

**Solutions:**

1. **Check Storage Permissions**:
   - The app creates fallback directories automatically
   - Verify the site has write permissions

2. **Manual Directory Creation**:
   - Contact support to create: `/home/frappe/frappe-bench/sites/[site]/private/files/face_embeddings/`
   - Ensure directory has proper permissions (755)

3. **Use Alternative Storage**:
   - The app will automatically use the fallback directory
   - Check `/setup` page to see which directories are available

### Issue: Camera Access in Browser

**Symptoms:**
- Camera not working on face check-in page
- Browser permission errors

**Solutions:**

1. **HTTPS Required**:
   - Frappe Cloud sites use HTTPS by default
   - Ensure you're accessing via `https://your-site.frappe.cloud`

2. **Browser Permissions**:
   - Allow camera access when prompted
   - Check browser settings for camera permissions
   - Try different browsers if issues persist

### Issue: Performance Problems

**Symptoms:**
- Slow face recognition
- Timeouts during enrollment

**Solutions:**

1. **Optimize Images**:
   - Resize employee photos to 300x300 pixels
   - Use JPEG format with moderate compression
   - Avoid very large image files

2. **Batch Processing**:
   - Process employees in smaller batches instead of "Bulk Enroll All"
   - Process during off-peak hours

3. **Contact Support**:
   - Request increased memory/CPU allocation if available
   - Consider upgrading Frappe Cloud plan for better performance

## Monitoring and Maintenance

### 1. Regular Checks
- Monitor face recognition accuracy
- Check storage usage for face embeddings
- Verify system status periodically at `/setup`

### 2. Updates
- App updates will be deployed through Frappe Cloud
- Dependencies may need to be reinstalled after major updates
- Contact support for dependency-related issues after updates

### 3. Backup Considerations
- Face recognition data is stored in site files
- Included in standard Frappe Cloud backups
- Can be manually backed up from the embeddings directory

## Support and Resources

### Getting Help

1. **App-Specific Issues**:
   - Check the `/setup` page for detailed system status
   - Review browser console for JavaScript errors
   - Check Frappe error logs in the dashboard

2. **Frappe Cloud Issues**:
   - Contact Frappe Cloud support for:
     - Dependency installation
     - Container configuration
     - Storage/permission issues
     - Performance optimization

3. **Feature Requests**:
   - Report issues and request features through the app repository
   - Provide Frappe Cloud-specific context when reporting

### Useful Commands (if you have access)

```bash
# Check app status
bench --site your-site console

# Check face recognition system
import face_checkin.api.face_api as face_api
face_api.check_system_status()

# Check enrollment status
face_api.check_enrollment_status()

# Manual face enrollment
face_api.upload_face('EMP-001')
```

## Security Considerations for Frappe Cloud

1. **Data Privacy**:
   - Face recognition data stays within your Frappe Cloud instance
   - Not shared with external services
   - Complies with Frappe Cloud's data protection policies

2. **Access Control**:
   - Use Frappe's built-in role-based permissions
   - Limit access to employee image management
   - Monitor check-in logs for security

3. **HTTPS/SSL**:
   - Automatically handled by Frappe Cloud
   - Required for camera access in browsers

This setup guide should help you successfully deploy the Face Check-in System on Frappe Cloud. Contact support if you encounter issues specific to the containerized environment.