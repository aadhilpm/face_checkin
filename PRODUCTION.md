# Production Deployment Guide

## Production Checklist

### Before Deployment

1. **System Requirements Met**
   - [ ] Python 3.7+ installed
   - [ ] System dependencies installed (cmake, build-essential, etc.)
   - [ ] Frappe/ERPNext 13.0+ running
   - [ ] Minimum 4GB RAM available
   - [ ] SSL certificate configured (required for camera access)

2. **Dependencies Installed**
   - [ ] face-recognition library installed
   - [ ] All requirements.txt packages installed
   - [ ] Custom field created in Employee Checkin

3. **Data Preparation**
   - [ ] Employee records created with clear photos
   - [ ] Projects created in ERPNext
   - [ ] User permissions configured

### Deployment Steps

1. **Install App**
   ```bash
   cd /path/to/your/bench
   bench get-app face_checkin
   bench --site your-site install-app face_checkin
   bench restart
   ```

2. **Verify Installation**
   - Visit `/setup` to check system status
   - Ensure face recognition dependencies show as available
   - Test project loading

3. **Configure SSL (Required)**
   Face recognition requires camera access, which browsers only allow over HTTPS.
   
   ```bash
   # Using Let's Encrypt
   sudo certbot --nginx -d your-domain.com
   
   # Or configure your reverse proxy (Nginx/Apache) for SSL
   ```

### Performance Optimization

1. **Face Recognition Optimization**
   - Optimize employee photos to 300x300px before upload
   - Use JPEG format with moderate compression
   - Limit to 500 active employees for best performance

2. **Database Optimization**
   ```sql
   -- Add indexes for better performance
   ALTER TABLE `tabEmployee Checkin` ADD INDEX idx_employee_time (employee, time);
   ALTER TABLE `tabEmployee Checkin` ADD INDEX idx_custom_project (custom_project);
   ```

3. **Caching Setup**
   Enable Redis for better performance:
   ```json
   // site_config.json
   {
     "redis_cache": "redis://localhost:13000",
     "redis_queue": "redis://localhost:11000"
   }
   ```

### Security Configuration

1. **Access Control**
   - Create dedicated roles for face check-in operators
   - Limit access to employee image management
   - Enable audit logs for face recognition events

2. **Data Privacy**
   - Face embeddings are stored locally in `face_store/embeddings/`
   - Consider encryption at rest for sensitive data
   - Implement data retention policies

3. **Network Security**
   - Use HTTPS only
   - Configure firewall rules
   - Limit API access to authenticated users only

### Monitoring & Maintenance

1. **System Monitoring**
   ```bash
   # Monitor face recognition performance
   tail -f logs/worker.error.log | grep face_checkin
   
   # Check disk usage for face embeddings
   du -sh apps/face_checkin/face_store/
   ```

2. **Regular Maintenance**
   - Weekly backup of face_store directory
   - Monthly cleanup of old check-in records
   - Quarterly update of face recognition libraries
   - Monitor API response times

3. **Performance Metrics**
   - Track face recognition accuracy
   - Monitor API response times (<2 seconds ideal)
   - Watch memory usage during bulk operations

### Troubleshooting Production Issues

1. **Face Recognition Fails**
   ```bash
   # Check dependencies
   bench --site your-site console
   >>> import face_recognition
   >>> print("Face recognition working")
   ```

2. **Camera Access Denied**
   - Ensure site is served over HTTPS
   - Check browser permissions
   - Verify camera hardware

3. **Project Loading Issues**
   ```bash
   # Check Project doctype exists
   bench --site your-site console
   >>> import frappe
   >>> frappe.db.exists("DocType", "Project")
   ```

4. **Performance Issues**
   - Monitor CPU/memory during face recognition
   - Consider image preprocessing
   - Implement queue for bulk operations

### Backup Strategy

1. **Critical Data**
   - ERPNext database (includes Employee Checkin records)
   - Face embeddings: `apps/face_checkin/face_store/`
   - Employee images in Files doctype

2. **Backup Commands**
   ```bash
   # Database backup
   bench --site your-site backup --with-files
   
   # Face embeddings backup
   tar -czf face_embeddings_$(date +%Y%m%d).tar.gz apps/face_checkin/face_store/
   ```

3. **Restore Process**
   ```bash
   # Restore database
   bench --site your-site restore /path/to/backup.sql.gz --with-private-files --with-public-files
   
   # Restore face embeddings
   tar -xzf face_embeddings_backup.tar.gz -C /
   ```

### Scaling Considerations

1. **High-Volume Environments**
   - Consider dedicated worker for face recognition
   - Implement queue system for bulk processing
   - Use load balancer for multiple instances

2. **Multi-Site Deployment**
   - Each site needs separate face_store directory
   - Configure site-specific paths in custom app config

3. **Performance Limits**
   - 500+ employees: Consider optimizations
   - 1000+ employees: Implement background processing
   - 2000+ employees: Consider cluster deployment

### Production Environment Variables

Create `apps/face_checkin/face_checkin/config.py`:
```python
# Production configuration
FACE_RECOGNITION_TOLERANCE = 0.6  # Adjust for accuracy vs speed
MAX_FACE_ENCODINGS = 1000  # Memory management
ENABLE_FACE_CACHE = True  # Cache embeddings in memory
LOG_LEVEL = "WARNING"  # Reduce log verbosity
```

### Health Checks

Implement monitoring endpoints:
- `/api/method/face_checkin.api.face_api.check_system_status` - System health
- Monitor response times and error rates
- Set up alerts for face recognition failures

### Support & Maintenance Contact

- Technical Issues: Check INSTALL.md troubleshooting section
- Performance Issues: Monitor logs and system resources
- Security Updates: Regularly update face-recognition library
- Feature Requests: Create GitHub issues with detailed requirements