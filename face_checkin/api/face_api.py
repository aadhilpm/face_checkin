import frappe
import os
from datetime import datetime
import base64
from io import BytesIO

# Optional imports for face recognition functionality
try:
    import face_recognition
    import numpy as np
    from PIL import Image
    FACE_RECOGNITION_AVAILABLE = True
except ImportError as e:
    frappe.log_error(f"Face recognition dependencies not available: {str(e)}")
    FACE_RECOGNITION_AVAILABLE = False
    # Create dummy modules to prevent crashes
    class DummyModule:
        pass
    face_recognition = DummyModule()
    np = DummyModule()
    Image = DummyModule()

EMBEDDING_DIR = frappe.get_app_path('face_checkin', 'face_store', 'embeddings')

@frappe.whitelist()
def upload_face(employee_id, image_base64=None):
    """
    Create face embedding from employee image
    If image_base64 is not provided, use employee's image from record
    """
    if not FACE_RECOGNITION_AVAILABLE:
        return {
            "status": "error",
            "message": "Face recognition dependencies not installed. Please install face-recognition, numpy, and pillow packages."
        }
    
    if not os.path.exists(EMBEDDING_DIR):
        os.makedirs(EMBEDDING_DIR)

    if image_base64:
        # Use provided image
        image_data = base64.b64decode(image_base64.split(",")[-1])
        img = Image.open(BytesIO(image_data)).convert("RGB")
    else:
        # Use employee record image
        employee = frappe.get_doc("Employee", employee_id)
        if not employee.image:
            return {
                "status": "error",
                "message": f"No image found for employee {employee_id}"
            }
        
        # Get image file from Frappe
        image_file = frappe.get_doc("File", {"file_url": employee.image})
        image_path = frappe.get_site_path() + image_file.file_url
        
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            return {
                "status": "error", 
                "message": f"Failed to load employee image: {str(e)}"
            }

    img_np = np.array(img)

    # Detect face
    face_locations = face_recognition.face_locations(img_np)
    if not face_locations:
        return {
            "status": "error",
            "message": "No face detected in image"
        }

    encodings = face_recognition.face_encodings(img_np, face_locations)
    if not encodings:
        return {
            "status": "error",
            "message": "Failed to extract face encoding"
        }

    embedding = encodings[0]

    # Save embedding to disk
    filepath = os.path.join(EMBEDDING_DIR, f"{employee_id}.npy")
    np.save(filepath, embedding)

    return {
        "status": "success",
        "message": f"Face embedding saved for {employee_id}"
    }

@frappe.whitelist()
def bulk_enroll_from_employee_images():
    """
    Create face embeddings for all employees who have images but no face encodings
    """
    if not FACE_RECOGNITION_AVAILABLE:
        return {
            "status": "error",
            "message": "Face recognition dependencies not installed. Please install required packages first."
        }
    
    if not os.path.exists(EMBEDDING_DIR):
        os.makedirs(EMBEDDING_DIR)

    employees_with_images = frappe.db.sql("""
        SELECT name, employee_name, image 
        FROM `tabEmployee` 
        WHERE image IS NOT NULL AND image != ''
    """, as_dict=True)

    results = []
    for employee in employees_with_images:
        embedding_file = os.path.join(EMBEDDING_DIR, f"{employee.name}.npy")
        
        # Skip if embedding already exists
        if os.path.exists(embedding_file):
            continue
            
        result = upload_face(employee.name)
        results.append({
            "employee": employee.name,
            "employee_name": employee.employee_name,
            "result": result
        })

    return {
        "status": "success",
        "message": f"Processed {len(results)} employees",
        "details": results
    }

@frappe.whitelist()
def recognize_and_checkin(image_base64, project=None, device_id=None, log_type=None):
    """
    Recognize employee from face and create checkin record
    """
    if not FACE_RECOGNITION_AVAILABLE:
        return {
            "status": "error",
            "message": "Face recognition system not available. Please contact administrator."
        }
    
    try:
        if not os.path.exists(EMBEDDING_DIR):
            return {
                "status": "error",
                "message": "No employee faces registered in system"
            }

        # Decode base64 image
        image_data = base64.b64decode(image_base64.split(",")[-1])
        img = Image.open(BytesIO(image_data)).convert("RGB")
        img_np = np.array(img)

        # Detect face in the image
        face_locations = face_recognition.face_locations(img_np)
        if not face_locations:
            return {
                "status": "error",
                "message": "No face detected in image"
            }

        # Get face encoding
        face_encodings = face_recognition.face_encodings(img_np, face_locations)
        if not face_encodings:
            return {
                "status": "error",
                "message": "Could not extract face features"
            }

        face_encoding = face_encodings[0]

        # Load all known employee faces
        known_encodings = []
        employee_ids = []
        
        for filename in os.listdir(EMBEDDING_DIR):
            if filename.endswith('.npy'):
                employee_id = filename[:-4]  # Remove .npy extension
                filepath = os.path.join(EMBEDDING_DIR, filename)
                
                # Verify employee exists in system
                if frappe.db.exists("Employee", employee_id):
                    encoding = np.load(filepath)
                    known_encodings.append(encoding)
                    employee_ids.append(employee_id)

        if not known_encodings:
            return {
                "status": "error",
                "message": "No registered employee faces found"
            }

        # Compare face with known faces
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.6)
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)

        if not any(matches):
            return {
                "status": "error", 
                "message": "Employee not recognized"
            }

        # Get the best match
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            recognized_employee = employee_ids[best_match_index]
            confidence = 1 - face_distances[best_match_index]
            
            # Get employee details
            employee = frappe.get_doc("Employee", recognized_employee)
            
            # Determine log type if not provided
            if not log_type:
                log_type = determine_log_type(recognized_employee)
            
            # Create Employee Checkin record
            checkin = frappe.new_doc("Employee Checkin")
            checkin.employee = recognized_employee
            checkin.employee_name = employee.employee_name
            checkin.time = datetime.now()
            checkin.log_type = log_type
            checkin.device_id = device_id or "Face Recognition System"
            if project:
                checkin.custom_project = project
            checkin.insert()
            
            return {
                "status": "success",
                "message": f"{log_type} recorded for {employee.employee_name}",
                "employee_id": recognized_employee,
                "employee_name": employee.employee_name,
                "log_type": log_type,
                "time": checkin.time,
                "confidence": round(confidence * 100, 2),
                "checkin_id": checkin.name
            }
        else:
            return {
                "status": "error",
                "message": "Employee not recognized"
            }
            
    except Exception as e:
        frappe.log_error(f"Face recognition error: {str(e)}")
        return {
            "status": "error",
            "message": "System error during face recognition"
        }

def determine_log_type(employee_id):
    """
    Determine if this should be check-in or check-out based on last log
    """
    last_checkin = frappe.db.get_value(
        "Employee Checkin",
        filters={
            "employee": employee_id,
        },
        fieldname=["log_type", "time"],
        order_by="time desc"
    )
    
    if not last_checkin:
        return "IN"  # First entry of the day
    
    # If last entry was IN, next should be OUT
    if last_checkin[0] == "IN":
        return "OUT"
    else:
        return "IN"

@frappe.whitelist()
def get_checkin_status(employee_id=None, project=None):
    """
    Get current checkin status for employee or all recent checkins
    """
    if employee_id:
        # Get last checkin for specific employee
        last_checkin = frappe.db.get_value(
            "Employee Checkin",
            filters={"employee": employee_id},
            fieldname=["employee_name", "log_type", "time", "name"],
            order_by="time desc"
        )
        
        if last_checkin:
            return {
                "employee_name": last_checkin[0],
                "last_log_type": last_checkin[1],
                "last_time": last_checkin[2],
                "checkin_id": last_checkin[3]
            }
        else:
            return {"message": "No checkin records found"}
    else:
        # Get recent checkins for display
        filters = "WHERE DATE(time) = CURDATE()"
        if project:
            filters += f" AND custom_project = '{project}'"
            
        recent_checkins = frappe.db.sql(f"""
            SELECT employee_name, log_type, time, employee, custom_project
            FROM `tabEmployee Checkin`
            {filters}
            ORDER BY time DESC
            LIMIT 10
        """, as_dict=True)
        
        return {"recent_checkins": recent_checkins}

@frappe.whitelist()
def get_projects():
    """
    Get list of active projects for selection
    """
    try:
        # Check if Project doctype exists
        if not frappe.db.exists("DocType", "Project"):
            return {
                "status": "error",
                "message": "Project doctype not found. Please ensure ERPNext is properly installed."
            }
        
        # Try different approaches to get projects
        projects = []
        
        # Method 1: Try to get projects with "Open" status
        try:
            projects = frappe.get_all(
                "Project",
                filters={
                    "status": "Open",
                    "is_active": "Yes"
                },
                fields=["name", "project_name", "status"],
                order_by="name asc"
            )
            
            # Ensure project_name field exists, use name if empty
            for project in projects:
                if not project.get("project_name"):
                    project["project_name"] = project["name"]
        except Exception as e:
            frappe.log_error(f"Method 1 failed: {str(e)}")
        
        # Method 2: If no projects found, try different status values
        if not projects:
            try:
                projects = frappe.db.sql("""
                    SELECT name, 
                           COALESCE(NULLIF(project_name, ''), name) as project_name, 
                           status
                    FROM `tabProject` 
                    WHERE (is_active = 'Yes' OR is_active IS NULL)
                    AND (status = 'Open' OR status = 'Completed' OR status = 'Template' OR status IS NULL)
                    ORDER BY 
                        CASE 
                            WHEN status = 'Open' THEN 1
                            WHEN status = 'Completed' THEN 2
                            WHEN status = 'Template' THEN 3
                            ELSE 4
                        END,
                        name ASC
                    LIMIT 100
                """, as_dict=True)
            except Exception as e:
                frappe.log_error(f"Method 2 failed: {str(e)}")
        
        # Method 3: If still no projects, get all active projects
        if not projects:
            try:
                projects = frappe.get_all(
                    "Project",
                    filters={
                        "is_active": "Yes"
                    },
                    fields=["name", "project_name", "status"],
                    order_by="creation desc",
                    limit=50
                )
                
                # Ensure project_name field exists, use name if empty
                for project in projects:
                    if not project.get("project_name"):
                        project["project_name"] = project["name"]
            except Exception as e:
                frappe.log_error(f"Method 3 failed: {str(e)}")
        
        # Method 4: If still no projects, get all projects (no filters)
        if not projects:
            try:
                projects = frappe.get_all(
                    "Project",
                    fields=["name", "project_name", "status"],
                    order_by="creation desc",
                    limit=50
                )
                
                # Ensure project_name field exists, use name if empty
                for project in projects:
                    if not project.get("project_name"):
                        project["project_name"] = project["name"]
            except Exception as e:
                frappe.log_error(f"Method 4 failed: {str(e)}")
        
        if not projects:
            return {
                "status": "error", 
                "message": "No projects found in the system. Please create projects in ERPNext first."
            }
        
        return {
            "status": "success",
            "projects": projects
        }
        
    except Exception as e:
        frappe.log_error(f"Error fetching projects: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to fetch projects: {str(e)}"
        }

@frappe.whitelist()
def check_system_status():
    """
    Check system dependencies and status
    """
    return {
        "face_recognition_available": FACE_RECOGNITION_AVAILABLE,
        "embedding_directory_exists": os.path.exists(EMBEDDING_DIR),
        "embedding_directory_path": EMBEDDING_DIR,
        "user": frappe.session.user,
        "user_roles": frappe.get_roles()
    }

@frappe.whitelist()
def check_enrollment_status(employee_ids=None):
    """
    Check face enrollment status for employees
    Returns status for specific employees or all employees with images
    """
    if not os.path.exists(EMBEDDING_DIR):
        return {
            "status": "error",
            "message": "Face embeddings directory not found"
        }
    
    try:
        # Get list of existing embedding files
        existing_embeddings = set()
        for filename in os.listdir(EMBEDDING_DIR):
            if filename.endswith('.npy'):
                employee_id = filename[:-4]  # Remove .npy extension
                existing_embeddings.add(employee_id)
        
        # If specific employee IDs provided, check only those
        if employee_ids:
            if isinstance(employee_ids, str):
                employee_ids = [employee_ids]
            
            results = {}
            for emp_id in employee_ids:
                # Verify employee exists in system
                if frappe.db.exists("Employee", emp_id):
                    employee = frappe.get_doc("Employee", emp_id)
                    has_image = bool(employee.image and employee.image.strip())
                    has_face_data = emp_id in existing_embeddings
                    
                    results[emp_id] = {
                        "employee_name": employee.employee_name,
                        "has_image": has_image,
                        "has_face_data": has_face_data,
                        "enrollment_complete": has_image and has_face_data
                    }
                else:
                    results[emp_id] = {
                        "error": "Employee not found"
                    }
            
            return {
                "status": "success",
                "enrollment_status": results
            }
        
        # Return status for all employees with images
        employees_with_images = frappe.db.sql("""
            SELECT name, employee_name, image 
            FROM `tabEmployee` 
            WHERE image IS NOT NULL AND image != ''
        """, as_dict=True)
        
        enrollment_summary = {
            "total_employees_with_images": len(employees_with_images),
            "employees_enrolled": 0,
            "employees_pending": 0,
            "details": {}
        }
        
        for employee in employees_with_images:
            has_face_data = employee.name in existing_embeddings
            enrollment_summary["details"][employee.name] = {
                "employee_name": employee.employee_name,
                "has_image": True,
                "has_face_data": has_face_data,
                "enrollment_complete": has_face_data
            }
            
            if has_face_data:
                enrollment_summary["employees_enrolled"] += 1
            else:
                enrollment_summary["employees_pending"] += 1
        
        return {
            "status": "success",
            "enrollment_summary": enrollment_summary
        }
        
    except Exception as e:
        frappe.log_error(f"Error checking enrollment status: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to check enrollment status: {str(e)}"
        }

@frappe.whitelist()
def debug_project_status():
    """
    Debug endpoint to help troubleshoot project loading issues
    Only available to System Manager role
    """
    if "System Manager" not in frappe.get_roles():
        return {"error": "Access denied. System Manager role required."}
    
    try:
        debug_info = {
            "user": frappe.session.user,
            "roles": frappe.get_roles(),
            "project_doctype_exists": frappe.db.exists("DocType", "Project"),
            "project_count": 0,
            "sample_projects": [],
            "error_details": []
        }
        
        if debug_info["project_doctype_exists"]:
            # Count total projects
            try:
                debug_info["project_count"] = frappe.db.count("Project")
            except Exception as e:
                debug_info["error_details"].append(f"Count error: {str(e)}")
            
            # Get sample projects with all details
            try:
                sample_projects = frappe.db.sql("""
                    SELECT name, project_name, status, is_active, creation
                    FROM `tabProject`
                    ORDER BY creation DESC
                    LIMIT 5
                """, as_dict=True)
                debug_info["sample_projects"] = sample_projects
            except Exception as e:
                debug_info["error_details"].append(f"Sample query error: {str(e)}")
                
            # Test project permissions
            try:
                debug_info["can_read_project"] = frappe.has_permission("Project", "read")
            except Exception as e:
                debug_info["error_details"].append(f"Permission error: {str(e)}")
        
        return debug_info
        
    except Exception as e:
        return {"error": str(e)}


@frappe.whitelist()
def upload_employee_image(employee_id, image_base64, filename="employee_photo.jpg"):
    """
    Upload image directly to employee record and create face embedding
    """
    try:
        # Check if employee exists
        if not frappe.db.exists("Employee", employee_id):
            return {
                "status": "error",
                "message": f"Employee {employee_id} not found"
            }

        # Decode base64 image
        import base64
        from io import BytesIO
        
        if "," in image_base64:
            image_data = base64.b64decode(image_base64.split(",")[1])
        else:
            image_data = base64.b64decode(image_base64)

        # Create file in Frappe
        file_doc = frappe.get_doc({
            "doctype": "File",
            "file_name": f"{employee_id}_{filename}",
            "content": image_data,
            "is_private": 0,
            "folder": "Home/Attachments"
        })
        file_doc.save()

        # Update employee record with image
        employee = frappe.get_doc("Employee", employee_id)
        employee.image = file_doc.file_url
        employee.save()

        # Create face embedding if face recognition is available
        if FACE_RECOGNITION_AVAILABLE:
            face_result = upload_face(employee_id)
        else:
            face_result = {"status": "info", "message": "Image uploaded but face recognition not available"}

        return {
            "status": "success",
            "message": f"Image uploaded successfully for {employee.employee_name}",
            "file_url": file_doc.file_url,
            "face_enrollment": face_result
        }

    except Exception as e:
        frappe.log_error(f"Error uploading employee image: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to upload image: {str(e)}"
        }
