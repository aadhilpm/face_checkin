import frappe
import os
from datetime import datetime
import base64
from io import BytesIO
import glob

# Optional imports for face recognition functionality
try:
    from face_checkin.utils.face_recognition_simple import (
        face_locations, face_encodings, compare_faces, face_distance,
        get_face_recognition, FACE_RECOGNITION_AVAILABLE
    )
    import numpy as np
    from PIL import Image
except ImportError as e:
    try:
        frappe.log_error(f"Face recognition dependencies not available: {str(e)}")
    except:
        print(f"Face recognition dependencies not available: {str(e)}")
    FACE_RECOGNITION_AVAILABLE = False
    # Create dummy modules to prevent crashes
    class DummyModule:
        pass
    face_locations = DummyModule()
    face_encodings = DummyModule()
    compare_faces = DummyModule()
    face_distance = DummyModule()
    np = DummyModule()
    Image = DummyModule()

def get_embedding_directory():
    """
    Get the embedding directory with fallback options
    """
    # List of possible embedding directories in order of preference
    possible_paths = []
    
    try:
        # Try the standard app path first
        app_embedding_dir = frappe.get_app_path('face_checkin', 'face_store', 'embeddings')
        possible_paths.append(app_embedding_dir)
    except:
        pass
    
    # Add fallback paths
    try:
        site_path = frappe.get_site_path()
        possible_paths.extend([
            os.path.join(site_path, 'private', 'files', 'face_embeddings'),
            os.path.join(site_path, 'public', 'files', 'face_embeddings'),
            os.path.join(site_path, 'face_embeddings')
        ])
    except:
        pass
    
    # Find the first existing directory with .npy files
    for path in possible_paths:
        if os.path.exists(path):
            try:
                # Check if directory has any .npy files
                files = [f for f in os.listdir(path) if f.endswith('.npy')]
                if files:  # If we found embedding files, use this directory
                    return path
                elif path == possible_paths[0]:  # Always prefer the app path even if empty
                    return path
            except:
                continue
    
    # If no existing directories with content found, create and return the app path
    if possible_paths:
        default_path = possible_paths[0]
        try:
            os.makedirs(default_path, exist_ok=True)
            return default_path
        except:
            pass
    
    # Last resort: create in site private files
    try:
        default_path = os.path.join(frappe.get_site_path(), 'private', 'files', 'face_embeddings')
        os.makedirs(default_path, exist_ok=True)
        return default_path
    except:
        # Ultimate fallback
        return os.path.join(os.getcwd(), 'face_embeddings')

@frappe.whitelist()
def upload_face(employee_id, image_base64=None):
    """
    Create face embedding from employee image
    """
    if not FACE_RECOGNITION_AVAILABLE:
        return {
            "status": "error",
            "message": "Face recognition dependencies not installed."
        }
    
    embedding_dir = get_embedding_directory()

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
        
        # Get image file from Frappe with better error handling
        try:
            if employee.image.startswith('/'):
                # Absolute URL - construct full path
                image_path = frappe.get_site_path() + employee.image
            else:
                # Relative URL - use frappe's get_file method
                try:
                    image_file = frappe.get_doc("File", {"file_url": employee.image})
                    if image_file.file_url.startswith('/'):
                        image_path = frappe.get_site_path() + image_file.file_url
                    else:
                        image_path = os.path.join(frappe.get_site_path(), 'public', 'files', image_file.file_url)
                except:
                    # Fallback: try direct file path construction
                    if employee.image.startswith('/files/'):
                        image_path = frappe.get_site_path() + employee.image
                    else:
                        image_path = os.path.join(frappe.get_site_path(), 'public', 'files', employee.image)
            
            # Check if file exists, if not try alternative paths
            if not os.path.exists(image_path):
                # Try different path combinations
                alt_paths = [
                    os.path.join(frappe.get_site_path(), 'public', employee.image.lstrip('/')),
                    os.path.join(frappe.get_site_path(), 'private', 'files', os.path.basename(employee.image)),
                    os.path.join(frappe.get_site_path(), 'public', 'files', os.path.basename(employee.image))
                ]
                
                for alt_path in alt_paths:
                    if os.path.exists(alt_path):
                        image_path = alt_path
                        break
                else:
                    return {
                        "status": "error",
                        "message": f"Employee image file not found."
                    }
            
            img = Image.open(image_path).convert("RGB")
            
        except Exception as e:
            return {
                "status": "error", 
                "message": f"Failed to load employee image: {str(e)}"
            }

    img_np = np.array(img)

    # Detect face
    face_locs = face_locations(img_np)
    if not face_locs:
        return {
            "status": "error",
            "message": "No face detected in image"
        }

    encodings = face_encodings(img_np, face_locs)
    if not encodings:
        return {
            "status": "error",
            "message": "Failed to extract face encoding"
        }

    embedding = encodings[0]

    # Save embedding to disk
    filepath = os.path.join(embedding_dir, f"{employee_id}.npy")
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
            "message": "Face recognition dependencies not installed."
        }
    
    embedding_dir = get_embedding_directory()

    # Clear employee cache to ensure fresh data
    frappe.clear_cache(doctype="Employee")
    
    employees_with_images = frappe.db.sql("""
        SELECT name, employee_name, image 
        FROM `tabEmployee` 
        WHERE image IS NOT NULL AND image != ''
    """, as_dict=True)

    results = []
    for employee in employees_with_images:
        embedding_file = os.path.join(embedding_dir, f"{employee.name}.npy")
        
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
        embedding_dir = get_embedding_directory()
        
        if not os.path.exists(embedding_dir):
            return {
                "status": "error",
                "message": "No employee faces registered in system"
            }

        # Check embeddings directory
        embedding_files = [f for f in os.listdir(embedding_dir) if f.endswith('.npy')]
        if not embedding_files:
            return {
                "status": "error",
                "message": "No face embeddings found. Please register employee faces first."
            }

        # Decode base64 image
        try:
            if not image_base64 or len(image_base64) < 50:
                return {
                    "status": "error",
                    "message": "Invalid or empty image data received"
                }
            
            image_data = base64.b64decode(image_base64.split(",")[-1])
            if len(image_data) < 1000:
                return {
                    "status": "error", 
                    "message": "Image data appears to be too small or corrupted"
                }
            
            img = Image.open(BytesIO(image_data)).convert("RGB")
            img_np = np.array(img)
            
            # Check image dimensions
            if img_np.shape[0] < 100 or img_np.shape[1] < 100:
                return {
                    "status": "error",
                    "message": f"Image too small for face detection: {img_np.shape[1]}x{img_np.shape[0]}"
                }
                
        except Exception as img_error:
            return {
                "status": "error",
                "message": f"Failed to process image: {str(img_error)}"
            }

        # Detect face in the image
        face_locs = face_locations(img_np)
        if not face_locs:
            return {
                "status": "error",
                "message": "No face detected in image"
            }

        # Get face encoding
        face_encs = face_encodings(img_np, face_locs)
        if not face_encs:
            return {
                "status": "error",
                "message": "Could not extract face features"
            }

        face_encoding = face_encs[0]

        # Load all known employee faces (reset arrays for each recognition)
        known_encodings = []
        employee_ids = []
        failed_loadings = []
        
        # Clear any previous data to prevent cross-contamination
        import gc
        gc.collect()
        
        for filename in os.listdir(embedding_dir):
            if filename.endswith('.npy'):
                employee_id = filename[:-4]  # Remove .npy extension
                filepath = os.path.join(embedding_dir, filename)
                
                try:
                    # Verify employee exists in system
                    if frappe.db.exists("Employee", employee_id):
                        encoding = np.load(filepath)
                        
                        # Validate encoding shape (ArcFace produces 512-dimensional embeddings)
                        if encoding.shape != (512,):
                            if encoding.shape == (128,):
                                failed_loadings.append(f"{employee_id}: Old face-recognition embedding detected (128-dim). Please re-enroll this employee.")
                            else:
                                failed_loadings.append(f"{employee_id}: Invalid encoding shape {encoding.shape}. Expected (512,)")
                            continue
                            
                        known_encodings.append(encoding)
                        employee_ids.append(employee_id)
                    else:
                        failed_loadings.append(f"{employee_id}: Employee not found in database")
                        
                except Exception as load_error:
                    failed_loadings.append(f"{employee_id}: {str(load_error)}")
        
        # Log failed loadings for debugging
        if failed_loadings:
            try:
                frappe.log_error(f"Face embedding loading issues: {'; '.join(failed_loadings)}")
            except:
                pass

        if not known_encodings:
            error_msg = "No registered employee faces found"
            if failed_loadings:
                error_msg += f". Issues with {len(failed_loadings)} face files - check logs for details."
            return {
                "status": "error",
                "message": error_msg
            }

        # Compare face with known faces
        try:
            # Ensure we have fresh data for comparison
            matches = compare_faces(known_encodings, face_encoding, tolerance=0.6)
            face_distances = face_distance(known_encodings, face_encoding)
            
            # Log comparison details for debugging
            frappe.log_error(f"Face comparison: {len(matches)} total matches, distances: {face_distances[:5] if len(face_distances) > 5 else face_distances}", "Face Recognition Debug")
            
        except Exception as compare_error:
            return {
                "status": "error",
                "message": f"Face comparison failed: {str(compare_error)}"
            }

        if not any(matches):
            return {
                "status": "error", 
                "message": "Employee not recognized"
            }

        # Get the best match
        best_match_index = np.argmin(face_distances)
        best_distance = face_distances[best_match_index]
        
        # Validate that the best match is actually a match and has good confidence
        if matches[best_match_index] and best_distance < 0.6:
            recognized_employee = employee_ids[best_match_index]
            confidence = 1 - best_distance
            
            # Log recognition for debugging
            frappe.log_error(f"Employee recognized: {recognized_employee}, confidence: {confidence:.2%}, distance: {best_distance:.3f}", "Face Recognition Success")
            
            # Get employee details (force reload to avoid cache issues)
            frappe.clear_document_cache("Employee", recognized_employee)
            employee = frappe.get_doc("Employee", recognized_employee)
            
            # Determine log type if not provided
            if not log_type:
                log_type = determine_log_type(recognized_employee)
            
            # Create Employee Checkin record
            checkin = frappe.new_doc("Employee Checkin")
            checkin.employee = recognized_employee
            checkin.employee_name = employee.employee_name
            checkin.time = frappe.utils.now_datetime()
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
        import traceback
        error_details = traceback.format_exc()
        
        try:
            frappe.log_error(f"Face recognition error: {str(e)}\n\nFull traceback:\n{error_details}")
        except:
            pass
        
        return {
            "status": "error",
            "message": f"System error during face recognition: {str(e)}"
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
        
        projects = frappe.get_all(
            "Project",
            filters={
                "is_active": "Yes"
            },
            fields=["name", "project_name", "status"],
            order_by="name asc"
        )
        
        # Ensure project_name field exists, use name if empty
        for project in projects:
            if not project.get("project_name"):
                project["project_name"] = project["name"]
        
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
        try:
            frappe.log_error(f"Error fetching projects: {str(e)}")
        except:
            pass
        return {
            "status": "error",
            "message": f"Failed to fetch projects: {str(e)}"
        }

@frappe.whitelist()
def check_enrollment_status(employee_ids=None):
    """
    Check face enrollment status for employees
    """
    embedding_dir = get_embedding_directory()
    
    # Debug logging
    debug_info = {
        "embedding_dir": embedding_dir,
        "dir_exists": os.path.exists(embedding_dir) if embedding_dir else False,
        "files_found": []
    }
    
    if not embedding_dir or not os.path.exists(embedding_dir):
        return {
            "status": "error",
            "message": "Face embeddings directory not found",
            "debug": debug_info
        }
    
    try:
        # Get list of existing embedding files
        existing_embeddings = set()
        all_files = os.listdir(embedding_dir)
        debug_info["all_files"] = all_files
        
        for filename in all_files:
            if filename.endswith('.npy'):
                employee_id = filename[:-4]  # Remove .npy extension
                existing_embeddings.add(employee_id)
                debug_info["files_found"].append({"filename": filename, "employee_id": employee_id})
        
        debug_info["existing_embeddings"] = list(existing_embeddings)
        
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
                "enrollment_status": results,
                "debug": debug_info
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
            
            # Debug logging
            debug_info["employee_checks"] = debug_info.get("employee_checks", [])
            debug_info["employee_checks"].append({
                "employee_id": employee.name,
                "employee_name": employee.employee_name,
                "has_face_data": has_face_data,
                "in_existing_embeddings": employee.name in existing_embeddings
            })
            
            if has_face_data:
                enrollment_summary["employees_enrolled"] += 1
            else:
                enrollment_summary["employees_pending"] += 1
        
        return {
            "status": "success",
            "enrollment_summary": enrollment_summary,
            "debug": debug_info
        }
        
    except Exception as e:
        try:
            frappe.log_error(f"Error checking enrollment status: {str(e)}")
        except:
            pass
        return {
            "status": "error",
            "message": f"Failed to check enrollment status: {str(e)}"
        }

@frappe.whitelist()
def test_enrollment_status():
    """
    Test function to debug enrollment status - returns raw data for inspection
    """
    try:
        embedding_dir = get_embedding_directory()
        
        test_result = {
            "embedding_dir": embedding_dir,
            "dir_exists": os.path.exists(embedding_dir) if embedding_dir else False,
            "files": []
        }
        
        if embedding_dir and os.path.exists(embedding_dir):
            all_files = os.listdir(embedding_dir)
            test_result["all_files"] = all_files
            
            for filename in all_files:
                if filename.endswith('.npy'):
                    filepath = os.path.join(embedding_dir, filename)
                    employee_id = filename[:-4]
                    
                    try:
                        # Load and check the embedding
                        import numpy as np
                        embedding = np.load(filepath)
                        test_result["files"].append({
                            "filename": filename,
                            "employee_id": employee_id,
                            "file_size": os.path.getsize(filepath),
                            "embedding_shape": embedding.shape,
                            "embedding_valid": embedding.size > 0
                        })
                    except Exception as e:
                        test_result["files"].append({
                            "filename": filename,
                            "employee_id": employee_id,
                            "error": str(e)
                        })
        
        # Test employee query
        try:
            employees = frappe.db.sql("""
                SELECT name, employee_name, image 
                FROM `tabEmployee` 
                WHERE image IS NOT NULL AND image != ''
            """, as_dict=True)
            test_result["employees_with_images"] = len(employees)
            test_result["employee_sample"] = employees[:3] if employees else []
        except Exception as e:
            test_result["employee_query_error"] = str(e)
        
        return {
            "status": "success",
            "test_result": test_result
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "traceback": frappe.utils.get_traceback()
        }

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
        try:
            frappe.log_error(f"Error uploading employee image: {str(e)}")
        except:
            pass
        return {
            "status": "error",
            "message": f"Failed to upload image: {str(e)}"
        }

@frappe.whitelist()
def delete_face_data(employee_id):
    """
    Delete face data for a specific employee
    """
    try:
        if not FACE_RECOGNITION_AVAILABLE:
            return {
                "status": "error",
                "message": "Face recognition libraries not available"
            }
        
        # Get embedding directory
        embedding_dir = get_embedding_directory()
        if not embedding_dir or not os.path.exists(embedding_dir):
            return {
                "status": "error", 
                "message": "Face data directory not found"
            }
        
        # Look for face data file
        face_file_path = os.path.join(embedding_dir, f"{employee_id}.npy")
        
        if not os.path.exists(face_file_path):
            return {
                "status": "warning",
                "message": f"No face data found for employee {employee_id}"
            }
        
        # Delete the face data file
        os.remove(face_file_path)
        
        frappe.log_error(f"Face data deleted for employee: {employee_id}", "Face Data Deletion")
        
        return {
            "status": "success",
            "message": f"Face data successfully deleted for employee {employee_id}",
            "employee_id": employee_id,
            "deleted_file": face_file_path
        }
        
    except Exception as e:
        frappe.log_error(f"Error deleting face data for {employee_id}: {str(e)}", "Face Data Deletion Error")
        return {
            "status": "error",
            "message": f"Failed to delete face data: {str(e)}"
        }

@frappe.whitelist()
def clear_employee_cache():
    """
    Clear all employee document cache to force refresh of employee data
    """
    try:
        # Clear all employee caches
        frappe.clear_cache(doctype="Employee")
        
        return {
            "status": "success",
            "message": "Employee cache cleared successfully"
        }
        
    except Exception as e:
        frappe.log_error(f"Error clearing employee cache: {str(e)}", "Employee Cache Clear Error")
        return {
            "status": "error",
            "message": f"Failed to clear employee cache: {str(e)}"
        }

@frappe.whitelist()
def bulk_delete_face_data():
    """
    Delete all stored face data files
    """
    try:
        if not FACE_RECOGNITION_AVAILABLE:
            return {
                "status": "error",
                "message": "Face recognition libraries not available"
            }
        
        # Get embedding directory
        embedding_dir = get_embedding_directory()
        if not embedding_dir or not os.path.exists(embedding_dir):
            return {
                "status": "error", 
                "message": "Face data directory not found"
            }
        
        # Find all .npy files
        npy_files = glob.glob(os.path.join(embedding_dir, "*.npy"))
        
        if not npy_files:
            return {
                "status": "warning",
                "message": "No face data files found to delete"
            }
        
        deleted_files = []
        failed_files = []
        
        for file_path in npy_files:
            try:
                employee_id = os.path.basename(file_path).replace('.npy', '')
                os.remove(file_path)
                deleted_files.append(employee_id)
            except Exception as e:
                failed_files.append({
                    "file": os.path.basename(file_path),
                    "error": str(e)
                })
        
        frappe.log_error(f"Bulk face data deletion: {len(deleted_files)} deleted, {len(failed_files)} failed", "Bulk Face Data Deletion")
        
        return {
            "status": "success",
            "message": f"Deleted face data for {len(deleted_files)} employees",
            "details": {
                "deleted_count": len(deleted_files),
                "deleted_employees": deleted_files,
                "failed_count": len(failed_files),
                "failed_files": failed_files,
                "embedding_directory": embedding_dir
            }
        }
        
    except Exception as e:
        frappe.log_error(f"Error in bulk face data deletion: {str(e)}", "Bulk Face Data Deletion Error")
        return {
            "status": "error",
            "message": f"Failed to delete face data: {str(e)}"
        }