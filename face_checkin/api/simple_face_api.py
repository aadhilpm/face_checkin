import frappe
import os
from datetime import datetime
import base64
from io import BytesIO
import glob
from typing import Tuple
import numpy as np

# Face recognition imports with fallback
try:
    from face_checkin.utils.face_recognition_simple import (
        face_locations, face_encodings, compare_faces, face_distance,
        get_face_recognition, FACE_RECOGNITION_AVAILABLE,
        validate_face_quality
    )
    from PIL import Image
except ImportError as e:
    FACE_RECOGNITION_AVAILABLE = False
    frappe.log_error(f"Face recognition dependencies not available: {str(e)}")

def get_embedding_directory():
    """Get the embedding directory - simplified version"""
    try:
        site_path = frappe.get_site_path()
        embedding_dir = os.path.join(site_path, 'private', 'files', 'face_embeddings')
        os.makedirs(embedding_dir, exist_ok=True)
        return embedding_dir
    except Exception as e:
        frappe.log_error(f"Failed to get embedding directory: {e}")
        return None

@frappe.whitelist()
def simple_upload_face(employee_id, image_base64=None):
    """
    Simplified face upload with relaxed quality requirements
    """
    if not FACE_RECOGNITION_AVAILABLE:
        return {
            "status": "error",
            "message": "Face recognition system not available"
        }
    
    embedding_dir = get_embedding_directory()
    if not embedding_dir:
        return {
            "status": "error",
            "message": "Cannot access face data storage"
        }

    try:
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
            
            # Get image file path
            try:
                if employee.image.startswith('/'):
                    image_path = frappe.get_site_path() + employee.image
                else:
                    image_path = os.path.join(frappe.get_site_path(), 'public', 'files', employee.image)
                
                if not os.path.exists(image_path):
                    return {
                        "status": "error",
                        "message": "Employee image file not found"
                    }
                
                img = Image.open(image_path).convert("RGB")
                
            except Exception as e:
                return {
                    "status": "error", 
                    "message": f"Failed to load employee image: {str(e)}"
                }

        img_np = np.array(img)

        # Detect face - simplified
        face_locs = face_locations(img_np)
        if not face_locs:
            return {
                "status": "error",
                "message": "No face detected in image. Please ensure face is clearly visible."
            }

        # Basic quality check - very lenient
        face_location = face_locs[0]
        top, right, bottom, left = face_location
        face_width = right - left
        face_height = bottom - top
        
        # Only reject if face is extremely small
        if face_width < 30 or face_height < 30:
            return {
                "status": "error",
                "message": "Face too small in image. Please use a clearer photo."
            }

        # Generate face encoding
        encodings = face_encodings(img_np, face_locs)
        if not encodings:
            return {
                "status": "error",
                "message": "Could not process face features"
            }

        embedding = encodings[0]

        # Save embedding
        filepath = os.path.join(embedding_dir, f"{employee_id}.npy")
        np.save(filepath, embedding)

        return {
            "status": "success",
            "message": f"Face data created successfully for {employee_id}"
        }

    except Exception as e:
        frappe.log_error(f"Simple face upload error: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to process face data: {str(e)}"
        }

@frappe.whitelist()
def simple_recognize_and_checkin(image_base64, project=None, device_id=None, log_type=None):
    """
    Simplified face recognition and checkin with relaxed matching
    """
    if not FACE_RECOGNITION_AVAILABLE:
        return {
            "status": "error",
            "message": "Face recognition system not available"
        }
    
    try:
        embedding_dir = get_embedding_directory()
        if not embedding_dir or not os.path.exists(embedding_dir):
            return {
                "status": "error",
                "message": "No employee faces registered in system"
            }

        # Check for face embeddings
        embedding_files = [f for f in os.listdir(embedding_dir) if f.endswith('.npy')]
        if not embedding_files:
            return {
                "status": "error",
                "message": "No employee face data found. Please register faces first."
            }

        # Decode image
        try:
            if not image_base64 or len(image_base64) < 50:
                return {
                    "status": "error",
                    "message": "Invalid image data"
                }
            
            image_data = base64.b64decode(image_base64.split(",")[-1])
            img = Image.open(BytesIO(image_data)).convert("RGB")
            img_np = np.array(img)
            
        except Exception:
            return {
                "status": "error",
                "message": "Failed to process image"
            }

        # Detect face
        face_locs = face_locations(img_np)
        if not face_locs:
            return {
                "status": "error",
                "message": "No face detected. Please position face clearly in camera."
            }
        
        # Handle multiple faces - just use the first one
        if len(face_locs) > 1:
            frappe.log_error(f"Multiple faces detected, using first one")
        
        face_location = face_locs[0]

        # Get face encoding
        face_encs = face_encodings(img_np, [face_location])
        if not face_encs:
            return {
                "status": "error",
                "message": "Could not process face features"
            }

        face_encoding = face_encs[0]

        # Load all employee faces
        known_encodings = []
        employee_ids = []
        
        for filename in os.listdir(embedding_dir):
            if filename.endswith('.npy'):
                employee_id = filename[:-4]
                filepath = os.path.join(embedding_dir, filename)
                
                try:
                    if frappe.db.exists("Employee", employee_id):
                        encoding = np.load(filepath)
                        # Basic validation - ensure it's a valid encoding
                        if len(encoding.shape) == 1 and encoding.shape[0] == 64:
                            known_encodings.append(encoding)
                            employee_ids.append(employee_id)
                except Exception:
                    continue

        if not known_encodings:
            return {
                "status": "error",
                "message": "No valid employee face data found"
            }

        # Compare faces with relaxed tolerance
        tolerance = 0.7  # More lenient than default 0.6
        matches = compare_faces(known_encodings, face_encoding, tolerance=tolerance)
        face_distances = face_distance(known_encodings, face_encoding)

        # Find best match
        best_match_index = None
        best_distance = float('inf')
        
        for i, (match, distance) in enumerate(zip(matches, face_distances)):
            if match and distance < best_distance:
                best_distance = distance
                best_match_index = i

        if best_match_index is None:
            return {
                "status": "error", 
                "message": "Employee not recognized. Please try again or contact administrator."
            }

        # Get recognized employee
        recognized_employee = employee_ids[best_match_index]
        confidence = max(0, min(100, (1 - best_distance) * 100))
        
        # Minimum confidence check - very lenient
        if confidence < 40:  # Much lower than original 75%
            return {
                "status": "error",
                "message": "Recognition confidence too low. Please try again."
            }

        # Get employee details
        employee = frappe.get_doc("Employee", recognized_employee)
        
        # Determine log type if not provided
        if not log_type:
            last_checkin = frappe.db.get_value(
                "Employee Checkin",
                filters={"employee": recognized_employee},
                fieldname="log_type",
                order_by="time desc"
            )
            log_type = "OUT" if last_checkin == "IN" else "IN"
        
        # Create checkin record - simplified
        checkin = frappe.new_doc("Employee Checkin")
        checkin.employee = recognized_employee
        checkin.employee_name = employee.employee_name
        checkin.time = frappe.utils.now_datetime()
        checkin.log_type = log_type
        checkin.device_id = device_id or "Face Recognition System"
        
        if project:
            checkin.custom_project = project
        
        try:
            checkin.insert()
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to record checkin: {str(e)}"
            }
        
        return {
            "status": "success",
            "message": f"{log_type} recorded for {employee.employee_name}",
            "employee_id": recognized_employee,
            "employee_name": employee.employee_name,
            "log_type": log_type,
            "time": checkin.time,
            "confidence": round(confidence, 1),
            "checkin_id": checkin.name
        }
            
    except Exception as e:
        frappe.log_error(f"Simple face recognition error: {str(e)}")
        return {
            "status": "error",
            "message": f"System error during face recognition"
        }

@frappe.whitelist()
def simple_bulk_enroll():
    """
    Simplified bulk enrollment for all employees with images
    """
    if not FACE_RECOGNITION_AVAILABLE:
        return {
            "status": "error",
            "message": "Face recognition system not available"
        }
    
    embedding_dir = get_embedding_directory()
    if not embedding_dir:
        return {
            "status": "error",
            "message": "Cannot access face data storage"
        }

    # Get employees with images
    employees_with_images = frappe.db.sql("""
        SELECT name, employee_name, image 
        FROM `tabEmployee` 
        WHERE image IS NOT NULL AND image != ''
    """, as_dict=True)

    results = []
    success_count = 0
    
    for employee in employees_with_images:
        embedding_file = os.path.join(embedding_dir, f"{employee.name}.npy")
        
        # Skip if embedding already exists
        if os.path.exists(embedding_file):
            continue
            
        result = simple_upload_face(employee.name)
        results.append({
            "employee": employee.name,
            "employee_name": employee.employee_name,
            "result": result
        })
        
        if result["status"] == "success":
            success_count += 1

    return {
        "status": "success",
        "message": f"Processed {len(results)} employees, {success_count} successful",
        "details": results
    }

@frappe.whitelist()
def get_simple_projects():
    """
    Simplified project listing
    """
    try:
        if not frappe.db.exists("DocType", "Project"):
            return {
                "status": "error",
                "message": "Projects not available in system"
            }
        
        projects = frappe.get_all(
            "Project",
            filters={"is_active": "Yes"},
            fields=["name", "project_name"],
            order_by="name asc"
        )
        
        if not projects:
            return {
                "status": "info",
                "message": "No active projects found"
            }
        
        return {
            "status": "success",
            "projects": projects
        }
        
    except Exception:
        return {
            "status": "error",
            "message": "Failed to load projects"
        }

@frappe.whitelist()
def simple_system_status():
    """
    Simplified system status check
    """
    try:
        embedding_dir = get_embedding_directory()
        embedding_count = 0
        
        if embedding_dir and os.path.exists(embedding_dir):
            embedding_files = [f for f in os.listdir(embedding_dir) if f.endswith('.npy')]
            embedding_count = len(embedding_files)
        
        return {
            "status": "success",
            "face_recognition_available": FACE_RECOGNITION_AVAILABLE,
            "embeddings_directory": embedding_dir,
            "registered_faces": embedding_count,
            "system_ready": FACE_RECOGNITION_AVAILABLE and embedding_count > 0
        }
        
    except Exception:
        return {
            "status": "error",
            "message": "System status check failed"
        }

@frappe.whitelist()
def simple_delete_face_data(employee_id):
    """
    Simplified face data deletion
    """
    try:
        embedding_dir = get_embedding_directory()
        if not embedding_dir:
            return {
                "status": "error",
                "message": "Cannot access face data storage"
            }
        
        face_file_path = os.path.join(embedding_dir, f"{employee_id}.npy")
        
        if not os.path.exists(face_file_path):
            return {
                "status": "warning",
                "message": f"No face data found for employee {employee_id}"
            }
        
        os.remove(face_file_path)
        
        return {
            "status": "success",
            "message": f"Face data deleted for employee {employee_id}"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to delete face data: {str(e)}"
        }

@frappe.whitelist()
def simple_upload_employee_image(employee_id, image_base64, filename="employee_photo.jpg"):
    """
    Simplified employee image upload with automatic face data creation
    """
    try:
        # Check if employee exists
        if not frappe.db.exists("Employee", employee_id):
            return {
                "status": "error",
                "message": f"Employee {employee_id} not found"
            }

        # Decode and validate image
        try:
            if "," in image_base64:
                image_data = base64.b64decode(image_base64.split(",")[1])
            else:
                image_data = base64.b64decode(image_base64)
            
            # Basic validation - check size
            if len(image_data) < 1000:
                return {
                    "status": "error",
                    "message": "Image data too small or corrupted"
                }
            
            # Validate it's a real image
            img = Image.open(BytesIO(image_data))
            if img.width < 50 or img.height < 50:
                return {
                    "status": "error",
                    "message": "Image too small - minimum 50x50 pixels"
                }
                
        except Exception:
            return {
                "status": "error",
                "message": "Invalid image data"
            }

        # Create file in Frappe
        file_doc = frappe.get_doc({
            "doctype": "File",
            "file_name": f"{employee_id}_{filename}",
            "content": image_data,
            "is_private": 0,
            "folder": "Home/Attachments"
        })
        file_doc.save()

        # Update employee record
        employee = frappe.get_doc("Employee", employee_id)
        employee.image = file_doc.file_url
        employee.save()

        # Create face data
        face_result = simple_upload_face(employee_id, image_base64)

        return {
            "status": "success",
            "message": f"Image uploaded for {employee.employee_name}",
            "file_url": file_doc.file_url,
            "face_enrollment": face_result
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to upload image: {str(e)}"
        }