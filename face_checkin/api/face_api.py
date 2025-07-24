import frappe
import os
from datetime import datetime
import base64
from io import BytesIO
import glob
from typing import Tuple

# Optional imports for face recognition functionality
try:
    from face_checkin.utils.face_recognition_simple import (
        face_locations, face_encodings, compare_faces, face_distance,
        get_face_recognition, FACE_RECOGNITION_AVAILABLE,
        validate_face_quality, get_best_face_from_multiple,
        create_multi_image_face_data, validate_multi_image_consistency
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
    Get the embedding directory, prioritizing site directory to persist across app updates
    Always creates the directory if it doesn't exist
    """
    # Primary site directory (preferred - persists across app updates)
    site_embedding_dir = None
    try:
        site_path = frappe.get_site_path()
        site_embedding_dir = os.path.join(site_path, 'private', 'files', 'face_embeddings')
    except Exception as e:
        frappe.log_error(f"Failed to get site path: {e}", "Embedding Directory Error")
    
    # Check if there are existing embeddings in other locations first
    possible_existing_paths = []
    
    try:
        site_path = frappe.get_site_path()
        possible_existing_paths.extend([
            os.path.join(site_path, 'private', 'files', 'face_embeddings'),  # Primary location
            os.path.join(site_path, 'public', 'files', 'face_embeddings'),   # Alternative location
            os.path.join(site_path, 'face_embeddings')  # Direct in site root
        ])
    except:
        if site_embedding_dir:
            possible_existing_paths.append(site_embedding_dir)
    
    # Check app directory for existing embeddings (legacy)
    try:
        app_embedding_dir = frappe.get_app_path('face_checkin', 'face_store', 'embeddings')
        possible_existing_paths.append(app_embedding_dir)
    except:
        pass
    
    # Look for existing directory with embedding files
    for path in possible_existing_paths:
        if os.path.exists(path):
            try:
                files = [f for f in os.listdir(path) if f.endswith('.npy')]
                if files:  # Found existing embeddings - use this directory
                    frappe.log_error(f"Using existing embedding directory with {len(files)} files: {path}", "Embedding Directory Info")
                    return path
            except Exception as e:
                continue
    
    # No existing embeddings found - create and use the primary site directory
    if site_embedding_dir:
        try:
            os.makedirs(site_embedding_dir, exist_ok=True)
            frappe.log_error(f"Created new embedding directory: {site_embedding_dir}", "Embedding Directory Info")
            return site_embedding_dir
        except Exception as e:
            frappe.log_error(f"Failed to create site embedding directory: {e}", "Embedding Directory Error")
    
    # Fallback: try to create in current working directory
    fallback_dir = os.path.join(os.getcwd(), 'face_embeddings')
    try:
        os.makedirs(fallback_dir, exist_ok=True)
        frappe.log_error(f"Using fallback embedding directory: {fallback_dir}", "Embedding Directory Warning")
        return fallback_dir
    except Exception as e:
        frappe.log_error(f"Failed to create fallback embedding directory: {e}", "Embedding Directory Error")
        return None

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

    # Check image quality with ultra-lenient mode for employee enrollment
    quality_result = validate_face_quality(img_np, face_locs[0], lenient_mode=True, strict_accuracy=False, employee_enrollment_mode=True)
    
    # For employee enrollment, only reject if quality is extremely poor
    if not quality_result["valid"] and quality_result.get("quality_score", 0) < 15:
        return {
            "status": "error",
            "message": f"Image quality too poor for face recognition: {', '.join(quality_result['issues'])}",
            "quality_score": quality_result.get("quality_score", 0),
            "suggestions": quality_result["issues"]
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
def recognize_and_checkin(image_base64, project=None, device_id=None, log_type=None, latitude=None, longitude=None):
    """
    Recognize employee from face and create checkin record
    
    Args:
        image_base64: Base64 encoded image for face recognition
        project: Project to associate with checkin
        device_id: Device identifier
        log_type: IN or OUT (auto-determined if not provided)
        latitude: Latitude coordinate (REQUIRED if HR Settings allow_geolocation_tracking is enabled)
        longitude: Longitude coordinate (REQUIRED if HR Settings allow_geolocation_tracking is enabled)
    
    Returns:
        dict: Response with status, message, and employee details
        - If geolocation_required=True in error response, coordinates must be provided
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
        
        # Handle multiple faces detected - this could indicate wrong person in frame
        if len(face_locs) > 1:
            frappe.log_error(f"Multiple faces detected in check-in image: {len(face_locs)} faces found", "Multiple Face Warning")
            
            # Check if multiple face detection is allowed (can be configured in site_config.json)
            allow_multiple_faces = frappe.conf.get("face_recognition_allow_multiple_faces", False)
            
            if not allow_multiple_faces:
                # For security, reject check-in if multiple faces detected
                # This prevents accidental check-in of wrong person
                return {
                    "status": "error",
                    "message": f"Multiple faces detected ({len(face_locs)} faces). Please ensure only one person is visible in the camera for security."
                }
            else:
                # Select best quality face if multiple faces are allowed
                best_face = get_best_face_from_multiple(img_np, face_locs)
                if best_face:
                    face_locs = [best_face]
                    frappe.log_error(f"Selected best face from {len(face_locs)} detected faces", "Multiple Face Handling")
                else:
                    return {
                        "status": "error",
                        "message": "Could not determine best face from multiple detected faces"
                    }
        
        # Validate face quality with strict accuracy enforcement
        face_quality = validate_face_quality(img_np, face_locs[0], lenient_mode=False, strict_accuracy=True)
        if not face_quality["valid"]:
            quality_issues = " and ".join(face_quality["issues"])
            return {
                "status": "error", 
                "message": f"Poor image quality: {quality_issues}"
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
                        
                        # Validate encoding shape - expecting 64-dimensional OpenCV features
                        if len(encoding.shape) == 1 and encoding.shape[0] == 64:
                            # Valid 64-dimensional OpenCV embedding
                            known_encodings.append(encoding)
                            employee_ids.append(employee_id)
                        elif len(encoding.shape) == 1:
                            # Invalid dimensions - needs re-enrollment
                            failed_loadings.append(f"{employee_id}: Invalid embedding size {encoding.shape[0]}. Expected 64-dim. Please re-enroll.")
                            continue
                        else:
                            failed_loadings.append(f"{employee_id}: Invalid encoding format {encoding.shape}. Expected 1D array.")
                            continue
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
            # STRICT ACCURACY ENFORCEMENT - Tighter thresholds for maximum accuracy
            initial_tolerance = frappe.conf.get("face_recognition_initial_tolerance", 0.5)  # Reduced from 0.7
            strict_tolerance = frappe.conf.get("face_recognition_strict_tolerance", 0.35)   # Reduced from 0.55
            min_quality_score = frappe.conf.get("face_recognition_min_quality", 75)        # Increased from 60
            min_confidence_threshold = frappe.conf.get("face_recognition_min_confidence", 75.0) # Increased from 65%
            required_confidence_gap = frappe.conf.get("face_recognition_confidence_gap", 10.0)  # Increased from 5%
            
            # Ensure we have fresh data for comparison
            matches = compare_faces(known_encodings, face_encoding, tolerance=initial_tolerance)
            face_distances = face_distance(known_encodings, face_encoding)
            
            # ACCURACY MONITORING: Log detailed comparison metrics
            accuracy_log = {
                "total_candidates": len(known_encodings),
                "initial_matches": sum(matches),
                "min_distance": min(face_distances) if face_distances else 1.0,
                "max_distance": max(face_distances) if face_distances else 1.0,
                "avg_distance": np.mean(face_distances) if face_distances else 1.0,
                "quality_score": face_quality["quality_score"],
                "thresholds": {
                    "initial_tolerance": initial_tolerance,
                    "strict_tolerance": strict_tolerance,
                    "min_confidence": min_confidence_threshold,
                    "confidence_gap": required_confidence_gap
                }
            }
            frappe.log_error(f"Face Recognition Accuracy Metrics: {accuracy_log}", "Face Recognition Accuracy")
            
        except Exception as compare_error:
            return {
                "status": "error",
                "message": f"Face comparison failed: {str(compare_error)}"
            }

        # Filter matches that meet minimum confidence before selecting best
        valid_matches = []
        for i, (match, distance) in enumerate(zip(matches, face_distances)):
            if match and distance < initial_tolerance:
                confidence = max(0, min(100, (1 - distance) * 100))
                if confidence >= min_confidence_threshold:
                    valid_matches.append({
                        'index': i,
                        'distance': distance,
                        'confidence': confidence,
                        'employee_id': employee_ids[i]
                    })

        if not valid_matches:
            return {
                "status": "error", 
                "message": "Employee not recognized - no confident matches found"
            }

        # Sort by confidence (highest first) and select the most confident match
        valid_matches.sort(key=lambda x: x['confidence'], reverse=True)
        best_match = valid_matches[0]
        
        # Apply quality-adjusted validation based on image quality
        quality_bonus = (face_quality["quality_score"] / 100) * 0.1  # Up to 0.1 bonus for high quality
        adjusted_tolerance = strict_tolerance + quality_bonus
        
        # Additional validation: ensure the best match is significantly better than others
        if len(valid_matches) > 1:
            second_best = valid_matches[1]
            confidence_gap = best_match['confidence'] - second_best['confidence']
            
            # Require significant confidence gap between best and second-best match for accuracy
            if confidence_gap < required_confidence_gap:
                frappe.log_error(f"Ambiguous face recognition: Best match {best_match['employee_id']} ({best_match['confidence']:.1f}%) vs {second_best['employee_id']} ({second_best['confidence']:.1f}%) - gap too small", "Face Recognition Warning")
                return {
                    "status": "error",
                    "message": "Multiple similar faces detected - unable to identify with confidence"
                }
        
        # MULTI-STAGE VERIFICATION for maximum accuracy
        
        # Stage 1: Basic threshold validation
        if best_match['distance'] >= adjusted_tolerance:
            return {
                "status": "error",
                "message": "Employee not recognized - insufficient match quality"
            }
        
        # Stage 2: Minimum confidence validation
        if best_match['confidence'] < min_confidence_threshold:
            return {
                "status": "error",
                "message": f"Recognition confidence too low ({best_match['confidence']:.1f}% < {min_confidence_threshold}%)"
            }
        
        # Stage 3: Image quality validation
        if face_quality["quality_score"] < min_quality_score:
            return {
                "status": "error",
                "message": f"Image quality insufficient for accurate recognition (score: {face_quality['quality_score']}/{min_quality_score})"
            }
        
        # Stage 4: Cross-validation with secondary features (if available)
        try:
            # Perform additional validation using different feature extraction methods
            secondary_confidence = _perform_secondary_validation(img_np, face_locs[0], best_match['employee_id'], embedding_dir)
            
            # Require both primary and secondary methods to agree
            confidence_difference = abs(best_match['confidence'] - secondary_confidence)
            if confidence_difference > 15.0:  # Allow 15% difference between methods
                frappe.log_error(f"Primary-secondary confidence mismatch: {best_match['confidence']:.1f}% vs {secondary_confidence:.1f}%", "Face Recognition Validation Warning")
                return {
                    "status": "error",
                    "message": "Recognition validation failed - inconsistent results between verification methods"
                }
        except Exception as secondary_error:
            frappe.log_error(f"Secondary validation failed: {str(secondary_error)}", "Secondary Validation Error")
            # Continue without secondary validation but log the issue
        
        # All validation stages passed
        recognized_employee = best_match['employee_id']
        confidence = best_match['confidence']
        
        # ACCURACY MONITORING: Log comprehensive recognition metrics
        recognition_metrics = {
            "employee_id": recognized_employee,
            "confidence": confidence,
            "distance": best_match['distance'],
            "quality_score": face_quality['quality_score'],
            "validation_stages_passed": 4,  # All 4 stages passed
            "total_candidates": len(known_encodings),
            "valid_matches_found": len(valid_matches),
            "confidence_gap": valid_matches[0]['confidence'] - valid_matches[1]['confidence'] if len(valid_matches) > 1 else "N/A",
            "thresholds_met": {
                "distance_threshold": best_match['distance'] < adjusted_tolerance,
                "confidence_threshold": confidence >= min_confidence_threshold,
                "quality_threshold": face_quality['quality_score'] >= min_quality_score,
                "ambiguity_check": len(valid_matches) == 1 or (valid_matches[0]['confidence'] - valid_matches[1]['confidence']) >= required_confidence_gap
            }
        }
        frappe.log_error(f"ACCURACY SUCCESS - Employee Recognition Metrics: {recognition_metrics}", "Face Recognition Accuracy Success")
        
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
        
        # Check HR Settings for geolocation tracking
        hr_settings = frappe.get_single("HR Settings")
        geolocation_provided = False
        if hr_settings.allow_geolocation_tracking:
            if latitude and longitude:
                try:
                    checkin.latitude = float(latitude)
                    checkin.longitude = float(longitude)
                    geolocation_provided = True
                except (ValueError, TypeError):
                    # Invalid coordinates - continue without geolocation but warn
                    frappe.log_error(f"Invalid geolocation coordinates provided: lat={latitude}, lng={longitude}")
            
        # Note: We allow check-ins without geolocation if coordinates aren't provided
        # The frontend should handle geolocation collection
        
        try:
            checkin.insert()
        except frappe.ValidationError as ve:
            # Handle validation errors from ERPNext (like distance validation)
            return {
                "status": "error",
                "message": f"Check-in validation failed: {str(ve)}",
                "geolocation_required": hr_settings.allow_geolocation_tracking
            }
        
        return {
            "status": "success",
            "message": f"{log_type} recorded for {employee.employee_name}",
            "employee_id": recognized_employee,
            "employee_name": employee.employee_name,
            "log_type": log_type,
            "time": checkin.time,
            "confidence": round(confidence, 2),
            "checkin_id": checkin.name
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

def _perform_secondary_validation(image: np.ndarray, face_location: Tuple[int, int, int, int], employee_id: str, embedding_dir: str) -> float:
    """
    Perform secondary validation using alternative feature extraction methods
    Returns confidence score from secondary method
    """
    try:
        top, right, bottom, left = face_location
        face_image = image[top:bottom, left:right]
        
        if face_image.size == 0:
            return 0.0
        
        # Load the stored embedding for this employee
        employee_embedding_path = os.path.join(embedding_dir, f"{employee_id}.npy")
        if not os.path.exists(employee_embedding_path):
            return 0.0
        
        stored_embedding = np.load(employee_embedding_path)
        
        # Extract features using alternative methods for cross-validation
        from face_checkin.utils.face_recognition_simple import SimpleFaceRecognition
        secondary_fr = SimpleFaceRecognition(production_mode=False)  # Use higher quality for validation
        
        if not secondary_fr.initialized:
            return 0.0
        
        # Extract features using secondary method
        secondary_features = secondary_fr._extract_face_features(face_image)
        
        # Calculate similarity using cosine similarity
        dot_product = np.dot(stored_embedding, secondary_features)
        norm_stored = np.linalg.norm(stored_embedding)
        norm_secondary = np.linalg.norm(secondary_features)
        
        if norm_stored == 0 or norm_secondary == 0:
            return 0.0
        
        cosine_similarity = dot_product / (norm_stored * norm_secondary)
        confidence = max(0, min(100, cosine_similarity * 100))
        
        return confidence
        
    except Exception as e:
        frappe.log_error(f"Secondary validation error for {employee_id}: {str(e)}", "Secondary Validation Error")
        return 0.0


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
def check_system_status():
    """
    Simple system status check for face recognition setup
    """
    try:
        # Check face recognition availability
        face_available = FACE_RECOGNITION_AVAILABLE
        
        # Get embedding directory
        embedding_dir = None
        dir_exists = False
        try:
            embedding_dir = get_embedding_directory()
            dir_exists = bool(embedding_dir and os.path.exists(embedding_dir))
        except Exception as e:
            frappe.log_error(f"Error getting embedding directory: {str(e)}", "System Status Check")
        
        # Check ONNX availability
        onnx_available = False
        onnx_models_ready = False
        try:
            import onnxruntime
            onnx_available = True
            try:
                from face_checkin.utils.onnx_face_recognition import get_onnx_face_recognition
                onnx_fr = get_onnx_face_recognition()
                onnx_models_ready = onnx_fr.is_available()
            except Exception as e:
                frappe.log_error(f"Error checking ONNX models: {str(e)}", "System Status Check")
        except ImportError:
            pass
        except Exception as e:
            frappe.log_error(f"Error checking ONNX runtime: {str(e)}", "System Status Check")
        
        # Get user info
        user = frappe.session.user if hasattr(frappe, 'session') else "Guest"
        roles = []
        try:
            roles = frappe.get_roles(user) if user != "Guest" else []
        except Exception as e:
            frappe.log_error(f"Error getting user roles: {str(e)}", "System Status Check")
        
        return {
            "face_recognition_available": face_available,
            "embedding_directory_exists": dir_exists,
            "embedding_directory": embedding_dir,
            "onnx_available": onnx_available,
            "onnx_models_ready": onnx_models_ready,
            "user": user,
            "user_roles": roles
        }
        
    except Exception as e:
        error_msg = f"System status check failed: {str(e)}"
        frappe.log_error(error_msg, "System Status Check Error")
        return {
            "face_recognition_available": False,
            "embedding_directory_exists": False,
            "embedding_directory": None,
            "onnx_available": False,
            "onnx_models_ready": False,
            "user": "Unknown",
            "user_roles": [],
            "error": error_msg
        }

@frappe.whitelist()
def diagnose_face_data():
    """
    Diagnose face embedding files and provide repair suggestions
    """
    try:
        embedding_dir = get_embedding_directory()
        if not os.path.exists(embedding_dir):
            return {
                "status": "error",
                "message": "Face data directory doesn't exist",
                "directory": embedding_dir
            }
        
        files = [f for f in os.listdir(embedding_dir) if f.endswith('.npy')]
        if not files:
            return {
                "status": "info",
                "message": "No face embedding files found",
                "directory": embedding_dir,
                "total_files": 0
            }
        
        diagnosis = {
            "total_files": len(files),
            "valid_files": [],
            "invalid_files": [],
            "missing_employees": [],
            "size_issues": [],
            "opencv_compatible": 0
        }
        
        for filename in files:
            employee_id = filename.replace('.npy', '')
            filepath = os.path.join(embedding_dir, filename)
            
            try:
                # Check if employee exists
                if not frappe.db.exists("Employee", employee_id):
                    diagnosis["missing_employees"].append({
                        "employee_id": employee_id,
                        "issue": "Employee not found in database"
                    })
                    continue
                
                # Load and analyze embedding
                encoding = np.load(filepath)
                file_size = os.path.getsize(filepath)
                
                if len(encoding.shape) != 1:
                    diagnosis["invalid_files"].append({
                        "employee_id": employee_id,
                        "issue": f"Invalid shape {encoding.shape}, expected 1D array",
                        "file_size": file_size
                    })
                    continue
                
                dim = encoding.shape[0]
                if dim == 64:
                    diagnosis["opencv_compatible"] += 1
                    diagnosis["valid_files"].append({
                        "employee_id": employee_id,
                        "dimensions": dim,
                        "type": "OpenCV (64-dim)",
                        "file_size": file_size
                    })
                else:
                    diagnosis["size_issues"].append({
                        "employee_id": employee_id,
                        "issue": f"Invalid {dim} dimensions - expected 64",
                        "dimensions": dim,
                        "file_size": file_size
                    })
                    
            except Exception as e:
                diagnosis["invalid_files"].append({
                    "employee_id": employee_id,
                    "issue": f"Failed to load: {str(e)}",
                    "file_size": os.path.getsize(filepath) if os.path.exists(filepath) else 0
                })
        
        # Generate recommendations
        recommendations = []
        if diagnosis["missing_employees"]:
            recommendations.append(f"Remove {len(diagnosis['missing_employees'])} orphaned face files")
        if diagnosis["size_issues"]:
            recommendations.append(f"Re-enroll {len(diagnosis['size_issues'])} employees with incompatible embeddings")
        if diagnosis["invalid_files"]:
            recommendations.append(f"Re-enroll {len(diagnosis['invalid_files'])} employees with corrupted files")
        
        return {
            "status": "success",
            "directory": embedding_dir,
            "diagnosis": diagnosis,
            "recommendations": recommendations
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Diagnosis failed: {str(e)}"
        }

@frappe.whitelist()
def cleanup_face_data():
    """
    Clean up problematic face embedding files
    """
    try:
        embedding_dir = get_embedding_directory()
        if not os.path.exists(embedding_dir):
            return {
                "status": "error",
                "message": "Face data directory doesn't exist"
            }
        
        files = [f for f in os.listdir(embedding_dir) if f.endswith('.npy')]
        if not files:
            return {
                "status": "info",
                "message": "No face files to clean up"
            }
        
        cleaned_files = []
        errors = []
        
        for filename in files:
            employee_id = filename.replace('.npy', '')
            filepath = os.path.join(embedding_dir, filename)
            
            try:
                # Check if employee exists
                if not frappe.db.exists("Employee", employee_id):
                    os.remove(filepath)
                    cleaned_files.append(f"{employee_id}: Removed orphaned file")
                    continue
                
                # Check file integrity
                encoding = np.load(filepath)
                if len(encoding.shape) != 1:
                    os.remove(filepath)
                    cleaned_files.append(f"{employee_id}: Removed corrupted file (invalid shape)")
                    continue
                
                # Remove files with wrong dimensions (only 64-dim is valid now)
                if encoding.shape[0] != 64:
                    os.remove(filepath)
                    cleaned_files.append(f"{employee_id}: Removed incompatible {encoding.shape[0]}-dim file")
                    continue
                    
            except Exception as e:
                errors.append(f"{employee_id}: Failed to process - {str(e)}")
        
        return {
            "status": "success",
            "cleaned_files": cleaned_files,
            "errors": errors,
            "total_cleaned": len(cleaned_files)
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Cleanup failed: {str(e)}"
        }

@frappe.whitelist()
def get_detailed_status():
    """
    Detailed system diagnosis - simplified version
    """
    try:
        status = {}
        
        # Test dependencies one by one
        try:
            import cv2
            status["opencv"] = "Available"
        except Exception as e:
            status["opencv"] = f"Error: {str(e)}"
        
        try:
            from PIL import Image
            status["pil"] = "Available"
        except Exception as e:
            status["pil"] = f"Error: {str(e)}"
        
        try:
            import numpy as np
            status["numpy"] = "Available"
        except Exception as e:
            status["numpy"] = f"Error: {str(e)}"
        
        # Check ONNX Runtime
        try:
            import onnxruntime as ort
            status["onnxruntime"] = "Available"
            status["onnx_version"] = ort.__version__
        except ImportError:
            status["onnxruntime"] = "Not installed"
            status["onnx_version"] = None
        except Exception as e:
            status["onnxruntime"] = f"Error: {str(e)}"
            status["onnx_version"] = None
        
        # Check ONNX face recognition integration
        try:
            from face_checkin.utils.onnx_face_recognition import get_onnx_face_recognition
            onnx_fr = get_onnx_face_recognition()
            status["onnx_models"] = "Available" if onnx_fr.is_available() else "Not loaded"
            status["onnx_models_dir"] = onnx_fr.models_dir
            
            # Check individual model files
            if hasattr(onnx_fr, 'model_urls'):
                model_status = {}
                for model_key, model_info in onnx_fr.model_urls.items():
                    model_path = os.path.join(onnx_fr.models_dir, model_info['filename'])
                    if os.path.exists(model_path):
                        size_mb = os.path.getsize(model_path) / (1024 * 1024)
                        model_status[model_key] = f"Downloaded ({size_mb:.1f}MB)"
                    else:
                        model_status[model_key] = "Not downloaded"
                status["onnx_model_files"] = model_status
        except Exception as e:
            frappe.log_error(f"Error checking ONNX face recognition: {str(e)}", "Detailed Status Check")
            status["onnx_models"] = f"Error: {str(e)}"
            status["onnx_models_dir"] = None
            status["onnx_model_files"] = {}
        
        # Check directory
        try:
            embedding_dir = get_embedding_directory()
            status["embedding_dir"] = embedding_dir
            status["dir_exists"] = os.path.exists(embedding_dir) if embedding_dir else False
        except Exception as e:
            frappe.log_error(f"Error getting embedding directory: {str(e)}", "Detailed Status Check")
            status["embedding_dir"] = None
            status["dir_exists"] = False
        
        # Check embeddings
        if status["dir_exists"]:
            try:
                files = [f for f in os.listdir(status["embedding_dir"]) if f.endswith('.npy')]
                status["embedding_files"] = len(files)
            except Exception as e:
                frappe.log_error(f"Error listing embedding files: {str(e)}", "Detailed Status Check")
                status["embedding_files"] = 0
        else:
            status["embedding_files"] = 0
        
        status["face_recognition_flag"] = FACE_RECOGNITION_AVAILABLE
        
        return status
        
    except Exception as e:
        error_msg = f"Detailed status check failed: {str(e)}"
        frappe.log_error(error_msg, "Detailed Status Check Error")
        return {
            "opencv": "Error during check",
            "pil": "Error during check", 
            "numpy": "Error during check",
            "onnxruntime": "Error during check",
            "onnx_version": None,
            "onnx_models": "Error during check",
            "onnx_models_dir": None,
            "onnx_model_files": {},
            "embedding_dir": None,
            "dir_exists": False,
            "embedding_files": 0,
            "face_recognition_flag": False,
            "error": error_msg
        }

@frappe.whitelist()
def get_geolocation_settings():
    """
    Get HR Settings geolocation tracking configuration
    """
    try:
        hr_settings = frappe.get_single("HR Settings")
        return {
            "status": "success",
            "allow_geolocation_tracking": bool(hr_settings.allow_geolocation_tracking)
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to get geolocation settings: {str(e)}"
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
        from PIL import Image
        import numpy as np
        
        if "," in image_base64:
            image_data = base64.b64decode(image_base64.split(",")[1])
        else:
            image_data = base64.b64decode(image_base64)

        # Validate image quality before saving
        if FACE_RECOGNITION_AVAILABLE:
            try:
                img = Image.open(BytesIO(image_data)).convert("RGB")
                img_np = np.array(img)
                
                # Detect face for quality validation
                face_locs = face_locations(img_np)
                if not face_locs:
                    return {
                        "status": "error",
                        "message": "No face detected in uploaded image"
                    }
                
                # Check image quality with ultra-lenient mode for employee uploads
                quality_result = validate_face_quality(img_np, face_locs[0], lenient_mode=True, strict_accuracy=False, employee_enrollment_mode=True)
                
                # For employee image uploads, only reject if quality is extremely poor
                if not quality_result["valid"] and quality_result.get("quality_score", 0) < 15:
                    return {
                        "status": "error",
                        "message": f"Image quality too poor for face recognition: {', '.join(quality_result['issues'])}",
                        "quality_score": quality_result.get("quality_score", 0),
                        "suggestions": quality_result["issues"]
                    }
            except Exception as e:
                return {
                    "status": "error",
                    "message": f"Failed to validate image quality: {str(e)}"
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
def get_accuracy_statistics(days=7):
    """
    Get face recognition accuracy statistics for monitoring
    """
    try:
        from datetime import datetime, timedelta
        
        # Get recent error logs related to face recognition
        end_date = datetime.now()
        start_date = end_date - timedelta(days=int(days))
        
        # Query error logs for face recognition metrics
        accuracy_logs = frappe.db.sql("""
            SELECT creation, error, title
            FROM `tabError Log`
            WHERE title LIKE '%Face Recognition Accuracy%'
            AND creation BETWEEN %s AND %s
            ORDER BY creation DESC
            LIMIT 100
        """, (start_date, end_date), as_dict=True)
        
        # Parse accuracy metrics
        stats = {
            "total_attempts": 0,
            "successful_recognitions": 0,
            "failed_recognitions": 0,
            "quality_failures": 0,
            "confidence_failures": 0,
            "ambiguity_failures": 0,
            "avg_confidence": 0,
            "avg_quality_score": 0,
            "recognition_rate": 0
        }
        
        confidence_scores = []
        quality_scores = []
        
        for log in accuracy_logs:
            stats["total_attempts"] += 1
            
            if "ACCURACY SUCCESS" in log.get("title", ""):
                stats["successful_recognitions"] += 1
                # Parse metrics from log (simplified - in production would use structured logging)
                try:
                    import re
                    confidence_match = re.search(r"'confidence': ([\d.]+)", log.get("error", ""))
                    quality_match = re.search(r"'quality_score': ([\d.]+)", log.get("error", ""))
                    
                    if confidence_match:
                        confidence_scores.append(float(confidence_match.group(1)))
                    if quality_match:
                        quality_scores.append(float(quality_match.group(1)))
                except:
                    pass
            else:
                stats["failed_recognitions"] += 1
                # Categorize failure types
                error_msg = log.get("error", "").lower()
                if "quality" in error_msg:
                    stats["quality_failures"] += 1
                elif "confidence" in error_msg:
                    stats["confidence_failures"] += 1
                elif "ambiguous" in error_msg or "similar" in error_msg:
                    stats["ambiguity_failures"] += 1
        
        # Calculate averages
        if confidence_scores:
            stats["avg_confidence"] = sum(confidence_scores) / len(confidence_scores)
        if quality_scores:
            stats["avg_quality_score"] = sum(quality_scores) / len(quality_scores)
        
        # Calculate recognition rate
        if stats["total_attempts"] > 0:
            stats["recognition_rate"] = (stats["successful_recognitions"] / stats["total_attempts"]) * 100
        
        return {
            "status": "success",
            "period_days": days,
            "statistics": stats,
            "recommendations": _generate_accuracy_recommendations(stats)
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to get accuracy statistics: {str(e)}"
        }

def _generate_accuracy_recommendations(stats):
    """Generate recommendations based on accuracy statistics"""
    recommendations = []
    
    if stats["recognition_rate"] < 85:
        recommendations.append("Recognition rate is below 85% - consider adjusting thresholds or improving image quality requirements")
    
    if stats["quality_failures"] > stats["successful_recognitions"] * 0.3:
        recommendations.append("High quality failure rate - provide better guidance to users on image capture")
    
    if stats["confidence_failures"] > stats["successful_recognitions"] * 0.2:
        recommendations.append("High confidence failure rate - may need to retrain face embeddings or adjust confidence thresholds")
    
    if stats["ambiguity_failures"] > stats["successful_recognitions"] * 0.1:
        recommendations.append("Ambiguity issues detected - consider retraining similar-looking employees with higher quality images")
    
    if stats["avg_confidence"] > 0 and stats["avg_confidence"] < 80:
        recommendations.append(f"Average confidence is {stats['avg_confidence']:.1f}% - consider improving face embedding quality")
    
    if not recommendations:
        recommendations.append("Face recognition accuracy is performing well within expected parameters")
    
    return recommendations

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

@frappe.whitelist()
def upload_multiple_face_images(employee_id, images_base64_list, validate_consistency=True):
    """
    Upload multiple images for a single employee to create robust face data
    This helps overcome strict quality requirements by using ensemble learning
    
    Args:
        employee_id: Employee ID
        images_base64_list: JSON string or list of base64 encoded images
        validate_consistency: Whether to check if all images are of the same person
    """
    try:
        if not FACE_RECOGNITION_AVAILABLE:
            return {
                "status": "error",
                "message": "Face recognition libraries not available"
            }
        
        # Parse images if provided as JSON string
        if isinstance(images_base64_list, str):
            import json
            try:
                images_base64_list = json.loads(images_base64_list)
            except json.JSONDecodeError:
                return {
                    "status": "error",
                    "message": "Invalid JSON format for images list"
                }
        
        if not isinstance(images_base64_list, list) or len(images_base64_list) < 1:
            return {
                "status": "error", 
                "message": "At least one image is required"
            }
        
        if len(images_base64_list) > 10:
            return {
                "status": "error",
                "message": "Maximum 10 images allowed per upload"
            }
        
        # Validate employee exists
        if not frappe.db.exists("Employee", employee_id):
            return {
                "status": "error",
                "message": f"Employee {employee_id} not found"
            }
        
        # Optional: Validate image consistency 
        consistency_result = None
        if validate_consistency and len(images_base64_list) > 1:
            consistency_result = validate_multi_image_consistency(
                images_base64_list, 
                similarity_threshold=0.5  # More lenient for multi-image uploads
            )
            
            if not consistency_result.get('consistent', False):
                return {
                    "status": "warning",
                    "message": "Images may be of different people",
                    "details": {
                        "consistency_check": consistency_result,
                        "suggestion": "Please ensure all images are of the same person, or set validate_consistency=false to skip this check"
                    }
                }
        
        # Create multi-image face data
        result = create_multi_image_face_data(
            images_base64_list, 
            employee_id=employee_id,
            use_lenient_quality=True
        )
        
        if not result.get('success', False):
            return {
                "status": "error",
                "message": result.get('message', 'Failed to create face data'),
                "details": result
            }
        
        # Save the face encoding
        try:
            embedding_dir = get_embedding_directory()
            if not embedding_dir:
                return {
                    "status": "error",
                    "message": "Could not determine embedding directory"
                }
            
            # Ensure directory exists
            os.makedirs(embedding_dir, exist_ok=True)
            
            # Save face encoding
            encoding_path = os.path.join(embedding_dir, f"{employee_id}.npy")
            np.save(encoding_path, result['face_encoding'])
            
            frappe.log_error(
                f"Multi-image face data created for {employee_id}: {result['images_processed']} images processed, {result['images_failed']} failed",
                "Multi-Image Face Upload Success"
            )
            
            return {
                "status": "success",
                "message": f"Face data created successfully from {result['images_processed']} images",
                "details": {
                    "employee_id": employee_id,
                    "images_processed": result['images_processed'],
                    "images_failed": result['images_failed'],
                    "average_quality": sum(result['quality_scores']) / len(result['quality_scores']) if result['quality_scores'] else 0,
                    "quality_scores": result['quality_scores'],
                    "processing_details": result.get('details', []),
                    "consistency_check": consistency_result,
                    "encoding_path": encoding_path
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to save face encoding: {str(e)}"
            }
            
    except Exception as e:
        frappe.log_error(f"Error in multi-image face upload for {employee_id}: {str(e)}", "Multi-Image Face Upload Error")
        return {
            "status": "error",
            "message": f"Multi-image face upload failed: {str(e)}"
        }

@frappe.whitelist()
def validate_face_image_consistency(images_base64_list, similarity_threshold=0.5):
    """
    Validate that multiple images are of the same person before processing
    Useful for frontend validation before actual upload
    
    Args:
        images_base64_list: JSON string or list of base64 encoded images
        similarity_threshold: Minimum similarity required (default 0.7)
    """
    try:
        if not FACE_RECOGNITION_AVAILABLE:
            return {
                "status": "error",
                "message": "Face recognition libraries not available"
            }
        
        # Parse images if provided as JSON string
        if isinstance(images_base64_list, str):
            import json
            try:
                images_base64_list = json.loads(images_base64_list)
            except json.JSONDecodeError:
                return {
                    "status": "error",
                    "message": "Invalid JSON format for images list"
                }
        
        if not isinstance(images_base64_list, list):
            return {
                "status": "error",
                "message": "Images must be provided as a list"
            }
        
        if len(images_base64_list) < 2:
            return {
                "status": "success",
                "message": "Single image provided - no consistency check needed",
                "consistent": True
            }
        
        # Perform consistency validation
        result = validate_multi_image_consistency(images_base64_list, similarity_threshold)
        
        return {
            "status": "success",
            "message": result.get('message', 'Consistency check completed'),
            "consistent": result.get('consistent', False),
            "details": {
                "min_similarity": result.get('min_similarity', 0),
                "avg_similarity": result.get('avg_similarity', 0),
                "threshold": similarity_threshold,
                "similarity_matrix": result.get('similarity_matrix', [])
            }
        }
        
    except Exception as e:
        frappe.log_error(f"Error in face consistency validation: {str(e)}", "Face Consistency Validation Error") 
        return {
            "status": "error",
            "message": f"Consistency validation failed: {str(e)}"
        }

@frappe.whitelist()
def preview_multi_image_face_data(images_base64_list):
    """
    Preview what would happen with multi-image face data creation without saving
    Useful for testing and validation before actual upload
    
    Args:
        images_base64_list: JSON string or list of base64 encoded images
    """
    try:
        if not FACE_RECOGNITION_AVAILABLE:
            return {
                "status": "error",
                "message": "Face recognition libraries not available"
            }
        
        # Parse images if provided as JSON string  
        if isinstance(images_base64_list, str):
            import json
            try:
                images_base64_list = json.loads(images_base64_list)
            except json.JSONDecodeError:
                return {
                    "status": "error",
                    "message": "Invalid JSON format for images list"
                }
        
        if not isinstance(images_base64_list, list) or len(images_base64_list) < 1:
            return {
                "status": "error",
                "message": "At least one image is required"
            }
        
        # Create preview of multi-image processing
        result = create_multi_image_face_data(
            images_base64_list,
            employee_id="preview",
            use_lenient_quality=True
        )
        
        # Also check consistency if multiple images
        consistency_result = None
        if len(images_base64_list) > 1:
            consistency_result = validate_multi_image_consistency(images_base64_list, 0.7)
        
        return {
            "status": "success" if result.get('success', False) else "warning",
            "message": result.get('message', 'Processing completed'),
            "preview": {
                "would_succeed": result.get('success', False),
                "images_processed": result.get('images_processed', 0),
                "images_failed": result.get('images_failed', 0),
                "average_quality": sum(result.get('quality_scores', [])) / len(result.get('quality_scores', [])) if result.get('quality_scores') else 0,
                "quality_scores": result.get('quality_scores', []),
                "processing_details": result.get('details', []),
                "consistency_check": consistency_result,
                "has_valid_encoding": result.get('face_encoding') is not None
            }
        }
        
    except Exception as e:
        frappe.log_error(f"Error in multi-image face data preview: {str(e)}", "Multi-Image Preview Error")
        return {
            "status": "error", 
            "message": f"Preview failed: {str(e)}"
        }