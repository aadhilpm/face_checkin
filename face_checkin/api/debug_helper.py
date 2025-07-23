import frappe
import os
import numpy as np

@frappe.whitelist()
def debug_face_files():
    """
    Debug helper to identify face file issues
    """
    try:
        # Get embedding directory
        def get_embedding_directory():
            try:
                site_path = frappe.get_site_path()
                return os.path.join(site_path, 'private', 'files', 'face_data')
            except:
                return None
        
        embedding_dir = get_embedding_directory()
        
        if not embedding_dir:
            return {"error": "Could not determine embedding directory"}
        
        if not os.path.exists(embedding_dir):
            return {"error": f"Embedding directory does not exist: {embedding_dir}"}
        
        # List all files
        all_files = os.listdir(embedding_dir)
        npy_files = [f for f in all_files if f.endswith('.npy')]
        
        debug_info = {
            "embedding_dir": embedding_dir,
            "all_files": all_files,
            "npy_files": npy_files,
            "file_details": []
        }
        
        for filename in npy_files:
            employee_id = filename.replace('.npy', '')
            filepath = os.path.join(embedding_dir, filename)
            
            file_info = {
                "filename": filename,
                "employee_id": employee_id,
                "file_size": os.path.getsize(filepath),
                "employee_exists": frappe.db.exists("Employee", employee_id)
            }
            
            try:
                # Try to load the embedding
                embedding = np.load(filepath)
                file_info["embedding_shape"] = str(embedding.shape)
                file_info["embedding_dtype"] = str(embedding.dtype)
                file_info["is_valid_shape"] = len(embedding.shape) == 1
                
                if len(embedding.shape) == 1:
                    file_info["dimensions"] = embedding.shape[0]
                    if embedding.shape[0] == 64:
                        file_info["type"] = "OpenCV (64-dim)"
                        file_info["compatible"] = True
                    else:
                        file_info["type"] = f"Invalid ({embedding.shape[0]}-dim)"
                        file_info["compatible"] = False
                        file_info["issue"] = f"Expected 64 dimensions, got {embedding.shape[0]}. Please re-enroll."
                else:
                    file_info["compatible"] = False
                    file_info["issue"] = f"Invalid shape: {embedding.shape}"
                    
            except Exception as e:
                file_info["load_error"] = str(e)
                file_info["compatible"] = False
                file_info["issue"] = f"Cannot load file: {str(e)}"
            
            debug_info["file_details"].append(file_info)
        
        return debug_info
        
    except Exception as e:
        return {"error": f"Debug failed: {str(e)}"}

@frappe.whitelist()
def fix_face_file_issue(employee_id):
    """
    Fix a specific face file issue by removing the problematic file
    """
    try:
        def get_embedding_directory():
            try:
                site_path = frappe.get_site_path()
                return os.path.join(site_path, 'private', 'files', 'face_data')
            except:
                return None
        
        embedding_dir = get_embedding_directory()
        filepath = os.path.join(embedding_dir, f"{employee_id}.npy")
        
        if os.path.exists(filepath):
            os.remove(filepath)
            return {
                "success": True,
                "message": f"Removed problematic face file for employee {employee_id}",
                "action": "Please re-enroll this employee"
            }
        else:
            return {
                "success": False,
                "message": f"Face file for employee {employee_id} not found"
            }
            
    except Exception as e:
        return {
            "success": False,
            "message": f"Failed to fix file: {str(e)}"
        }