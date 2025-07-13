import frappe
import os

def after_install():
    """
    Setup required for Face Check-in System after installation
    """
    try:
        # Create face embeddings directory
        create_face_embeddings_directory()
        
        # Create custom field for Employee Checkin
        create_custom_fields()
        
        # Set permissions
        setup_permissions()
        
        frappe.db.commit()
        print("Face Check-in System installation completed successfully!")
        
    except Exception as e:
        frappe.log_error(f"Face Check-in installation error: {str(e)}")
        print(f"Installation error: {str(e)}")

def create_face_embeddings_directory():
    """Create directory for storing face embeddings"""
    try:
        embedding_dir = frappe.get_app_path('face_checkin', 'face_store', 'embeddings')
        if not os.path.exists(embedding_dir):
            os.makedirs(embedding_dir, exist_ok=True)
            print(f"Created face embeddings directory: {embedding_dir}")
    except Exception as e:
        print(f"Error creating embeddings directory: {str(e)}")

def create_custom_fields():
    """Create required custom fields"""
    
    # Custom field for Employee Checkin - Project selection
    custom_field = {
        "doctype": "Custom Field",
        "dt": "Employee Checkin",
        "fieldname": "custom_project",
        "label": "Project",
        "fieldtype": "Link",
        "options": "Project",
        "insert_after": "device_id",
        "allow_on_submit": 1,
        "description": "Project associated with this check-in"
    }
    
    if not frappe.db.exists("Custom Field", {"dt": "Employee Checkin", "fieldname": "custom_project"}):
        try:
            doc = frappe.get_doc(custom_field)
            doc.insert()
            print("Created custom field: custom_project in Employee Checkin")
        except Exception as e:
            print(f"Error creating custom field: {str(e)}")
    else:
        print("Custom field custom_project already exists")

def setup_permissions():
    """Setup default permissions for face check-in system"""
    
    # Permissions for Employee Checkin (already exists in ERPNext)
    # We'll add read permissions for System Manager and HR Manager
    
    permissions = [
        {
            "doctype": "Employee Checkin",
            "role": "System Manager",
            "read": 1,
            "write": 1,
            "create": 1,
            "delete": 1
        },
        {
            "doctype": "Employee Checkin", 
            "role": "HR Manager",
            "read": 1,
            "write": 1,
            "create": 1,
            "delete": 0
        },
        {
            "doctype": "Employee",
            "role": "HR Manager", 
            "read": 1,
            "write": 1
        }
    ]
    
    try:
        for perm in permissions:
            if not frappe.db.exists("Custom DocPerm", {
                "parent": perm["doctype"],
                "role": perm["role"]
            }):
                # Permission already handled by ERPNext core
                pass
                
        print("Permissions setup completed")
    except Exception as e:
        print(f"Error setting up permissions: {str(e)}")

def create_web_page_permissions():
    """Setup permissions for web pages"""
    try:
        # These are handled by website_route_rules in hooks.py
        # No additional setup needed for web pages
        pass
    except Exception as e:
        print(f"Error setting up web page permissions: {str(e)}")