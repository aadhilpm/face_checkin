import frappe

def create_sample_projects():
    """Create sample projects for face checkin testing"""
    
    sample_projects = [
        {
            "project_name": "Default Office Project",
            "status": "Open",
            "project_type": "Internal",
            "is_active": "Yes"
        },
        {
            "project_name": "Development Team",
            "status": "Open", 
            "project_type": "Internal",
            "is_active": "Yes"
        },
        {
            "project_name": "Marketing Department",
            "status": "Open",
            "project_type": "Internal", 
            "is_active": "Yes"
        }
    ]
    
    created_projects = []
    
    for project_data in sample_projects:
        # Check if project already exists
        if not frappe.db.exists("Project", {"project_name": project_data["project_name"]}):
            try:
                project = frappe.get_doc({
                    "doctype": "Project",
                    **project_data
                })
                project.insert()
                created_projects.append(project.name)
                print(f"Created project: {project.project_name}")
            except Exception as e:
                print(f"Error creating project {project_data['project_name']}: {str(e)}")
        else:
            print(f"Project already exists: {project_data['project_name']}")
    
    return created_projects

if __name__ == "__main__":
    frappe.init(site="localhost")
    frappe.connect()
    create_sample_projects()
    frappe.db.commit()