import frappe

def get_context(context):
    # Check if user is logged in and has admin permissions
    if frappe.session.user == "Guest":
        frappe.throw("Please login to access this page", frappe.PermissionError)
    
    # Check if user has System Manager or HR Manager role
    if not ("System Manager" in frappe.get_roles() or "HR Manager" in frappe.get_roles()):
        frappe.throw("You don't have permission to access this page", frappe.PermissionError)
    
    # Add user context
    context.user = frappe.session.user
    context.csrf_token = frappe.sessions.get_csrf_token()
    
    return context