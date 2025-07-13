import frappe

def get_context(context):
    # Allow access to setup page for all logged in users
    if frappe.session.user == "Guest":
        frappe.throw("Please login to access this page", frappe.PermissionError)
    
    # Add user context
    context.user = frappe.session.user
    
    return context