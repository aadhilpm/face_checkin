#!/usr/bin/env python3
"""
Quick test script to verify project functionality
Run this in the Frappe console to test project loading
"""

import frappe

def test_projects():
    print("=" * 50)
    print("FACE CHECK-IN PROJECT TEST")
    print("=" * 50)
    
    # Test 1: Check if Project doctype exists
    print("\n1. Testing Project DocType existence...")
    project_exists = frappe.db.exists("DocType", "Project")
    print(f"   Project DocType exists: {project_exists}")
    
    if not project_exists:
        print("   ❌ ERPNext not properly installed or Project module missing")
        return
    
    # Test 2: Count total projects
    print("\n2. Counting projects...")
    try:
        total_count = frappe.db.count("Project")
        print(f"   Total projects in system: {total_count}")
    except Exception as e:
        print(f"   ❌ Error counting projects: {e}")
        return
    
    # Test 3: Get sample projects
    print("\n3. Getting sample projects...")
    try:
        projects = frappe.db.sql("""
            SELECT name, project_name, status, disabled
            FROM `tabProject`
            ORDER BY creation DESC
            LIMIT 5
        """, as_dict=True)
        
        print(f"   Found {len(projects)} sample projects:")
        for project in projects:
            status_text = f"Status: {project.status or 'None'}, Disabled: {project.disabled or 0}"
            print(f"   • {project.project_name or project.name} ({project.name}) - {status_text}")
            
    except Exception as e:
        print(f"   ❌ Error getting sample projects: {e}")
        return
    
    # Test 4: Test the actual API
    print("\n4. Testing get_projects API...")
    try:
        from face_checkin.api.face_api import get_projects
        result = get_projects()
        print(f"   API Response: {result}")
        
        if result.get('status') == 'success':
            project_count = len(result.get('projects', []))
            print(f"   ✅ API returned {project_count} projects successfully")
            
            if project_count > 0:
                print("   Sample projects from API:")
                for project in result['projects'][:3]:
                    print(f"   • {project.get('project_name', project.get('name'))} ({project.get('name')})")
        else:
            print(f"   ❌ API returned error: {result.get('message')}")
            
    except Exception as e:
        print(f"   ❌ API test failed: {e}")
    
    # Test 5: Check permissions
    print("\n5. Testing permissions...")
    try:
        can_read = frappe.has_permission("Project", "read")
        print(f"   Current user can read projects: {can_read}")
        print(f"   Current user: {frappe.session.user}")
        print(f"   User roles: {frappe.get_roles()}")
    except Exception as e:
        print(f"   ❌ Permission test failed: {e}")
    
    print("\n" + "=" * 50)
    print("TEST COMPLETE")
    print("=" * 50)

if __name__ == "__main__":
    test_projects()