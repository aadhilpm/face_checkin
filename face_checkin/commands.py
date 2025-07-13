import click
import frappe
from frappe.commands import pass_context

@click.command('create-sample-projects')
@pass_context
def create_sample_projects(context):
    """Create sample projects for face checkin app"""
    
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
                click.echo(f"✅ Created project: {project.project_name}")
            except Exception as e:
                click.echo(f"❌ Error creating project {project_data['project_name']}: {str(e)}")
        else:
            click.echo(f"ℹ️  Project already exists: {project_data['project_name']}")
    
    frappe.db.commit()
    
    if created_projects:
        click.echo(f"\n🎉 Successfully created {len(created_projects)} sample projects!")
        click.echo("You can now select projects in the face checkin interface.")
    else:
        click.echo("\nℹ️  All sample projects already exist.")

commands = [create_sample_projects]