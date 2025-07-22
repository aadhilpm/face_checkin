app_name = "face_checkin"
app_title = "Face Checkin"
app_publisher = "Aadhil Badeel Technology"
app_description = "Employee attendance tracking using facial recognition"
app_email = "aadhil@badeeltechnology.com"
app_license = "MIT"
app_version = "1.0.0"

# Required apps
required_apps = ["frappe", "erpnext"]

# Website Route Rules
website_route_rules = [
	{"from_route": "/checkin", "to_route": "checkin"},
	{"from_route": "/setup", "to_route": "setup"},
	{"from_route": "/employee-images", "to_route": "employee_images"},
	{"from_route": "/offline", "to_route": "offline"},
]

# Installation
after_install = "face_checkin.install.after_install"