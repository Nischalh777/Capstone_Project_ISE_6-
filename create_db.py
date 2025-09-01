# create_db.py

from app import app, db

print("--- Initializing Database ---")
print("This script will create the 'detections.db' file and the necessary tables.")

# The 'with app.app_context()' is crucial. It tells SQLAlchemy
# which Flask application it should create the database for.
# This ensures that the database connection and configuration are
# correctly loaded from your main app.
with app.app_context():
    print("Creating all database tables...")
    
    # This command inspects your models (like the 'Detection' class in app.py)
    # and creates a matching table in the SQLite database.
    db.create_all()
    
    print("✅ Database tables created successfully.")
    print("You should now see a 'detections.db' file in your project folder.")