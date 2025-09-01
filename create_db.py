from app import app, db

print("--- Initializing Database ---")
with app.app_context():
    print("Creating all database tables based on models in app.py...")
    db.create_all()
    print("✅ Database tables created successfully.")
    print("A 'detections.db' file should now be in your project folder.")