import sqlite3
import os

# Path to the database file
db_path = 'experiments.db'

# Check if the database file exists
if not os.path.exists(db_path):
    print(f"Database file {db_path} not found.")
    exit(1)

try:
    # Connect to the database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check if the error_message column already exists
    cursor.execute("PRAGMA table_info(experiments)")
    columns = cursor.fetchall()
    column_names = [column[1] for column in columns]
    
    if 'error_message' not in column_names:
        # Add the error_message column
        cursor.execute("ALTER TABLE experiments ADD COLUMN error_message TEXT")
        conn.commit()
        print("Added 'error_message' column to the experiments table.")
    else:
        print("The 'error_message' column already exists in the experiments table.")
    
    # Close the connection
    conn.close()
    print("Database update completed successfully.")
    
except sqlite3.Error as e:
    print(f"SQLite error: {e}")
except Exception as e:
    print(f"Error: {e}")
