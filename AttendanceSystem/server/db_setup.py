import sqlite3

def setup_database():
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()
    
    # Create attendance table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_name TEXT,
        timestamp REAL
    )
    ''')
    
    conn.commit()
    conn.close()

if __name__ == '__main__':
    setup_database()
