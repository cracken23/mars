import sqlite3

def clear_attendance_db():
    """Clear all records from the attendance database."""
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()
    
    # Delete all records from the attendance table
    cursor.execute('DELETE FROM attendance')
    
    conn.commit()
    conn.close()
    print("Attendance records have been cleared.")

if __name__ == '__main__':
    clear_attendance_db()
