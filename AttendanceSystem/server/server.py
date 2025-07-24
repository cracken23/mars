from flask import Flask, request, jsonify, render_template
import sqlite3

app = Flask(__name__)

# Route to mark attendance
@app.route('/attendance', methods=['POST'])
def receive_data():
    try:
        # Try to get JSON data from the request
        data = request.json
        student_name = data['student_name']
        timestamp = data['timestamp']
        
        # Call the function to mark attendance
        mark_attendance(student_name, timestamp)
        
        # Return success response
        return jsonify({'message': 'Attendance marked successfully'}), 200
    
    except KeyError as e:
        # Handle missing key in JSON
        return jsonify({'error': f'Missing key: {str(e)}'}), 400
    
    except Exception as e:
        # Handle any other error
        return jsonify({'error': str(e)}), 500

# Function to insert attendance data into the database
def mark_attendance(student_name, timestamp):
    try:
        # Connect to the SQLite database
        conn = sqlite3.connect('attendance.db')
        cursor = conn.cursor()
        
        # Insert the attendance data
        cursor.execute("INSERT INTO attendance (student_name, timestamp) VALUES (?, ?)", (student_name, timestamp))
        
        # Commit the changes
        conn.commit()
    
    except sqlite3.Error as e:
        # Log and raise any SQLite errors
        raise Exception(f"Database error: {str(e)}")
    
    finally:
        # Close the connection to the database
        conn.close()

# Route to display the attendance records in HTML
@app.route('/')
def index():
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM attendance")
    records = cursor.fetchall()
    conn.close()
    return render_template('index.html', records=records)

# Route to get attendance records for AJAX updates
@app.route('/get_records', methods=['GET'])
def get_records():
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()
    
    # Fetch the most recent record for each student
    cursor.execute('''
        SELECT student_name, MAX(timestamp) as timestamp
        FROM attendance
        GROUP BY student_name
        ORDER BY timestamp DESC
    ''')
    
    records = cursor.fetchall()
    conn.close()
    return jsonify(records)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)
