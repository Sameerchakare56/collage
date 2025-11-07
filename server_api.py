from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from typing import Optional, List
from datetime import date, datetime, time
import mysql.connector
from mysql.connector import Error
import struct
import json
import base64
import numpy as np
import uvicorn

# =========================
# FastAPI App
# =========================
app = FastAPI(
    title="Face Recognition Attendance System API",
    version="2.0.0",
    description="Complete API for face recognition based attendance system"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Database Configuration
# =========================
DB_CONFIG = {
    'host': 'localhost',
    'database': 'college_attendance',
    'user': 'sameer',
    'password': '12345'
}

def get_db_connection():
    """Create database connection"""
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        return connection
    except Error as e:
        raise HTTPException(status_code=500, detail=f"Database connection failed: {str(e)}")

# =========================
# Helper Functions
# =========================

def embedding_to_binary(embedding_list):
    """Convert embedding list to binary format for VARBINARY storage"""
    return struct.pack(f'{len(embedding_list)}f', *embedding_list)

def binary_to_embedding(binary_data):
    """Convert binary data back to embedding list"""
    if not binary_data:
        return None
    num_floats = len(binary_data) // 4
    return list(struct.unpack(f'{num_floats}f', binary_data))

def calculate_face_distance(embedding1, embedding2):
    """Calculate Euclidean distance between two face embeddings"""
    emb1 = np.array(embedding1)
    emb2 = np.array(embedding2)
    return float(np.linalg.norm(emb1 - emb2))

# =========================
# Pydantic Models
# =========================

class StudentCreate(BaseModel):
    roll_no: str
    name: str
    email: Optional[EmailStr] = None
    phone: Optional[str] = None
    department: Optional[str] = None
    semester: Optional[int] = None

class StudentUpdate(BaseModel):
    roll_no: Optional[str] = None
    name: Optional[str] = None
    email: Optional[EmailStr] = None
    phone: Optional[str] = None
    department: Optional[str] = None
    semester: Optional[int] = None

class TeacherCreate(BaseModel):
    emp_code: str
    name: str
    email: Optional[EmailStr] = None
    phone: Optional[str] = None
    department: Optional[str] = None
    designation: Optional[str] = None

class TeacherUpdate(BaseModel):
    emp_code: Optional[str] = None
    name: Optional[str] = None
    email: Optional[EmailStr] = None
    phone: Optional[str] = None
    department: Optional[str] = None
    designation: Optional[str] = None

class CourseCreate(BaseModel):
    course_code: str
    course_name: str
    department: Optional[str] = None
    semester: Optional[int] = None
    teacher_id: int

class CourseUpdate(BaseModel):
    course_code: Optional[str] = None
    course_name: Optional[str] = None
    department: Optional[str] = None
    semester: Optional[int] = None
    teacher_id: Optional[int] = None

class EnrollmentCreate(BaseModel):
    course_id: int
    student_id: int
    enrollment_date: Optional[date] = None

class AttendanceCreate(BaseModel):
    student_id: int
    course_id: int
    date: date
    check_in: datetime
    check_out: Optional[datetime] = None
    status: str = 'present'

class AttendanceUpdate(BaseModel):
    check_out: Optional[datetime] = None
    status: Optional[str] = None

# =========================
# ROOT & HEALTH ENDPOINTS
# =========================

@app.get("/")
async def root():
    """API root endpoint with system information"""
    return {
        "message": "Face Recognition Attendance System API",
        "version": "2.0.0",
        "status": "running",
        "features": [
            "Face recognition based attendance",
            "Real-time face matching with 128D embeddings",
            "Student and course management",
            "Attendance tracking and reports",
            "Recognition confidence scoring",
            "Duplicate attendance prevention"
        ],
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "students": "/students/",
            "teachers": "/teachers/",
            "courses": "/courses/",
            "attendance": "/attendance/",
            "recognition": "/recognize-face"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.fetchone()
        cursor.close()
        conn.close()
        return {
            "status": "healthy",
            "database": "connected",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "database": "disconnected",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# =========================
# STUDENT ENDPOINTS
# =========================

@app.post("/students/", status_code=201)
async def create_student(student: StudentCreate):
    """Create a new student"""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        query = """
            INSERT INTO students (roll_no, name, email, phone, department, semester) 
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        cursor.execute(query, (
            student.roll_no, student.name, student.email,
            student.phone, student.department, student.semester
        ))
        conn.commit()
        return {
            "message": "Student created successfully",
            "student_id": cursor.lastrowid,
            "roll_no": student.roll_no
        }
    except Error as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        cursor.close()
        conn.close()

@app.get("/students/")
async def get_all_students():
    """Get all students with their statistics"""
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        query = """
            SELECT s.*,
                   COUNT(DISTINCT si.image_id) as image_count,
                   SUM(CASE WHEN si.face_embedding IS NOT NULL THEN 1 ELSE 0 END) as embeddings_count
            FROM students s
            LEFT JOIN student_images si ON s.student_id = si.student_id
            GROUP BY s.student_id
            ORDER BY s.roll_no
        """
        cursor.execute(query)
        return cursor.fetchall()
    finally:
        cursor.close()
        conn.close()

@app.get("/students/{student_id}")
async def get_student(student_id: int):
    """Get student details by ID"""
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT * FROM students WHERE student_id = %s", (student_id,))
        student = cursor.fetchone()
        
        if not student:
            raise HTTPException(status_code=404, detail="Student not found")
        
        cursor.execute("""
            SELECT 
                COUNT(*) as image_count,
                SUM(CASE WHEN face_embedding IS NOT NULL THEN 1 ELSE 0 END) as embeddings_count,
                AVG(quality_score) as avg_quality_score
            FROM student_images 
            WHERE student_id = %s
        """, (student_id,))
        stats = cursor.fetchone()
        
        cursor.execute("""
            SELECT c.course_id, c.course_code, c.course_name, c.department
            FROM courses c
            JOIN course_student cs ON c.course_id = cs.course_id
            WHERE cs.student_id = %s
        """, (student_id,))
        courses = cursor.fetchall()
        
        student.update(stats)
        student['enrolled_courses'] = courses
        
        return student
    finally:
        cursor.close()
        conn.close()

@app.put("/students/{student_id}")
async def update_student(student_id: int, student: StudentUpdate):
    """Update student information"""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        updates = []
        values = []
        
        for key, value in student.dict(exclude_unset=True).items():
            updates.append(f"{key} = %s")
            values.append(value)
        
        if not updates:
            raise HTTPException(status_code=400, detail="No fields to update")
        
        values.append(student_id)
        query = f"UPDATE students SET {', '.join(updates)} WHERE student_id = %s"
        cursor.execute(query, values)
        conn.commit()
        
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Student not found")
        
        return {"message": "Student updated successfully"}
    except Error as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        cursor.close()
        conn.close()

@app.delete("/students/{student_id}")
async def delete_student(student_id: int):
    """Delete a student"""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM students WHERE student_id = %s", (student_id,))
        conn.commit()
        
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Student not found")
        
        return {"message": "Student deleted successfully"}
    finally:
        cursor.close()
        conn.close()

# =========================
# TEACHER ENDPOINTS
# =========================

@app.post("/teachers/", status_code=201)
async def create_teacher(teacher: TeacherCreate):
    """Create a new teacher"""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        query = """
            INSERT INTO teachers (emp_code, name, email, phone, department, designation) 
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        cursor.execute(query, (
            teacher.emp_code, teacher.name, teacher.email,
            teacher.phone, teacher.department, teacher.designation
        ))
        conn.commit()
        return {
            "message": "Teacher created successfully",
            "teacher_id": cursor.lastrowid
        }
    except Error as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        cursor.close()
        conn.close()

@app.get("/teachers/")
async def get_all_teachers():
    """Get all teachers"""
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT * FROM teachers ORDER BY name")
        return cursor.fetchall()
    finally:
        cursor.close()
        conn.close()

@app.get("/teachers/{teacher_id}")
async def get_teacher(teacher_id: int):
    """Get teacher details by ID"""
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT * FROM teachers WHERE teacher_id = %s", (teacher_id,))
        teacher = cursor.fetchone()
        
        if not teacher:
            raise HTTPException(status_code=404, detail="Teacher not found")
        
        cursor.execute("""
            SELECT course_id, course_code, course_name, department, semester
            FROM courses
            WHERE teacher_id = %s
        """, (teacher_id,))
        courses = cursor.fetchall()
        
        teacher['courses'] = courses
        return teacher
    finally:
        cursor.close()
        conn.close()

@app.put("/teachers/{teacher_id}")
async def update_teacher(teacher_id: int, teacher: TeacherUpdate):
    """Update teacher information"""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        updates = []
        values = []
        
        for key, value in teacher.dict(exclude_unset=True).items():
            updates.append(f"{key} = %s")
            values.append(value)
        
        if not updates:
            raise HTTPException(status_code=400, detail="No fields to update")
        
        values.append(teacher_id)
        query = f"UPDATE teachers SET {', '.join(updates)} WHERE teacher_id = %s"
        cursor.execute(query, values)
        conn.commit()
        
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Teacher not found")
        
        return {"message": "Teacher updated successfully"}
    except Error as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        cursor.close()
        conn.close()

@app.delete("/teachers/{teacher_id}")
async def delete_teacher(teacher_id: int):
    """Delete a teacher"""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM teachers WHERE teacher_id = %s", (teacher_id,))
        conn.commit()
        
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Teacher not found")
        
        return {"message": "Teacher deleted successfully"}
    finally:
        cursor.close()
        conn.close()

# =========================
# COURSE ENDPOINTS
# =========================

@app.post("/courses/", status_code=201)
async def create_course(course: CourseCreate):
    """Create a new course"""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        query = """
            INSERT INTO courses (course_code, course_name, department, semester, teacher_id) 
            VALUES (%s, %s, %s, %s, %s)
        """
        cursor.execute(query, (
            course.course_code, course.course_name, course.department,
            course.semester, course.teacher_id
        ))
        conn.commit()
        return {
            "message": "Course created successfully",
            "course_id": cursor.lastrowid
        }
    except Error as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        cursor.close()
        conn.close()

@app.get("/courses/")
async def get_all_courses():
    """Get all courses with statistics"""
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        query = """
            SELECT c.*, t.name as teacher_name, t.emp_code,
                   COUNT(DISTINCT cs.student_id) as enrolled_students
            FROM courses c
            LEFT JOIN teachers t ON c.teacher_id = t.teacher_id
            LEFT JOIN course_student cs ON c.course_id = cs.course_id
            GROUP BY c.course_id
            ORDER BY c.course_code
        """
        cursor.execute(query)
        return cursor.fetchall()
    finally:
        cursor.close()
        conn.close()

@app.get("/courses/{course_id}")
async def get_course(course_id: int):
    """Get course details by ID"""
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        query = """
            SELECT c.*, t.name as teacher_name, t.emp_code, t.email as teacher_email
            FROM courses c
            LEFT JOIN teachers t ON c.teacher_id = t.teacher_id
            WHERE c.course_id = %s
        """
        cursor.execute(query, (course_id,))
        course = cursor.fetchone()
        
        if not course:
            raise HTTPException(status_code=404, detail="Course not found")
        
        return course
    finally:
        cursor.close()
        conn.close()

@app.get("/courses/{course_id}/students")
async def get_course_students(course_id: int):
    """Get all students enrolled in a course"""
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        query = """
            SELECT s.*, cs.enrollment_date,
                   COUNT(DISTINCT si.image_id) as image_count,
                   SUM(CASE WHEN si.face_embedding IS NOT NULL THEN 1 ELSE 0 END) as embeddings_count
            FROM course_student cs
            JOIN students s ON cs.student_id = s.student_id
            LEFT JOIN student_images si ON s.student_id = si.student_id
            WHERE cs.course_id = %s
            GROUP BY s.student_id
            ORDER BY s.roll_no
        """
        cursor.execute(query, (course_id,))
        return cursor.fetchall()
    finally:
        cursor.close()
        conn.close()

@app.put("/courses/{course_id}")
async def update_course(course_id: int, course: CourseUpdate):
    """Update course information"""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        updates = []
        values = []
        
        for key, value in course.dict(exclude_unset=True).items():
            updates.append(f"{key} = %s")
            values.append(value)
        
        if not updates:
            raise HTTPException(status_code=400, detail="No fields to update")
        
        values.append(course_id)
        query = f"UPDATE courses SET {', '.join(updates)} WHERE course_id = %s"
        cursor.execute(query, values)
        conn.commit()
        
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Course not found")
        
        return {"message": "Course updated successfully"}
    except Error as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        cursor.close()
        conn.close()

@app.delete("/courses/{course_id}")
async def delete_course(course_id: int):
    """Delete a course"""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM courses WHERE course_id = %s", (course_id,))
        conn.commit()
        
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Course not found")
        
        return {"message": "Course deleted successfully"}
    finally:
        cursor.close()
        conn.close()

# =========================
# ENROLLMENT ENDPOINTS
# =========================

@app.post("/enrollments/", status_code=201)
async def create_enrollment(enrollment: EnrollmentCreate):
    """Enroll a student in a course"""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        query = """
            INSERT INTO course_student (course_id, student_id, enrollment_date) 
            VALUES (%s, %s, %s)
        """
        cursor.execute(query, (
            enrollment.course_id,
            enrollment.student_id,
            enrollment.enrollment_date or date.today()
        ))
        conn.commit()
        return {
            "message": "Enrollment created successfully",
            "enrollment_id": cursor.lastrowid
        }
    except Error as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        cursor.close()
        conn.close()

@app.get("/enrollments/student/{student_id}")
async def get_student_courses(student_id: int):
    """Get all courses a student is enrolled in"""
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        query = """
            SELECT cs.*, c.course_code, c.course_name, c.department, c.semester,
                   t.name as teacher_name
            FROM course_student cs
            JOIN courses c ON cs.course_id = c.course_id
            LEFT JOIN teachers t ON c.teacher_id = t.teacher_id
            WHERE cs.student_id = %s
            ORDER BY c.course_code
        """
        cursor.execute(query, (student_id,))
        return cursor.fetchall()
    finally:
        cursor.close()
        conn.close()

@app.delete("/enrollments/{enrollment_id}")
async def delete_enrollment(enrollment_id: int):
    """Remove a student enrollment"""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM course_student WHERE id = %s", (enrollment_id,))
        conn.commit()
        
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Enrollment not found")
        
        return {"message": "Enrollment deleted successfully"}
    finally:
        cursor.close()
        conn.close()

# =========================
# STUDENT IMAGE UPLOAD
# =========================

@app.post("/student-images/{student_id}", status_code=201)
async def upload_student_image(
    student_id: int,
    file: UploadFile = File(...),
    face_embedding: Optional[str] = Form(None),
    quality_score: Optional[float] = Form(None),
    is_primary: Optional[bool] = Form(False)
):
    """Upload student face image with embedding"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("SELECT student_id FROM students WHERE student_id = %s", (student_id,))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="Student not found")
        
        image_data = await file.read()
        
        embedding_binary = None
        embedding_dimensions = None
        
        if face_embedding:
            try:
                embedding_vector = json.loads(face_embedding)
                embedding_binary = embedding_to_binary(embedding_vector)
                embedding_dimensions = len(embedding_vector)
            except (json.JSONDecodeError, struct.error) as e:
                print(f"⚠️  Invalid embedding format: {e}")
        
        query = """
            INSERT INTO student_images 
            (student_id, face_image, face_embedding, embedding_dimensions, quality_score, is_primary) 
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        cursor.execute(query, (
            student_id,
            image_data,
            embedding_binary,
            embedding_dimensions,
            quality_score,
            is_primary
        ))
        conn.commit()
        
        return {
            "message": "Image uploaded successfully",
            "image_id": cursor.lastrowid,
            "student_id": student_id,
            "has_embedding": embedding_binary is not None,
            "embedding_dimensions": embedding_dimensions if embedding_dimensions else 0,
            "quality_score": quality_score
        }
        
    except Error as e:
        conn.rollback()
        raise HTTPException(status_code=400, detail=f"Database error: {str(e)}")
    finally:
        cursor.close()
        conn.close()

@app.get("/student-images/{student_id}")
async def get_student_images(student_id: int):
    """Get all images for a student (metadata only)"""
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        query = """
            SELECT 
                image_id, student_id, created_at, embedding_dimensions,
                quality_score, is_primary,
                CASE WHEN face_embedding IS NOT NULL THEN TRUE ELSE FALSE END as has_embedding
            FROM student_images 
            WHERE student_id = %s
            ORDER BY is_primary DESC, quality_score DESC, created_at DESC
        """
        cursor.execute(query, (student_id,))
        return cursor.fetchall()
    finally:
        cursor.close()
        conn.close()

@app.get("/student-images/download/{image_id}")
async def download_student_image(image_id: int):
    """Download a specific image"""
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        query = "SELECT face_image, student_id FROM student_images WHERE image_id = %s"
        cursor.execute(query, (image_id,))
        result = cursor.fetchone()
        
        if not result:
            raise HTTPException(status_code=404, detail="Image not found")
        
        image_base64 = base64.b64encode(result['face_image']).decode('utf-8')
        return {
            "image_id": image_id,
            "student_id": result['student_id'],
            "image_data": image_base64
        }
    finally:
        cursor.close()
        conn.close()

@app.delete("/student-images/{image_id}")
async def delete_student_image(image_id: int):
    """Delete a student image"""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM student_images WHERE image_id = %s", (image_id,))
        conn.commit()
        
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Image not found")
        
        return {"message": "Image deleted successfully"}
    finally:
        cursor.close()
        conn.close()

# =========================
# FACE RECOGNITION & ATTENDANCE
# =========================

@app.post("/recognize-face", status_code=200)
async def recognize_face(
    file: UploadFile = File(...),
    face_embedding: str = Form(...),
    course_id: int = Form(...),
    threshold: float = Form(0.6)
):
    """Recognize face and mark attendance"""
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    try:
        try:
            new_embedding = json.loads(face_embedding)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid embedding format")
        
        query = """
            SELECT 
                si.image_id, si.student_id, si.face_embedding, si.quality_score,
                s.roll_no, s.name, s.department, s.email
            FROM student_images si
            JOIN students s ON si.student_id = s.student_id
            JOIN course_student cs ON s.student_id = cs.student_id
            WHERE cs.course_id = %s AND si.face_embedding IS NOT NULL
            ORDER BY si.quality_score DESC
        """
        cursor.execute(query, (course_id,))
        student_images = cursor.fetchall()
        
        if not student_images:
            raise HTTPException(
                status_code=404,
                detail="No enrolled students with face embeddings found for this course"
            )
        
        best_match = None
        best_distance = float('inf')
        
        for img in student_images:
            stored_embedding = binary_to_embedding(img['face_embedding'])
            if stored_embedding:
                distance = calculate_face_distance(new_embedding, stored_embedding)
                if distance < best_distance:
                    best_distance = distance
                    best_match = img
        
        if best_distance > threshold:
            log_query = """
                INSERT INTO attendance_recognition_log 
                (student_id, course_id, recognition_confidence, face_match_distance, status)
                VALUES (NULL, %s, %s, %s, 'failed')
            """
            cursor.execute(log_query, (course_id, 0, best_distance))
            conn.commit()
            
            return {
                "recognized": False,
                "attendance_marked": False,
                "message": "No matching face found in database",
                "best_distance": best_distance,
                "threshold": threshold
            }
        
        student_id = best_match['student_id']
        confidence = max(0, min(100, (1 - best_distance) * 100))
        
        today = date.today()
        check_query = """
            SELECT attendance_id, check_in FROM student_attendance 
            WHERE student_id = %s AND course_id = %s AND date = %s
        """
        cursor.execute(check_query, (student_id, course_id, today))
        existing = cursor.fetchone()
        
        if existing:
            log_query = """
                INSERT INTO attendance_recognition_log 
                (student_id, course_id, recognition_confidence, face_match_distance, matched_image_id, status)
                VALUES (%s, %s, %s, %s, %s, 'duplicate')
            """
            cursor.execute(log_query, (
                student_id, course_id, confidence, best_distance, best_match['image_id']
            ))
            conn.commit()
            
            return {
                "recognized": True,
                "attendance_marked": False,
                "message": "Attendance already marked for today",
                "student": {
                    "student_id": student_id,
                    "roll_no": best_match['roll_no'],
                    "name": best_match['name'],
                    "department": best_match['department'],
                    "email": best_match['email']
                },
                "existing_attendance": {
                    "attendance_id": existing['attendance_id'],
                    "check_in": existing['check_in'].isoformat()
                },
                "confidence": round(confidence, 2),
                "distance": best_distance
            }
        
        attendance_query = """
            INSERT INTO student_attendance 
            (student_id, course_id, date, check_in, status, marked_by)
            VALUES (%s, %s, %s, %s, 'present', 'face_recognition')
        """
        current_time = datetime.now()
        cursor.execute(attendance_query, (student_id, course_id, today, current_time))
        attendance_id = cursor.lastrowid
        
        log_query = """
            INSERT INTO attendance_recognition_log 
            (student_id, course_id, recognition_confidence, face_match_distance, matched_image_id, status)
            VALUES (%s, %s, %s, %s, %s, 'success')
        """
        cursor.execute(log_query, (
            student_id, course_id, confidence, best_distance, best_match['image_id']
        ))
        conn.commit()
        
        return {
            "recognized": True,
            "attendance_marked": True,
            "message": "Attendance marked successfully",
            "student": {
                "student_id": student_id,
                "roll_no": best_match['roll_no'],
                "name": best_match['name'],
                "department": best_match['department'],
                "email": best_match['email']
            },
            "attendance": {
                "attendance_id": attendance_id,
                "check_in": current_time.isoformat(),
                "date": today.isoformat(),
                "status": "present"
            },
            "confidence": round(confidence, 2),
            "distance": best_distance
        }
        
    except HTTPException:
        raise
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Recognition failed: {str(e)}")
    finally:
        cursor.close()
        conn.close()

# =========================
# ATTENDANCE ENDPOINTS
# =========================

@app.post("/attendance/", status_code=201)
async def create_attendance(attendance: AttendanceCreate):
    """Manually create attendance record"""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        query = """
            INSERT INTO student_attendance 
            (student_id, course_id, date, check_in, check_out, status, marked_by)
            VALUES (%s, %s, %s, %s, %s, %s, 'manual')
        """
        cursor.execute(query, (
            attendance.student_id,
            attendance.course_id,
            attendance.date,
            attendance.check_in,
            attendance.check_out,
            attendance.status
        ))
        conn.commit()
        return {
            "message": "Attendance created successfully",
            "attendance_id": cursor.lastrowid
        }
    except Error as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        cursor.close()
        conn.close()

@app.get("/attendance/")
async def get_all_attendance(
    course_id: Optional[int] = None,
    student_id: Optional[int] = None,
    date_from: Optional[date] = None,
    date_to: Optional[date] = None
):
    """Get attendance records with filters"""
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        query = """
            SELECT 
                sa.*,
                s.roll_no, s.name as student_name, s.department,
                c.course_code, c.course_name
            FROM student_attendance sa
            JOIN students s ON sa.student_id = s.student_id
            JOIN courses c ON sa.course_id = c.course_id
            WHERE 1=1
        """
        params = []
        
        if course_id:
            query += " AND sa.course_id = %s"
            params.append(course_id)
        
        if student_id:
            query += " AND sa.student_id = %s"
            params.append(student_id)
        
        if date_from:
            query += " AND sa.date >= %s"
            params.append(date_from)
        
        if date_to:
            query += " AND sa.date <= %s"
            params.append(date_to)
        
        query += " ORDER BY sa.date DESC, sa.check_in DESC"
        
        cursor.execute(query, params)
        return cursor.fetchall()
    finally:
        cursor.close()
        conn.close()

@app.get("/attendance/{attendance_id}")
async def get_attendance(attendance_id: int):
    """Get specific attendance record"""
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        query = """
            SELECT 
                sa.*,
                s.roll_no, s.name as student_name, s.email, s.department,
                c.course_code, c.course_name
            FROM student_attendance sa
            JOIN students s ON sa.student_id = s.student_id
            JOIN courses c ON sa.course_id = c.course_id
            WHERE sa.attendance_id = %s
        """
        cursor.execute(query, (attendance_id,))
        attendance = cursor.fetchone()
        
        if not attendance:
            raise HTTPException(status_code=404, detail="Attendance record not found")
        
        return attendance
    finally:
        cursor.close()
        conn.close()

@app.put("/attendance/{attendance_id}")
async def update_attendance(attendance_id: int, attendance: AttendanceUpdate):
    """Update attendance record (e.g., check-out time)"""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        updates = []
        values = []
        
        for key, value in attendance.dict(exclude_unset=True).items():
            updates.append(f"{key} = %s")
            values.append(value)
        
        if not updates:
            raise HTTPException(status_code=400, detail="No fields to update")
        
        values.append(attendance_id)
        query = f"UPDATE student_attendance SET {', '.join(updates)} WHERE attendance_id = %s"
        cursor.execute(query, values)
        conn.commit()
        
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Attendance record not found")
        
        return {"message": "Attendance updated successfully"}
    except Error as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        cursor.close()
        conn.close()

@app.delete("/attendance/{attendance_id}")
async def delete_attendance(attendance_id: int):
    """Delete attendance record"""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM student_attendance WHERE attendance_id = %s", (attendance_id,))
        conn.commit()
        
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Attendance record not found")
        
        return {"message": "Attendance deleted successfully"}
    finally:
        cursor.close()
        conn.close()

# =========================
# ATTENDANCE REPORTS
# =========================

@app.get("/attendance/report/student/{student_id}")
async def get_student_attendance_report(student_id: int, course_id: Optional[int] = None):
    """Get attendance statistics for a student"""
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        query = """
            SELECT 
                c.course_id, c.course_code, c.course_name,
                COUNT(*) as total_classes,
                SUM(CASE WHEN sa.status = 'present' THEN 1 ELSE 0 END) as present_count,
                SUM(CASE WHEN sa.status = 'absent' THEN 1 ELSE 0 END) as absent_count,
                SUM(CASE WHEN sa.status = 'late' THEN 1 ELSE 0 END) as late_count,
                ROUND((SUM(CASE WHEN sa.status = 'present' THEN 1 ELSE 0 END) * 100.0 / COUNT(*)), 2) as attendance_percentage
            FROM student_attendance sa
            JOIN courses c ON sa.course_id = c.course_id
            WHERE sa.student_id = %s
        """
        params = [student_id]
        
        if course_id:
            query += " AND sa.course_id = %s"
            params.append(course_id)
        
        query += " GROUP BY c.course_id"
        
        cursor.execute(query, params)
        return cursor.fetchall()
    finally:
        cursor.close()
        conn.close()

@app.get("/attendance/report/course/{course_id}")
async def get_course_attendance_report(
    course_id: int,
    date_from: Optional[date] = None,
    date_to: Optional[date] = None
):
    """Get attendance statistics for a course"""
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        query = """
            SELECT 
                s.student_id, s.roll_no, s.name, s.department,
                COUNT(*) as total_classes,
                SUM(CASE WHEN sa.status = 'present' THEN 1 ELSE 0 END) as present_count,
                SUM(CASE WHEN sa.status = 'absent' THEN 1 ELSE 0 END) as absent_count,
                SUM(CASE WHEN sa.status = 'late' THEN 1 ELSE 0 END) as late_count,
                ROUND((SUM(CASE WHEN sa.status = 'present' THEN 1 ELSE 0 END) * 100.0 / COUNT(*)), 2) as attendance_percentage
            FROM students s
            JOIN course_student cs ON s.student_id = cs.student_id
            LEFT JOIN student_attendance sa ON s.student_id = sa.student_id AND sa.course_id = cs.course_id
            WHERE cs.course_id = %s
        """
        params = [course_id]
        
        if date_from:
            query += " AND sa.date >= %s"
            params.append(date_from)
        
        if date_to:
            query += " AND sa.date <= %s"
            params.append(date_to)
        
        query += " GROUP BY s.student_id ORDER BY s.roll_no"
        
        cursor.execute(query, params)
        return cursor.fetchall()
    finally:
        cursor.close()
        conn.close()

@app.get("/attendance/report/daily/{course_id}/{date}")
async def get_daily_attendance_report(course_id: int, date: date):
    """Get attendance report for a specific date"""
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        query = """
            SELECT 
                s.student_id, s.roll_no, s.name, s.department,
                sa.attendance_id, sa.check_in, sa.check_out, sa.status, sa.marked_by,
                CASE WHEN sa.attendance_id IS NOT NULL THEN TRUE ELSE FALSE END as is_present
            FROM students s
            JOIN course_student cs ON s.student_id = cs.student_id
            LEFT JOIN student_attendance sa ON s.student_id = sa.student_id 
                AND sa.course_id = cs.course_id 
                AND sa.date = %s
            WHERE cs.course_id = %s
            ORDER BY s.roll_no
        """
        cursor.execute(query, (date, course_id))
        return cursor.fetchall()
    finally:
        cursor.close()
        conn.close()

# =========================
# RECOGNITION LOG ENDPOINTS
# =========================

@app.get("/recognition-logs/")
async def get_recognition_logs(
    student_id: Optional[int] = None,
    course_id: Optional[int] = None,
    status: Optional[str] = None,
    limit: int = 100
):
    """Get recognition attempt logs"""
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        query = """
            SELECT 
                arl.*,
                s.roll_no, s.name as student_name,
                c.course_code, c.course_name
            FROM attendance_recognition_log arl
            LEFT JOIN students s ON arl.student_id = s.student_id
            LEFT JOIN courses c ON arl.course_id = c.course_id
            WHERE 1=1
        """
        params = []
        
        if student_id:
            query += " AND arl.student_id = %s"
            params.append(student_id)
        
        if course_id:
            query += " AND arl.course_id = %s"
            params.append(course_id)
        
        if status:
            query += " AND arl.status = %s"
            params.append(status)
        
        query += " ORDER BY arl.timestamp DESC LIMIT %s"
        params.append(limit)
        
        cursor.execute(query, params)
        return cursor.fetchall()
    finally:
        cursor.close()
        conn.close()

@app.get("/recognition-logs/stats")
async def get_recognition_stats():
    """Get overall recognition statistics"""
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        query = """
            SELECT 
                COUNT(*) as total_attempts,
                SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as successful,
                SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed,
                SUM(CASE WHEN status = 'duplicate' THEN 1 ELSE 0 END) as duplicates,
                AVG(recognition_confidence) as avg_confidence,
                AVG(face_match_distance) as avg_distance
            FROM attendance_recognition_log
        """
        cursor.execute(query)
        return cursor.fetchone()
    finally:
        cursor.close()
        conn.close()

# =========================
# RUN SERVER
# =========================

