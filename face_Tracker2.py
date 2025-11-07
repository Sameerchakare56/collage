import cv2
import requests
import numpy as np
import json
import io
import face_recognition
from datetime import datetime, timedelta
import time
import threading
from ultralytics import YOLO

"""
=============================================================================
FACE RECOGNITION ATTENDANCE SYSTEM - YOLOv8 INTEGRATION
=============================================================================

MODELS USED:
1. YOLOv8 (Ultralytics):
   - Fast and accurate face detection
   - Real-time performance (60+ FPS on GPU)
   - Better detection than Haar Cascade
   
2. face_recognition library (dlib CNN):
   - Face Encoding: Deep Convolutional Neural Network
   - 128-dimensional face embeddings
   - Pre-trained on millions of faces

IMPROVEMENTS:
- Replaced Haar Cascade with YOLOv8 for better face detection
- Improved accuracy and speed
- Better handling of various face angles and lighting
=============================================================================
"""

# Configuration
API_BASE_URL = "http://localhost:8005"
RECOGNIZE_ENDPOINT = "/recognize-face"
COURSES_ENDPOINT = "/courses"
ATTENDANCE_UPDATE_ENDPOINT = "/attendance"

class CameraManager:
    """Manages camera initialization and frame capture"""
    def __init__(self):
        self.cap = None
        
    def initialize_camera(self, index=0):
        """Initialize camera using OpenCV VideoCapture"""
        self.cap = cv2.VideoCapture(index)
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            print(f"✅ Camera {index} initialized (640x480)")
            return True
        else:
            print(f"❌ Camera {index} not available")
            return False
    
    def read_frame(self):
        """Read frame from camera"""
        if self.cap:
            return self.cap.read()
        return False, None
    
    def release(self):
        """Release camera resources"""
        if self.cap:
            self.cap.release()

class YOLOv8FaceDetector:
    """
    YOLOv8-based face detection
    Much faster and more accurate than Haar Cascade
    """
    def __init__(self, model_path='yolov8n-face.pt'):
        """
        Initialize YOLOv8 face detection model
        
        Args:
            model_path: Path to YOLOv8 face model
                       'yolov8n-face.pt' - Nano (fastest)
                       'yolov8s-face.pt' - Small
                       'yolov8m-face.pt' - Medium
                       'yolov8l-face.pt' - Large (most accurate)
        """
        try:
            # Try to load face-specific model
            self.model = YOLO(model_path)
            print(f"✅ YOLOv8 Face Model loaded: {model_path}")
        except:
            # Fallback to general YOLOv8 model (will detect faces as 'person')
            print("⚠️  Face-specific model not found. Using YOLOv8n for person detection.")
            self.model = YOLO('yolov8n.pt')
        
        self.conf_threshold = 0.5  # Confidence threshold
        
    def detect_faces(self, frame):
        """
        Detect faces using YOLOv8
        Returns: List of (x, y, width, height, confidence) for each face
        """
        results = self.model(frame, verbose=False, conf=self.conf_threshold)
        
        faces = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0])
                
                # Convert to (x, y, w, h) format
                x = int(x1)
                y = int(y1)
                w = int(x2 - x1)
                h = int(y2 - y1)
                
                faces.append((x, y, w, h, confidence))
        
        return faces
    
    def draw_faces(self, frame, faces, labels=None):
        """Draw rectangles around detected faces with labels"""
        for i, face in enumerate(faces):
            x, y, w, h = face[:4]
            confidence = face[4] if len(face) > 4 else 0
            
            label = labels[i] if labels and i < len(labels) else f"Face {confidence:.2f}"
            color = (0, 255, 0) if label.startswith("Face") else (0, 255, 255)
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return frame

class FaceRecognizer:
    """
    Face recognition using face_recognition library (dlib CNN)
    Generates 128-dimensional face embeddings
    """
    def __init__(self, api_base_url):
        self.api_base_url = api_base_url
        self.model = 'hog'  # 'hog' or 'cnn'
        self.last_recognition_attempt = {}
        self.recognition_cooldown = 2
        
    def extract_embedding(self, frame):
        """Extract 128-dimensional face embedding"""
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame, model=self.model)
            
            if len(face_locations) == 0:
                return None, None
            
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            
            if len(face_encodings) > 1:
                print(f"⚠️  Multiple faces detected. Using the first one.")
            
            return face_encodings[0].tolist(), face_locations[0]
            
        except Exception as e:
            print(f"❌ Embedding extraction error: {e}")
            return None, None
    
    def can_attempt_recognition(self):
        """Check if cooldown period has passed"""
        current_time = time.time()
        if 'last_attempt' in self.last_recognition_attempt:
            elapsed = current_time - self.last_recognition_attempt['last_attempt']
            if elapsed < self.recognition_cooldown:
                return False
        return True
    
    def recognize_and_mark_attendance(self, frame, course_id, threshold=0.6):
        """Recognize face and mark attendance via API"""
        try:
            self.last_recognition_attempt['last_attempt'] = time.time()
            
            embedding, location = self.extract_embedding(frame)
            
            if embedding is None:
                return {
                    "success": False,
                    "message": "No face detected",
                    "recognized": False
                }
            
            # Encode frame as JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            image_file = io.BytesIO(buffer.tobytes())
            
            # Prepare multipart form data
            files = {'file': ('face.jpg', image_file, 'image/jpeg')}
            data = {
                'face_embedding': json.dumps(embedding),
                'course_id': str(course_id),
                'threshold': str(threshold)
            }
            
            # Send to API
            url = f"{self.api_base_url}{RECOGNIZE_ENDPOINT}"
            print(f"🔗 Sending request to: {url}")
            print(f"📊 Data: course_id={course_id}, threshold={threshold}, embedding_size={len(embedding)}")
            
            response = requests.post(url, files=files, data=data, timeout=10)
            
            print(f"📡 Response status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "success": True,
                    "data": result,
                    "recognized": result.get('recognized', False)
                }
            else:
                print(f"❌ API Error: {response.status_code}")
                print(f"Response: {response.text}")
                return {
                    "success": False,
                    "message": f"API Error: {response.status_code} - {response.text}",
                    "recognized": False
                }
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Network error: {e}")
            return {
                "success": False,
                "message": f"Network error: {str(e)}",
                "recognized": False
            }
        except Exception as e:
            print(f"❌ Recognition error: {e}")
            return {
                "success": False,
                "message": str(e),
                "recognized": False
            }

class AttendanceTracker:
    """Tracks student attendance with statistics"""
    def __init__(self, api_base_url):
        self.api_base_url = api_base_url
        self.active_students = {}
        self.update_interval = 30
        
        self.total_recognitions = 0
        self.successful_recognitions = 0
        self.failed_recognitions = 0
        self.session_start = datetime.now()
        
        self.recognition_history = []
        self.hourly_stats = {}
        
    def mark_first_seen(self, student_id, student_info, attendance_id, check_in_time):
        """Mark first time student is seen (check-in)"""
        self.active_students[student_id] = {
            'attendance_id': attendance_id,
            'first_seen': check_in_time,
            'last_seen': check_in_time,
            'last_update': time.time(),
            'student_info': student_info,
            'recognition_count': 1
        }
        
        self.total_recognitions += 1
        self.successful_recognitions += 1
        
        self.recognition_history.insert(0, {
            'time': check_in_time,
            'student': student_info['name'],
            'status': 'Check-in',
            'success': True
        })
        if len(self.recognition_history) > 10:
            self.recognition_history.pop()
        
        hour = check_in_time.hour
        self.hourly_stats[hour] = self.hourly_stats.get(hour, 0) + 1
        
        print(f"📍 First seen: {student_info['name']} at {check_in_time.strftime('%H:%M:%S')}")
    
    def update_last_seen(self, student_id):
        """Update last seen time (check-out)"""
        if student_id not in self.active_students:
            return False
        
        current_time = time.time()
        student_data = self.active_students[student_id]
        
        student_data['recognition_count'] += 1
        self.total_recognitions += 1
        
        if current_time - student_data['last_update'] < self.update_interval:
            return False
        
        try:
            attendance_id = student_data['attendance_id']
            check_out_time = datetime.now()
            
            url = f"{self.api_base_url}{ATTENDANCE_UPDATE_ENDPOINT}/{attendance_id}"
            data = {'check_out': check_out_time.isoformat()}
            
            response = requests.put(url, json=data, timeout=5)
            
            if response.status_code == 200:
                student_data['last_seen'] = check_out_time
                student_data['last_update'] = current_time
                
                self.recognition_history.insert(0, {
                    'time': check_out_time,
                    'student': student_data['student_info']['name'],
                    'status': 'Update',
                    'success': True
                })
                if len(self.recognition_history) > 10:
                    self.recognition_history.pop()
                
                print(f"🔄 Updated: {student_data['student_info']['name']}")
                return True
            else:
                return False
                
        except Exception as e:
            print(f"❌ Error updating: {e}")
            return False
    
    def increment_failed(self):
        """Increment failed recognition counter"""
        self.failed_recognitions += 1
        self.total_recognitions += 1
        
        self.recognition_history.insert(0, {
            'time': datetime.now(),
            'student': 'Unknown',
            'status': 'Failed',
            'success': False
        })
        if len(self.recognition_history) > 10:
            self.recognition_history.pop()
    
    def get_student_session_duration(self, student_id):
        """Get duration of student's session in seconds"""
        if student_id not in self.active_students:
            return None
        
        student_data = self.active_students[student_id]
        duration = student_data['last_seen'] - student_data['first_seen']
        return duration.total_seconds()
    
    def get_active_students(self):
        """Get all currently tracked students"""
        return self.active_students
    
    def get_statistics(self):
        """Get comprehensive statistics for dashboard"""
        avg_duration = 0
        if self.active_students:
            durations = [self.get_student_session_duration(sid) for sid in self.active_students]
            avg_duration = sum(durations) / len(durations) if durations else 0
        
        session_duration = (datetime.now() - self.session_start).total_seconds()
        
        return {
            'total_students': len(self.active_students),
            'total_recognitions': self.total_recognitions,
            'successful': self.successful_recognitions,
            'failed': self.failed_recognitions,
            'success_rate': (self.successful_recognitions / max(1, self.total_recognitions)) * 100,
            'avg_session_duration': avg_duration,
            'session_duration': session_duration,
            'recognition_history': self.recognition_history,
            'hourly_stats': self.hourly_stats
        }

class CameraUI:
    """UI overlay for camera feed window"""
    def draw_camera_ui(self, frame, course_info, status_message="Ready"):
        """Draw UI elements on camera frame"""
        height, width = frame.shape[:2]
        
        # Top bar
        cv2.rectangle(frame, (0, 0), (width, 80), (40, 40, 40), -1)
        cv2.putText(frame, f"Course: {course_info['course_name']}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Code: {course_info['course_code']}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Status bar
        cv2.rectangle(frame, (0, height-50), (width, height), (40, 40, 40), -1)
        cv2.putText(frame, status_message, 
                   (10, height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        current_time = datetime.now().strftime("%H:%M:%S")
        cv2.putText(frame, current_time, 
                   (width-150, height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame

class StatisticsUI:
    """Statistics dashboard with data visualization"""
    def __init__(self, width=900, height=700):
        self.width = width
        self.height = height
        self.bg_color = (30, 30, 30)
        self.text_color = (255, 255, 255)
        self.accent_color = (0, 255, 255)
        self.success_color = (0, 255, 0)
        self.warning_color = (0, 165, 255)
        self.error_color = (0, 0, 255)
        
    def create_dashboard(self, course_info, tracker):
        """Create comprehensive statistics dashboard"""
        dashboard = np.full((self.height, self.width, 3), self.bg_color, dtype=np.uint8)
        
        y_offset = 25
        
        # Header
        cv2.putText(dashboard, "ATTENDANCE STATISTICS DASHBOARD", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, self.accent_color, 2)
        y_offset += 35
        
        # Course info
        cv2.line(dashboard, (20, y_offset), (self.width-20, y_offset), (100, 100, 100), 1)
        y_offset += 25
        
        cv2.putText(dashboard, f"Course: {course_info['course_name']}", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.text_color, 1)
        y_offset += 25
        cv2.putText(dashboard, f"Code: {course_info['course_code']} | Teacher: {course_info.get('teacher_name', 'N/A')}", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_offset += 30
        
        # Statistics
        stats = tracker.get_statistics()
        
        cv2.line(dashboard, (20, y_offset), (self.width-20, y_offset), (100, 100, 100), 1)
        y_offset += 25
        
        cv2.putText(dashboard, "LIVE STATISTICS", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.accent_color, 2)
        y_offset += 35
        
        # Stat boxes row 1
        box_y = y_offset
        self._draw_stat_box(dashboard, 20, box_y, 200, 85, 
                          str(stats['total_students']), "Active Students", self.success_color)
        
        self._draw_stat_box(dashboard, 240, box_y, 200, 85, 
                          str(stats['total_recognitions']), "Total Scans", self.accent_color)
        
        self._draw_stat_box(dashboard, 460, box_y, 200, 85, 
                          f"{stats['success_rate']:.1f}%", "Success Rate", self.success_color)
        
        self._draw_stat_box(dashboard, 680, box_y, 200, 85, 
                          str(stats['failed']), "Failed Scans", self.error_color)
        
        y_offset += 110
        
        # Stat boxes row 2
        box_y = y_offset
        avg_duration = stats['avg_session_duration']
        avg_dur_str = f"{int(avg_duration // 60)}:{int(avg_duration % 60):02d}"
        
        self._draw_stat_box(dashboard, 20, box_y, 200, 85, 
                          avg_dur_str, "Avg Session", self.warning_color)
        
        session_dur = stats['session_duration']
        session_str = f"{int(session_dur // 60)}:{int(session_dur % 60):02d}"
        
        self._draw_stat_box(dashboard, 240, box_y, 200, 85, 
                          session_str, "Session Time", self.accent_color)
        
        successful = stats['successful']
        self._draw_stat_box(dashboard, 460, box_y, 200, 85, 
                          str(successful), "Successful", self.success_color)
        
        rate = (stats['total_recognitions'] / max(1, session_dur / 60))
        self._draw_stat_box(dashboard, 680, box_y, 200, 85, 
                          f"{rate:.1f}", "Scans/Min", self.warning_color)
        
        y_offset += 110
        
        # Active students table
        cv2.line(dashboard, (20, y_offset), (self.width-20, y_offset), (100, 100, 100), 1)
        y_offset += 25
        
        cv2.putText(dashboard, "ACTIVE STUDENTS", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.accent_color, 2)
        y_offset += 30
        
        active_students = tracker.get_active_students()
        
        if not active_students:
            cv2.putText(dashboard, "No active students detected yet", 
                       (40, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
            cv2.putText(dashboard, "Waiting for face recognition...", 
                       (40, y_offset + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
            y_offset += 60
        else:
            # Table header
            cv2.rectangle(dashboard, (20, y_offset-20), (self.width-20, y_offset+5), (50, 50, 50), -1)
            cv2.putText(dashboard, "Name", (30, y_offset-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.text_color, 1)
            cv2.putText(dashboard, "Roll", (260, y_offset-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.text_color, 1)
            cv2.putText(dashboard, "Check-in", (370, y_offset-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.text_color, 1)
            cv2.putText(dashboard, "Last Seen", (520, y_offset-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.text_color, 1)
            cv2.putText(dashboard, "Duration", (680, y_offset-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.text_color, 1)
            cv2.putText(dashboard, "Scans", (810, y_offset-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.text_color, 1)
            
            y_offset += 15
            
            # Student rows
            for i, (student_id, data) in enumerate(active_students.items()):
                if y_offset > self.height - 150:
                    break
                
                student = data['student_info']
                first_seen = data['first_seen']
                last_seen = data['last_seen']
                duration = (last_seen - first_seen).total_seconds()
                rec_count = data['recognition_count']
                
                if i % 2 == 0:
                    cv2.rectangle(dashboard, (20, y_offset-12), (self.width-20, y_offset+8), 
                                (40, 40, 40), -1)
                
                name = student['name'][:20]
                cv2.putText(dashboard, name, (30, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.text_color, 1)
                
                cv2.putText(dashboard, student['roll_no'][:12], (260, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.text_color, 1)
                
                cv2.putText(dashboard, first_seen.strftime("%H:%M:%S"), (370, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 200, 255), 1)
                
                cv2.putText(dashboard, last_seen.strftime("%H:%M:%S"), (520, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 100), 1)
                
                duration_str = f"{int(duration // 60)}m {int(duration % 60)}s"
                cv2.putText(dashboard, duration_str, (680, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.success_color, 1)
                
                cv2.putText(dashboard, str(rec_count), (820, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.accent_color, 1)
                
                y_offset += 28
            
            y_offset += 15
        
        # Recent activity
        cv2.line(dashboard, (20, y_offset), (self.width-20, y_offset), (100, 100, 100), 1)
        y_offset += 25
        
        cv2.putText(dashboard, "RECENT ACTIVITY (Last 10)", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.accent_color, 2)
        y_offset += 30
        
        history = stats['recognition_history']
        if not history:
            cv2.putText(dashboard, "No activity yet", 
                       (40, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        else:
            for item in history[:6]:
                time_str = item['time'].strftime("%H:%M:%S")
                status = item['status']
                student = item['student'][:25]
                success = item['success']
                
                color = self.success_color if success else self.error_color
                status_icon = "✓" if success else "✗"
                
                cv2.putText(dashboard, f"{status_icon} {time_str} | {student} - {status}", 
                           (40, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
                y_offset += 22
        
        # Footer
        footer_y = self.height - 25
        cv2.line(dashboard, (20, footer_y-10), (self.width-20, footer_y-10), (100, 100, 100), 1)
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(dashboard, f"Last Updated: {current_time}", 
                   (20, footer_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        cv2.putText(dashboard, "Press 'q' to quit", 
                   (self.width-180, footer_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        return dashboard
    
    def _draw_stat_box(self, img, x, y, width, height, value, label, color):
        """Draw a statistics box"""
        cv2.rectangle(img, (x, y), (x+width, y+height), (50, 50, 50), -1)
        cv2.rectangle(img, (x, y), (x+width, y+height), color, 2)
        
        text_size = cv2.getTextSize(value, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
        text_x = x + (width - text_size[0]) // 2
        text_y = y + height // 2 + 5
        cv2.putText(img, value, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
        
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        label_x = x + (width - label_size[0]) // 2
        label_y = y + height - 15
        cv2.putText(img, label, (label_x, label_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

def get_courses():
    """Fetch available courses from API"""
    try:
        response = requests.get(f"{API_BASE_URL}{COURSES_ENDPOINT}", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ Failed to fetch courses: {response.status_code}")
            return []
    except Exception as e:
        print(f"❌ Error fetching courses: {e}")
        return []

def select_course(courses):
    """Interactive course selection"""
    if not courses:
        print("❌ No courses available")
        return None
    
    print("\n" + "="*60)
    print("Available Courses:")
    print("="*60)
    for i, course in enumerate(courses, 1):
        print(f"{i}. {course['course_code']} - {course['course_name']}")
        print(f"   Teacher: {course.get('teacher_name', 'N/A')} | Students: {course.get('enrolled_students', 0)}")
    print("="*60)
    
    while True:
        try:
            choice = input("\n📝 Select course number (or 'q' to quit): ").strip()
            if choice.lower() == 'q':
                return None
            
            choice = int(choice)
            if 1 <= choice <= len(courses):
                return courses[choice - 1]
            else:
                print(f"❌ Please enter a number between 1 and {len(courses)}")
        except ValueError:
            print("❌ Please enter a valid number")
        except KeyboardInterrupt:
            return None

def main():


    print("🎓 Dual Window Face Recognition Attendance System")
    print("="*60)
    print("\nMODELS USED:")
    print("1. Haar Cascade (OpenCV) - Fast face detection")
    print("2. dlib CNN - 128D face embeddings for recognition")
    print("3. Euclidean distance matching with threshold 0.6")
    print("="*60)
    
    # Fetch and select course
    courses = get_courses()
    selected_course = select_course(courses)
    
    if not selected_course:
        print("👋 Exiting...")
        return
    
    course_id = selected_course['course_id']
    print(f"\n✅ Selected: {selected_course['course_name']} ({selected_course['course_code']})")
    
    # Initialize components
    camera_manager = CameraManager()
    face_detector = YOLOv8FaceDetector()
    face_recognizer = FaceRecognizer(API_BASE_URL)
    attendance_tracker = AttendanceTracker(API_BASE_URL)
    camera_ui = CameraUI()
    stats_ui = StatisticsUI(width=900, height=700)
    
    if not camera_manager.initialize_camera(0):
        print("❌ Failed to initialize camera")
        return
    
    print("\nControls:")
    print("  • Press 'q' in either window → Quit")
    print("="*60)
    
    status_message = "🤖 AUTO MODE - Scanning..."
    
    while True:
        ret, frame = camera_manager.read_frame()
        
        if not ret:
            print("❌ Failed to read frame")
            break
        
        # STEP 1: Detect faces using Haar Cascade
        faces = face_detector.detect_faces(frame)
        
        # Draw detected faces
        display_frame = face_detector.draw_faces(frame.copy(), faces)
        
        # STEP 2: Auto-recognize when face is detected
        if len(faces) > 0 and face_recognizer.can_attempt_recognition():
            status_message = "🔍 Recognizing..."
            
            # STEP 3: Extract 128D embedding and send to API
            result = face_recognizer.recognize_and_mark_attendance(frame, course_id)
            
            if result['success'] and result.get('data'):
                data = result['data']
                
                if data.get('recognized'):
                    student = data['student']
                    student_id = student['student_id']
                    
                    if data.get('attendance_marked'):
                        # STEP 4a: First time - mark check-in
                        check_in_time = datetime.fromisoformat(data['attendance']['check_in'])
                        attendance_id = data['attendance']['attendance_id']
                        
                        attendance_tracker.mark_first_seen(
                            student_id, student, attendance_id, check_in_time
                        )
                        
                        status_message = f"✅ {student['name'][:25]} - Check-in"
                        print(f"\n✅ NEW: {student['roll_no']} - {student['name']}")
                        print(f"   Time: {check_in_time.strftime('%H:%M:%S')}")
                        print(f"   Confidence: {data.get('confidence', 0):.2f}%")
                    else:
                        # STEP 4b: Already marked - update last seen
                        if attendance_tracker.update_last_seen(student_id):
                            duration = attendance_tracker.get_student_session_duration(student_id)
                            duration_str = f"{int(duration // 60)}m"
                            status_message = f"🔄 {student['name'][:25]} - Updated ({duration_str})"
                        else:
                            status_message = f"👁️  {student['name'][:25]} - Tracking"
                else:
                    # STEP 5: Recognition failed
                    attendance_tracker.increment_failed()
                    status_message = "❌ Face not recognized"
                    print("❌ No match found in database")
            else:
                if result.get('message') != 'No face detected':
                    attendance_tracker.increment_failed()
                    status_message = f"⚠️  Recognition failed"
        elif len(faces) == 0:
            status_message = "🤖 Scanning for faces..."
        
        # STEP 6: Draw UI overlays
        display_frame = camera_ui.draw_camera_ui(display_frame, selected_course, status_message)
        
        # STEP 7: Create statistics dashboard
        #stats_dashboard = stats_ui.create_dashboard(selected_course, attendance_tracker)
        
        # STEP 8: Display both windows
        cv2.imshow('Camera Feed', display_frame)
        #cv2.imshow('Statistics Dashboard', stats_dashboard)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("\n👋 Shutting down...")
            print("\n📊 FINAL SESSION SUMMARY")
            print("="*60)
            
            stats = attendance_tracker.get_statistics()
            print(f"\n📈 Overall Statistics:")
            print(f"   Total Students: {stats['total_students']}")
            print(f"   Total Recognitions: {stats['total_recognitions']}")
            print(f"   Successful: {stats['successful']}")
            print(f"   Failed: {stats['failed']}")
            print(f"   Success Rate: {stats['success_rate']:.2f}%")
            print(f"   Session Duration: {int(stats['session_duration'] // 60)} minutes")
            
            active_students = attendance_tracker.get_active_students()
            if active_students:
                print(f"\n📋 Student Sessions ({len(active_students)} students):")
                print("-" * 60)
                for student_id, data in active_students.items():
                    student = data['student_info']
                    first_seen = data['first_seen']
                    last_seen = data['last_seen']
                    duration = (last_seen - first_seen).total_seconds()
                    rec_count = data['recognition_count']
                    
                    print(f"\n  👤 {student['name']}")
                    print(f"     Roll No: {student['roll_no']}")
                    print(f"     Check-in:  {first_seen.strftime('%H:%M:%S')}")
                    print(f"     Check-out: {last_seen.strftime('%H:%M:%S')}")
                    print(f"     Duration:  {int(duration // 60)}m {int(duration % 60)}s")
                    print(f"     Recognitions: {rec_count}")
            else:
                print("\n⚠️  No students tracked in this session")
            
            print("="*60)
            break
    
    # Cleanup
    camera_manager.release()
    cv2.destroyAllWindows()
    print("\n✅ System shutdown complete!")
    print("✅ All windows closed")

if __name__ == "__main__":
 
    main()