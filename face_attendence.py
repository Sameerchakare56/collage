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
FACE RECOGNITION ATTENDANCE SYSTEM - CROPPED FACE VERSION
=============================================================================

IMPROVEMENTS:
- Only detects and processes faces (no body detection)
- Crops face region before sending to API
- Better visualization with face-only rectangles
- Improved accuracy with focused face detection
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
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            print(f"✅ Camera {index} initialized (1280x720)")
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
    YOLOv8-based face detection - FACE ONLY
    """
    def __init__(self, model_path='yolov8n-face.pt'):
        try:
            self.model = YOLO(model_path)
            print(f"✅ YOLOv8 Face Model loaded: {model_path}")
        except:
            print("⚠️  Face-specific model not found. Using YOLOv8n for person detection.")
            self.model = YOLO('yolov8n.pt')
        
        self.conf_threshold = 0.5
        self.padding = 20  # Padding for face crop
        
    def detect_faces(self, frame):
        """
        Detect ONLY faces using YOLOv8
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
    
    def crop_face(self, frame, x, y, w, h):
        """
        Crop face region with padding
        Returns: cropped face image and crop coordinates
        """
        height, width = frame.shape[:2]
        
        # Add padding
        x1 = max(0, x - self.padding)
        y1 = max(0, y - self.padding)
        x2 = min(width, x + w + self.padding)
        y2 = min(height, y + h + self.padding)
        
        # Crop face
        face_crop = frame[y1:y2, x1:x2]
        
        return face_crop, (x1, y1, x2, y2)
    
    def draw_face_only(self, frame, x, y, w, h, label="Face", color=(0, 255, 0)):
        """Draw rectangle ONLY around face with label"""
        # Main rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
        # Label background
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(frame, (x, y-35), (x + label_size[0] + 10, y), color, -1)
        
        # Label text
        cv2.putText(frame, label, (x+5, y-12), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame

class FaceRecognizer:
    """
    Face recognition using face_recognition library
    Works with CROPPED FACES for better accuracy
    """
    def __init__(self, api_base_url):
        self.api_base_url = api_base_url
        self.model = 'hog'
        self.last_recognition_attempt = {}
        self.recognition_cooldown = 2
        
    def extract_embedding(self, face_crop):
        """Extract 128-dimensional face embedding from cropped face"""
        try:
            rgb_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_face, model=self.model)
            
            if len(face_locations) == 0:
                return None, None
            
            face_encodings = face_recognition.face_encodings(rgb_face, face_locations)
            
            if len(face_encodings) == 0:
                return None, None
            
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
    
    def recognize_and_mark_attendance(self, face_crop, course_id, threshold=0.6):
        """Recognize face from CROPPED IMAGE and mark attendance"""
        try:
            self.last_recognition_attempt['last_attempt'] = time.time()
            
            embedding, location = self.extract_embedding(face_crop)
            
            if embedding is None:
                return {
                    "success": False,
                    "message": "No face detected in crop",
                    "recognized": False
                }
            
            # Encode cropped face as JPEG
            _, buffer = cv2.imencode('.jpg', face_crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            image_file = io.BytesIO(buffer.tobytes())
            
            # Prepare multipart form data
            files = {'file': ('face_crop.jpg', image_file, 'image/jpeg')}
            data = {
                'face_embedding': json.dumps(embedding),
                'course_id': str(course_id),
                'threshold': str(threshold)
            }
            
            # Send to API
            url = f"{self.api_base_url}{RECOGNIZE_ENDPOINT}"
            print(f"🔗 Sending cropped face to API...")
            print(f"📊 Crop size: {face_crop.shape[1]}x{face_crop.shape[0]} | Embedding: {len(embedding)}D")
            
            response = requests.post(url, files=files, data=data, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "success": True,
                    "data": result,
                    "recognized": result.get('recognized', False)
                }
            else:
                print(f"❌ API Error: {response.status_code}")
                return {
                    "success": False,
                    "message": f"API Error: {response.status_code}",
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
        self.update_interval = 3
        
        self.total_recognitions = 0
        self.successful_recognitions = 0
        self.failed_recognitions = 0
        self.session_start = datetime.now()
        
        self.recognition_history = []
        
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
        """Get comprehensive statistics"""
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
            'recognition_history': self.recognition_history
        }

class CameraUI:
    """UI overlay for camera feed window"""
    def draw_camera_ui(self, frame, course_info, status_message="Ready", stats=None):
        """Draw UI elements on camera frame"""
        height, width = frame.shape[:2]
        
        # Top bar with transparency
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 100), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        cv2.putText(frame, f"Course: {course_info['course_name']}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"Code: {course_info['course_code']} | Teacher: {course_info.get('teacher_name', 'N/A')}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        if stats:
            cv2.putText(frame, f"Active: {stats['total_students']} | Success: {stats['successful']} | Failed: {stats['failed']}", 
                       (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 1)
        
        # Status bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, height-60), (width, height), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        status_color = (0, 255, 0) if "✅" in status_message else (0, 165, 255) if "🔍" in status_message else (255, 255, 255)
        cv2.putText(frame, status_message, 
                   (10, height-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        current_time = datetime.now().strftime("%H:%M:%S")
        cv2.putText(frame, current_time, 
                   (width-150, height-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Help text
        cv2.putText(frame, "Press 'q' to quit", 
                   (width-150, height-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        return frame

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
    print("🎓 Face Recognition Attendance System - CROPPED FACE VERSION")
    print("="*60)
    print("\nFEATURES:")
    print("✓ Face-only detection (no body)")
    print("✓ Automatic face cropping before recognition")
    print("✓ 128D face embeddings")
    print("✓ Real-time attendance tracking")
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
    
    if not camera_manager.initialize_camera(0):
        print("❌ Failed to initialize camera")
        return
    
    print("\nControls:")
    print("  • Press 'q' → Quit")
    print("  • Automatic face detection and recognition")
    print("="*60)
    
    status_message = "🤖 Scanning for faces..."
    
    while True:
        ret, frame = camera_manager.read_frame()
        
        if not ret:
            print("❌ Failed to read frame")
            break
        
        # STEP 1: Detect ONLY faces
        faces = face_detector.detect_faces(frame)
        
        # Create display frame
        display_frame = frame.copy()
        
        # STEP 2: Draw rectangles ONLY around faces
        for i, face in enumerate(faces):
            x, y, w, h = face[:4]
            confidence = face[4] if len(face) > 4 else 0
            
            label = f"Face {i+1} ({confidence:.2f})"
            color = (0, 255, 0) if len(faces) == 1 else (0, 255, 255)
            display_frame = face_detector.draw_face_only(display_frame, x, y, w, h, label, color)
        
        # STEP 3: Auto-recognize when single face is detected
        if len(faces) == 1 and face_recognizer.can_attempt_recognition():
            status_message = "🔍 Recognizing face..."
            
            x, y, w, h = faces[0][:4]
            
            # STEP 4: Crop face region
            face_crop, crop_coords = face_detector.crop_face(frame, x, y, w, h)
            
            # STEP 5: Extract embedding and recognize from CROPPED FACE
            result = face_recognizer.recognize_and_mark_attendance(face_crop, course_id)
            
            if result['success'] and result.get('data'):
                data = result['data']
                
                if data.get('recognized'):
                    student = data['student']
                    student_id = student['student_id']
                    
                    if data.get('attendance_marked'):
                        # First time - mark check-in
                        check_in_time = datetime.fromisoformat(data['attendance']['check_in'])
                        attendance_id = data['attendance']['attendance_id']
                        
                        attendance_tracker.mark_first_seen(
                            student_id, student, attendance_id, check_in_time
                        )
                        
                        status_message = f"✅ {student['name'][:30]} - Check-in"
                        print(f"\n✅ RECOGNIZED: {student['roll_no']} - {student['name']}")
                        print(f"   Confidence: {data.get('confidence', 0):.2f}%")
                    else:
                        # Already marked - update last seen
                        if attendance_tracker.update_last_seen(student_id):
                            duration = attendance_tracker.get_student_session_duration(student_id)
                            status_message = f"🔄 {student['name'][:30]} - Updated"
                        else:
                            status_message = f"👁️  {student['name'][:30]} - Tracking"
                else:
                    # Recognition failed
                    attendance_tracker.increment_failed()
                    status_message = "❌ Face not recognized"
                    print("❌ No match in database")
            else:
                if result.get('message') != 'No face detected in crop':
                    attendance_tracker.increment_failed()
                    status_message = "⚠️  Recognition failed"
        elif len(faces) > 1:
            status_message = f"⚠️  Multiple faces detected ({len(faces)}) - Please ensure only one person"
        elif len(faces) == 0:
            status_message = "🤖 Scanning for faces..."
        
        # STEP 6: Draw UI overlay
        stats = attendance_tracker.get_statistics()
        display_frame = camera_ui.draw_camera_ui(display_frame, selected_course, status_message, stats)
        
        # STEP 7: Display frame
        cv2.imshow('Face Recognition Attendance - Cropped Face Mode', display_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("\n👋 Shutting down...")
            print("\n📊 FINAL SESSION SUMMARY")
            print("="*60)
            
            stats = attendance_tracker.get_statistics()
            print(f"\n📈 Statistics:")
            print(f"   Total Students: {stats['total_students']}")
            print(f"   Total Scans: {stats['total_recognitions']}")
            print(f"   Successful: {stats['successful']}")
            print(f"   Failed: {stats['failed']}")
            print(f"   Success Rate: {stats['success_rate']:.2f}%")
            
            active_students = attendance_tracker.get_active_students()
            if active_students:
                print(f"\n📋 Tracked Students ({len(active_students)}):")
                for student_id, data in active_students.items():
                    student = data['student_info']
                    duration = attendance_tracker.get_student_session_duration(student_id)
                    print(f"  • {student['name']} ({student['roll_no']}) - {int(duration // 60)}m {int(duration % 60)}s")
            
            print("="*60)
            break
    
    # Cleanup
    camera_manager.release()
    cv2.destroyAllWindows()
    print("\n✅ System shutdown complete!")

if __name__ == "__main__":
    main()