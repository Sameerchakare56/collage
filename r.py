import cv2
import requests
import numpy as np
from datetime import datetime
import json
import io
import face_recognition

# Configuration
API_BASE_URL = "http://localhost:8005"
UPLOAD_ENDPOINT = "/student-images/{student_id}"

class CameraManager:
    def __init__(self):
        self.cameras = {}
        self.active_camera = None
        self.current_index = 0
        
    def initialize_camera(self, index):
        """Initialize a camera by index"""
        if index not in self.cameras:
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                self.cameras[index] = cap
                print(f"✅ Camera {index} initialized")
                return True
            else:
                print(f"❌ Camera {index} not available")
                return False
        return True
    
    def switch_camera(self, index):
        """Switch to a different camera"""
        if self.initialize_camera(index):
            self.active_camera = self.cameras[index]
            self.current_index = index
            print(f"🎥 Switched to Camera {index}")
            return True
        return False
    
    def read_frame(self):
        """Read frame from active camera"""
        if self.active_camera:
            return self.active_camera.read()
        return False, None
    
    def release_all(self):
        """Release all cameras"""
        for cap in self.cameras.values():
            cap.release()
        cv2.destroyAllWindows()

class FaceDetector:
    def __init__(self):
        # Load Haar Cascade for fast detection (for display)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
    def detect_faces(self, frame):
        """Detect faces in frame using Haar Cascade"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        return faces
    
    def draw_faces(self, frame, faces):
        """Draw rectangles around detected faces"""
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, 'Face', (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        return frame

class FaceEmbedding:
    def __init__(self):
        """Initialize face embedding extractor"""
        self.model = 'large'  # Use 'large' model for better accuracy
        
    def extract_embedding(self, frame):
        """
        Extract face embedding using face_recognition library (dlib)
        Returns: list of embeddings (128-dimensional vectors) and face locations
        """
        try:
            # Convert BGR to RGB (face_recognition uses RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Find all face locations and encodings
            face_locations = face_recognition.face_locations(rgb_frame, model='hog')
            
            if len(face_locations) == 0:
                print("⚠️  No faces detected for embedding extraction")
                return None, None
            
            # Extract embeddings (128-dimensional vectors)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            
            if len(face_encodings) > 1:
                print(f"⚠️  Multiple faces detected ({len(face_encodings)}). Using the first one.")
            
            # Return first face's embedding and location
            embedding = face_encodings[0]
            location = face_locations[0]
            
            print(f"✅ Embedding extracted: {len(embedding)}-dimensional vector")
            return embedding.tolist(), location
            
        except Exception as e:
            print(f"❌ Embedding extraction error: {e}")
            return None, None
    
    def get_embedding_quality_score(self, frame, location):
        """
        Calculate quality score based on face size and position
        """
        top, right, bottom, left = location
        face_width = right - left
        face_height = bottom - top
        frame_height, frame_width = frame.shape[:2]
        
        # Face size ratio
        size_ratio = (face_width * face_height) / (frame_width * frame_height)
        
        # Centering score
        face_center_x = (left + right) / 2
        face_center_y = (top + bottom) / 2
        frame_center_x = frame_width / 2
        frame_center_y = frame_height / 2
        
        center_distance = np.sqrt(
            ((face_center_x - frame_center_x) / frame_width) ** 2 +
            ((face_center_y - frame_center_y) / frame_height) ** 2
        )
        
        # Quality score (0-100)
        quality = (size_ratio * 100 * 2 + (1 - center_distance) * 100) / 2
        quality = min(100, max(0, quality))
        
        return round(quality, 2)

class APIUploader:
    def __init__(self, base_url):
        self.base_url = base_url
        self.upload_count = 0
        
    def upload_student_image(self, student_id, frame, embedding=None):
        """Upload student image with face embedding using multipart form data"""
        try:
            # Encode frame to JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            
            # Create file-like object
            image_file = io.BytesIO(buffer.tobytes())
            
            # Prepare multipart form data
            files = {
                'file': ('face_image.jpg', image_file, 'image/jpeg')
            }
            
            # Add embedding as form data (if available)
            data = {}
            if embedding is not None:
                # Convert embedding to JSON string
                data['face_embedding'] = json.dumps(embedding)
            
            # Send POST request
            url = f"{self.base_url}{UPLOAD_ENDPOINT.format(student_id=student_id)}"
            
            if data:
                response = requests.post(url, files=files, data=data)
            else:
                response = requests.post(url, files=files)
            
            if response.status_code == 201:
                self.upload_count += 1
                result = response.json()
                print(f"✅ Image {self.upload_count} uploaded - Image ID: {result.get('image_id')}")
                return True
            else:
                print(f"❌ Upload failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Upload error: {e}")
            return False
    
    def reset_count(self):
        """Reset upload count for new student"""
        self.upload_count = 0

def get_student_id():
    """Prompt for student ID"""
    while True:
        try:
            print("\n" + "="*50)
            student_id = input("📝 Enter Student ID (number): ").strip()
            if student_id.isdigit():
                return int(student_id)
            else:
                print("❌ Please enter a valid number")
        except KeyboardInterrupt:
            print("\n👋 Exiting...")
            return None

def main():
    global camera_manager
    
    # Initialize components
    print("🔧 Initializing components...")
    camera_manager = CameraManager()
    face_detector = FaceDetector()
    face_embedding_extractor = FaceEmbedding()
    api_uploader = APIUploader(API_BASE_URL)
    
    # Start with camera 0
    print("🎥 Starting Cam1 (0)")
    if not camera_manager.switch_camera(0):
        print("❌ Failed to open camera 0")
        return
    
    # Get initial student ID
    current_student_id = get_student_id()
    if current_student_id is None:
        return
    
    print(f"\n✅ Capturing images for Student ID: {current_student_id}")
    print("="*50)
    print("Controls:")
    print("  • Press 's' → Capture & upload image with embedding")
    print("  • Press 'f' → Finish current student (switch to new student)")
    print("  • Press '0-9' → Switch cameras")
    print("  • Press 'q' → Quit")
    print("="*50)
    
    quality_score = 0
    
    while True:
        ret, frame = camera_manager.read_frame()
        
        if not ret:
            print("❌ Failed to read frame")
            break
        
        # Detect faces (for display)
        faces = face_detector.detect_faces(frame)
        
        # Draw faces on frame
        display_frame = face_detector.draw_faces(frame.copy(), faces)
        
        # Display info
        cv2.putText(display_frame, f'Student ID: {current_student_id}', 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(display_frame, f'Camera: {camera_manager.current_index}', 
                   (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(display_frame, f'Faces: {len(faces)}', 
                   (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(display_frame, f'Uploaded: {api_uploader.upload_count}', 
                   (10, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        if quality_score > 0:
            color = (0, 255, 0) if quality_score > 70 else (0, 255, 255) if quality_score > 40 else (0, 0, 255)
            cv2.putText(display_frame, f'Quality: {quality_score}%', 
                       (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Display frame
        cv2.imshow('Face Detection - Multi Camera', display_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        # Quit
        if key == ord('q'):
            print("👋 Exiting...")
            break
        
        # Switch cameras (0-9)
        elif key >= ord('0') and key <= ord('9'):
            cam_index = key - ord('0')
            camera_manager.switch_camera(cam_index)
        
        # Capture and upload with embedding
        elif key == ord('s'):
            if len(faces) > 0:
                print(f"\n📸 Capturing image for Student ID: {current_student_id}...")
                print("🔍 Extracting face embedding...")
                
                # Extract embedding
                embedding, location = face_embedding_extractor.extract_embedding(frame)
                
                if embedding is not None:
                    # Calculate quality score
                    quality_score = face_embedding_extractor.get_embedding_quality_score(frame, location)
                    print(f"📊 Face quality score: {quality_score}%")
                    
                    # Upload with embedding
                    api_uploader.upload_student_image(current_student_id, frame, embedding)
                else:
                    print("⚠️  Uploading without embedding (face detection failed)")
                    api_uploader.upload_student_image(current_student_id, frame, None)
            else:
                print("\n❌ No faces detected. Please ensure face is visible.")
        
        # Finish with current student
        elif key == ord('f'):
            print(f"\n✅ Finished with Student ID: {current_student_id}")
            print(f"   Total images uploaded: {api_uploader.upload_count}")
            
            # Get new student ID
            api_uploader.reset_count()
            quality_score = 0
            current_student_id = get_student_id()
            
            if current_student_id is None:
                break
            
            print(f"\n✅ Now capturing images for Student ID: {current_student_id}")
            print("="*50)
    
    # Cleanup
    camera_manager.release_all()
    print("✅ All cameras released")
    print(f"✅ Session complete!")

if __name__ == "__main__":
    main()