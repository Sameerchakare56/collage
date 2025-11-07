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
                # Set higher resolution for better face quality
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                self.cameras[index] = cap
                print(f"✅ Camera {index} initialized (1280x720)")
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
        # Load Haar Cascade for face detection only
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.padding = 20  # Padding around face for better crop
        
    def detect_faces(self, frame):
        """Detect only faces in frame using Haar Cascade"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(80, 80)  # Minimum face size for better quality
        )
        return faces
    
    def crop_face(self, frame, x, y, w, h):
        """
        Crop face from frame with padding
        Returns: cropped face image
        """
        height, width = frame.shape[:2]
        
        # Add padding around face
        x1 = max(0, x - self.padding)
        y1 = max(0, y - self.padding)
        x2 = min(width, x + w + self.padding)
        y2 = min(height, y + h + self.padding)
        
        # Crop face region
        face_crop = frame[y1:y2, x1:x2]
        
        return face_crop, (x1, y1, x2, y2)
    
    def draw_face_rectangle(self, frame, x, y, w, h, label="Face", color=(0, 255, 0)):
        """Draw rectangle only around face"""
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
        # Add label above face
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(frame, (x, y-30), (x + label_size[0] + 10, y), color, -1)
        cv2.putText(frame, label, (x+5, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame

class FaceEmbedding:
    def __init__(self):
        """Initialize face embedding extractor"""
        self.model = 'hog'  # Use 'hog' for speed, 'cnn' for accuracy
        
    def extract_embedding(self, face_crop):
        """
        Extract face embedding from cropped face image
        Returns: embedding (128-dimensional vector) and face location
        """
        try:
            # Convert BGR to RGB
            rgb_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            
            # Find face locations in cropped image
            face_locations = face_recognition.face_locations(rgb_face, model=self.model)
            
            if len(face_locations) == 0:
                print("⚠️  No face detected in cropped region")
                return None, None
            
            # Extract embeddings
            face_encodings = face_recognition.face_encodings(rgb_face, face_locations)
            
            if len(face_encodings) == 0:
                print("⚠️  Failed to extract embedding")
                return None, None
            
            # Return first face's embedding
            embedding = face_encodings[0]
            location = face_locations[0]
            
            print(f"✅ Embedding extracted: {len(embedding)}-dimensional vector")
            return embedding.tolist(), location
            
        except Exception as e:
            print(f"❌ Embedding extraction error: {e}")
            return None, None
    
    def get_embedding_quality_score(self, face_crop):
        """
        Calculate quality score based on face crop characteristics
        """
        height, width = face_crop.shape[:2]
        
        # Check if face is well-sized (not too small)
        min_size = 100
        size_score = min(100, (min(width, height) / min_size) * 50)
        
        # Check sharpness using Laplacian variance
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_score = min(50, laplacian_var / 10)
        
        # Total quality score
        quality = size_score + sharpness_score
        quality = min(100, max(0, quality))
        
        return round(quality, 2)

class APIUploader:
    def __init__(self, base_url):
        self.base_url = base_url
        self.upload_count = 0
        
    def upload_student_image(self, student_id, face_crop, embedding=None, quality_score=None):
        """Upload cropped face image with embedding"""
        try:
            # Encode cropped face to JPEG
            _, buffer = cv2.imencode('.jpg', face_crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            # Create file-like object
            image_file = io.BytesIO(buffer.tobytes())
            
            # Prepare multipart form data
            files = {
                'file': ('face_crop.jpg', image_file, 'image/jpeg')
            }
            
            # Add embedding and quality score as form data
            data = {}
            if embedding is not None:
                data['face_embedding'] = json.dumps(embedding)
            if quality_score is not None:
                data['quality_score'] = str(quality_score)
            
            # Send POST request
            url = f"{self.base_url}{UPLOAD_ENDPOINT.format(student_id=student_id)}"
            
            response = requests.post(url, files=files, data=data)
            
            if response.status_code == 201:
                self.upload_count += 1
                result = response.json()
                print(f"✅ Face crop {self.upload_count} uploaded - Image ID: {result.get('image_id')}")
                print(f"   Quality Score: {quality_score}%")
                print(f"   Embedding: {result.get('has_embedding')}")
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
    print("🔧 Initializing Face Capture System...")
    print("="*50)
    print("Features:")
    print("  ✓ Face detection only (no body)")
    print("  ✓ Automatic face cropping")
    print("  ✓ 128D face embedding extraction")
    print("  ✓ Quality score calculation")
    print("  ✓ Cropped face upload to database")
    print("="*50)
    
    camera_manager = CameraManager()
    face_detector = FaceDetector()
    face_embedding_extractor = FaceEmbedding()
    api_uploader = APIUploader(API_BASE_URL)
    
    # Start with camera 0
    print("\n🎥 Starting Camera 0...")
    if not camera_manager.switch_camera(0):
        print("❌ Failed to open camera 0")
        return
    
    # Get initial student ID
    current_student_id = get_student_id()
    if current_student_id is None:
        return
    
    print(f"\n✅ Capturing face images for Student ID: {current_student_id}")
    print("="*50)
    print("Controls:")
    print("  • Press 's' → Capture & upload CROPPED FACE with embedding")
    print("  • Press 'p' → Preview cropped face before upload")
    print("  • Press 'f' → Finish current student")
    print("  • Press '0-9' → Switch cameras")
    print("  • Press 'q' → Quit")
    print("="*50)
    
    quality_score = 0
    preview_crop = None
    show_preview = False
    
    while True:
        ret, frame = camera_manager.read_frame()
        
        if not ret:
            print("❌ Failed to read frame")
            break
        
        # Detect faces only
        faces = face_detector.detect_faces(frame)
        
        # Create display frame
        display_frame = frame.copy()
        
        # Draw rectangles only around faces
        for i, (x, y, w, h) in enumerate(faces):
            label = f"Face {i+1}"
            color = (0, 255, 0) if len(faces) == 1 else (0, 255, 255)
            display_frame = face_detector.draw_face_rectangle(
                display_frame, x, y, w, h, label, color
            )
            
            # Show crop area with padding
            face_crop, (x1, y1, x2, y2) = face_detector.crop_face(frame, x, y, w, h)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 255), 1)
        
        # Display info overlay
        info_y = 30
        cv2.rectangle(display_frame, (0, 0), (400, 200), (0, 0, 0), -1)
        
        cv2.putText(display_frame, f'Student ID: {current_student_id}', 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        info_y += 35
        
        cv2.putText(display_frame, f'Camera: {camera_manager.current_index}', 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        info_y += 30
        
        face_status = f'Faces Detected: {len(faces)}'
        face_color = (0, 255, 0) if len(faces) == 1 else (0, 165, 255) if len(faces) > 1 else (0, 0, 255)
        cv2.putText(display_frame, face_status, 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, face_color, 2)
        info_y += 30
        
        cv2.putText(display_frame, f'Uploaded: {api_uploader.upload_count}', 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        info_y += 30
        
        if quality_score > 0:
            quality_color = (0, 255, 0) if quality_score > 70 else (0, 255, 255) if quality_score > 40 else (0, 0, 255)
            cv2.putText(display_frame, f'Quality: {quality_score:.1f}%', 
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, quality_color, 2)
        
        # Display main frame
        cv2.imshow('Face Capture - Cropped Upload', display_frame)
        
        # Show preview window if requested
        if show_preview and preview_crop is not None:
            cv2.imshow('Face Crop Preview', preview_crop)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        # Quit
        if key == ord('q'):
            print("\n👋 Exiting...")
            break
        
        # Switch cameras (0-9)
        elif key >= ord('0') and key <= ord('9'):
            cam_index = key - ord('0')
            camera_manager.switch_camera(cam_index)
        
        # Preview cropped face
        elif key == ord('p'):
            if len(faces) > 0:
                x, y, w, h = faces[0]
                preview_crop, _ = face_detector.crop_face(frame, x, y, w, h)
                show_preview = True
                print("\n👁️  Preview window opened")
            else:
                print("\n❌ No face detected for preview")
        
        # Capture and upload cropped face
        elif key == ord('s'):
            if len(faces) > 0:
                if len(faces) > 1:
                    print(f"\n⚠️  Multiple faces detected ({len(faces)}). Using the first face.")
                
                print(f"\n📸 Capturing CROPPED FACE for Student ID: {current_student_id}...")
                
                # Get first face
                x, y, w, h = faces[0]
                
                # Crop face region
                face_crop, crop_coords = face_detector.crop_face(frame, x, y, w, h)
                
                print(f"✂️  Face cropped: {face_crop.shape[1]}x{face_crop.shape[0]} pixels")
                print("🔍 Extracting face embedding from cropped region...")
                
                # Extract embedding from cropped face
                embedding, location = face_embedding_extractor.extract_embedding(face_crop)
                
                # Calculate quality score
                quality_score = face_embedding_extractor.get_embedding_quality_score(face_crop)
                print(f"📊 Face quality score: {quality_score}%")
                
                if quality_score < 30:
                    print("⚠️  Low quality face image. Consider better lighting or positioning.")
                
                # Upload cropped face with embedding
                if embedding is not None:
                    api_uploader.upload_student_image(
                        current_student_id, 
                        face_crop, 
                        embedding, 
                        quality_score
                    )
                else:
                    print("⚠️  Uploading cropped face without embedding")
                    api_uploader.upload_student_image(
                        current_student_id, 
                        face_crop, 
                        None, 
                        quality_score
                    )
            else:
                print("\n❌ No face detected. Please ensure face is clearly visible.")
        
        # Finish with current student
        elif key == ord('f'):
            print(f"\n✅ Finished with Student ID: {current_student_id}")
            print(f"   Total face crops uploaded: {api_uploader.upload_count}")
            
            # Get new student ID
            api_uploader.reset_count()
            quality_score = 0
            show_preview = False
            current_student_id = get_student_id()
            
            if current_student_id is None:
                break
            
            print(f"\n✅ Now capturing faces for Student ID: {current_student_id}")
            print("="*50)
    
    # Cleanup
    camera_manager.release_all()
    print("\n✅ All cameras released")
    print(f"✅ Face capture session complete!")

if __name__ == "__main__":
    main()