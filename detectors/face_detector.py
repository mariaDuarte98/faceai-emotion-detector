import cv2
from mtcnn import MTCNN

def get_face_detector(detector_type="FER"):
    """
    Returns the appropriate face detector based on the detector_type.
    
    Args:
        detector_type (str): 'deepface' or 'FER' to select the detector type.
        
    Returns:
        face detector object (cv2.CascadeClassifier or MTCNN instance)
    """
    if detector_type == "deepface":
        # Haar Cascade detector from OpenCV (classic face detector)
        return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    else:
        # MTCNN detector (deep learning based)
        return MTCNN()

def detect_faces(detector, frame, detector_type="FER"):
    """
    Detects faces in a frame using the given detector.
    
    Args:
        detector: face detector object.
        frame: image frame (BGR).
        detector_type (str): detector type to decide detection method.
        
    Returns:
        list of detected face bounding boxes or face info.
    """
    if detector_type == "deepface":
        # Convert to grayscale for Haar cascade
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    else:
        # MTCNN returns list of dicts with 'box' key
        return detector.detect_faces(frame)
