from deepface import DeepFace
from fer import FER

def get_emotion_detector(detector_type="FER"):
    """
    Returns the emotion detector object or module based on detector_type.
    
    Args:
        detector_type (str): 'deepface' or 'FER'
        
    Returns:
        Emotion detector (DeepFace module or FER instance)
    """
    if detector_type == "deepface":
        # DeepFace is a module, used statically
        return DeepFace
    else:
        # FER is instantiated with mtcnn option enabled
        return FER(mtcnn=True)

def detect_emotion(emotion_detector, face_img, detector_type="FER"):
    """
    Predicts the dominant emotion from a face image.
    
    Args:
        emotion_detector: emotion detector object/module
        face_img: cropped face image (RGB or BGR depending on detector)
        detector_type (str): which detector is used
        
    Returns:
        str: dominant emotion label or None if detection failed
    """
    if detector_type == "deepface":
        # DeepFace returns a list of analysis results
        analysis = emotion_detector.analyze(face_img, actions=['emotion'], enforce_detection=False)
        return analysis[0]['dominant_emotion']
    else:
        # FER returns a list of detected emotions with scores
        emotions = emotion_detector.detect_emotions(face_img)
        if emotions:
            # Return the emotion with the highest score for first face
            return max(emotions[0]["emotions"], key=emotions[0]["emotions"].get)
        else:
            return None
