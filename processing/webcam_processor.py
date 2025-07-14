import cv2
from fer import FER
from collections import deque

# Initialize emotion detector
emotion_detector = FER(mtcnn=True)
emotion_history = deque(maxlen=10)

def process_webcam_frame(frame):
    """
    Process a single frame from webcam, detect emotions and draw boxes.

    Args:
        frame (numpy.ndarray): BGR image frame from webcam.

    Returns:
        frame with bounding box and emotion label drawn.
    """
    emotions = emotion_detector.detect_emotions(frame)
    if emotions:
        emotion_scores = emotions[0]["emotions"]
        dominant_emotion = max(emotion_scores, key=emotion_scores.get)
        emotion_history.append(dominant_emotion)

        common_emotion = max(set(emotion_history), key=emotion_history.count)
        x, y, w, h = emotions[0]["box"]

        # Draw rectangle and emotion label
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, common_emotion, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return frame
