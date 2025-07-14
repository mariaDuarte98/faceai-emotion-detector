import cv2

def draw_box_and_label(frame, label, x, y, w, h):
    """
    Draws a rectangle around the face and puts the emotion label on the frame.
    
    Args:
        frame: image frame (BGR)
        label: string to display (emotion)
        x, y: top-left coordinates of the face bounding box
        w, h: width and height of bounding box
    """
    # Draw rectangle around face
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # Put label above face rectangle
    cv2.putText(frame, label, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
