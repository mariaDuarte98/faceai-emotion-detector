import cv2
from collections import deque
import logging
from multiprocessing import Pool, cpu_count

from detectors.face_detector import get_face_detector, detect_faces
from detectors.emotion_detector import get_emotion_detector, detect_emotion
from processing.utils import draw_box_and_label

logger = logging.getLogger(__name__)

EMOTION_HISTORY_LENGTH = 10

def process_video_chunk(args):
    """
    Processes a chunk of video frames to detect faces and emotions.
    
    Args:
        args: tuple containing (video_path, start_frame, end_frame, detector_type)
        
    Returns:
        list of processed frames for the chunk
    """
    video_path, start_frame, end_frame, detector_type = args
    
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    emotion_detector = get_emotion_detector(detector_type)
    face_detector = get_face_detector(detector_type)
    
    frames = []
    emotion_history = deque(maxlen=EMOTION_HISTORY_LENGTH)
    current_frame = start_frame
    
    while current_frame < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        
        faces = detect_faces(face_detector, frame, detector_type)
        
        for face_el in faces:
            # Extract face bounding box and face image based on detector type
            if detector_type == "deepface":
                x, y, w, h = face_el
                face_img = frame[y:y+h, x:x+w]
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            else:
                x, y, w, h = face_el['box']
                x, y = max(0, x), max(0, y)
                face_img = frame[y:y+h, x:x+w]
            
            try:
                emotion = detect_emotion(emotion_detector, face_img, detector_type)
                if emotion:
                    emotion_history.append(emotion)
                
                # Get most common emotion in history
                if emotion_history:
                    common_emotion = max(set(emotion_history), key=emotion_history.count)
                    draw_box_and_label(frame, common_emotion, x, y, w, h)
            
            except Exception as e:
                logger.error(f"Emotion detection failed: {e}")
                continue
        
        frames.append(frame)
        current_frame += 1
    
    cap.release()
    return frames

def process_video(video_path, skip_frames=True, detector_type="FER"):
    """
    Main function to process the entire video, splitting into chunks to parallelize.
    
    Args:
        video_path (str): path to input video file
        skip_frames (bool): whether to skip frames for speed
        detector_type (str): which detector to use ("FER" or "deepface")
        
    Returns:
        path to processed video file
    """
    if not video_path:
        raise ValueError("No video file provided.")
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 20  # fallback to 20 if fps=0
    cap.release()
    
    # Adjust frame count if skipping frames
    frames_to_process = total_frames // 2 if skip_frames else total_frames
    
    # Define chunk size based on CPU cores for multiprocessing
    cpu_cores = cpu_count()
    chunk_size = frames_to_process // cpu_cores
    
    chunks = []
    start = 0
    for i in range(cpu_cores):
        end = start + chunk_size
        if i == cpu_cores - 1:
            end = frames_to_process  # last chunk processes remaining frames
        
        chunks.append((video_path, start, end, detector_type))
        start = end
    
    logger.info(f"Processing video in {cpu_cores} chunks...")
    
    with Pool(processes=cpu_cores) as pool:
        processed_chunks = pool.map(process_video_chunk, chunks)
    
    # Flatten list of frames
    all_frames = [frame for chunk in processed_chunks for frame in chunk]
    
    # Write the processed frames to output video
    output_path = "processed_video.mp4"
    height, width, _ = all_frames[0].shape
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    for frame in all_frames:
        out.write(frame)
    out.release()
    
    logger.info(f"Video processed and saved to {output_path}")
    return output_path
