import gradio as gr
from processing.video_processor import process_video
from processing.webcam_processor import process_webcam_frame
import cv2

def webcam_emotion_detection(frame):
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    processed_frame = process_webcam_frame(frame_bgr)
    processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
    return processed_frame_rgb

print(gr.__file__)
print(gr.__version__)

with gr.Blocks() as demo:
    gr.Markdown("# Facial Emotion Recognition")

    with gr.Tab("Video Upload"):
        video_input = gr.Video(label="Upload Video")
        skip_checkbox = gr.Checkbox(label="Skip every other frame for speed?", value=True)
        detector_radio = gr.Radio(choices=["FER", "deepface"], label="Emotion Detector", value="FER")
        processed_video = gr.Video(label="Processed Video")

        video_button = gr.Button("Process Video")
        video_button.click(
            fn=process_video,
            inputs=[video_input, skip_checkbox, detector_radio],
            outputs=processed_video
        )

    with gr.Tab("Webcam"):
        # Troquei só essa linha pra gr.Image com webcam, pra realtime funcionar
        webcam_input = gr.Image(sources=["webcam"], label="Webcam Feed")
        output_image = gr.Image(label="Webcam Emotion Detection", streaming=True)

        webcam_input.stream(
            fn=webcam_emotion_detection,
            inputs=webcam_input,
            outputs=output_image
        )

