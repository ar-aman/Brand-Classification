import streamlit as st
from paddleocr import PaddleOCR
import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox
import tempfile
import os
from PIL import Image

# Initialize PaddleOCR
ocr = PaddleOCR(lang='en')

# Streamlit app title
st.title("Real-Time Object Detection and OCR App")

# Placeholder for video feed
frame_placeholder = st.empty()
ocr_result_placeholder = st.empty()

# State management for the camera
if "run_camera" not in st.session_state:
    st.session_state.run_camera = False

# Start and stop buttons
col1, col2 = st.columns(2)
with col1:
    if st.button("Start Camera"):
        st.session_state.run_camera = True

with col2:
    if st.button("Stop Camera"):
        st.session_state.run_camera = False

def draw_bbox_without_labels(frame, bbox):
    for box in bbox:
        x1, y1, x2, y2 = box
        # Draw the bounding box (rectangle) with a specific color (e.g., green) and thickness
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return frame

# Open webcam and process frames
if st.session_state.run_camera:
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Error: Could not open webcam.")
        st.session_state.run_camera = False
    else:
        try:
            while st.session_state.run_camera:
                ret, frame = cap.read()
                if not ret or frame is None:
                    st.error("Failed to capture frame.")
                    break

                # Perform object detection
                bbox, labels, confidences = cv.detect_common_objects(frame)
                output_image = draw_bbox_without_labels(frame, bbox)

                # Convert BGR to RGB for displaying in Streamlit
                output_image_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)

                # Display the frame with detected objects
                frame_placeholder.image(output_image_rgb, caption="Real-Time Object Detection", use_container_width=True)

                # Process each detected object for OCR
                cropped_objects = []
                for box in bbox:
                    x1, y1, x2, y2 = box
                    cropped_object = frame[y1:y2, x1:x2]
                    if cropped_object.size != 0:
                        cropped_objects.append(cropped_object)

                # Display OCR results
                ocr_results = []
                for idx, obj in enumerate(cropped_objects):
                    result = ocr.ocr(obj, cls=True)
                    if result and result[0]:
                        detected_text = [line[1][0] for line in result[0]]
                        ocr_results.append(f"Detected Text: {', '.join(detected_text)}")
                    else:
                        ocr_results.append(f"No text detected")

                ocr_result_placeholder.write("\n".join(ocr_results))

                # Stop the loop when 'Stop Camera' button is pressed
                if not st.session_state.run_camera:
                    break

        except Exception as e:
            st.error(f"Error during processing: {e}")
