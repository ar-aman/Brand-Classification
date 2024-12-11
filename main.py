import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox
from paddleocr import PaddleOCR
import numpy as np
import pandas as pd

cap = cv2.VideoCapture(0)

ocr = PaddleOCR(lang='en')

try:
    while True:
        try:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Failed to capture frame. Retrying...")
                continue

            try:
                bbox, labels, confidences = cv.detect_common_objects(frame)
            except Exception as e:
                print(f"Error during object detection: {e}")
                continue

            output_image = draw_bbox(frame, bbox, labels, confidences)

            cropped_objects = []
            for box in bbox:
                x1, y1, x2, y2 = box
                try:
                    cropped_object = frame[y1:y2, x1:x2]
                    if cropped_object.size != 0:
                        cropped_objects.append(cropped_object)
                    else:
                        print("Warning: Cropped object is empty.")
                except Exception as e:
                    print(f"Error while cropping object: {e}")

            for idx, obj in enumerate(cropped_objects):
                try:
                    result = ocr.ocr(obj, cls=True)

                    if result and result[0]:
                        detected_text = [line[1][0] for line in result[0]]
                        print(f"Detected text in object {idx + 1}: {detected_text}")
                    else:
                        print(f"No text detected in object {idx + 1}")

                except Exception as e:
                    print(f"Error during OCR processing: {e}")

            cv2.imshow("Object Detection", output_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Terminating application...")
                break

        except KeyboardInterrupt:
            print("Interrupted by user. Exiting...")
            break

except Exception as e:
    print(f"Unexpected error: {e}")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Resources released. Windows closed.")
