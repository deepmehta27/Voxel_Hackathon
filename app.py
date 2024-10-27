
import streamlit as st
import cv2
import torch
import numpy as np
import os

# Load YOLOv5 model (pre-trained on COCO dataset for detecting people)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Hardcoded coordinates for the virtual fence (top-left, top-right, bottom-right, bottom-left)
virtual_fence = np.array([(415, 75), (610, 100), (510, 310), (170, 180)], np.int32)

# Function to check if an object crosses the virtual fence (polygon)
def is_trespassing(bbox, fence_polygon):
    x1, y1, x2, y2 = bbox
    
    # Create a bounding box polygon for the detected object
    bbox_polygon = np.array([(x1, y1), (x2, y1), (x2, y2), (x1, y2)], np.int32)
    
    # Check if any of the corners of the bounding box are inside the virtual fence
    inside = False
    for point in bbox_polygon:
        point_int = (int(point[0]), int(point[1]))  # Ensure point is an integer tuple
        if cv2.pointPolygonTest(fence_polygon, point_int, False) >= 0:
            inside = True
            break

    return inside

# Function to process video and apply object detection + virtual fence
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    output_frames = []  # List to store processed frames
    trespass_alerts = 0  # Counter for trespassing alerts

    # Check if video opened successfully
    if not cap.isOpened():
        st.error("Error: Cannot open video")
        return None, 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run object detection
        results = model(frame)

        # Draw the virtual fence (polygon)
        cv2.polylines(frame, [virtual_fence], isClosed=True, color=(0, 255, 0), thickness=2)

        # Process results and draw bounding boxes
        for *xyxy, conf, cls in results.xyxy[0]:  # Bounding box coordinates
            x1, y1, x2, y2 = map(int, xyxy)
            label = f'{model.names[int(cls)]} {conf:.2f}'

            # Draw bounding box for detected object
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Check if object crosses the virtual fence (only detect people - class '0')
            if int(cls) == 0 and is_trespassing([x1, y1, x2, y2], virtual_fence):
                # Trigger alert: Draw red bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(frame, 'ALERT: Trespassing!', (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                trespass_alerts += 1

        # Append the processed frame to the output list
        output_frames.append(frame)

    cap.release()
    return output_frames, trespass_alerts

# Function to save processed frames to video file
def save_to_video(output_frames, output_video_path, fps=30):
    height, width, _ = output_frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for frame in output_frames:
        out.write(frame)

    out.release()

# Streamlit UI
st.title("Virtual Fence Detection using YOLOv5")

# Upload video file
uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

# If a video is uploaded
if uploaded_video is not None:
    st.video(uploaded_video)
    
    # Save uploaded video to temporary location
    temp_video_path = "temp_video.mp4"
    with open(temp_video_path, "wb") as f:
        f.write(uploaded_video.getbuffer())

    # Process the video and display the output
    output_frames, trespass_alerts = process_video(temp_video_path)

    # Save the processed frames to a video file
    output_video_path = "processed_video.mp4"
    save_to_video(output_frames, output_video_path)

    # Display the processed video in Streamlit
    st.write("Processed video with virtual fence detection:")
    st.video(output_video_path)

    # Provide a download link for the processed video
    with open(output_video_path, "rb") as f:
        st.download_button("Download Processed Video", f, file_name="processed_video.mp4")

    # Show the number of alerts
    if trespass_alerts > 0:
        st.error(f"ALERT: {trespass_alerts} trespassing events detected!")
    else:
        st.success("No trespassing detected.")
else:
    st.warning("Please upload a video for processing.")