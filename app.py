# app.py

import streamlit as st
import os
import uuid
from detect import detect_objects
from PIL import Image

# Setup
st.title("👀 YOLOv8 Object Detection App")
st.write("Upload an image to detect objects using YOLOv8.")

UPLOAD_DIR = "test_images"
OUTPUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save uploaded image
    file_ext = uploaded_file.name.split('.')[-1]
    file_id = str(uuid.uuid4())
    input_path = os.path.join(UPLOAD_DIR, f"{file_id}.{file_ext}")
    output_path = os.path.join(OUTPUT_DIR, f"detected_{file_id}.{file_ext}")

    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image(input_path, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Running YOLOv8 detection..."):
        detections = detect_objects(input_path, target_class="person", save_path=output_path)

    st.success("Detection complete!")
    
    if detections:
        st.image(output_path, caption="Detected Image", use_column_width=True)
        st.json(detections)
    else:
        st.warning("No 'person' detected in the image.")
