import streamlit as st
import cv2
import easyocr
import numpy as np
import os

def preprocess_and_extract(image_path):
    reader = easyocr.Reader(['en'])

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    # Define sharpening kernel
    kernel_sharpening = np.array([[-1, -1, -1],
                                  [-1,  9, -1],
                                  [-1, -1, -1]])

    # Apply sharpening kernel
    sharpened = cv2.filter2D(gray, -1, kernel_sharpening)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(sharpened)

    _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    glucose_value = None
    bbox = None

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        aspect_ratio = w / float(h)
        area = cv2.contourArea(contour)

        if 0.5 < aspect_ratio < 2.0 and area > 1000:
            roi = image_rgb[y:y+h, x:x+w]

            results = reader.readtext(roi)

            for (box, text, prob) in results:
                numeric_text = ''.join(c for c in text if c.isdigit() or c == '.')
                try:
                    value = float(numeric_text)
                    if 20 <= value <= 600:
                        glucose_value = f"{value} mg/dL"
                        bbox = box
                        bbox = [(x + bx, y + by) for (bx, by) in bbox]
                        break
                except ValueError:
                    continue

            if glucose_value:
                break

    return glucose_value, bbox

def main():
    st.title('Glucometer Glucose Value Extractor')
    st.write('Upload an image of a glucometer to extract the glucose value.')

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        temp_image_path = "temp_image.jpg"
        with open(temp_image_path, "wb") as f:
            f.write(uploaded_file.read())

        glucose_value, bbox = preprocess_and_extract(temp_image_path)

        if glucose_value:
            image = cv2.imread(temp_image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if bbox is not None:
                bbox = np.array(bbox).astype(np.int32)
                cv2.rectangle(image_rgb, tuple(bbox[0]), tuple(bbox[2]), (0, 255, 0), 2)

            st.write(f"Detected Glucose Value: **{glucose_value}**")
            st.image(image_rgb, caption=f"Detected Glucose Value: {glucose_value}")
        else:
            st.error("Unable to detect glucose value. Please try again with a clearer image or a different angle.")

        os.remove(temp_image_path)