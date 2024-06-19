import streamlit as st
import cv2
import easyocr
import numpy as np
import pandas as pd
import os
from io import BytesIO
import base64
 
# Define a function to preprocess and extract glucose value from an image
def preprocess_and_extract(image_path):
    reader = easyocr.Reader(['en'])
 
    image = cv2.imread(image_path)
    if image is None:
        st.error(f"Unable to read image from path: {image_path}")
        return []
 
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
 
    kernel_sharpening = np.array([[-1, -1, -1],
                                  [-1,  9, -1],
                                  [-1, -1, -1]])
 
    sharpened = cv2.filter2D(gray, -1, kernel_sharpening)
 
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(sharpened)
 
    _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
 
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
 
    glucose_values = []
 
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
                        glucose_values.append({'Image': image_path, 'Glucose Value': f"{value} mg/dL"})
                        break
                except ValueError:
                    continue
 
    return glucose_values
 
def main():
    st.title('Glucometer Glucose Value Extractor')
    st.write('Upload an image of a glucometer to extract the glucose value.')
 
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
 
    if uploaded_file is not None:
        temp_image_path = "temp_image.jpg"
        with open(temp_image_path, "wb") as f:
            f.write(uploaded_file.read())
 
        glucose_values = preprocess_and_extract(temp_image_path)
 
        if glucose_values:
            st.write("### Detected Glucose Values:")
            for glucose_value in glucose_values:
                st.write(f"- {glucose_value['Image']} {glucose_value['Glucose Value']}")
        else:
            st.error("Unable to detect glucose value. Please try again with a clearer image or a different angle.")
 
        try:
            os.remove(temp_image_path)
        except Exception as e:
            st.warning(f"Failed to delete temporary file {temp_image_path}: {e}")
 
        # Save detected glucose values to CSV file
        save_glucose_data(glucose_values)
 
        # Display the button to download all glucose values as CSV
        if st.button("Download All Glucose Values as CSV"):
            download_csv()
 
def save_glucose_data(glucose_values):
    file_path = "all_glucose_values.csv"
    if glucose_values:
        new_df = pd.DataFrame(glucose_values)
        if os.path.exists(file_path):
            all_glucose_df = pd.read_csv(file_path)
            all_glucose_df = pd.concat([all_glucose_df, new_df], ignore_index=True)
        else:
            all_glucose_df = new_df.copy()
 
        # Remove duplicates
        all_glucose_df.drop_duplicates(inplace=True)
 
        # Save to CSV
        all_glucose_df.to_csv(file_path, index=False)
 
def download_csv():
    file_path = "all_glucose_values.csv"
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            csv_data = f.read()
        b64 = base64.b64encode(csv_data).decode('utf-8')
        href = f'<a href="data:file/csv;base64,{b64}" download="all_glucose_values.csv">Download All Glucose Values CSV File</a>'
        st.markdown(href, unsafe_allow_html=True)
    else:
        st.warning("No glucose values detected yet.")
 
if __name__ == "__main__":
    main()
 
