import streamlit as st
import cv2
import easyocr
import numpy as np
import pandas as pd
import os
from io import BytesIO
import base64

# Initialize a DataFrame to store all data across sessions
all_glucose_df = pd.DataFrame(columns=['Image', 'Glucose Value'])

# Define a SessionState class to manage session-specific data
class SessionState:
    def __init__(self):
        self.glucose_data = []
        self.all_glucose_df = pd.DataFrame(columns=['Image', 'Glucose Value'])

# Function to preprocess and extract glucose value from an image
def preprocess_and_extract(image_path, session_state):
    reader = easyocr.Reader(['en'])

    image = cv2.imread(image_path)
    if image is None:
        st.error(f"Unable to read image from path: {image_path}")
        return None, None

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

    # Store detected glucose value in the session-specific list
    session_state.glucose_data.append({'Image': image_path, 'Glucose Value': glucose_value})

    return glucose_value, bbox

def main():
    global all_glucose_df
    
    # Initialize session state
    if 'session_state' not in st.session_state:
        st.session_state.session_state = SessionState()

    st.title('Glucometer Glucose Value Extractor')
    st.write('Upload an image of a glucometer to extract the glucose value.')

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        temp_image_path = "temp_image.jpg"
        with open(temp_image_path, "wb") as f:
            f.write(uploaded_file.read())

        glucose_value, bbox = preprocess_and_extract(temp_image_path, st.session_state.session_state)

        if glucose_value:
            image = cv2.imread(temp_image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if bbox is not None:
                bbox = np.array(bbox).astype(np.int32)
                cv2.rectangle(image_rgb, tuple(bbox[0]), tuple(bbox[2]), (0, 255, 0), 2)

            st.write(f"Detected Glucose Value: **{glucose_value}**")
            # st.image(image_rgb, caption=f"Detected Glucose Value: {glucose_value}")
        else:
            st.error("Unable to detect glucose value. Please try again with a clearer image or a different angle.")

        try:
            os.remove(temp_image_path)
        except Exception as e:
            st.warning(f"Failed to delete temporary file {temp_image_path}: {e}")

        # Display the table of all detected glucose values in the current session
        if st.session_state.session_state.glucose_data:
            session_df = pd.DataFrame(st.session_state.session_state.glucose_data)
            st.write("### Current Session Detected Glucose Values")
            st.write(session_df)

    # Button to download all glucose values as CSV
    if st.button("Download All Glucose Values as CSV"):
        # Concatenate current session data with all stored data
        all_glucose_df = pd.concat([all_glucose_df, pd.DataFrame(st.session_state.session_state.glucose_data)], ignore_index=True)
        all_glucose_df.drop_duplicates(subset=['Image'], keep='last', inplace=True)
        
        # Generate CSV file for download
        csv_data = all_glucose_df.to_csv(index=False).encode('utf-8')
        b64 = base64.b64encode(csv_data).decode('utf-8')
        href = f'<a href="data:file/csv;base64,{b64}" download="all_glucose_values.csv">Download All Glucose Values CSV File</a>'
        st.markdown(href, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
