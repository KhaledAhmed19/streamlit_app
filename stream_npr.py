import cv2
import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from datetime import datetime
import pandas as pd
import os

st.set_page_config(
    page_title="Automatic Number Plate License Detection",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

chars_en_ar_mapping = {
    '-': '-', '1': '1', '2': '2', '3': '3', '4': '4', '5': '5', '6': '6', '7': '7', '8': '8', '9': '9', '0': '0',
    'alf': 'ا', 'baa': 'ب', 'dal': 'د', 'ein': 'ع', 'faa': 'ف', 'gem': 'ج', 'haa': 'هـ', 'kaff': 'ك', 'lam': 'ل',
    'meem': 'م', 'noon': 'ن', 'raa': 'ر', 'sadd': 'ص', 'seen': 'س', 'taa': 'ط', 'waw': 'و', 'yaa': 'ي', 'yea': 'ي', 'zal': 'ذ'
}


letters_mapping = {
    'alf': 'ا', 'baa': 'ب', 'dal': 'د', 'ein': 'ع', 'faa': 'ف', 'gem': 'ج', 'haa': 'هـ', 'kaff': 'ك', 'lam': 'ل',
    'meem': 'م', 'noon': 'ن', 'raa': 'ر', 'sadd': 'ص', 'seen': 'س', 'taa': 'ط', 'waw': 'و', 'yaa': 'ي', 'yea': 'ي', 'zal': 'ذ'
}


numbers_mapping = {
    '1': '1', '2': '2', '3': '3', '4': '4', '5': '5', '6': '6', '7': '7', '8': '8', '9': '9', '0': '0'
}


model_plate = YOLO('best_plates_new.pt')
model_charc = YOLO('best_char.pt')
class_list = model_charc.names
confidence_threshold = 0.5

def check_plate(detected_plate, plate_list, file):
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    result = "Allowed entry  " if detected_plate in plate_list else "Not allowed  "
    file.write(f"{detected_plate}\t{result}\t{current_datetime}\n")
    return result

def draw_text(img, text, pos, font_size=20, font_path="arial.ttf", color=(255, 255, 0)):
    font = ImageFont.truetype(font_path, font_size)
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    draw.text(pos, text, font=font, fill=color)
    return np.array(img_pil)

def main():
    st.title("License Plate Recognition System")
    video_source = st.selectbox("Select video source", ["Upload a video file", "RTMP Stream", "DroidCam"])

    if video_source == "Upload a video file":
        uploaded_file = st.file_uploader("Upload a video file", type=["mp4"])
        if uploaded_file is not None:
            if not os.path.exists("uploads"):
                os.makedirs("uploads")
            file_path = os.path.join("uploads", uploaded_file.name)
            with open(file_path, "wb") as file:
                file.write(uploaded_file.getbuffer())
            process_video(file_path)
    elif video_source == "RTMP Stream":
        rtmp_url = st.text_input("Enter RTMP stream URL")
        if rtmp_url:
            process_video(rtmp_url, is_rtmp=True)
    elif video_source == "DroidCam":
        st.write("Using DroidCam as video source.")
        process_droidcam()

def process_droidcam():
    with open("car_plate_data.txt", "w") as file:
        predefined_plates = ["4 8 7 1 sadd ein meem"]

        cap = cv2.VideoCapture(0) 

        frame_placeholder = st.empty()
        result_placeholder = st.empty()  # Create a single placeholder for the result
        processed_numbers = set()
        frame_width, frame_height = 480, 800

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (frame_width, frame_height))

            results_plate = model_plate.predict(frame)
            a = results_plate[0].boxes.data
            a = a.cpu()
            px = pd.DataFrame(a).astype("float")

            for index, row in px.iterrows():
                x1, y1, x2, y2, _, d = map(int, row)
                if d >= len(class_list):
                    st.warning(f"Warning: Detected class index {d} is out of bounds for class_list")
                    continue

                crop = frame[y1:y2, x1:x2]
                out_char = model_charc.predict(crop)
                out_rest = out_char[0].boxes.data
                out_rest = out_rest.cpu()
                predicted_chars = []

                for obj in out_rest:
                    confidence = obj[4]
                    if confidence < confidence_threshold:
                        continue
                    char_class = int(obj[5])
                    if char_class < len(class_list):
                        char_x1, char_y1, char_x2, char_y2, _, char_class = map(int, obj)
                        predicted_chars.append((char_x1, class_list[char_class]))
                        cv2.rectangle(crop, (char_x1, char_y1), (char_x2, char_y2), (255, 0, 0), 2)
                        class_name = class_list[char_class] if char_class < len(class_list) else "Unknown"
                        arabic_class_name = chars_en_ar_mapping.get(class_name, class_name)
                        crop = draw_text(crop, arabic_class_name, (char_x1, char_y1 - 10))

                predicted_chars.sort(key=lambda x: x[0])
                predicted_text = " ".join([char for _, char in predicted_chars])

                if predicted_text:
                    arabic_printable = ""
                    letters_to_reverse = ""

                    for charch in predicted_text.split():
                        if charch in numbers_mapping.keys():
                            trans = chars_en_ar_mapping.get(charch, charch)
                            arabic_printable += trans + " "

                        elif charch in letters_mapping.keys():
                            trans = chars_en_ar_mapping.get(charch, charch)
                            letters_to_reverse += trans + " "

                    arabic_text = arabic_printable + " " + letters_to_reverse[::-1]

                    arabic_predefined = " ".join(chars_en_ar_mapping.get(letter, letter) for letter in predefined_plates[0].split())

                    if predicted_text not in processed_numbers:
                        processed_numbers.add(predicted_text)

                    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                    result = check_plate(predicted_text, predefined_plates, file)

                    # Update the result placeholder with the new result

                    result_color = "color: green;" if result == "Allowed entry  " else "color: red;"

                    result_placeholder.markdown(f"""<h3 style='display: flex; align-items: center;'>
                    <span style='{result_color}'>Gate Access: {result} </span>
                    &nbsp;&nbsp;&nbsp;
                    <span style='color: blue; margin-right: 10px;'>  {arabic_text}</span>
                    </h3>""", unsafe_allow_html=True)


                for char_x1, char in predicted_chars:
                    arabic_char = chars_en_ar_mapping.get(char, char)
                    frame = draw_text(frame, arabic_char, (x1 + char_x1, y1 - 10))

            frame_placeholder.image(frame, channels="BGR")

        cap.release()

if __name__ == '__main__':
    main()
