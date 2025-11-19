# registration.py
import streamlit as st
import numpy as np
from PIL import Image
import io
import face_recognition
from pymongo import MongoClient
import datetime
import pandas as pd
import os

# === Page Config ===
st.set_page_config(page_title="Find ME", layout="centered")

# === Paths ===
DATA_DIR = "data"
IMAGES_DIR = "images"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

CSV_PATH = os.path.join(DATA_DIR, "registrations.csv")

# === MongoDB (Optional) ===
client = MongoClient("mongodb://localhost:27017/")
db = client["event_db"]
collection = db["registrations"]

# === Load Existing CSV (if any) ===
if os.path.exists(CSV_PATH):
    df_existing = pd.read_csv(CSV_PATH)
else:
    df_existing = pd.DataFrame(columns=["Name", "Phone", "Age", "Gender", "Registered_At", "Front_Image", "Left_Image", "Right_Image"])

# === UI ===
st.title("Find ME")
st.markdown("### Register attendee for face recognition")

with st.form("reg_form"):
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Name *", placeholder="John Doe")
        phone = st.text_input("Phone *", placeholder="+1234567890")
    with col2:
        age = st.number_input("Age *", min_value=1, max_value=120, value=25, step=1)
        gender = st.selectbox("Gender *", ["Male", "Female", "Other"])

    st.markdown("### Capture 3 Face Angles")
    c1, c2, c3 = st.columns(3)
    with c1: front = st.camera_input("Front Face", key="front")
    with c2: left = st.camera_input("Left Profile", key="left")
    with c3: right = st.camera_input("Right Profile", key="right")

    submit = st.form_submit_button("Register", type="primary")

# === Face Encoding Function ===
def get_encoding(img_bytes):
    try:
        img = Image.open(io.BytesIO(img_bytes))
        arr = np.array(img)
        encodings = face_recognition.face_encodings(arr)
        return encodings[0].tolist() if encodings else None
    except:
        return None

# === Save to Folders + CSV + MongoDB ===
if submit:
    if not all([name.strip(), phone.strip(), front, left, right]):
        st.error("All fields are required!")
    else:
        e1 = get_encoding(front.getvalue())
        e2 = get_encoding(left.getvalue())
        e3 = get_encoding(right.getvalue())

        if None in (e1, e2, e3):
            st.error("Face not detected! Try better lighting.")
        else:
            # === 1. Save Images to `images/` ===
            safe_name = "".join(c for c in name if c.isalnum() or c in " _-").rstrip()
            front_path = os.path.join(IMAGES_DIR, f"{safe_name}_front.png")
            left_path = os.path.join(IMAGES_DIR, f"{safe_name}_left.png")
            right_path = os.path.join(IMAGES_DIR, f"{safe_name}_right.png")

            Image.open(io.BytesIO(front.getvalue())).save(front_path)
            Image.open(io.BytesIO(left.getvalue())).save(left_path)
            Image.open(io.BytesIO(right.getvalue())).save(right_path)

            # === 2. Append to CSV ===
            new_row = {
                "Name": name.strip(),
                "Phone": phone.strip(),
                "Age": int(age),
                "Gender": gender,
                "Registered_At": datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                "Front_Image": front_path,
                "Left_Image": left_path,
                "Right_Image": right_path
            }
            df_new = pd.DataFrame([new_row])
            df_updated = pd.concat([df_existing, df_new], ignore_index=True)
            df_updated.to_csv(CSV_PATH, index=False)

            # === 3. Save to MongoDB (Optional) ===
            doc = {
                "name": name.strip(),
                "phone": phone.strip(),
                "age": int(age),
                "gender": gender,
                "encodings": [e1, e2, e3],
                "image_paths": {
                    "front": front_path,
                    "left": left_path,
                    "right": right_path
                },
                "registered_at": datetime.datetime.utcnow()
            }
            collection.insert_one(doc)

            st.success(f"Registered {name}! Saved to `data/` and `images/`.")
            st.balloons()