# detection.py
import cv2
import face_recognition
import numpy as np
from pymongo import MongoClient
import streamlit as st

# === MongoDB ===
client = MongoClient("mongodb://localhost:27017/")
collection = client["event_db"]["registrations"]

# Load all known people
known_people = []
for doc in collection.find():
    encodings = [np.array(e) for e in doc["encodings"]]
    known_people.append({
        "name": doc["name"],
        "phone": doc["phone"],
        "age": doc["age"],
        "gender": doc["gender"],
        "encodings": encodings
    })

if not known_people:
    st.warning("No registered people found. Register someone first!")
    st.stop()

# === Streamlit Search UI ===
st.title("Find ME – Live Detection")
st.markdown("### Search for a person to highlight")

# Search box
search = st.text_input("Search by Name or Phone", "")

# Filter people
filtered = [
    p for p in known_people
    if search.lower() in p["name"].lower() or search in p["phone"]
]

if not filtered:
    st.info("No matches. Try another name/phone.")
    options = []
else:
    # Create dropdown options: "Name (Phone)"
    options = [f"{p['name']} ({p['phone']})" for p in filtered]

# Dropdown
selected = st.selectbox("Select person to highlight", options, index=None)

# Get selected person's encodings
target_encodings = None
target_name = None
if selected:
    name_phone = selected.split(" (")
    target_name = name_phone[0]
    target_phone = name_phone[1][:-1] if len(name_phone) > 1 else ""
    for p in filtered:
        if p["name"] == target_name and p["phone"] == target_phone:
            target_encodings = p["encodings"]
            break

# === Webcam ===
st.markdown("### Live Detection")
frame_placeholder = st.empty()

cap = cv2.VideoCapture(0)
TOLERANCE = 0.48

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    locs = face_recognition.face_locations(rgb, model="hog")
    encs = face_recognition.face_encodings(rgb, locs)

    for (top, right, bottom, left), enc in zip(locs, encs):
        color = (128, 128, 128)  # Gray = unknown
        label = "Unknown"

        if target_encodings:
            # Only compare with selected person
            distances = face_recognition.face_distance(target_encodings, enc)
            best_dist = min(distances)
            if best_dist < TOLERANCE:
                color = (0, 255, 0)  # Green = match
                label = f"{target_name}"
            else:
                color = (100, 100, 100)  # Dim gray
                label = ""
        else:
            # No target → show all (optional)
            pass

        # Draw box
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        if label:
            cv2.putText(frame, label, (left, top - 10),
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)

    # Show in Streamlit
    frame_placeholder.image(frame, channels="BGR")

    # Press 'q' in terminal to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
