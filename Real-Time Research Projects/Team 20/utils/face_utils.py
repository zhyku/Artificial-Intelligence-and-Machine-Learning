import cv2
import face_recognition
import os
import numpy as np
import csv
from datetime import datetime

DATA_DIR = 'face_data'
CSV_PATH = 'employees.csv'

def register_employee(camera, emp_id, name):
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    ret, frame = camera.read()
    if not ret:
        return "Failed to capture frame."

    # Save image
    img_path = os.path.join(DATA_DIR, f'{emp_id}.jpg')
    cv2.imwrite(img_path, frame)

    # Save to CSV
    with open(CSV_PATH, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([emp_id, name])

    return f"{name} registered with ID {emp_id}."

def load_known_faces():
    known_encodings = []
    known_ids = []
    id_name_map = {}

    if not os.path.exists(CSV_PATH):
        return known_encodings, known_ids, id_name_map

    # Load ID-name map
    with open(CSV_PATH, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            id_name_map[row[0]] = row[1]

    for filename in os.listdir(DATA_DIR):
        emp_id = filename.split('.')[0]
        path = os.path.join(DATA_DIR, filename)
        image = face_recognition.load_image_file(path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_encodings.append(encodings[0])
            known_ids.append(emp_id)
    
    return known_encodings, known_ids, id_name_map

def recognize_face(camera):
    ret, frame = camera.read()
    if not ret:
        return "Failed to capture frame."

    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    known_encodings, known_ids, id_name_map = load_known_faces()

    for encoding in face_encodings:
        matches = face_recognition.compare_faces(known_encodings, encoding)
        if True in matches:
            matched_index = matches.index(True)
            emp_id = known_ids[matched_index]
            name = id_name_map.get(emp_id, "Unknown")
            log_attendance(emp_id, name)
            return f"Access Granted: {name} (Employee ID: {emp_id})"
    return "Face not recognized."

def log_attendance(emp_id, name):
    with open('attendance.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([emp_id, name, datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
