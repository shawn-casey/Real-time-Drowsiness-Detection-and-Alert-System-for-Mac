import cv2
import os
import time
import dlib
from plyer import notification

current_dir = os.path.dirname(os.path.abspath(__file__))
face_cascade_path = os.path.join(current_dir, 'haarcascade_frontalface_default.xml')
eye_cascade_path = os.path.join(current_dir, 'haarcascade_eye.xml')
fullbody_cascade_path = os.path.join(current_dir, 'haarcascade_fullbody.xml')

face_classifier = cv2.CascadeClassifier(face_cascade_path)
eye_classifier = cv2.CascadeClassifier(eye_cascade_path)
fullbody_classifier = cv2.CascadeClassifier(fullbody_cascade_path)

def eye_aspect_ratio(eye):
    # Compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = abs(eye[1][1] - eye[5][1])
    B = abs(eye[2][1] - eye[4][1])

    # Compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = abs(eye[0][0] - eye[3][0])

    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    return ear

def detect_sleep(frame, predictor, face_tracker=None, max_failed_frames=20, sleep_threshold=0.2):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Face detection
    if face_tracker is None:
        faces = face_classifier.detectMultiScale(gray_frame, 1.1, 4)
        if len(faces) > 0:
            x, y, w, h = faces[0]  # Use the first detected face as the initial bounding box
            face_tracker = cv2.TrackerCSRT_create()
            face_tracker.init(frame, (x, y, w, h))

    if face_tracker:
        success, face_box = face_tracker.update(gray_frame)
        if success:
            x, y, w, h = [int(i) for i in face_box]
            roi_gray = gray_frame[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]

            # Eye detection using dlib facial landmarks
            predictor_results = predictor(gray_frame, dlib.rectangle(x, y, x + w, y + h))
            left_eye_points = [(predictor_results.part(i).x, predictor_results.part(i).y) for i in range(36, 42)]
            right_eye_points = [(predictor_results.part(i).x, predictor_results.part(i).y) for i in range(42, 48)]

            # Calculate the eye aspect ratio for both eyes
            left_ear = eye_aspect_ratio(left_eye_points)
            right_ear = eye_aspect_ratio(right_eye_points)

            # Average the eye aspect ratio for both eyes
            ear = (left_ear + right_ear) / 2.0

            # Check if eyes are closed based on the eye aspect ratio threshold
            if ear < sleep_threshold:
                global last_closed_time
                if last_closed_time is None:
                    last_closed_time = time.time()
                else:
                    current_time = time.time()
                    elapsed_time = current_time - last_closed_time
                    if elapsed_time >= sleep_threshold:
                        # Trigger the alarm/notification
                        wake_up()
                        last_closed_time = None
            else:
                last_closed_time = None

    return frame, face_tracker

def wake_up():
    # Display a notification using the plyer library
    notification_title = "Wake Up!"
    notification_message = "You might have fallen asleep. Wake up!"
    notification.notify(title=notification_title, message=notification_message)

def main():
    cap = cv2.VideoCapture(0)

    face_tracker = None
    failed_frames = 0

    # Initialize the dlib facial landmark predictor (you may need to download the shape_predictor_68_face_landmarks.dat file)
    predictor_path = os.path.join(current_dir, 'shape_predictor_68_face_landmarks.dat')
    predictor = dlib.shape_predictor(predictor_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame, face_tracker = detect_sleep(frame, predictor, face_tracker)

        # If face is not detected for multiple frames, reset the face tracker
        if face_tracker is None:
            failed_frames += 1
            if failed_frames >= 20:
                failed_frames = 0
                face_tracker = None

        cv2.imshow('Real-time Face, Eye, and Full-Body Detection', processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Global variable to store the last time eyes were closed
last_closed_time = None

if __name__ == "__main__":
    main()
