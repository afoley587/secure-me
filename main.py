"""Runs Face Detection On A Webcam Video Stream
"""
import os
import sys

from cv2 import cv2

from face_detector import FaceDetector
from utils import (
    gather_files,
    draw_rectangle,
    resize_and_recolor,
    open_shell,
    do_login
)

DATA_PATH = os.environ["DATA_PATH"]
DB_SALT   = os.environ["DATABASE_SALT"].encode('utf-8')
DB_PATH   = os.path.join(DATA_PATH, "database", "users.db")

def main():
    """Main entrypoint / loop of the system
    """
    list_of_files, names = gather_files()
    face_detector = FaceDetector()
    face_detector.train(list_of_files, names)

    video_capture = cv2.VideoCapture(0)

    while True:
        # Read the raw CV2 Frame from our Webcam
        _, frame      = video_capture.read()
        # Resize it and convert the color to a format
        # that face_recognition library and our data
        # recognize
        rgb_small_frame = resize_and_recolor(frame)
        # Search the image based on our training data
        face_names, face_locations = face_detector.search(rgb_small_frame)

        draw_rectangle(
            frame,
            locations=face_locations,
            labels=face_names
        )

        # Display the resulting image
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Boolean flag indicating whether or not we found any
        # faces
        found_faces = (
            len(face_names) > 0 and
            len(face_locations) > 0
        )

        if found_faces:
            # If we found a face we recognize,
            # lets try to do a login based on our database
            success = do_login(face_names[0], DB_SALT, DB_PATH)
            if success:
                # If successful, we can exit
                print(f"Hello, {face_names[0]}! Welcome back to the system....")
                open_shell()
                sys.exit(0)
            else:
                # If incorrect, we can try again
                print("Incorrect Password, please try again")


if __name__ == "__main__":
    main()
