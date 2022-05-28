"""The facial data class uses the face_recognition
    module to train itself using the files passed to it
"""

import face_recognition
import numpy as np

class FaceDetector:
    """The facial data class uses the face_recognition
    module to train itself using the files passed to it

    names (list): A list of names that we understand
     encodings (list): A list of facial encoding data
    images (list): A list of raw images passed in
    """
    def __init__(self):
        self.names     = []
        self.encodings = []
        self.images    = []

    def train(self, file_list: list, names: list):
        """Reads each image passed to it and fills out the names,
        encodings, and images lists

        Args:
            file_list: A list of files to read
            names: A lit of names to correspond to the files

        Returns:
            None
        """
        if len(file_list) != len(names):
            print("ERR! The file list and name list are not equal")
            return

        for index, img_file in enumerate(file_list):
            img      = face_recognition.load_image_file(img_file)
            encoding = face_recognition.face_encodings(img)[0]
            self.images.append(img)
            self.encodings.append(encoding)
            self.names.append(names[index])

    def search(self, frame):
        """Searches a frame for a face based on our loaded encodings

        Args:
            frame: TODO

        Returns:
            face_names: TODO
            face_locations: TODO
        """
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)
        face_names     = []
        name           = ""

        for index, face_encoding in enumerate(face_encodings):
            matches          = face_recognition.compare_faces(self.encodings, face_encoding)
            face_distances   = face_recognition.face_distance(self.encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = self.names[best_match_index]
                face_names.append(name)
            else:
                face_locations.pop(index)
        return (face_names, face_locations)
