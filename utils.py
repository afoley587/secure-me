"""Functions to be shared through out the main loop
"""
import os
import glob
from getpass import getpass
import pty
import sys

import bcrypt
from cv2 import cv2

from db import DBInterface

def gather_files(extension_glob='*.jpg', location="") -> (list, list):
    """Gathers files based on file globs and returns a tuple of (files, filenames)

    Arguments:
      extension_glob - A file glob to look for (png / jpg / etc.)
      location - Starting directory, defaults to None
    """
    # If None passed, use cwd/data/faces
    if not location:
        script_dir = os.path.dirname(
            os.path.realpath(
                os.path.join(os.getcwd(), os.path.expanduser(__file__))
            )
        )
        location = os.path.join(script_dir, 'data', 'faces')

    file_list     = []
    filename_list = []
    for _file in glob.glob(os.path.join(location, extension_glob)):
        file_list.append(_file)
        filename_list.append(_file.split(os.path.sep)[-1].split('.')[0])
    return (file_list, filename_list)

def draw_rectangle(frame, locations=None, labels=None):
    """Draws a rectangle on an openCV Frame
    """

    if not locations or not labels:
        return

    for (top, right, bottom, left), label in zip(locations, labels):
        if label == "":
            continue
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        # Draw a rectangle around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        # Input text label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, label, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

def resize_and_recolor(frame):
    """Resizes and colors an image
    """
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]
    return rgb_small_frame

def open_shell():
    """Opens a python shell
    """
    shell = sys.executable
    print("Opening your shell of choice sir")
    pty.spawn(shell)


def get_password():
    """Gets a password from the user
    """
    plaintext = getpass()
    return plaintext

def do_login(username: str, salt: str, database: str):
    """Performs a login action

    Arguments:
        username - username of the user

    Returns:
        True  - If successful
        False - If unsuccessful
    """
    dbi       = DBInterface(database)
    password  = bcrypt.hashpw(
        get_password().encode('utf-8'),
        salt.encode('utf-8')
    ).decode('utf-8')

    statement = """SELECT name FROM user WHERE name = ? AND password = ?"""
    params    = (username, password)

    matches = dbi.run_closing_query(statement, params=params)
    if len(matches) < 1:
        return False
    return True
