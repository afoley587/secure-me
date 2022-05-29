# LinkedIn

# Blog
## Build Your Own Biometrics (in only a few lines of code)

BYOB? Bring your own what? No, today we are doing 
__B__uild __Y__our __O__wn __B__iometerics. We are all familiar
with iPhone FaceID or some of the really neat Unix Face recognition
tools that are available today. But, have you ever built your own?
I haven't, so I decided to today and hope you enjoy the ride!

From a high level, most pieces of software work by:

1. Registering your face
2. Asking for a Pin or some sort of password

After which, they give you the keys to your kingdom. So, we will need to 
build those two things:

1. Something that takes input from our webcam, processes the faces,
    and see if it knows who those faces belong to.
2. Once the faces are recognized, something must ask for a username as
    input and see if its the proper password.
  
Lets get started!

## Dependencies
One of the reasons our code is going to be so light is because of a very
cool python library called [`face_recognition`](https://github.com/ageitgey/face_recognition).

Of course, its a non-standard library, and is only one out a few that we are going to use
today. It also has a few OS dependencies. If you're running on Mac, you might need these
four brew installs (`Brewfile` included [here]()!):
```shell
brew install "cmake"
brew install "dlib"
brew install "boost"
brew install "boost-python"
```

Once those are installed, we can begin installing our python dependencies. If you're using a
`toml` file, you can use the following dependencies:
```shell
[tool.poetry.dependencies]
python = "^3.8"
cmake = "3.21.2"
dlib = "19.24.0"
opencv-python = "4.5.3.56"
bcrypt = "3.2.2"
face-recognition = "1.3.0"
```

If you're using `pip` to install:
```shell
bcrypt==3.2.2
cffi==1.15.0
click==8.1.3
cmake==3.21.2
colorama==0.4.4; platform_system == "Windows" or sys_platform == "win32"
dlib==19.24.0
face-recognition==1.3.0
face-recognition-models==0.3.0
numpy==1.22.4
opencv-python==4.5.3.56
pillow==9.1.1
pycparser==2.21
```

Now, we are ready to start building! Let's do the following:

* Gather Facial Data
* Build Our Face Detector
* Generate Our Database
* Build Our Authentication Layer
* Bring It All Together

## Gather Facial Data
This part sounds tedious, but guess what, its not. All
we need is one single image for each face we want to detect.

So, open up your webcam and take a selfie! And put it in the 
`data/faces` directory, saved as `<YOUR_NAME>.jpg`.

For example, in my directory, you'll see:
```shell
| - data
|     - faces
|         - alex.jpg
```

And that's all we have to do for our training set!

## Building our face detector
Now we can move to building our face detector! Create a file called
`face_detector.py`. Let's start tackling that from top to bottom.

First, we need our imports:
```python
import face_recognition
import numpy as np
```

We are going to need both the `face_recognition` and `numpy` libraries.

Then we can define our class:
```python
# face_detector.py
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
```

Our class is going to save three different lists:

1. The names of the faces it can recognize
2. The encodings (face recognition data) of those faces
3. The raw images themselves

We will then need two methods for our class:

1. A trainer to set the aforementioned lists
2. A searcher to search a given image for any known faces

We are going to start with the trainer:

```python
    # face_detector.py
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
```

As you can see, we are just going to walk through every file in a given
data list. We are then going to load the image using the `face_recognition`
library, and then we are going to get the facial data fro those images.
Finally, we are just going to store the image, encoding, and name in our
parallel arrays.

Our search isn't much more complicated.

The first two lines of the function should look very similar. We are 
first using the `face_recognition.face_locations` function to search the give 
for any faces. These faces may or may not be recognized by the system, but at least
we know if there are or are not faces.

Next, we use the `face_recognition.face_encodings` function to search each face in the
image and create the facial encodings from those locations. Essentially, we are decomposing 
the face into a set of data that our system can use as a comparator.

```python
    # face_detector.py
    def search(self, frame):
        """Searches a frame for a face based on our loaded encodings

        Args:
            frame: The image frame to search

        Returns:
            face_names: The names belonging to the faces
            face_locations: The locations of the faces in the image
        """
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)
        ret_names      = []
        ret_locations  = []
```


Then, we begin looping through each face encoding that we found. We use the
`face_recognition.compare_faces` function to see if any of the images in our
frame match any of our known images. The `matches` array will be a list of booleans, indicating
whether the face in our frame matches the faces we know or not.

Next, we find the face distances. A face distance is a measure of difference in the faces.
So if a face distance is high (close to 1), it means that two faces are not very similar. We 
can then pull the minimum value from the `face_distances` array, meaning the most prominent
face in our frame

```python
        # face_detector.py
        for index, face_encoding in enumerate(face_encodings):
            matches          = face_recognition.compare_faces(self.encodings, face_encoding)
            face_distances   = face_recognition.face_distance(self.encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

```

Finally, we do a quick sanity check: did our most prominent face actually match?
If it did, lets pull the name out of our `names` array and store both the name
and the location.

Finally, we return the found faces and the locations of the faces.

```python
            # face_detector.py
            if matches[best_match_index]:
                ret_names.append(self.names[best_match_index])
                ret_locations.append(face_locations[best_match_index])
        return (ret_names, ret_locations)
```

That's it! We can now find faces in any frames, compare them to 
a list of known faces, and then return the names and locations 
of those faces!

## Generate Our Database
We are now DONE with our face recognizer. But, we have to pull in one more
thing: password authentication. So, we will need a database. Our seed data
will go in `data/users/passwords.json`:
```json
{
  "passwords": [
    { "name": "alex", "pass": "foobar" }
  ]
}
```

Of course, dont store anything super sensitive in Git, but I think its okay
for these demonstrative purposes!

Now, let's go through a helper script to build or database for us:

First, we import some libraries and set some globally usable
variables:

  * DATA_PATH - Path to our data directory
  * DB_SALT - A salt to hash passwords with
  * DB_PATH - The database path

```python
# generate_db.py
import json
import sqlite3 as sql
import os
import bcrypt

DATA_PATH = os.environ["DATA_PATH"]
DB_SALT   = os.environ["DATABASE_SALT"].encode('utf-8')
DB_PATH   = os.path.join(DATA_PATH, "database", "users.db")
```

We then remove any old database and then create our base statement and 
our user statement. The base statement will be used to initialize the
database table while the user statement will add each user from the above
json file into that table:
```python
# generate_db.py
# remove the old DB in favor of our newly seeded one
if os.path.exists(DB_PATH):
    os.remove(DB_PATH)

# Creating a basic table for us to store users in
BASE_STATEMENT = """
CREATE TABLE IF NOT EXISTS user (
  name varchar(50) DEFAULT NULL,
  password varchar(100) DEFAULT NULL
);
"""
USER_STATEMENT = "INSERT INTO user (name, password) VALUES"
```

We then read our json config file, hash each password we found with bcrypt, 
and then add the user/password pair to our user statement:
```python
# generate_db.py
with open(os.path.join(DATA_PATH, "users", "passwords.json"), "r", encoding="utf-8") as config:
    json_config = json.loads(config.read())

for user in json_config["passwords"]:
    # Hash to password using Bcrypt
    hashed = bcrypt.hashpw(
        user['pass'].encode('utf-8'),
        DB_SALT
    ).decode('utf-8')
    USER_STATEMENT += f"\n\t('{user['name']}', '{hashed}')"

USER_STATEMENT += ";"
```

Finally, we just execute our statements:
```python
# generate_db.py
# Run the SQL on our database
with sql.connect(DB_PATH) as con:
    cur = con.cursor()
    cur.execute(BASE_STATEMENT)

with sql.connect(DB_PATH) as con:
    cur = con.cursor()
    cur.execute(USER_STATEMENT)
```

Our database is now initialized and ready for use.

## Build Our Authentication Layer
So we have a database, but we need something to interact with it - queue a python
class!

```python
# db.py
import sqlite3 as sql

class DBInterface:
    """A basic SQLite interface

          Attributes:
              sqlite_db - Path to a SQLite DB
    """
    def __init__(self, sqlite_db: str):
        self.sqlite_db = sqlite_db
```

Our `DBInterface` will act as our connection(s) to our authentication database. For our
purposes, we just need to be able to execute queries, but we dont need anything fancy.

The observant reader will notice that we have two functions:

* `run_closing_query`
* `_run_closing_query`

One is meant to be public, and one is private. Of course, the public one (`run_closing_query`)
just calls the private one `_run_closing_query`.

Again, these are very simple functions. The `_run_closing_query` first opens a connection
with the database. It then initializes a cursor to use for query execution, and then it
runs the query and returns the results. 

```python
    # db.py
    def run_closing_query(self, query: str, params=()):
        """Public version of _run_closing_query

        Arguments:
            query    - Query to run
            params - Parameters to the query

        Returns:
            results - Results from the query
        """
        return self._run_closing_query(query, params=params)

    def _run_closing_query(self, query, params=()):
        """Public version of _run_closing_query

        Arguments:
            query    - Query to run
            params - Parameters to the query

        Returns:
            results - Results from the query
        """
        results = None
        with sql.connect(self.sqlite_db) as con:
            cur = con.cursor()
            cur.execute(query, tuple(params))
            results = cur.fetchall()
        return results
```

Finally, something is going to have to call this interface
with the proper queries! That's where our `do_login` function comes in!

`do_login` creates a new database interface with the above class,
then uses our `get_password` function to retrieve a password from the
user. We have to then hash the password with the same salt value that 
we used in our `generate_db.py` python script. Finally, we see if a user
with that name and password exist in our database. If they do, then they
entered the proper password and we True, otherwise the password was incorrect
and we return False.

Note: `get_password` is simply a wrapper around python's `getpass` function!

```python
# utils.py
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
        get_password().encode('utf-8'), salt
    ).decode('utf-8')

    statement = """SELECT name FROM user WHERE name = ? AND password = ?"""
    params    = (username, password)

    matches = dbi.run_closing_query(statement, params=params)
    if len(matches) < 1:
        return False
    return True
```

## Bring It All Together
We have it all now! Let's glue all of our pieces together!

First, lets import our modules:
```python
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
```

Let's also pull our data path, database, and salt
from environment variables:
```python
DATA_PATH = os.environ["DATA_PATH"]
DB_SALT   = os.environ["DATABASE_SALT"].encode('utf-8')
DB_PATH   = os.path.join(DATA_PATH, "database", "users.db")
```

And let's enter our main function! First, in our main function,
we will use some helper functions to gather all of the files and 
names of the files from our data directory. Then we will train our
new `FaceDetector` class and train it on those images. Once our model
is trained, we can open up a video stream from our webcam:

Note: `gather_files` is a helper function defined in the extra notes below!

```python
def main():
    """Main entrypoint / loop of the system
    """
    list_of_files, names = gather_files()
    face_detector = FaceDetector()
    face_detector.train(list_of_files, names)

    video_capture = cv2.VideoCapture(0)
```

Next, we begin reading from our video capture frame-by-frame
and search it for any relevant faces:

Note: `resize_and_recolor` is a helper function defined in the extra notes below!
```python
    while True:
        # Read the raw CV2 Frame from our Webcam
        _, frame      = video_capture.read()
        # Resize it and convert the color to a format
        # that face_recognition library and our data
        # recognize
        rgb_small_frame = resize_and_recolor(frame)
        # Search the image based on our training data
        face_names, face_locations = face_detector.search(rgb_small_frame)
```

Next, we draw a rectangle over any relevant faces and show the frame to the user:

Note: `draw_rectangle` is a helper function defined in the extra notes below!

```python
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
```

FInally, if we found any relevant faces, we can do the login flow. If the login 
is successful, we can will drop a user into a python shell, signifying success!

```python
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
```

## Running
To run:

Note: The DATABASE_SALT was generated with `bcrypt.gensalt()`

```shell
#!/bin/bash

GIT_ROOT=$(git rev-parse --show-toplevel)
export DATABASE_PATH="${GIT_ROOT}/data/database/users.db"
export DATABASE_SALT='$2b$12$l0U9KMzQ52as6nnm5W6XJu'
export DATA_PATH="${GIT_ROOT}/data"

poetry run python main.py
```

<video width="320" height="240" controls>
  <source src="example.mov" type="video/mp4">
</video>

## Extra Notes

Some functions were left out for the sake of brevity:
`gather_files`:

`resize_and_recolor`:

`draw_rectangle`: 