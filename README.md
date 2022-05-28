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
* Build Our Authentication Layer
* Bring It All Together

## Gather Facial Data
## Building our face detector
