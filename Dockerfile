FROM tensorflow/tensorflow:2.10.0

WORKDIR /app

RUN ["apt-get", "update"]

RUN ["apt-get", "install", "ffmpeg", "libsm6", "libxext6", "-y"]

RUN ["python", "-m", "pip", "install", "--upgrade", "pip"]

RUN ["pip", "install", "opencv-python-headless"]

RUN ["pip", "install", "numpy"]

RUN ["pip", "install", "matplotlib"]

RUN ["pip", "install", "scipy"]

RUN ["pip", "install", "pyqt5"]

COPY . .

ENTRYPOINT ["python", "main.py"]