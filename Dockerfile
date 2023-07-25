FROM tensorflow/tensorflow:2.10.0

WORKDIR /app

COPY . .

RUN ["pip", "install", "opencv-python"]

RUN ["pip", "install", "numpy"]

RUN ["pip", "install", "matplotlib"]

RUN ["pip", "install", "scipy"]

CMD ["python", "main.py"]