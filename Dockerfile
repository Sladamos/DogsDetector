FROM tensorflow/tensorflow:latest-gpu

WORKDIR /app

COPY . .

CMD ["python", "main.py"]