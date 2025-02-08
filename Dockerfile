# Use an official Python runtime as a parent image
FROM python:3.12-slim

WORKDIR /app

COPY . /app
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["python", "demo.py"]