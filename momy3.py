import socket
import struct
import io
from PIL import Image, ImageDraw
from flask import Flask, Response, render_template, request, jsonify
from transformers import YolosImageProcessor, YolosForObjectDetection, AutoImageProcessor, DPTForDepthEstimation
import torch
import numpy as np
import requests
import matplotlib.pyplot as plt
import time

app = Flask(__name__)

HOST = '0.0.0.0'
PORT = 8000

dpt_dinov2_small_kitti = DPTForDepthEstimation.from_pretrained("facebook/dpt-dinov2-small-kitti")
dpt_dinov2_small_kitti_image_processor = AutoImageProcessor.from_pretrained("facebook/dpt-dinov2-small-kitti")

def depth_estimation(image):
    inputs = dpt_dinov2_small_kitti_image_processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = dpt_dinov2_small_kitti(**inputs)
        predicted_depth = outputs.predicted_depth

    # Interpolate to original size
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    )

    # Convert the prediction to a numpy array
    output_array = prediction.squeeze().numpy()

    return output_array


def receive_image(conn):
    data_length = struct.unpack('I', conn.recv(4))[0]
    image_data = b''
    while len(image_data) < data_length:
        image_data += conn.recv(data_length - len(image_data))
    try:
        image = Image.open(io.BytesIO(image_data))
        return image
    except Exception as e:
        print(f"Error processing image: {e}")


@app.route('/')
def index():
    return render_template('index.html')


def generate():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((HOST, PORT))
    server_socket.listen()

    print(f"Server listening on {HOST}:{PORT}")

    while True:
        conn, addr = server_socket.accept()
        print(f"Accepted connection from {addr}")

        try:
            image = receive_image(conn)

            results = object_detection(image, )
            # Estrai le informazioni necessarie dai risultati
            scores = results["scores"]
            labels = results["labels"]
            boxes = results["boxes"]

            # Disegna i rettangoli sull'immagine
            detect_img = draw_boxes_on_image(image, boxes, labels, scores)

            if image:
                # Converti l'immagine in un formato supportato da Flask (come JPEG)
                img_byte_array = io.BytesIO()
                detect_img.save(img_byte_array, format='JPEG')
                img_byte_array = img_byte_array.getvalue()

                # Invia l'immagine al client come streaming
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + img_byte_array + b'\r\n\r\n')
        except Exception as e:
            print(f"Error processing image: {e}")
        finally:
            conn.close()


@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
