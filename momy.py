import socket
import struct
import io
from PIL import Image, ImageDraw
from flask import Flask, Response, render_template, request, jsonify
from transformers import YolosImageProcessor, YolosForObjectDetection, AutoModelForCausalLM, AutoTokenizer
import torch

app = Flask(__name__)

HOST = '0.0.0.0'
PORT = 8000

yolos_model = YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny')
yolos_processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")

# Verifica la disponibilità di CUDA
if torch.cuda.is_available():
    DialoGPT_tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    DialoGPT_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
else:
    raise RuntimeError("CUDA is not available. Make sure your GPU and CUDA are properly configured.")

def object_detection(image, threshold=0.9):
    inputs = yolos_processor(images=image, return_tensors="pt")
    outputs = yolos_model(**inputs)

    # model predicts bounding boxes and corresponding COCO classes
    logits = outputs.logits
    bboxes = outputs.pred_boxes

    # print results
    target_sizes = torch.tensor([image.size[::-1]])
    results = yolos_processor.post_process_object_detection(outputs, threshold, target_sizes=target_sizes)[0]
    return results


def draw_boxes_on_image(image, boxes, labels, scores):
    # Crea una copia dell'immagine originale
    drawn_image = image.copy()

    # Prepara l'oggetto per disegnare sulla copia dell'immagine
    draw = ImageDraw.Draw(drawn_image)

    # Itera attraverso le previsioni e disegna i rettangoli sull'immagine
    for box, label, score in zip(boxes, labels, scores):
        box = [round(i, 2) for i in box.tolist()]

        # Estrai le coordinate del rettangolo
        x, y, w, h = box
        # Calcola le coordinate degli angoli del rettangolo
        x1, y1, x2, y2 = x, y, x + w, y + h

        # Disegna il rettangolo sull'immagine
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

        # Aggiungi etichetta e punteggio
        label_text = f"{yolos_model.config.id2label[label.item()]}: {round(score.item(), 3)}"
        draw.text((x1, y1 - 10), label_text, fill="red")

    return drawn_image


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


# Initialize the chat history variable
chat_history_ids = None


@app.route('/chat', methods=['POST'])
def chat():
    global chat_history_ids

    user_message = request.form['user_message']
    print(user_message)

    # Verifica la disponibilità di CUDA
    if torch.cuda.is_available():
        if chat_history_ids is None:
            # Initialize chat_history_ids if it's None
            chat_history_ids = torch.tensor([], dtype=torch.long)

        # encode the new user input, add the eos_token and return a tensor in PyTorch
        new_user_input_ids = DialoGPT_tokenizer.encode(user_message + DialoGPT_tokenizer.eos_token, return_tensors='pt')

        # append the new user input tokens to the chat history
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)

        # generate a response while limiting the total chat history to 1000 tokens
        chat_history_ids = DialoGPT_model.generate(bot_input_ids, max_length=1000, pad_token_id=DialoGPT_tokenizer.eos_token_id)

        # pretty print last output tokens from bot
        bot_response = DialoGPT_tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

        return {'bot_response': bot_response}
    else:
        bot_response = "ciao non stai usando cuda"

    print(bot_response)
    return {'bot_response': bot_response}


if __name__ == '__main__':
    app.run(debug=True)
