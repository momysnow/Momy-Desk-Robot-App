import socket
import struct
import io
from PIL import Image

HOST = '0.0.0.0'  # Accetta connessioni da qualsiasi interfaccia di rete
PORT = 8000

def receive_image(conn):
    # Ricevi la lunghezza dei dati come 4 byte non firmati
    data_length = struct.unpack('I', conn.recv(4))[0]

    # Ricevi i dati dell'immagine
    image_data = b''
    while len(image_data) < data_length:
        image_data += conn.recv(data_length - len(image_data))

    # Converti i dati dell'immagine in un oggetto immagine utilizzando il modulo PIL
    try:
        image = Image.open(io.BytesIO(image_data))
        image.show()
    except Exception as e:
        print(f"Error processing image: {e}")

def main():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((HOST, PORT))
    server_socket.listen()

    print(f"Server listening on {HOST}:{PORT}")

    while True:
        conn, addr = server_socket.accept()
        print(f"Accepted connection from {addr}")

        try:
            receive_image(conn)
        except Exception as e:
            print(f"Error processing image: {e}")
        finally:
            conn.close()

if __name__ == "__main__":
    main()
