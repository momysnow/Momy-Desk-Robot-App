from flask import Flask, jsonify, request, render_template

app = Flask(__name__)

# Esempio di route per una richiesta API GET
@app.route('/api/saluta', methods=['GET'])
def saluta():
    return jsonify({'messaggio': 'Ciao, benvenuto nella mia API!'})

# Esempio di route per una richiesta API POST
@app.route('/api/sommare', methods=['POST'])
def sommare():
    dati = request.get_json()

    if 'numero1' not in dati or 'numero2' not in dati:
        return jsonify({'errore': 'Fornisci entrambi i numeri'}), 400

    numero1 = dati['numero1']
    numero2 = dati['numero2']
    somma = numero1 + numero2

    return jsonify({'risultato': somma})

# Esempio di route per la pagina principale
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
