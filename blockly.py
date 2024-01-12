# Importa le librerie necessarie
from flask import Flask, request, render_template_string

# Crea un'istanza di Flask
app = Flask(__name__)

# Codice Python di default
default_code = """\

# Il tuo codice Python qui
print("ciao")
"""


# Definisci la route principale
@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Se il modulo è stato inviato, ottieni il codice Python da Blockly
        code = request.form['python_code']
        # Esegui o elabora il codice Python come desiderato
        print("Codice Python ricevuto:\n", code)
        return "Codice Python ricevuto con successo!"

    else:
        # Crea il codice HTML per incorporare Blockly
        html = """
        <html>
          <head>
            <script src="https://unpkg.com/blockly/blockly.min.js"></script>
          </head>
          <body>
            <h1>Benvenuto in Blockly!</h1>
            <div id="blocklyDiv" style="height: 480px; width: 600px;"></div>
            <xml id="toolbox" style="display: none">
              <block type="controls_if"></block>
              <block type="logic_compare"></block>
              <block type="controls_repeat_ext"></block>
              <block type="math_number"></block>
              <block type="math_arithmetic"></block>
              <block type="text"></block>
              <block type="text_print"></block>
            </xml>
            <form method="post" action="/">
              <input type="hidden" id="python_code" name="python_code" value="">
              <button type="submit">Stampa codice Python</button>
            </form>
            <script>
              // Crea un'istanza di Blockly
              var workspace = Blockly.inject('blocklyDiv',
                  {toolbox: document.getElementById('toolbox')});

              function updateCode() {
                  // Converte il workspace in codice Python
                  var code = Blockly.Python.workspaceToCode(workspace);
                  // Aggiorna il valore del campo nascosto con il codice Python
                  document.getElementById('python_code').value = code;
                  // Stampa il codice Python nella console
                  console.log(code); // Aggiungi questa riga
                }


              // Aggiungi un listener per l'evento di clic sul pulsante
              document.querySelector('button').addEventListener('click', updateCode);
            </script>
          </body>
        </html>
        """
        # Restituisci il codice HTML come risposta
        return render_template_string(html)


# Esegui la tua app
if __name__ == "__main__":
    app.run()
