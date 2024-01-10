# Importare le librerie necessarie
import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt

# Definire i parametri del modello
input_size = 40 # Dimensione delle caratteristiche MFCC
hidden_size = 256 # Dimensione dello stato nascosto della RNN
output_size = 2 # Dimensione dell'output: parola di attivazione o non parola di attivazione
num_layers = 2 # Numero di strati della RNN
learning_rate = 0.01 # Tasso di apprendimento
num_epochs = 10 # Numero di epoche di addestramento
batch_size = 32 # Dimensione del batch
keyword = "ciao" # Parola di attivazione da riconoscere

# Definire il modello
class KeywordModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(KeywordModel, self).__init__()
        self.rnn = torch.nn.GRU(input_size, hidden_size, num_layers, batch_first=True) # Usare una rete neurale ricorrente di tipo GRU
        self.fc = torch.nn.Linear(hidden_size, output_size) # Usare uno strato lineare per l'output
        self.softmax = torch.nn.Softmax(dim=1) # Usare la funzione softmax per la classificazione

    def forward(self, x):
        h0 = torch.zeros(num_layers, x.size(0), hidden_size) # Inizializzare lo stato nascosto
        out, _ = self.rnn(x, h0) # Ottenere l'output della RNN
        out = out[:, -1, :] # Prendere solo l'ultimo output temporale
        out = self.fc(out) # Ottenere l'output dello strato lineare
        out = self.softmax(out) # Ottenere la probabilità di appartenenza alle due classi
        return out

# Creare un'istanza del modello
model = KeywordModel(input_size, hidden_size, output_size, num_layers)

# Definire la funzione di perdita e l'ottimizzatore
criterion = torch.nn.CrossEntropyLoss() # Usare la cross entropy come funzione di perdita
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # Usare Adam come algoritmo di ottimizzazione

# Definire una funzione per estrarre le caratteristiche MFCC da un file audio
def extract_features(file):
    waveform, sample_rate = torchaudio.load(file) # Caricare il file audio
    mfcc = torchaudio.transforms.MFCC(sample_rate)(waveform) # Calcolare le caratteristiche MFCC
    mfcc = mfcc.mean(0).transpose(0, 1) # Fare la media sulle dimensioni del canale e trasporre le dimensioni del tempo e delle caratteristiche
    return mfcc

# Definire una funzione per creare il dataset di addestramento e di test
def create_dataset(path, keyword):
    files = torchaudio.datasets.SPEECHCOMMANDS(path) # Caricare il dataset Speech Commands
    features = [] # Lista per memorizzare le caratteristiche MFCC
    labels = [] # Lista per memorizzare le etichette
    for file in files:
        if file[2] == keyword or file[2] == "_background_noise_": # Selezionare solo i file che contengono la parola di attivazione o il rumore di fondo
            feature = extract_features(file[0]) # Estrarre le caratteristiche MFCC dal file audio
            label = 1 if file[2] == keyword else 0 # Assegnare l'etichetta 1 se il file contiene la parola di attivazione, altrimenti 0
            features.append(feature)
            labels.append(label)
    features = torch.nn.utils.rnn.pad_sequence(features, batch_first=True) # Padding delle sequenze per avere la stessa lunghezza
    labels = torch.tensor(labels) # Convertire le etichette in tensori
    dataset = torch.utils.data.TensorDataset(features, labels) # Creare il dataset come coppia di tensori
    return dataset

# Creare il dataset di addestramento e di test
train_dataset = create_dataset("./data/SpeechCommands/speech_commands_v0.02", keyword) # Usare la cartella speech_commands_v0.02 come dataset di addestramento
test_dataset = create_dataset("./data/SpeechCommands/testing_list.txt", keyword) # Usare il file testing_list.txt come dataset di test

# Creare i dataloader per il batch
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # Mescolare i dati di addestramento
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False) # Non mescolare i dati di test

# Addestrare il modello
for epoch in range(num_epochs):
    model.train() # Impostare il modello in modalità di addestramento
    train_loss = 0 # Inizializzare la perdita di addestramento
    train_acc = 0 # Inizializzare l'accuratezza di addestramento
    for batch, (features, labels) in enumerate(train_loader):
        optimizer.zero_grad() # Azzerare i gradienti
        outputs = model(features) # Ottenere gli output del modello
        loss = criterion(outputs, labels) # Calcolare la perdita
        loss.backward() # Calcolare i gradienti
        optimizer.step() # Aggiornare i parametri
        train_loss += loss.item() # Aggiungere la perdita al totale
        _, preds = torch.max(outputs, 1) # Ottenere le predizioni
        train_acc += torch.sum(preds == labels).item() / len(labels) # Calcolare l'accuratezza
    train_loss = train_loss / len(train_loader) # Calcolare la perdita media
    train_acc = train_acc / len(train_loader) # Calcolare l'accuratezza media
    print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")

    model.eval() # Impostare il modello in modalità di valutazione
    test_loss = 0 # Inizializzare la perdita di test
    test_acc = 0 # Inizializzare l'accuratezza di test
    with torch.no_grad(): # Non calcolare i gradienti
        for features, labels in test_loader:
            outputs = model(features) # Ottenere gli output del modello
            loss = criterion(outputs, labels) # Calcolare la perdita
            test_loss += loss.item() # Aggiungere la perdita al totale
            _, preds = torch.max(outputs, 1) # Ottenere le predizioni
            test_acc += torch.sum(preds == labels).item() / len(labels) # Calcolare l'accuratezza
    test_loss = test_loss / len(test_loader) # Calcolare la perdita media
    test_acc = test_acc / len(test_loader) # Calcolare l'accuratezza media
    print(f"Epoch {epoch+1}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
