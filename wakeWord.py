# Importare le librerie
import torch
import torchaudio
import numpy as np

# Caricare e preprocessare i dati
train_data = torchaudio.datasets.SPEECHCOMMANDS("./data", download=True, subset="training")
test_data = torchaudio.datasets.SPEECHCOMMANDS("./data", download=True, subset="testing")

# Creare una funzione per convertire i file wav in tensori
def wav2tensor(wav, label):
  # Normalizzare i valori tra -1 e 1
  wav = wav / torch.max(torch.abs(wav))
  # Aggiungere una dimensione di canale
  wav = wav.unsqueeze(0)
  # Creare un tensore binario per la label
  label = torch.tensor(label == "hey")
  return wav, label

# Applicare la funzione a tutti i dati
train_data = [(wav2tensor(wav, label)) for wav, sample_rate, label, speaker_id, utterance_number in train_data]
test_data = [(wav2tensor(wav, label)) for wav, sample_rate, label, speaker_id, utterance_number in test_data]

# Creare un dataloader per il batch
train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)

# Definire la rete neurale
class Net(torch.nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    # Primo strato convoluzionale con 16 filtri e kernel di dimensione 3
    self.conv1 = torch.nn.Conv1d(1, 16, 3)
    # Secondo strato convoluzionale con 32 filtri e kernel di dimensione 3
    self.conv2 = torch.nn.Conv1d(16, 32, 3)
    # Strato di pooling con dimensione 2
    self.pool = torch.nn.MaxPool1d(2)
    # Strato lineare con 64 neuroni
    self.fc1 = torch.nn.Linear(32 * 248, 64)
    # Strato lineare con 1 neurone
    self.fc2 = torch.nn.Linear(64, 1)
    # Funzione di attivazione sigmoide
    self.sigmoid = torch.nn.Sigmoid()

  def forward(self, x):
    # Applicare il primo strato convoluzionale e la funzione di attivazione ReLU
    x = torch.nn.functional.relu(self.conv1(x))
    # Applicare il secondo strato convoluzionale e la funzione di attivazione ReLU
    x = torch.nn.functional.relu(self.conv2(x))
    # Applicare il pooling
    x = self.pool(x)
    # Ridimensionare il tensore per il livello lineare
    x = x.view(-1, 32 * 248)
    # Applicare il primo strato lineare e la funzione di attivazione ReLU
    x = torch.nn.functional.relu(self.fc1(x))
    # Applicare il secondo strato lineare
    x = self.fc2(x)
    # Applicare la sigmoide per produrre l'output tra 0 e 1
    x = self.sigmoid(x)
    return x

# Creare un'istanza della rete
net = Net()

# Definire la funzione di perdita e l'ottimizzatore
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

# Addestrare il modello
for epoch in range(10):
  # Inizializzare la perdita e l'accuratezza
  running_loss = 0.0
  running_acc = 0.0
  # Iterare sui dati di train
  for i, data in enumerate(train_loader):
    # Ottenere il batch di input e label
    inputs, labels = data
    # Azzerare i gradienti
    optimizer.zero_grad()
    # Calcolare le previsioni
    outputs = net(inputs)
    # Calcolare la perdita
    loss = criterion(outputs, labels.float())
    # Calcolare i gradienti
    loss.backward()
    # Aggiornare i pesi
    optimizer.step()
    # Calcolare l'accuratezza
    acc = torch.sum((outputs > 0.5) == labels) / labels.shape[0]
    # Stampare le statistiche ogni 200 batch
    running_loss += loss.item()
    running_acc += acc.item()
    if i % 200 == 199:
      print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 200:.3f}, acc: {running_acc / 200:.3f}")
      running_loss = 0.0
      running_acc = 0.0

# Testare il modello
# Inizializzare la perdita e l'accuratezza
test_loss = 0.0
test_acc = 0.0
# Disabilitare il calcolo dei gradienti
with torch.no_grad():
  # Iterare sui dati di test
  for data in test_loader:
    # Ottenere il batch di input e label
    inputs, labels = data
    # Calcolare le previsioni
    outputs = net(inputs)
    # Calcolare la perdita
    loss = criterion(outputs, labels.float())
    # Calcolare l'accuratezza
    acc = torch.sum((outputs > 0.5) == labels) / labels.shape[0]
    # Aggiungere la perdita e l'accuratezza
    test_loss += loss.item()
    test_acc += acc.item()
# Stampare le statistiche
print(f"Test loss: {test_loss / len(test_loader):.3f}, Test acc: {test_acc / len(test_loader):.3f}")
