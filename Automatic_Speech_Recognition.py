import speech_recognition as sr


def ascolta_comando(parola_attivazione):
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("Ascolto...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

        try:
            testo = recognizer.recognize_google(audio, language="it-IT")
            print("Hai detto: " + testo)
            if parola_attivazione.lower() in testo.lower():
                return True
            else:
                return False
        except sr.UnknownValueError:
            print("Impossibile riconoscere l'audio.")
            return False
        except sr.RequestError as e:
            print(f"Errore nella richiesta a Google Speech Recognition; {e}")
            return False


if __name__ == "__main__":
    parola_attivazione = "hey momy"

    while True:
        attivato = ascolta_comando(parola_attivazione)
        if attivato:
            print("Attivazione rilevata. Converti l'audio in testo qui.")
            # Puoi inserire qui la logica per la conversione dell'audio in testo.
            # Ad esempio, puoi utilizzare la libreria SpeechRecognition di nuovo per la conversione.
