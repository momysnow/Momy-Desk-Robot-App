from bark import SAMPLE_RATE, generate_audio, preload_models
from pydub import AudioSegment
from pydub.playback import play

# download and load all models
preload_models()

# generate audio from text
text_prompt = """
     Hello, my name is Suno. And, uh — and I like pizza. [laughs] 
     But I also have other interests such as playing tic tac toe.
"""
speech_array = generate_audio(text_prompt)

# convert NumPy array to AudioSegment
audio_segment = AudioSegment(
    speech_array.tobytes(),
    frame_rate=SAMPLE_RATE,
    sample_width=speech_array.dtype.itemsize,
    channels=1
)

# play audio
play(audio_segment)
