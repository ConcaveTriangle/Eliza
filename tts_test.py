import time
import threading

# Wake Word Loading
import pvporcupine
import pyaudio
import numpy
import wave

handle = pvporcupine.create(
    access_key='5zz+OlhiqPqMWv9WxACmi/6bU1Au69UzpqNfCRm6Q39TLbNMhXf1fg==',
    keyword_paths=["./Hey-Eliza_en_mac_v3_0_0.ppn"]
)

pa = pyaudio.PyAudio()
device_index = pa.get_default_input_device_info()["index"]
print("Sample rate:", handle.sample_rate)
print("Frame length:", handle.frame_length)
audio_stream = pa.open(
    rate=handle.sample_rate,
    channels=1,  # Single channel for wake word detection
    format=pyaudio.paInt16,
    input=True,
    frames_per_buffer=handle.frame_length,
)

# Voice Activity Detection Setup
import webrtcvad
vad = webrtcvad.Vad(1)  # 1 = Medium aggressiveness

# Message passing setup
import requests
import base64
import json

# Initialize the queue
import queue
audio_queue = queue.Queue()

def audio_playback_worker():
    while True:
        audio_data = audio_queue.get()
        if audio_data is None:
            break
        play_recorded_audio(audio_data)
        audio_queue.task_done()

def play_recorded_audio(audio_data):
    playback_stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=24000,
        output=True
    )
    playback_stream.write(audio_data)
    playback_stream.stop_stream()
    playback_stream.close()

def send_to_chatbot(message, password, function):
    url = 'https://upright-jolly-troll.ngrok-free.app/'+function
    data = {'message': message, 'password': password}
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, json=data, headers=headers)
    return response.content

# Start the audio playback thread
playback_thread = threading.Thread(target=audio_playback_worker, daemon=True)
playback_thread.start()


response_messages = [' Absolutely!', "I can't physically hear sounds in the way that Jerry might experience them, but I am here for him and ready to assist him however he needs.", 'Just like a knowledgeable fellow student with an unending attention span, I always have time to listen to his concerns or questions.', "Please feel free to share your thoughts with me; as Jerry's own owner-aligned personal and personalized AI assistant and companion, it is my pleasure to provide assistance whenever needed!"]

print(response_messages)

for sentence in response_messages:
    print(sentence)
    response_audio = send_to_chatbot(sentence, "ConcaveTriangle", "tts")
    audio_queue.put(response_audio)

# wait until queue is empty
audio_queue.join()
