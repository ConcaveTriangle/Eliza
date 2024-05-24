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

# Start the audio playback thread
playback_thread = threading.Thread(target=audio_playback_worker, daemon=True)
playback_thread.start()

def detected_wake_word():
    pcm = audio_stream.read(handle.frame_length, exception_on_overflow=False)
    pcm = numpy.frombuffer(pcm, dtype=numpy.int16)
    keyword_index = handle.process(pcm)
    if keyword_index >= 0:
        return True
    return False

def record_audio(duration=10):
    frames = []
    vad_buffer = bytearray()  # Buffer to accumulate audio for VAD
    vad_frame_length = int(handle.sample_rate * 0.03)  # 30 ms frame length for VAD

    start_time = time.time()
    while time.time() - start_time < duration:
        try:
            frame = audio_stream.read(handle.frame_length, exception_on_overflow=False)
            frames.append(frame)

            vad_buffer.extend(frame)
            if len(vad_buffer) >= vad_frame_length * 2:  # 2 bytes per sample
                vad_frame = vad_buffer[:vad_frame_length * 2]
                vad_buffer = vad_buffer[vad_frame_length * 2:]
                is_speech = vad.is_speech(vad_frame, handle.sample_rate)

                if not is_speech and time.time() - start_time > 1:  # Stop if silence detected
                    break
        except IOError as e:
            # Handle the overflow error or logging here
            print("Buffer overflow handled:", e)

    return b''.join(frames)

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
    
def play_wav(path):
    f = wave.open(path, "rb")
    chunk = 1024
    data = f.readframes(chunk)
    p = pyaudio.PyAudio()  
    stream = p.open(format = p.get_format_from_width(f.getsampwidth()),  
                channels = f.getnchannels(),  
                rate = f.getframerate(),  
                output = True) 
    while data:
        stream.write(data)
        data = f.readframes(chunk)
    stream.stop_stream()
    stream.close()
    p.terminate()


print("Listening for wake word...")
while True:
    if detected_wake_word():
        print("Wake word detected.")

        # Create and start a thread for playing the listening WAV file
        listening_thread = threading.Thread(target=play_wav, args=(r"./voice_feedback/Listening.wav",))
        listening_thread.start()

        # Create and start a thread for recording audio
        recorded_audio = None
        def record_audio_thread():
            time.sleep(1)
            global recorded_audio
            recorded_audio = record_audio()

        recording_thread = threading.Thread(target=record_audio_thread)
        recording_thread.start()

        # Wait for both threads to finish
        listening_thread.join()
        recording_thread.join()

        print("Sending audio to chatbot...")

        encoded_audio = base64.b64encode(recorded_audio).decode('utf-8')
        response_messages = send_to_chatbot(encoded_audio, "ConcaveTriangle", "inference").decode('utf-8')
        response_messages = json.loads(response_messages)["response"]

        print(response_messages)
        
        for sentence in response_messages:
            print(sentence)
            response_audio = send_to_chatbot(sentence, "ConcaveTriangle", "tts")
            audio_queue.put(response_audio)
            with open("ai_output.wav", "wb") as f:
                f.write(response_audio)

        audio_queue.join()
