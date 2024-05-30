# Message passing setup
from flask import Flask, request, jsonify, send_file
from hashlib import sha256
import base64

app = Flask(__name__)

# Text to Speech Setup
import torch
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import wave
import pyaudio
import time
import os

print("Loading TTS model...")
config = XttsConfig()
config.load_json("./models/XTTS-v2/config.json")
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir="./models/XTTS-v2/", use_deepspeed=True)
model.cuda()

print("Computing speaker latents...")
audio_list = []
for path in os.listdir("./audio_samples/mary/"):
    if path.endswith(".wav"):
        audio_list.append(os.path.join("./audio_samples/mary/", path))
placeholder = ["./test_1.wav", "./test_2.wav", "./test_3.wav"]
gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=placeholder)

# Functions

def text_to_speech(text):
    text = text.replace("\n", "")
    print("Starting:" + text)
    counter = -1 * time.time()
    out = model.inference(
    text,
    "en",
    gpt_cond_latent,
    speaker_embedding,
    temperature=0.7, # Add custom parameters here
    )
    torchaudio.save("ai_output.wav", torch.tensor(out["wav"]).unsqueeze(0), 24000) 
    print("Audio generated in " + str(time.time() + counter) + " seconds.")
    
    
@app.route('/tts', methods=['POST'])
def handle_tts():
    data = request.get_json()
    message = data.get('message')
    password = data.get('password')
    password = sha256(password.encode('utf-8')).hexdigest()
    print("Received " + message)
    if password == "d12e12eb84e22e182504f945c5235c9d0a8a3662709e6db222f9d31f41222b0a": 
        print("Password valid")
        text_to_speech(message)
        return send_file("./ai_output.wav", mimetype="audio/wav", as_attachment=True)
    else: 
        print("Password invalid")
        return jsonify({'error': 'Wrong password'}), 403

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6969)