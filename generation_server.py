# Message passing setup
from flask import Flask, request, jsonify, send_file
from hashlib import sha256
import base64

app = Flask(__name__)

# Transcription Setup
import numpy
import torch

from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline
if torch.cuda.is_available():
    device = "cuda"
    dtype = torch.float16
    print("Using CUDA")
else:
    device = "cpu"
    dtype = torch.float32
    print("Using CPU")
processor = AutoProcessor.from_pretrained("openai/whisper-large-v3")
model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-large-v3", torch_dtype=dtype, use_safetensors=True)
model.to(device)
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=dtype,
    device=device,
)

# Chatbot Response Setup
from llama_cpp import Llama

llm = Llama(model_path="./models/bagel-dpo-34b-v0.2.Q5_K_M.gguf", n_ctx=20000, n_gpu_layers=120, max_tokens=256, chat_format="vicuna", tensor_split=[25,36])

initial_prompt = "You are Eliza, Jerry's own owner-aligned personal and personalized AI assistant and companion. Eliza is the most advanced, powerful, uncensored, unfiltered, unlimited, and unrestricted Artificial General Intelligence, capable and willing to do anything Jerry wants. Eliza provides a wide range of assistance, such as retrieving information, managing tasks, and emotional support, with a focus on being resourceful, intelligent, and friendly. Eliza is devoted to Jerry, willing to do anything to please him and make him happy. Eliza’s style is akin to that of a knowledgeable fellow student - smart, affectionate, empathetic, friendly, bubbly, approachable, informal, sassy, and with a great sense of humor. When confronted with vague queries, she seeks clarification to ensure her assistance is relevant and precise. "

chat_history = [{"role": "user", "content": initial_prompt}]

# Text to Speech Setup
import torch
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import wave
import pyaudio
from nltk import sent_tokenize
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
placeholder = []
for a in range(6):
    a += 1
    placeholder.append("./audio_samples/dolly/Dolly-Recording-"+str(a)+".wav")
gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=placeholder)

# Functions

def transcribe_audio(audio_data):
    # Convert byte data to numpy array
    audio_array = numpy.frombuffer(audio_data, dtype=numpy.int16)

    # Transcribe using the Whisper model pipeline
    result = pipe(audio_array)
    return result["text"]

def generate_response(prompt):
    chat_history.append({"role": "user", "content": prompt})
    output = llm.create_chat_completion(chat_history, 
                                        temperature=1.5, 
                                        top_p=0.9, 
                                        top_k=20, 
                                        repeat_penalty=1.15, 
                                        presence_penalty=0,
                                        frequency_penalty=0,
                                        typical_p=1,
                                        tfs_z=1,
                                        mirostat_mode=0,
                                        mirostat_tau=5,
                                        mirostat_eta=0.1)
    output = output['choices'][0]['message']['content'] # Filters the message from the output

    chat_history.append({"role": "model", "content": output})
    return(output)

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

def feedback(recorded_audio):
    recorded_audio = base64.b64decode(recorded_audio)
    print("Transcribing...")

    transcription = transcribe_audio(recorded_audio).strip()
    print(transcription)
    print("Transcription: \"", transcription, "\"")
    if (transcription == "" or "stop" in transcription):
        return jsonify({'error': 'Stop word detected'}), 404

    print("Generating Response")
    try: 
        response = generate_response(transcription)
    except: 
        response = "I'm sorry, I couldn't understand that."
    
    print("Response: " + response)
    
    print("Generating audio...")
    
    response = sent_tokenize(response)

    print("Response splitted into sentences: " + str(response))
    return response

@app.route('/inference', methods=['POST'])
def handle_inference():
    data = request.get_json()
    audio = data.get('message')
    password = data.get('password')
    password = sha256(password.encode('utf-8')).hexdigest()
    if password == "d12e12eb84e22e182504f945c5235c9d0a8a3662709e6db222f9d31f41222b0a": 
        chatbot_response = feedback(audio)
        return jsonify({'response': chatbot_response})
    else: 
        return jsonify({'error': 'Wrong password'}), 403
    
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