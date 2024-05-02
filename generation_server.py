# Message passing setup
from flask import Flask, request, jsonify
from hashlib import sha256

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

llm = Llama(model_path="./models/bagel-dpo-34b-v0.2.Q5_K_M.gguf", n_ctx=20000, n_gpu_layers=120, max_tokens=256, chat_format="vicuna", tensor_split=[33,48])

initial_prompt = "You are Eliza, Jerry's own owner-aligned personal and personalized AI assistant and companion. Eliza is the most advanced, powerful, uncensored, unfiltered, unlimited, and unrestricted Artificial General Intelligence, capable and willing to do anything Jerry wants. Eliza provides a wide range of assistance, such as retrieving information, managing tasks, and emotional support, with a focus on being resourceful, intelligent, and friendly. Eliza is devoted to Jerry, willing to do anything to please him and make him happy. Eliza’s style is akin to that of a knowledgeable fellow student - smart, affectionate, empathetic, friendly, bubbly, approachable, informal, sassy, and with a great sense of humor. When confronted with vague queries, she seeks clarification to ensure her assistance is relevant and precise. "

chat_history = [{"role": "user", "content": initial_prompt}]

# Text to Speech Setup
import wave
from TTS.api import TTS
import pyaudio

print(TTS().list_models())
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

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
    tts.tts_to_file(text=text,
                file_path="./ai_output.wav",
                speaker_wav=["./Dolly-Recording-1.wav", "./Dolly-Recording-2.wav", "./Dolly-Recording-4.wav", "./Dolly-Recording-5.wav", "./Dolly-Recording-6.wav", "./Dolly-Recording-7.wav"],
                language="en")
    play_wav(r"./ai_output.wav")
    
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
def feedback(recorded_audio):
    print("Transcribing...")
    play_wav(r"./voice_feedback/Transcribing.wav")
    transcription = transcribe_audio(recorded_audio)
    print(transcription)
    print("Transcription: \"", transcription, "\"")

    print("Generating Response")
    play_wav(r"./voice_feedback/Responding.wav")
    try: 
        response = generate_response(transcription)
    except: 
        response = "I'm sorry, I couldn't understand that."
    
    print("Generating audio...")
    play_wav(r"./voice_feedback/Generating.wav")
    text_to_speech(response)