from flask import Flask, request, jsonify, send_file
from hashlib import sha256
import base64
import numpy
import torch
import torchaudio
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline
from llama_cpp import Llama
import wave
import pyaudio
import time
import wave
from TTS.api import TTS
import pyaudio

# Text-to-Speech streaming setup
print("Loading TTS model...")
print(TTS().list_models())
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True).to("cuda")

print("Computing speaker latents...")
gpt_cond_latent, speaker_embedding = tts.get_conditioning_latents(audio_path=["reference.wav"])

def text_to_speech_streaming(text):
    print("Starting TTS streaming...")
    t0 = time.time()
    chunks = tts.inference_stream(
        text,
        "en",
        gpt_cond_latent,
        speaker_embedding
    )

    wav_chunks = []
    for i, chunk in enumerate(chunks):
        if i == 0:
            print(f"Time to first chunk: {time.time() - t0}")
        print(f"Received chunk {i} of audio length {chunk.shape[-1]}")
        wav_chunks.append(chunk)

    wav = torch.cat(wav_chunks, dim=0)
    output_path = "./ai_output.wav"
    torchaudio.save(output_path, wav.squeeze().unsqueeze(0).cpu(), 24000)
    return output_path

while True:
    text_to_speech_streaming(input("Testing: "))