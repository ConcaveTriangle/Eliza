import os
import torch
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

print("Loading model...")
config = XttsConfig()
config.load_json("./models/XTTS-v2/config.json")
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir="./models/XTTS-v2/", use_deepspeed=True)
model.cuda()

print("Computing speaker latents...")
gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=["./audio_samples/Dolly-Recording-1.wav", "./audio_samples/Dolly-Recording-2.wav", "./audio_samples/Dolly-Recording-4.wav", "./audio_samples/Dolly-Recording-5.wav", "./audio_samples/Dolly-Recording-6.wav", "./audio_samples/Dolly-Recording-7.wav"])

print("Inference...")
out = model.inference(
    "It took me quite a long time to develop a voice and now that I have it I am not going to be silent.",
    "en",
    gpt_cond_latent,
    speaker_embedding,
    temperature=0.7, # Add custom parameters here
)
torchaudio.save("ai_output.wav", torch.tensor(out["wav"]).unsqueeze(0), 24000)