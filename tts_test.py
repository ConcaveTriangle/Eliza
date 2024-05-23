import torch
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import time
import os

print("Loading model...")
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
for a in range(10):
    a += 1
    placeholder.append("./audio_samples/mary/Mary-Recording-"+str(a)+".wav")
gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=placeholder)

print("Inference...")
def text_to_speech(text_list):
    final = []
    for text in text_list:
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
        final.append(torch.tensor(out["wav"]).unsqueeze(0))
        print("Response generated in " + str(time.time() + counter) + " seconds.")
    final = torch.cat(final, dim=1)
    torchaudio.save("ai_output.wav", final, 24000)

text_to_speech(['i sent twon planes into the twin towers!'])