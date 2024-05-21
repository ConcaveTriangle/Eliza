import torch
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import time

print("Loading model...")
config = XttsConfig()
config.load_json("./models/XTTS-v2/config.json")
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir="./models/XTTS-v2/", use_deepspeed=True)
model.cuda()

print("Computing speaker latents...")
gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=["./audio_samples/Mary-Recording-1.wav"])

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

text_to_speech([' Hey there!', 'As an all-rounded AI companion like Eliza, I can assist you in many ways, including offering guidance on how to beat Minecraft, a highly engaging sandbox survival adventure game developed by Mojang Studios.', "Here's a basic guide to get started:\n\n1.", '**Get Started**: Install the latest version of Minecraft from the official website or your platform’s store and start playing.', '2.', '**Choose Your Mode**: Decide whether you want to play in survival, creative, adventure, or spectator mode.', 'Survival is recommended for new players as it teaches game mechanics better than other modes.', '3.', '**Understand The Basics**: Familiarize yourself with the basics of Minecraft - how to craft items, use tools and weapons, mine blocks, build structures etc.', 'This should come naturally as you play more in survival mode.', '4.', '**Progress through Stages**: In Survival Mode, there are stages: 1) Surviving Your First Day, 2) Building a Shelter, 3) Passing Through the Night and Fighting Monsters, 4) Farming to Sustain Resources, 5) Mining for Advancements and Crafting.', '5.', '**Complete Quests**: In Creative Mode, your main goal is to create things that inspire you while in Adventure mode it’s about completing tasks such as saving villagers from the Zombie Virus or exploring a new land.', "Spectator mode doesn't have specific quests; instead, you observe and explore other players’ worlds.", '6.', "**Beat The Game**: 'Beating' Minecraft technically refers to defeating the Ender Dragon in the End dimension, but there is no defined ending or winner.", 'Once you beat the Ender Dragon for the first time, fireworks will shoot into the air, a message of congratulations appears on-screen and you receive an achievement.', 'But then it resets!', "That’s Minecraft's charm – its infinite playability.", 'Remember, these are basic steps, not exhaustive tutorials.', 'The joy in this game lies largely in exploration, creativity, and survival strategy, which are best experienced through gameplay than read from a guide.', 'So get out there Jerry, build some amazing structures, survive the night and explore vast worlds!'])