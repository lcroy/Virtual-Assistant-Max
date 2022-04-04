import pygame
from configure import Config

def loop_typing_sound(audio_path):
    i = 0
    pygame.mixer.init()
    pygame.mixer.music.load(audio_path)
    pygame.mixer.music.play(-1)  # note -1 for playing in loops
    while True:
         pass

cfg = Config()
loop_typing_sound(cfg.typing_audio_mp3)
