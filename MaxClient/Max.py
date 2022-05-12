import speech_recognition as sr
import random
import playsound
import pyttsx3
import time
import requests
import pygame
import subprocess

from configure import Config
from update_conversation import *
from robot_control_agent.call_other_service import call_other_service
from robot_control_agent.call_gpt import call_gpt2, call_gpt3
from robot_control_agent.robot_service_execution.mir.control_MiR import call_mir

class Max:

    def __init__(self, cfg):
        self.sample_rate = cfg.sample_rate
        self.chunk_size = cfg.chunk_size
        self.hint_sound = cfg.hint_sound
        self.voice = cfg.voice_id
        self.converter = pyttsx3.init()
        self.converter.setProperty('rate', 150)
        self.converter.setProperty('volume', 0.7)
        self.host = cfg.max_server_host
        self.headers = {'Content-Type': 'application/json', 'Accept-Language': 'en_US'}

    # Google service to recognize the speech
    def speech_to_text_google(self):
        # Set American English
        r = sr.Recognizer()
        with sr.Microphone() as source:
            # Adjusts the energy threshold dynamically using audio from source (an AudioSource instance) to account for ambient noise.
            print("Please wait one second for calibrating microphone...")
            r.pause_threshold = 0.8
            r.dynamic_energy_threshold = True
            r.adjust_for_ambient_noise(source, duration=1)
            print("Ok, microphone is ready...")
            # p = vlc.MediaPlayer(self.hint_sound)
            # p.play()
            playsound.playsound(self.hint_sound, True)
            audio = r.listen(source, timeout = None)
            transcript = ""
            try:
                transcript = r.recognize_google(audio, language="en-US")
                print('You: ' + transcript)
            except:
                print('Max: I did not hear anything....')

        return transcript.lower()

    #  pyttsx3 text to speech
    def text_to_speech_local(self, text):
        print('Max: ' + text)
        self.converter.say(text)
        self.converter.runAndWait()


    # Call max service
    def get_response(self, text, requested_service, client_slot_result):
        parameters = {'message':text, 'requested_service': requested_service, 'client_slot_result':client_slot_result}
        result = requests.get(self.host + 'get_service/', params=parameters, headers=self.headers)

        return result.json()

    def get_file(self):
        result = requests.get(self.host + 'download', headers=self.headers)

        return result.json()

    def call_max(self, cfg):
        # new a configure file
        cfg = Config()
        update_user(cfg, "...")
        update_service(cfg, "home")
        update_max(cfg, "Waiting operator's command...")
        #read response template from json file
        with open(cfg.response_template) as json_file:
            response_template = json.load(json_file)
        while True:
            # 1. check the trigger word
            print("You may talk to Max now...")
            text = self.speech_to_text_google().casefold()
            # if it detects speech...
            if len(text) > 0:
                update_user(cfg, text)
            # if system detects the trigger word - Max
            if any(key in text.casefold() for key in cfg.trigger_word_max):
                text = text.replace("Macs", "Max")
                text = random.choice(response_template['init_speak'])
                update_max(cfg, text)
                self.text_to_speech_local(text)
                while True:
                    # 2. wait for human command
                    text = self.speech_to_text_google().casefold()
                    # BotX does heard something not some random noise
                    if len(text) > 0:
                        # call mir service
                        if any(key in text.casefold() for key in cfg.trigger_word_mir):
                            text = text.replace(str(["mia","Mia"]), "MiR")
                            update_user(cfg, text)
                            update_mir(cfg, "Unknow...", "Unknow...", "Waiting...")
                            call_mir(self, cfg, 'mir', response_template)
                            continue

                        # call franka service
                        if any(key in text.casefold() for key in cfg.trigger_word_franka):
                            text = text.replace(str(["franka", "franca", "frankia", "frank"]), "Franka")
                            update_user(cfg, text)
                            update_franka(cfg, "Unknow...", "Unknow...", "Waiting...")
                            call_franka(self.cfg)
                            continue

                        # call GPT for a small talk
                        if any(key in text.casefold() for key in cfg.trigger_word_gpt):
                            update_user(cfg, text)
                            # call_gpt2(self, cfg, text)
                            call_gpt3(self, cfg, text)
                            continue

                        # call other service (remember if you are asking other service not registered in Max client,
                        # then it will ask you to confirm and checking the server side)
                        update_user(cfg, text)
                        call_other_service(self, cfg, text, response_template)
                        continue



