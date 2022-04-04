from transformers import pipeline, Conversation
from update_conversation import *
import openai


def call_gpt2(max, cfg, text):

    conversational_pipeline = pipeline("conversational")
    max.text_to_speech_local("Ok, let's have a chat then. This is supported by GPT2")
    change_topic = 0
    pre_response = ''
    conv1_start = max.speech_to_text_google()
    conv1 = Conversation(conv1_start)
    result = str(conversational_pipeline([conv1]))
    print(result)
    index = result.rfind('>>')
    max.text_to_speech_local(result[index+2:])
    print("===========================================")
    while True:
        conv1_2 = max.speech_to_text_google()

        if len(conv1_2)>0:
            if any(key in text.casefold() for key in cfg.trigger_word_quit_gpt2):
                max.text_to_speech_local(
                        "Sure, nice to talk to you.")
                break
            else:
                if change_topic == 0:
                    conv1.add_user_input(conv1_2)
                else:
                    conv1 = Conversation(conv1_2)
                    change_topic = 0
                    pre_response = ''
                result = str(conversational_pipeline([conv1]))
                print(result)
                index = result.rfind('>>')
                print("===========================================")
                if len(result[index + 2:]) <= 3:
                    max.text_to_speech_local("Sorry, I cannot support such deeper conversation at this moment. Can we talk something else?")
                    change_topic = 1
                    continue
                if pre_response == result[index + 2:]:
                    max.text_to_speech_local(
                        "Well, my answer is the same as the previous. Please ask me something else instead.")
                    change_topic = 1
                    continue
                else:
                    pre_response = result[index + 2:]
                max.text_to_speech_local(result[index + 2:])

def call_gpt3(max, cfg, text):
    # Max responses to the small talk
    max_response = "Ok, let's have a chat then. This is supported by GPT3"
    update_max(cfg, max_response)
    max.text_to_speech_local(max_response)
    openai.api_key = cfg.api_key
    while True:
        # waiting for operator's command...
        text = max.speech_to_text_google()
        update_user(cfg, text)
        # quit the small talk service
        if any(key in text.casefold() for key in cfg.trigger_word_quit_gpt):
            text = "Sure, nice to talk to you."
            update_max(cfg, text)
            max.text_to_speech_local(text)
            break

        response = openai.Completion.create(
            engine="davinci",
            prompt= "Human:" + text + "\nAI:",
            temperature=0.9,
            max_tokens=150,
            top_p=1,
            frequency_penalty=0.0,
            presence_penalty=0.6,
            stop=["\n", " Human:", " AI:"]
        )
        print(response)
        if len(response['choices'][0]['text'])>0:
            max_response = response['choices'][0]['text']
        else:
            max_response = "I have no comments on this."
        update_max(cfg, max_response)
        max.text_to_speech_local(max_response)