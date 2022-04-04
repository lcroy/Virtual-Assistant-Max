########################################################################################################################
# Knowledge repository for Virtual Assistant Bot-X
# Date: 29/9/2019 -
########################################################################################################################
# trigger word detection
trigger_word = ["max","macs"]
MiR_word = ["mir", "mia", "me"]

# Initial speaking
init_speak = ["hey, this is Max, what can I do for you.",
              "Yes, I am ready.",
              "what's up.",
              "Max is here. How can I help."]

init_speak_mia = ['Hello, MiR services are running now. I am ready to work.']

mia_key_words = {'battery': 'battery', 'target': 'warehouse'}

mia_response = {
    "battery": ["Mir's battery level is ", "Here is MiR's battery info "],
    "greeting":["All good. Thanks.", "Not bad. Thank you."],
    "mirfree":["Well, MiR is totally free now.", "em, nothing on my schedule."],
    "mirbusy":["Well, MiR is on the way to ", "Yeah, MiR is on the way to "],
    "mirfeetohelp":["Sure, how can I help you?", "Yes, tell me what you need?"],
    "mirbusybuthelp":["Well, I am currently running on a mission. But sure, how can I help you.", "em, a bit busy right now. But what can I do for you then."],
    "mirnottalkinmission":["Sorry, I can not do the small talk now. If you don't have any mission want me to do.", "Well, can I catch up with you later. I need to carry on my mission."],
    "delivery_person":["Can you repeat the name please.", "Sorry, I did not catch the name, can you repeat."],
    "delivery_object":["Sorry, what you want me to delivery? Can you repeat?", "Sorry, can you repeat what you need me to send?"],
    "delivery_color_or_size":["Sorry, I did not hear clearly, Can you describe a bit of it? the color and size.", "Sorry, would you please repeat the color and size of it, please"],
    "delivery_des": ["Sorry, I did not hear clearly. Can you repeat the destination?"],
    "delivery_des_not_reach": ["Sorry, the destination is not reachable. Please choose another one.", "Unfortunately, the destination is not registered in the system. You need to choose another one."],
    "delivery_order":["Ok, I add it on my schedule. I will ", "Got it. I will "],
    "go_position": ["ok, wait me here and I will back in a second.", "will do."],
    "name_position": ["Sure, but I need your give me a name for this place.", "No problem, how would you like me to call this position."],
    "create_mission_q_name": ["Ok, name your mission first", "Sure, give me a name for your mission"],
    "create_mission_q_pos": ["Sure, mission is created. but do you have pre-defined position?", "Ok, I just create a mission for you. have you created positions yet?"],
    "create_mission_act": ["Ok, tell me which position you would like to link to a moving action.", "Sure, give me the name of the position. I will make a link with an action."],
    "create_mission_link_to_position": ["ok, the action is created and the position is linked to the mission. More action?", "Job is done. Would you like to add another action?"],
    "execute_mission": ["Sure, which mission you want me to execute?", "Ok, give me the mission name please."],
    "location": ["sure, Let me check...Well, MiR currently is close to ", "well, MiR is close to ", "I believe MiR is near the ", "Ok, wait for a second, MiR is close to "],
    "wait": ["Sure, what is it ?", "Yes, I am listening. what do you want", "Ok, I will let MiR stop."],
    "continue": ["Ok, have a nice day.", "see you then."]
}

mia_done_task = ['Ok, done!', "great, it is done!"]

###################################################################################

# botx normal response
Botx_Res = {
    # greeting
    "greeting": ["Good to see you again! How about you.",
                 "hi I am pretty good. Thanks. I hope you are doing great as well.",
                 "not bad, thanks. How are you",
                 "I am ok, thank you. Anything I can help you today.",
                 "Still alive, just a joke. Thanks. How was your day",
                 "Well, depends, a bit tired. But how about you"],
    # Who made Jason.
    "who_made_you": ['Well, I was born in AAU Robotics and Automation group.',
                     'I am from the smartest research group. Robotics and Automation.'],
    # goodbye
    "goodbye": ["Sad to see you go",
                "Sure, talk to you later",
                "Goodbye!",
                "Ok, Enjoying your day"],
    # age
    "how_old_are_you": ["That's the secret I never tell.",
                        "Well, I guess I am younger than you.",
                        "technically, less than one-year old. So, I am still a baby."],
    # joke
    "tell_joke": ["If the opposite of pro is con, isn't the opposite of progress, congress?",
                  "What do you call a guy with no arms or legs floating in the ocean? Bob",
                  "A bishop, a priest, and a Rabbi walk into a bar. The bartender looks at them and says, What is this, a joke?"],
    # bot
    "are_you_a_bot": ['What do you think? Yes, I am.',
                      'Yes, I am a virtual assistant.',
                      ' Yes, I am a bot.'],
    # what_are_your_hobbies
    "what_are_your_hobbies": ['Well, I like reading, talking and learning',
                              'I do have a lot of hobbies but I like work with human most.'],
    # webot UR5
    "webot_UR": ["Sure, let's start it. Give me a second to connect with the simulator. Just let you know, I can do "
                 "simulator now but I will be able to control the real bot soon.",
                  "Ok, let's start to work then. what's your first command. I just need a second to connect with simulator."]
}

# BotX do not understand or something wrong
botx_do_not_know = ["sorry, I think I did not hear that clearly.",
                    "I might miss understanding something here",
                    "Well, I am not sure if I understand."]



# Authoritarian
AUTHADMIN = ["Chen"]
AUTHGUEST = ["Jinha", "Hahyeon", "Dimitri", "Simon"]
AUTHNO = ['Sorry, access denied. Please try to contact to the administrator.',
          'Well, you are not authorized to use virtual assistant. Please contact to the administrator.']

# BotX guess
GUESSING = ["I am guessing you want to ask something related to ",
            "Well,I can see you want to know something related to ",
            "Sorry, hard to catch what you said, are you asking something relatd to "]

# General
MISSUNDERSTANDING = ["sorry, I think I did not hear that clearly", "I might miss understanding something here",
                     "Well, I am not sure if I understand."]
YESORNO = ["Please just simply answer yes or no."]
GUESSING = ["probably I am saying a wrong things, but here is my response",
            "Sorry, I might get the wrong intents, but this is my reply",
            "Hope I don't miss understand you, here is my answer"]
DONOTUNDERSTAND = ["Sorry, I don't understand.", "Well, I don't know what's you said"]
REPEAT = ["Please repeat. Thanks", "Can you say it again, thanks", "I might need you to say it again. Sorry"]

# emotion
EMOTIONSAD = ['Well, what I can tell is you are a bit sad', "Well, you looks sad. I hope everything is ok.",
              "Well, you looks a little depress. Is everything ok?"]
EMOTIONHAPPY = ['Ok, you looks happy. I guess something good happens to you',
                "Wow, is there some good news. you looks glad.", "You do looks happy. I am glad"]

# stop
CALLFORJASON = 'Jason'
STOP = 'stop'
YES = 'yes'
LAURALETJASONSPEAK = ['Sure, just wait for a second', 'Yes, hold on', 'Well, he will be here soon']
JASONOPENING = ['Hi, this is Jason. How can I help you.',
                'hello, I am here. nice to talk to you again. what can I do for you']
JASONLETLAURASPEAK = ['ok, I will transfer to the Laura then', 'sure, I will let Laura take over here',
                      'ok, Laura will be back']

#  sentiment
STARTSENTIMENT = ['Sure, which topic you would like to check', 'No problem, tell the topic you want to know',
                  'Ok, what do you want to search?']
ANSWERSENTIMENT = ['So, here is what I got.', 'Well, here are the results.']
QUESTIONSENTIMENT = ['Would you like to know more details?',
                     'I can tell you more details of the results. Would you like to know?']
ANSWERPOS = ['The positive results are', 'Here are some positive results']
ANSWERNEG = ['The negative results are', 'Here are some negative results']
ANSWERNEU = ['The Neutral results are', 'Here are some Neutral results']
QUESTIONSENTIMENTCONTINUE = ['would you like to continue', 'do you want to check other topics']
