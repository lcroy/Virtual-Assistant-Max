import os

class Config:
    def __init__(self):
        # project path
        self.project_path = os.path.dirname(os.path.abspath(__file__))

        # models path
        self.model_path = os.path.join(self.project_path, 'models')
        self.main_intent_model_path = os.path.join(self.model_path, 'main_intent')
        self.main_intent_model_LSTM_path = os.path.join(self.main_intent_model_path, 'main.pt')
        self.main_intent_label_csv_path = os.path.join(self.main_intent_model_path, 'main_intent_label.tsv')
        self.main_intent_vocab_json = os.path.join(self.main_intent_model_path, 'vocab.json')
        # self.main_model_BERT_path = os.path.join(self.model_path, 'main_intent/main_intent.pt')
        self.mir_model_path = os.path.join(self.model_path, 'mir')

        # intent_recognition path
        self.predictor_path = os.path.join(self.project_path, 'services/language_service/intent_recognition')

        # Robot service
        # mir-------------------------------------------------------------------------------------
        self.MiR_host = 'http://192.168.12.20/api/v2.0.0/'
        self.headers = {'Content-Type': 'application/json', 'Accept-Language': 'en_US',
                        'Authorization': 'Basic YWRtaW46OGM2OTc2ZTViNTQxMDQxNWJkZTkwOGJkNGRlZTE1ZGZiMTY3YTljODczZmM0YmI4YTgxZjZmMmFiNDQ4YTkxOA=='}
        self.mir_path = os.path.join(self.predictor_path, 'mir')
        self.mir_dataset_path = os.path.join(self.mir_path, 'data')
        self.mir_task = 'mir'
        self.mir_intent_label = 'intent_label.txt'
        self.mir_slot_label = 'slot_label.txt'
        self.best_distance = 1.2

        self.main_intent_predictor_path = os.path.join(self.predictor_path, 'main_intent')
        # self.swarm_json = os.path.join(self.dataset_path, 'swarm/polybottest-firebase-key.json')

        # services
        self.service_list = os.path.join(self.project_path, 'services/service_list.json')
        self.required_slot = os.path.join(self.project_path, 'services/language_service/state_tracker/required_slot.json')
        self.confidence = 3.8

        # speak to text
        self.sample_rate = 16000
        self.chunk_size = 2048
        self.hint_sound = os.path.join(self.project_path, 'hint.mp3')

        # text to speech
        self.voice_id = 'Matthew'
        self.voice_path = os.path.join(self.project_path, 'voice_audio/')

        # conversation json file
        self.conv_json = os.path.join(self.project_path, 'static/data/conv.json')

        # trigger words
        self.trigger_word_max = ["max", "macs"]
        self.trigger_word_mir = ["mobile robot"]
        self.trigger_word_quit_mir = ['home', 'back']
        self.trigger_word_franka = ["franka", "franca", "frankia", "frank"]
        self.trigger_word_quit_franka = ['stop service']
        self.trigger_word_gpt = ["have a talk", "small talk", "chat"]
        self.trigger_word_quit_gpt = ['stop']
        

        # response template
        self.response_template = os.path.join(self.project_path, 'response_template/response_template.json')

        # Max client host
        # self.max_server_host = "http://172.27.15.18:5001/"
        self.max_server_host = "http://127.0.0.1:5000/"

        # robot control agent
        self.robot_control_agent_path = os.path.join(self.project_path, 'robot_control_agent')
        self.robot_service_execution_path = os.path.join(self.robot_control_agent_path, 'robot_service_execution')
        self.robot_service_management_path = os.path.join(self.robot_control_agent_path, 'robot_service_management')
        self.waiting_music_path = os.path.join(self.robot_control_agent_path, 'load.mp3')
        self.client_service_list_path = os.path.join(self.robot_service_management_path, 'service_list.json')

        #audio=====================================
        self.typing_audio_mp3 = os.path.join(self.project_path, 'static/audio/static_sounds_typing.mp3')
        self.typing_audio_script = os.path.join(self.project_path, 'static/audio/typing_audio.py')

        #open ai key===============================
        self.api_key = ""

        # LSTM Approach
        # parameters for LSTM Model
        self.embedding_dim = 100
        self.hidden_dim = 30
        self.droupout = 0.2
        self.num_class = 10
        self.batch_size = 30
        self.epochs = 30
        self.lr = 0.001
        # default sentence length
        self.input_length = 100
