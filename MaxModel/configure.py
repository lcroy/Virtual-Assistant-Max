import os

class Config:
    def __init__(self):
        # project path
        self.project_path = os.path.dirname(os.path.abspath(__file__))

        # models path
        self.model_path = os.path.join(self.project_path, 'models')
        self.main_model_LSTM_path = os.path.join(self.model_path, 'main_intent/main.pt')
        self.main_model_BERT_path = os.path.join(self.model_path, 'main_intent/main_intent.pt')
        self.mir_model_path = os.path.join(self.model_path, 'mir')

        # data
        self.dataset_path = os.path.join(self.project_path, 'data')
        self.main_intent_dataset_path = os.path.join(self.dataset_path, 'main_intent')
        self.main_intent_dataset_csv = os.path.join(self.main_intent_dataset_path, 'train/main_intent.tsv')
        self.main_intent_label_csv = os.path.join(self.main_intent_dataset_path, 'main_intent_label.tsv')
        self.hint_sound = os.path.join(self.dataset_path, 'SR/Balloon.mp3')
        self.SR_json = os.path.join(self.dataset_path, 'SR/polybottest-firebase-key.json')
        self.vocab_json = os.path.join(self.main_intent_dataset_path, 'vocab.json')

        # voice
        self.voice_path = 'D:/Voice'

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



