from train.main_intent.train_BERT import predict
from configure import Config
import time

cfg = Config()

def intent_identify(text):
    intent_label = predict(cfg.main_model_BERT_path, text, cfg.main_intent_label_csv)
    print("The predict intent is " + intent_label[0] + " and the confidence is " + str(intent_label[1]))

if __name__ == "__main__":
    start_time = time.time()
    # intent_identify('can you say it again')
    # print("---The BERT model costs {:.3f} seconds to predict the final results---".format((time.time() - start_time)))

    print("Response time for BERT model:")

    print("Intent - Greeting")
    intent_identify('Hi max, how are you today')
    print("---The BERT model costs {:.3f} seconds to predict the final results---".format((time.time() - start_time)))

    start_time = time.time()
    print("Intent - who_made_you")
    intent_identify('Hi max, who made you?')
    print("---The BERT model costs {:.3f} seconds to predict the final results---".format((time.time() - start_time)))

    start_time = time.time()
    print("Intent - repeat")
    intent_identify('Hi max, can you say it again')
    print("---The BERT model costs {:.3f} seconds to predict the final results---".format((time.time() - start_time)))

    start_time = time.time()
    print("Intent - how_old_are_you")
    intent_identify('Hi max, how old are you')
    print("---The BERT model costs {:.3f} seconds to predict the final results---".format((time.time() - start_time)))

    start_time = time.time()
    print("Intent - tell_joke")
    intent_identify('Hi max, can you tell me a joke')
    print("---The BERT model costs {:.3f} seconds to predict the final results---".format((time.time() - start_time)))

    start_time = time.time()
    print("Intent - are_you_a_bot")
    intent_identify('Hi max, what are you')
    print("---The BERT model costs {:.3f} seconds to predict the final results---".format((time.time() - start_time)))

    start_time = time.time()
    print("Intent - what_are_your_hobbies")
    intent_identify('Hi max, what do you do in your free time')
    print("---The BERT model costs {:.3f} seconds to predict the final results---".format((time.time() - start_time)))

    start_time = time.time()
    print("Intent - goodbye")
    intent_identify('Hi max, see you later')
    print("---The BERT model costs {:.3f} seconds to predict the final results---".format((time.time() - start_time)))