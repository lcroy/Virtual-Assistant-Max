from configure import Config
from services.language_service.intent_recognition.main_intent.predict_main_intent import pred_intent
from services.language_service.intent_recognition.mir.predict_mir_service import run_pred_mir
from services.language_service.state_tracker.max_dst import max_dst
import json

def pred_main(cfg, user_utterance):
    intent, confidence = pred_intent(cfg, user_utterance)
    pred_result = {'intent': intent, 'confidence': confidence}
    print(pred_result)
    # if confidence is lower than threshold
    # don't understand what user said
    if confidence > cfg.confidence:
        return pred_result, 'main', 1
    else:
        return pred_result, 'main', 0

def pred_mir(cfg, user_utterance):
    pred_result, confidence = run_pred_mir(cfg, user_utterance, 1)
    print(user_utterance)
    # if confidence is lower than threshold
    # don't understand what user said
    if confidence > cfg.confidence:
        return pred_result, 'mir', 1
    else:
        return pred_result, 'mir', 0


def pred_intent_slot(user_utterance, client_slot_result, requested_service):
    # read service list
    cfg = Config()
    pred_result, pred_service, language_service_result_flag, updated_client_slot_result, required_slot_list = 'none', 'none', 0, client_slot_result, []
    # check the service
    with open(cfg.service_list) as json_file:
        data = json.load(json_file)
        for item in data['service']:
            # loop the service
            if item['service name'] == requested_service:
                if requested_service == 'main':
                    pred_result, pred_service, language_service_result_flag = pred_main(cfg, user_utterance)
                elif requested_service == 'mir':
                    pred_result, pred_service, language_service_result_flag = pred_mir(cfg, user_utterance)
                    updated_client_slot_result, required_slot_list = max_dst(cfg, pred_result, client_slot_result, pred_service)
                elif requested_service == 'franka':
                    pass
                elif requested_service == 'swarm':
                    pass

    return pred_result, pred_service, language_service_result_flag, updated_client_slot_result, required_slot_list