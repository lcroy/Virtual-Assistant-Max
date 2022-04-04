import random
from update_conversation import *
import time

def call_other_service(max, cfg, text, response_template):
    # prepare parameters for GET request
    client_slot_result = json.dumps({'slot':'none'})
    requested_service = 'main'
    # Send the user utterance to the Max Server
    pred_result = max.get_response(text, requested_service, client_slot_result)
    print(pred_result)
    # Max does not understand
    if pred_result['result'] == 'do_not_understand':
        max_response = random.choice(response_template['max_do_not_understand'])
        update_max(cfg, max_response)
        max.text_to_speech_local(max_response)
        # checking if local services are updated
        max_response = random.choice(response_template['max_check_service'])
        update_max(cfg, max_response)
        max.text_to_speech_local(max_response)

        print("--- Start to checking the server side ---")
        start_time = time.time()
        # reading the service_ist.json file from server side
        server_service_list = max.get_file()
        with open(cfg.client_service_list_path) as json_file:
            client_service_list = json.load(json_file)
        json_file.close()
        # compare the services
        if server_service_list == client_service_list:
            max_response = random.choice(response_template['no_new_service'])
            update_max(cfg, max_response)
            max.text_to_speech_local(max_response)
        else:
            max_response = random.choice(response_template['find_new_service'])
            update_max(cfg, max_response)
            with open(cfg.client_service_list_path, 'w') as json_file:
                json.dump(server_service_list, json_file)
            json_file.close()
            max.text_to_speech_local(max_response)

        print("---It takes %s seconds to update client side---" % (time.time() - start_time))

    if pred_result['result'] == 'good_result':
        pred_intent = pred_result['intent']
        max_response = random.choice(response_template[pred_intent])
        update_max(cfg, max_response)
        max.text_to_speech_local(max_response)

