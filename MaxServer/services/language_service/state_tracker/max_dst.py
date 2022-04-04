import json

def max_dst(cfg, pred_result, client_slot_result, pred_service):
    # check the required slot for the required intent
    # print("Client slot state before the prediction:====================================== ")
    # print( client_slot_result)
    required_slot_list = []
    find_client_slot = 'init'
    with open(cfg.required_slot) as json_file:
        data = json.load(json_file)
        for item in data['service']:
            # get the service name
            if item['service name'] == pred_service:
                for service_intent_item in item['intent']:
                    for service_intent in service_intent_item:
                        if service_intent == pred_result['intent']:
                            for service_slot in service_intent_item[service_intent]:
                                for service_slot_key in service_slot:
                                    # if the slot is required
                                    if service_slot[service_slot_key] == 'yes':
                                        required_slot_list.append(service_slot_key)
                                        # compare with the client slot results
                                        # if the client slot value is none
                                        if client_slot_result['slot'] == 'none':
                                            client_slot_result['slot'] = pred_result['slot']
                                            break
                                        else:
                                            find_client_slot = 'no'
                                            for client_slot in client_slot_result['slot']:
                                                # if the slot key is found
                                                if client_slot[1] == service_slot_key:
                                                    # this slot value is missing and see if new slot can be embedded
                                                    find_client_slot = 'yes'
                                                    if client_slot[0] == '':
                                                        for pred_slot_item in pred_result['slot']:
                                                            if pred_slot_item[1] == service_slot_key:
                                                                if pred_slot_item[0] != '':
                                                                    client_slot[0] = pred_slot_item[0]
                                                                break
                                                    break
                                            if find_client_slot == 'no':
                                                for pred_slot_item in pred_result['slot']:
                                                    if pred_slot_item[1] == service_slot_key:
                                                        client_slot_result['slot'].append([pred_slot_item[0], service_slot_key])

                            break
                break

    # print("Fill the missing slots to client slot state after the prediction ====================================== ")
    # print(client_slot_result)
    if client_slot_result['slot'] != 'none':
        for item in client_slot_result['slot']:
            if item[1] in required_slot_list:
                required_slot_list.remove(item[1])

    return client_slot_result, required_slot_list
