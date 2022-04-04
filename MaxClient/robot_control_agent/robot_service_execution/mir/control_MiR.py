import random
from robot_control_agent.robot_service_execution.mir.Mir import MiR
from update_conversation import *
from pygame import *
import pygame

mission = []
# !!!!mission + timestamp from mir server and save it to the json
######################################################################################################
# State Tracking - Package Delivery
# {intent:DELIVERY; slot:{"object": yes, "size": yes, "color": yes, "person": yes, "destination": yes}
######################################################################################################
def obtain_order(max, mir, cfg, pred_result, response_template, requested_service):
    global mission
    person = ''
    object = ''
    size = ''
    color = ''
    destination = ''

    # if there are some slots not filled
    required_slot = pred_result['required_slot']
    client_slot_result = pred_result['slot']

    while True:
        #find slots need to be filled
        print(required_slot)
        if len(required_slot) > 0:

            for item in required_slot:
                #update the max interface - remind operator need more slots
                text = random.choice(response_template[item])
                update_max(cfg, text)
                update_mir(cfg, 'Connected with MiR.', 'Good!', 'Delivery service: requiring' + item)
                max.text_to_speech_local(text)

                # waiting for operator to give more slots
                text = max.speech_to_text_google()
                update_user(cfg, text)
                # send all the information to the Max server to get required slot
                client_slot_result = json.dumps(client_slot_result)
                pred_result = max.get_response(text, requested_service, client_slot_result)
                # save the slot results and required slots from the server side
                client_slot_result = pred_result['slot']
                required_slot = pred_result['required_slot']
                break
        else:
            # add to mission queue
            for item in client_slot_result['slot']:
                if item[1] == 'B-DELIVERY_PERSON':
                    person = item[0]
                    continue
                if item[1] == 'B-DELIVERY_OBJECT':
                    if item[0] != 'it':
                        object = item[0]
                    continue
                if item[1] == 'B-DELIVERY_POSITION':
                    destination = item[0]
                    continue
                if item[1] == 'B-DELIVERY_OBJECT_COLOR':
                    color = item[0]
                    continue
                if item[1] == 'B-DELIVERY_OBJECT_SIZE':
                    size = item[0]
                    continue
            break

    # max confirm the task
    text = random.choice(response_template['confirm_order']).replace('#size', size).replace('#colour', color).replace('#object', object).replace('#destination', destination).replace('#person', person).replace('?', '')
    # update the interface
    update_max(cfg, text)
    update_mir(cfg, 'Connected with MiR.', 'Good!', 'Executing delivery service.')
    max.text_to_speech_local(text)
    # post a mission
    destination = destination.casefold()
    mission_id = mir.get_mission_guid(destination)
    mir.post_to_mission_queue(mission_id)
    # add the task to the mission list
    mission.append([person, object, size, color, destination])


def call_mir(max, cfg, requested_service, response_template):
    global mission
    mir = MiR()

    # Max friendly ask operator to wait for a second
    text = random.choice(response_template['checking_mir'])
    update_max(cfg, text)
    max.text_to_speech_local(text)
    update_load_process(cfg, 'yes')
    # user-friendly music starts...
    mixer.init()
    mixer.music.load(cfg.waiting_music_path)
    mixer.music.play(-1)

    # Check if MiR is connected
    if (mir.get_system_info() != 'No Connection'):
        update_mir(cfg, 'Connected with MiR.', 'Good!', 'Waiting...')
        if pygame.mixer.music.get_busy():
            mixer.music.stop()
        update_load_process(cfg, 'no')
    else:
        # The program will return if the connection can not be established
        update_mir(cfg, 'Not able to connect to MiR...', 'Unknow...', 'Waiting...')
        text = random.choice(response_template['disconnect_mir'])
        update_max(cfg, text)
        if pygame.mixer.music.get_busy():
            mixer.music.stop()
        update_load_process(cfg, 'no')
        max.text_to_speech_local(text)
        return

    # MiR service starts now...
    update_service(cfg, "mir")
    text = random.choice(response_template['init_speak_mir'])
    update_max(cfg, text)
    max.text_to_speech_local(text)

    while True:
        # waiting for operator's command...
        text = max.speech_to_text_google()
        update_user(cfg, text)

        # prepare parameters for GET request
        client_slot_result = json.dumps({'slot': 'none'})

        # quit the MiR service
        if any(key in text.casefold() for key in cfg.trigger_word_quit_mir):
            update_user(cfg, text)
            text = random.choice(response_template['quit_mir_service'])
            update_max(cfg, text)
            max.text_to_speech_local(text)
            # init Max interface.
            update_user(cfg, "...")
            update_service(cfg, "home")
            update_max(cfg, "Waiting operator's command...")
            break

        # intent identification
        pred_result = max.get_response(text, requested_service, client_slot_result)
        print(pred_result)

        # Max does not understand
        if pred_result['result'] == 'do_not_understand':
            max_response = random.choice(response_template['do_not_understand'])
            update_max(cfg, max_response)
            max.text_to_speech_local(cfg, max_response)


        if pred_result['result'] == 'good_result':

            if pred_result['intent'] == 'GREETING':
                text = random.choice(response_template['greeting'])
                update_max(cfg, text)
                update_mir(cfg, 'Connected with MiR.', 'Good!', 'User wants to greet with MiR.')
                max.text_to_speech_local(text)
                continue

            if pred_result['intent'] == 'POSITIOUPDATE':
                text = random.choice(response_template['name_position'])
                update_max(cfg, text)
                max.text_to_speech_local(text)
                while True:
                    # Operator give a name of the position
                    text = max.speech_to_text_google()
                    update_user(cfg, text)
                    if len(text) > 0:
                        # Add the position on the map
                        mir.post_position(text)
                        text = random.choice(response_template['POSITIOUPDATE']).replace('#', text)
                        update_max(cfg, text)
                        update_mir(cfg, 'Connected with MiR.', 'Good!', 'Adding a new position to map.')
                        max.text_to_speech_local(text)
                        break
                    else:
                        # Max did not catch the name
                        text = random.choice(response_template['did_not_catch'])
                        update_max(cfg, text)
                        update_mir(cfg, 'Connected with MiR.', 'Good!', 'Adding a new position to map.')
                        max.text_to_speech_local(text)
                continue

            if pred_result['intent'] == 'MISSIONCHECK':
                # get the state's id
                sys_info = mir.get_system_info()
                state_id = sys_info['state_id']
                # pause
                if state_id == 4:
                    # check the close place marketed on the map
                    best_distance, cloest_location = mir.check_reach_des()
                    # print(best_distance, cloest_location, mission[0][4])
                    # if MiR reach the destination
                    if (best_distance <= cfg.best_distance) and (cloest_location == mission[0][4]) and (len(mission) > 0):
                        text = random.choice(response_template['DELIVERY']).replace('#size', mission[0][3]).replace('#colour', mission[0][2]).replace('#object', mission[0][1]).replace('#destination', mission[0][4]).replace('#person', mission[0][0]).replace('?', '')
                        update_max(cfg, text)
                        update_mir(cfg, 'Connected with MiR.', 'Good!', 'Checking what mission MiR is working on.')
                        max.text_to_speech_local(text)
                        # moving on to the next mission
                        mission = mission[1:]
                    else:
                        text = random.choice(response_template['mirbusy']).replace('#destination', mission[0][4])
                        update_max(cfg, text)
                        update_mir(cfg, 'Connected with MiR.', 'Good!', 'Checking what mission MiR is working on.')
                        max.text_to_speech_local(text)
                # ready
                if state_id == 3:
                    # check if there is any pending missions on the list
                    pending_mission = mir.get_pending_mission()
                    if pending_mission == False:
                        text = random.choice(response_template['mirfree'])
                        update_max(cfg, text)
                        update_mir(cfg, 'Connected with MiR.', 'Good!', 'Checking what mission MiR is working on.')
                        max.text_to_speech_local(text)
                # executing
                if state_id == 5:
                    # if the mir is running on a mission, pause the mir first
                    mir.put_state_to_pause()
                    text = random.choice(response_template['mirbusy']).replace('#destination', mission[0][4])
                    update_max(cfg, text)
                    update_mir(cfg, 'Connected with MiR.', 'Good!', 'Checking what mission MiR is working on.')
                    max.text_to_speech_local(text)
                    # continue MiR
                    mir.put_state_to_execute()
                continue

            # check the Mir's location
            if pred_result['intent'] == 'POSITIONCHECK':
                # get the nearest location
                dist, name = mir.get_nearest_position()
                text = random.choice(response_template['location']).replace('#location', name)
                update_max(cfg, text)
                update_mir(cfg, 'Connected with MiR.', 'Good!', 'Checking the location of MiR')
                max.text_to_speech_local(text)
                continue

            # check the battery information
            if pred_result['intent'] == 'BATTERYCHECK':
                # get the system status
                result = mir.get_system_info()
                battery_per = str(int(result['battery_percentage'])) + '%'
                text = random.choice(response_template['battery']).replace('#battery', battery_per)
                update_max(cfg, text)
                update_mir(cfg, 'Connected with MiR.', 'Good!', 'Checking battery level.')
                max.text_to_speech_local(text)
                continue

            # Require delivery service
            if pred_result['intent'] == 'ASKHELP':
                # get queued missions
                result = mir.get_exe_mission()
                # if MiR is free
                if result == 'None':
                    text = random.choice(response_template['mirfreetohelp'])
                    update_max(cfg, text)
                    update_mir(cfg, 'Connected with MiR.', 'Good!', 'Requiring delivery service.')
                    max.text_to_speech_local(text)

                    while True:
                        # wait the operator's command
                        text = max.speech_to_text_google()
                        update_user(cfg, text)

                        # call Max server to get intent
                        pred_result = max.get_response(text, requested_service, client_slot_result)

                        if pred_result['result'] == 'good_result':
                            if pred_result['intent'] == 'DELIVERY':
                                obtain_order(max, mir, cfg, pred_result, response_template, requested_service)
                                break
                            else:
                                text = random.choice(response_template['mirnottalkinmission'])
                                update_max(cfg, text)
                                update_mir(cfg, 'Connected with MiR.', 'Good!', 'Executing delivery service.')
                                max.text_to_speech_local(text)
                                break
                        elif pred_result['result'] == 'do_not_understand':
                            max_response = random.choice(response_template['do_not_understand'])
                            update_max(cfg, max_response)
                            max.text_to_speech_local(cfg, max_response)
                else:
                    # pause MiR
                    mir.put_state_to_pause()
                    text = random.choice(response_template['mirbusybuthelp'])
                    update_max(cfg, text)
                    update_mir(cfg, 'Connected with MiR.', 'Good!', 'Executing delivery service.')
                    max.text_to_speech_local(text)
                    # enter a loop to get new mission
                    while True:
                        text = max.speech_to_text_google()
                        update_user(cfg, text)
                        # call Max server to get intent
                        pred_result = max.get_response(text, requested_service, client_slot_result)
                        if pred_result['result'] == 'good_result':
                            if pred_result['intent'] == 'DELIVERY':
                                obtain_order(max, mir, cfg, pred_result, response_template, requested_service)
                                break
                            else:
                                text = random.choice(response_template['mirnottalkinmission'])
                                update_max(cfg, text)
                                update_mir(cfg, 'Connected with MiR.', 'Good!', 'Executing delivery service.')
                                max.text_to_speech_local(text)
                                break
                        elif pred_result['result'] == 'do_not_understand':
                            max_response = random.choice(response_template['do_not_understand'])
                            update_max(cfg, max_response)
                            max.text_to_speech_local(cfg, max_response)

                    mir.put_state_to_execute()
                continue


            if pred_result['intent'] == 'STATESTOP':
                mir.put_state_to_pause()
                text = random.choice(response_template['wait'])
                update_max(cfg, text)
                max.text_to_speech_local(text)
                continue


            if pred_result['intent'] == 'STATERUN':
                mir.put_state_to_execute()
                text = random.choice(response_template['continue'])
                update_max(cfg, text)
                max.text_to_speech_local(text)
                continue

        else:
            update_user(cfg, "...")
            text = random.choice(response_template['do_not_understand'])
            update_max(cfg, text)
            update_mir(cfg, 'Connected with MiR.', 'Good!', 'Waiting...')
            #max.text_to_speech_local(text)
            pass
