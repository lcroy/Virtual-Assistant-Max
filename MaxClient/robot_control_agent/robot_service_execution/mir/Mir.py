import requests, json
from geopy.distance import geodesic
import math
import time

class MiR():

    def __init__(self):

        self.host = "http://192.168.12.20/api/v2.0.0/"
        self.headers = {}
        self.headers['Content-Type'] = 'application/json'
        self.headers['Accept-Language'] = 'en_US'
        # self.headers[
        #     'Authorization'] = 'Basic RGlzdHJpYnV0b3I6NjJmMmYwZjFlZmYxMGQzMTUyYzk1ZjZmMDU5NjU3NmU0ODJiYjhlNDQ4MDY0MzNmNGNmOTI5NzkyODM0YjAxNA=='
        # self.headers[
        #     'Authorization'] = 'Basic RGlzdHJpYnV0b3I6ZTNiMGM0NDI5OGZjMWMxNDlhZmJmNGM4OTk2ZmI5MjQyN2FlNDFlNDY0OWI5MzRjYTQ5NTk5MWI3ODUyYjg1NQ=='
        #
        # self.headers[
        #     'Authorization'] = 'Basic ZGlzdHJpYnV0b3I6NjJmMmYwZjFlZmYxMGQzMTUyYzk1ZjZmMDU5NjU3NmU0ODJiYjhlNDQ4MDY0MzNmNGNmOTI5NzkyODM0YjAxNA=='
        # self.headers[
        #     'Authorization'] = 'Basic RGlzdHJpYnV0b3I6ZTNiMGM0NDI5OGZjMWMxNDlhZmJmNGM4OTk2ZmI5MjQyN2FlNDFlNDY0OWI5MzRjYTQ5NTk5MWI3ODUyYjg1NQ=='
        # self.group_id = 'mirconst-guid-0000-0004-users0000000'
        self.headers[
            'Authorization'] = 'Basic YWRtaW46OGM2OTc2ZTViNTQxMDQxNWJkZTkwOGJkNGRlZTE1ZGZiMTY3YTljODczZmM0YmI4YTgxZjZmMmFiNDQ4YTkxOA=='
        # self.group_id = 'mirconst-guid-0000-0011-missiongroup'
        # self.session_id = 'a2f5b1e6-d558-11ea-a95c-0001299f04e5'

    # get the system information
    def get_system_info(self):
        try:
            result = requests.get(self.host + 'status', headers=self.headers)
        except:
            return 'No Connection'

        return result.json()
        # print(str(int(info['battery_percentage'])) + "%")

    # get all missions
    def get_all_missions(self):
        result = requests.get(self.host + 'missions', headers=self.headers)

        return result.json()

    # get missions
    def get_specific_mission(self, guid):
        result = requests.get(self.host + 'missions/' + guid, headers=self.headers)

        return result.json()

    # get actions of a missions
    def get_actions_of_mission(self, guid):
        result = requests.get(self.host + 'missions/' + guid + '/actions', headers=self.headers)

        return result.json()

    # get all maps infomation
    def get_maps(self):
        result = requests.get(self.host + 'maps', headers=self.headers)

        return result.json()

    def get_specific_maps(self, guid):
        result = requests.get(self.host + 'maps/' + guid, headers=self.headers)

        return result.json()

    def get_register(self):
        result = requests.get(self.host + 'registers', headers=self.headers)

        return result.json()


    # get specific map by the map name
    def get_map_positions(self, map_name):
        result = requests.get(self.host + 'maps/' + map_name + '/positions', headers=self.headers)

        return result.json()

        # get positions details

    def get_all_position(self):
        result = requests.get(self.host + 'positions', headers=self.headers)

        return result.json()

    # get positions details
    def get_specific_position(self, guid):
        result = requests.get(self.host + 'positions/' + guid, headers=self.headers)

        return result.json()

    # post a new mission
    def post_mission(self, name):
        parameters = {"name": name, "hidden": False, "group_id": self.group_id, 'session_id': self.session_id}
        post_mission = requests.post(self.host + 'missions', json=parameters, headers=self.headers)

        return post_mission.json()

    # post actions to mission
    def post_action_to_mission(self, mission_id, position_id, action_type):
        parameters = {'action_type': action_type, 'mission_id': mission_id, 'parameters': [{'id': 'position', 'input_name': None, 'value': position_id}, {'id': 'retries', 'input_name': None, 'value': 10}, {'id': 'distance_threshold', 'input_name': None, 'value': 0.1}], 'priority': 1}

        result = requests.post(self.host + 'missions/' + mission_id + '/actions', json=parameters, headers=self.headers)

        return result.json()

    # post position
    def post_position(self, name):

        system_info = self.get_system_info()
        position = system_info['position']
        pos_x = position['x']
        pos_y = position['y']
        orientation = position['orientation']
        map_id = system_info['map_id']

        parameters = {"name": name, "pos_x": pos_x, "pos_y": pos_y, "orientation": orientation, "type_id": 0,
                      "map_id": map_id}
        post_position = requests.post(self.host + 'positions', json=parameters, headers=self.headers)


    def post_to_mission_queue(self, mission_id):
        mission_id = {"mission_id": mission_id}
        post_mission = requests.post(self.host + 'mission_queue', json=mission_id, headers=self.headers)

    def get_mission_queue(self):
        result = requests.get(self.host + 'mission_queue', headers=self.headers)

        return result.json()

    def get_spe_mission_from_queue(self, queue_id):
        result = requests.get(self.host + 'mission_queue/' + queue_id, headers=self.headers)

        return result.json()

    # pause robot executing
    def put_state_to_pause(self):
        parameters = {"state_id": 4}
        requests.put(self.host + 'status', json=parameters, headers=self.headers)

    # start executing
    def put_state_to_execute(self):
        parameters = {"state_id": 3}
        requests.put(self.host + 'status', json=parameters, headers=self.headers)

    # start mir
    def put_state_to_start(self):
        parameters = {"state_id": 1}
        requests.put(self.host + 'status', json=parameters, headers=self.headers)

    # start shutdown
    def put_state_to_shutdown(self):
        parameters = {"state_id": 2}
        requests.put(self.host + 'status', json=parameters, headers=self.headers)

    # Abort Mission
    def put_state_to_abort(self):
        parameters = {"state_id": 6}
        requests.put(self.host + 'status', json=parameters, headers=self.headers)

    # get the details of a mission
    def get_mission_guid(self, name):
        missions = self.get_all_missions()
        mission = {"guid":"none"}
        for item in missions:
            if item['name'] == name:
                mission = self.get_specific_mission(item['guid'])
                break

        return mission['guid']

    def get_details_mission_actions(self, guid):
        actions = self.get_actions_of_mission(guid)
        len_actions = len(actions)
        text = f"There are total {len_actions} actions founded."
        i = 1
        for item in actions:
            guid = item['parameters'][0]['value']
            result = self.get_specific_position(guid)
            text = text + f'The {i} action type is ' + item['action_type'] + "." + f'The position name is ' + str(result['name']) + "."
            i = i + 1

        return text


    def create_mission(self, name):
        # create a mission with the given name
        result = self.post_mission(name)
        # return a mission id for create actions
        mission_id = result['guid']

        return mission_id

    def create_action(self, mission_id, position_name, action_type ='move'):
        # get position guid
        all_position = self.get_all_position()
        for item in all_position:
            if item['name'] == position_name:
                guid = item['guid']
                print(guid)
                break

        # create an action with the specific type
        result = self.post_action_to_mission(mission_id, guid, action_type)

        return result

    def cal_distance(self, origin, dist):

        return geodesic(origin, dist).meters


    def get_current_position(self):
        sys_info = self.get_system_info()
        origin = (sys_info['position']['x'], sys_info['position']['y'])

        return origin

    def get_nearest_position(self):
        origin = self.get_current_position()
        all_positions = self.get_all_position()
        best_distance = float("inf")
        cloest_location = ''
        for item in all_positions:
            if item['name'] != 'Config position':
                position = self.get_specific_position(item['guid'])
                dist = (position['pos_x'], position['pos_y'])
                temp_distance = self.cal_distance(origin, dist)
                if temp_distance < best_distance:
                    best_distance = temp_distance
                    cloest_location = item['name']

        return (best_distance, cloest_location)

    def check_reach_des(self):
        origin = self.get_current_position()
        all_positions = self.get_all_position()
        best_distance = float("inf")
        cloest_location = ''
        for item in all_positions:
            if item['name'] != 'Config position':
                position = self.get_specific_position(item['guid'])
                temp_distance = math.sqrt(
                    math.pow(
                        origin[0] -
                        position['pos_x'],
                        2) +
                    math.pow(
                        origin[1] -
                        position['pos_y'],
                        2))
                if temp_distance < best_distance:
                    best_distance = temp_distance
                    cloest_location = item['name']

        return best_distance, cloest_location


    def get_exe_mission(self):
        exe_mission = self.get_mission_queue()
        mission_name = 'None'
        for item in exe_mission:
            if item['state'] == 'Executing':
                mission_gen = self.get_spe_mission_from_queue(str(item['id']))
                mission_detail = self.get_specific_mission(mission_gen['mission_id'])
                mission_name = mission_detail['name']

        return mission_name


    def get_pending_mission(self):
        mission_queue = self.get_mission_queue()
        for item in mission_queue:
            if item['state'] == 'Pending':
                return True

        return False

    def get_to_be_executed_mission(self):
        mission_queue = self.get_mission_queue()
        num_of_mission = 0;
        for item in mission_queue:
            if item['state'] == 'Pending':
                num_of_mission = num_of_mission + 1
        return num_of_mission

    def get_mission_done_or_not(self, id):
        exe_mission = self.get_mission_queue()
        for item in exe_mission:
            if (item['id'] == id) and (item['state'] == 'Done'):
                return True

        return False

    def move_mir(self, state_id, velocity, joystick_web_session_id):
        parameters = {"velocity":velocity}
        requests.put(self.host + 'status', json=parameters, headers=self.headers)

    def chang_manual(self, state_id):
        parameters = {"state_id": state_id}
        requests.put(self.host + 'status', json=parameters, headers=self.headers)
#
# mir = MiR()
#
# result = mir.get_all_position()
# for val in result:
#     if(val['map'] == '/v2.0.0/maps/6aaad226-2987-11eb-9423-0001299f16e3'):
#         print(val['name'])
#

#
#mir = MiR()

#result = mir.get_all_position()

#print(result)

# # # result = mir.get_mission_guid("acti")
# #
# # print(result)
# #
# # # #
# # mir = MiR()
# # result = mir.get_mission_guid('workshop')
# #
# # mir.post_to_mission_queue(result)
#
# # mir.post_to_mission_queue('5efd89d7-978a-11eb-aa21-0001299f16e3')
# #
# # result = mir.get_system_info()
# # result = mir.get_mission_queue()
# # result = mir.get_spe_mission_from_queue('1065')
# # print(result)
# # # state_id = 11
# # joystick_web_session_id = "vg5kk8rer9sab3bnlkllvqo5b3"
# #
# # velocity = {'angular': -0.8470000624656677, 'linear': 0.11378716677427292}
# #
# # mir.put_s
# # tate_to_pause()

# while True:
#     print("11")
#     mir.move_mir(velocity)



# mir.chang_manual(state_id)
#
# result = mir.get_system_info()
# print(result)
#
# # mir.put_state_to_pause()
# # print(mir.get_system_info())
#
# mir = MiR()
# result = mir.get_system_info()
# battery_per = str(int(result['battery_percentage'])) + '%'
#
# print(battery_per)

# 85cd7f3f-f2b7-11ea-ad20-0001299f16e3

# 8349bd76-978a-11eb-aa21-0001299f16e3

# 5efd89d7-978a-11eb-aa21-0001299f16e3

# 5efd89d7-978a-11eb-aa21-0001299f16e3

# 8349bd76-978a-11eb-aa21-0001299f16e3
