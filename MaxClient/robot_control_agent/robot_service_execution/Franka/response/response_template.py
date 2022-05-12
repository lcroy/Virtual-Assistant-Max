import json
from configure import Config

# Max response
do_not_understand = ["Sorry, I don't understand the question. If you have any question regarding the phone assembly, I can help you with that."]
go_home = ['Ok, thank you for using Max service. Hope to see you soon!']
go_mir = ['sure, I will call mir service then.']
go_swarm = ['sure, calling swarm robot service now.']
go_franka = ['sure, let me wake up franka robot.']
# no connection
no_conn = ["Sorry, Franka robot is not reachable at the moment. Please check the network connection and try it again."]

# Franka response
franka_ready = ['ok, Franka robot is ready now.']

# Franka service
service_info = ['Well, I can answer the questions regarding the assembly customized smart phones. I can also demonstrate how to assemble it. ']

# confirm cover
confirm_cover = ['I am sorry. Do you mean bottom cover? Or top Cover?']

# action
move_home = ["Sure, I will do that."]
get_PCB = ["Ok, let me bring the PCB to you then."]
get_top_cover = ["Yes, give me a second, I will get you the top cover"]
get_bottom_cover = ["No problem, let me get the house."]
get_fuse_one = ["Sure, fuse is coming."]
get_fuse_two = ["Ok, I will bring another one to you."]
move_home_two = ["Please remember, you always need to push the gripper when I bring a part to you. Then, I will release it."]

# load knowledge graph
cfg = Config()
with open(cfg.knowledge, "rb") as jsonFile:
    data = json.load(jsonFile)
    product = data["service"]["smart phone"][0]["definition"][0]["product"]
    pcb = data["service"]["smart phone"][0]["definition"][1]["PCB"]
    fuses = data["service"]["smart phone"][0]["definition"][2]["fuses"]
    house = data["service"]["smart phone"][0]["definition"][3]["house"]
    cover = data["service"]["smart phone"][0]["definition"][4]["cover"]
    lab = data["service"]["smart phone"][0]["definition"][5]["lab"]
    process = data["service"]["smart phone"][0]["definition"][6]["process"]

    PCBtoFuses = data["service"]["smart phone"][1]["relationship"][0]["PCBtoFuses"]
    PCBtoHouse = data["service"]["smart phone"][1]["relationship"][1]["PCBtoHouse"]
    HousetoCover = data["service"]["smart phone"][1]["relationship"][2]["HousetoCover"]
    FusestoHouse = data["service"]["smart phone"][1]["relationship"][3]["FusestoHouse"]
    FusestoCover = data["service"]["smart phone"][1]["relationship"][4]["FusestoCover"]
    PCBtoCover = data["service"]["smart phone"][1]["relationship"][5]["PCBtoCover"]
    guide = data["service"]["smart phone"][2]["guide"]
    demo = data["service"]["smart phone"][3]["demo"]
