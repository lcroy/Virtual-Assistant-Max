import json

def update_service(cfg, text):
    with open(cfg.conv_json, "r") as jsonFile:
        data = json.load(jsonFile)
        data[0]["service"] = text
    jsonFile.close()
    with open(cfg.conv_json, "w") as jsonFile:
        json.dump(data, jsonFile)
    jsonFile.close()

def update_mir(cfg, internet, system, task):
    with open(cfg.conv_json, "r") as jsonFile:
        data = json.load(jsonFile)
        data[0]["mir_internet"] = internet
        data[0]["mir_system"] = system
        data[0]["mir_task"] = task
    jsonFile.close()
    with open(cfg.conv_json, "w") as jsonFile:
        json.dump(data, jsonFile)
    jsonFile.close()


def update_franka(cfg, internet, system, task):
    with open(cfg.conv_json, "r") as jsonFile:
        data = json.load(jsonFile)
        data[0]["franka_internet"] = internet
        data[0]["franka_system"] = system
        data[0]["franka_task"] = task
    jsonFile.close()
    with open(cfg.conv_json, "w") as jsonFile:
        json.dump(data, jsonFile)
    jsonFile.close()


def update_swarm(cfg, internet, system, task):
    with open(cfg.conv_json, "r") as jsonFile:
        data = json.load(jsonFile)
        data[0]["swarm_internet"] = internet
        data[0]["swarm_system"] = system
        data[0]["swarm_task"] = task
    jsonFile.close()
    with open(cfg.conv_json, "w") as jsonFile:
        json.dump(data, jsonFile)
    jsonFile.close()


def update_max(cfg, text):
    with open(cfg.conv_json, "r") as jsonFile:
        data = json.load(jsonFile)
        data[0]["Max"] = text
    jsonFile.close()
    with open(cfg.conv_json, "w") as jsonFile:
        json.dump(data, jsonFile)
    jsonFile.close()


def update_user(cfg, text):
    with open(cfg.conv_json, "r") as jsonFile:
        data = json.load(jsonFile)
        data[0]["User"] = text
    jsonFile.close()
    with open(cfg.conv_json, "w") as jsonFile:
        json.dump(data, jsonFile)
    jsonFile.close()


def update_load_process(cfg, text):
    with open(cfg.conv_json, "r") as jsonFile:
        data = json.load(jsonFile)
        data[0]["load_process"] = text
    jsonFile.close()
    with open(cfg.conv_json, "w") as jsonFile:
        json.dump(data, jsonFile)
    jsonFile.close()