import os
import random
import playsound

from franka_web_API import franka_open_brakes, franka_execute_task
from Max_speak import *
from FrankaMax import FrankaMax
from update_conversation import *
from configure import Config
# from google_stt import *
from response.response_template import *
from knowledgegraph.keywords import *

def franka_service(cfg, Franka):

    update_franka("Connected...", "good", "Waiting...")

    while True:

        text = speech_to_text()

        tap_suggestion = 0

        if len(text) > 0:

            no_answer = 0

            update_user(text)

            # back to Max home service
            if any(key in text for key in service_home):
                update_service('home')
                update_max(go_home[0])
                playsound.playsound(os.path.join(cfg.voice_path, 'call_home.mp3'))
                break

            # Question service
            if any(key in text for key in ser_list):
                update_max(service_info[0])
                update_franka("Connected to Franka", "Good.", "Service info.")
                playsound.playsound(os.path.join(cfg.voice_path, 'service.mp3'))
                no_answer = 1
                continue

            # calling franka service
            # user requested action
            task_name = 'none'

            if any(key in text for key in action_list):
                if any(key in text for key in pcb_list):
                    #task_name = 'assembly_pcb'
                    task_name = 'assembly_pcb_danny'
                    update_max(get_PCB[0])
                    update_franka("Connected to Franka", "Good.", "Grasping - PCB")
                    playsound.playsound(os.path.join(cfg.voice_path, 'diy_pcb.mp3'))
                    no_answer = 1
                elif any(key in text for key in cover_list):
                    #task_name = 'assembly_cover'
                    task_name = 'assembly_cover_danny'
                    update_max(get_top_cover[0])
                    update_franka("Connected to Franka", "Good.", "Grasping - Top Cover")
                    playsound.playsound(os.path.join(cfg.voice_path, 'diy_cover.mp3'))
                    no_answer = 1
                elif any(key in text for key in house_list):
                    #task_name = 'assembly_body'
                    task_name = 'assembly_body_danny'
                    update_max(get_bottom_cover[0])
                    update_franka("Connected to Franka", "Good.", "Grasping - Bottom Cover")
                    playsound.playsound(os.path.join(cfg.voice_path, 'diy_house.mp3'))
                    no_answer = 1
                elif any(key in text for key in fuse_list):
                    if any(key in text for key in fuse_one):
                        #task_name = 'assembly_fuse_1'
                        task_name = 'assembly_fuse_1_danny'
                        update_max(get_fuse_one[0])
                        update_franka("Connected to Franka", "Good.", "Grasping - Fuse One")
                        playsound.playsound(os.path.join(cfg.voice_path, 'diy_fuse_one.mp3'))
                        no_answer = 1
                    if any(key in text for key in fuse_two):
                        #task_name = 'assembly_fuse_2'
                        task_name = 'assembly_fuse_2_danny'
                        update_max(get_fuse_two[0])
                        update_franka("Connected to Franka", "Good.", "Grasping - Fuse Two")
                        playsound.playsound(os.path.join(cfg.voice_path, 'diy_fuse_two.mp3'))
                        no_answer = 1
                if task_name != 'none':
                    franka_execute_task(cfg.HOSTNAME, cfg.LOGIN, cfg.PASSWORD, task_name)
                    if (tap_suggestion == 0):
                        update_max(move_home_two[0])
                        playsound.playsound(os.path.join(cfg.voice_path, 'gripper.mp3'))
                        no_answer = 1
                        tap_suggestion = 1
                    continue

            # relationship
            if any(key in text for key in pcb_list) and any(key in text for key in fuse_list):
                update_max(PCBtoFuses)
                update_franka("Connected to Franka", "Good.", "Task Reasoning - PCB and Fuse")
                playsound.playsound(os.path.join(cfg.voice_path, 'pcbtofuse.mp3'))
                no_answer = 1
                continue

            if any(key in text for key in pcb_list) and any(key in text for key in cover_list):
                update_max(PCBtoCover)
                update_franka("Connected to Franka", "Good.", "Task Reasoning - PCB and Top Cover")
                playsound.playsound(os.path.join(cfg.voice_path, 'pcbtocover.mp3'))
                no_answer = 1
                continue
            elif any(key in text for key in pcb_list) and ("cover" in text):
                update_max(confirm_cover[0])
                update_franka("Connected to Franka", "Good.", "Task Reasoning - PCB and ?")
                playsound.playsound(os.path.join(cfg.voice_path, 'confirm_cover.mp3'))
                while True:
                    text = speech_to_text()
                    if len(text) > 0:
                        if "top" in text:
                            update_max(PCBtoCover)
                            update_franka("Connected to Franka", "Good.", "Task Reasoning - PCB and Top Cover")
                            playsound.playsound(os.path.join(cfg.voice_path, 'pcbtocover.mp3'))
                            no_answer = 1
                            break
                        elif ("bottom" in text) or ("house" in text):
                            update_max(PCBtoHouse)
                            update_franka("Connected to Franka", "Good.", "Task Reasoning - PCB and Bottom Cover")
                            playsound.playsound(os.path.join(cfg.voice_path, 'pcbtohouse.mp3'))
                            no_answer = 1
                            break
                        elif "stop" in text:
                            break
                continue

            if any(key in text for key in pcb_list) and any(key in text for key in house_list):
                update_max(PCBtoHouse)
                update_franka("Connected to Franka", "Good.", "Task Reasoning - PCB and Bottom Cover")
                playsound.playsound(os.path.join(cfg.voice_path, 'pcbtohouse.mp3'))
                no_answer = 1
                continue
            elif any(key in text for key in pcb_list) and ("cover" in text):
                update_max(confirm_cover[0])
                update_franka("Connected to Franka", "Good.", "Task Reasoning - PCB and ?")
                playsound.playsound(os.path.join(cfg.voice_path, 'confirm_cover.mp3'))
                while True:
                    text = speech_to_text()
                    if len(text) > 0:
                        if "top" in text:
                            update_max(PCBtoCover)
                            update_franka("Connected to Franka", "Good.", "Task Reasoning - PCB and Top Cover")
                            playsound.playsound(os.path.join(cfg.voice_path, 'pcbtocover.mp3'))
                            no_answer = 1
                            break
                        elif ("bottom" in text) or ("house" in text):
                            update_max(PCBtoHouse)
                            update_franka("Connected to Franka", "Good.", "Task Reasoning - PCB and Bottom Cover")
                            playsound.playsound(os.path.join(cfg.voice_path, 'pcbtohouse.mp3'))
                            no_answer = 1
                            break
                        elif "stop" in text:
                            break

                continue

            if any(key in text for key in fuse_list) and any(key in text for key in house_list):
                update_max(FusestoHouse)
                update_franka("Connected to Franka", "Good.", "Task Reasoning - Bottom Cover and Fuse")
                playsound.playsound(os.path.join(cfg.voice_path, 'fusetohouse.mp3'))
                no_answer = 1
                continue
            elif any(key in text for key in fuse_list) and ("cover" in text):
                update_max(confirm_cover[0])
                update_franka("Connected to Franka", "Good.", "Task Reasoning - Fuse and ?")
                playsound.playsound(os.path.join(cfg.voice_path, 'confirm_cover.mp3'))
                while True:
                    text = speech_to_text()
                    if len(text) > 0:
                        if "top" in text:
                            update_max(PCBtoCover)
                            update_franka("Connected to Franka", "Good.", "Task Reasoning - Fuse and Top Cover")
                            playsound.playsound(os.path.join(cfg.voice_path, 'fusetocover.mp3'))
                            no_answer = 1
                            break
                        elif ("bottom" in text) or ("house" in text):
                            update_max(PCBtoHouse)
                            update_franka("Connected to Franka", "Good.", "Task Reasoning - Fuse and Bottom Cover")
                            playsound.playsound(os.path.join(cfg.voice_path, 'fusetohouse.mp3'))
                            no_answer = 1
                            break
                        elif "stop" in text:
                            break
                continue

            if any(key in text for key in fuse_list) and any(key in text for key in cover_list):
                update_max(FusestoCover)
                update_franka("Connected to Franka", "Good.", "Task Reasoning - Top Cover and Fuse")
                playsound.playsound(os.path.join(cfg.voice_path, 'fusetocover.mp3'))
                no_answer = 1
                continue
            elif any(key in text for key in fuse_list) and ("cover" in text):
                update_max(confirm_cover[0])
                update_franka("Connected to Franka", "Good.", "Task Reasoning - Fuse and ?")
                playsound.playsound(os.path.join(cfg.voice_path, 'confirm_cover.mp3'))
                while True:
                    text = speech_to_text()
                    if len(text) > 0:
                        if "top" in text:
                            update_max(PCBtoCover)
                            update_franka("Connected to Franka", "Good.", "Task Reasoning - Fuse and Top Cover")
                            playsound.playsound(os.path.join(cfg.voice_path, 'fusetocover.mp3'))
                            no_answer = 1
                            break
                        elif ("bottom" in text) or ("house" in text):
                            update_max(PCBtoHouse)
                            update_franka("Connected to Franka", "Good.", "Task Reasoning - Fuse and Bottom Cover")
                            playsound.playsound(os.path.join(cfg.voice_path, 'fusetohouse.mp3'))
                            no_answer = 1
                            break
                        elif "stop" in text:
                            break
                continue

            if any(key in text for key in house_list) and any(key in text for key in cover_list):
                update_max(HousetoCover)
                update_franka("Connected to Franka", "Good.", "Task Reasoning - Top and Bottom Cover")
                playsound.playsound(os.path.join(cfg.voice_path, 'housetocover.mp3'))
                no_answer = 1
                continue

            # knowledge - terminology
            if any(key in text for key in prod_list):
                update_max(product)
                update_franka("Connected to Franka", "Good.", "Query materials - product")
                playsound.playsound(os.path.join(cfg.voice_path, 'prod.mp3'))
                no_answer = 1
                continue

            if any(key in text for key in lab_list):
                update_max(lab)
                update_franka("Connected to Franka", "Good.", "Query Lab info. - Smart Lab")
                playsound.playsound(os.path.join(cfg.voice_path, 'lab.mp3'))
                no_answer = 1
                continue

            if any(key in text for key in process_list):
                update_max(process)
                update_franka("Connected to Franka", "Good.", "Query process - Processes of phone assembly")
                playsound.playsound(os.path.join(cfg.voice_path, 'process.mp3'))
                no_answer = 1
                continue

            if any(key in text for key in pcb_list):
                update_max(pcb)
                update_franka("Connected to Franka", "Good.", "Query materials - PCB")
                playsound.playsound(os.path.join(cfg.voice_path, 'pcb.mp3'))
                no_answer = 1
                continue

            if any(key in text for key in fuse_list):
                update_max(fuses)
                update_franka("Connected to Franka", "Good.", "Query materials - fuse")
                playsound.playsound(os.path.join(cfg.voice_path, 'fuses.mp3'))
                no_answer = 1
                continue

            if any(key in text for key in house_list):
                update_max(house)
                update_franka("Connected to Franka", "Good.", "Query materials - bottom cover/house")
                playsound.playsound(os.path.join(cfg.voice_path, 'house.mp3'))
                no_answer = 1
                continue

            if any(key in text for key in cover_list):
                update_max(cover)
                update_franka("Connected to Franka", "Good.", "Query materials - top cover")
                playsound.playsound(os.path.join(cfg.voice_path, 'cover.mp3'))
                no_answer = 1
                continue

            # assist
            if any(key in text for key in action_assist) and any(key in text for key in phone_assembly):
                update_max(guide)
                update_franka("Connected to Franka", "Good.", "Assist Phone Assembly")
                playsound.playsound(os.path.join(cfg.voice_path, 'guide.mp3'))
                task = {'Intent': 'FullAssembly', 'Color': 'blue'}
                Franka.DoAction(task)
                no_answer = 1
                continue

            # demonstration
            if any(key in text for key in action_demo) and any(key in text for key in phone_assembly):
                update_max(demo)
                update_franka("Connected...", "Good", "Demonstration")
                playsound.playsound(os.path.join(cfg.voice_path, 'demo.mp3'))
                task = {'Intent': 'AutoAssembly', 'Color': 'blue'}
                Franka.DoAction(task)
                no_answer = 1
                continue

            # back to original position
            if any(key in text for key in default_position_list) and any(key in text for key in action_default_position):
                task_name = '_move_home'
                franka_execute_task(cfg.HOSTNAME, cfg.LOGIN, cfg.PASSWORD, task_name)
                update_max(move_home[0])
                update_franka("Connected to Franka", "Good.", "Move - Default position")
                playsound.playsound(os.path.join(cfg.voice_path, 'home.mp3'))
                no_answer = 1
                continue

            if no_answer == 0:
                update_max(do_not_understand[0])
                update_franka("Connected...", "good", "Waiting...")
                # playsound.playsound(os.path.join(cfg.voice_path, "noanswer.mp3"))
                continue

def Max_Home(cfg):

    # initial interface
    update_service('home')
    update_user("...")
    update_max("waitting for user's command...")

    while True:
    # time.sleep(10)
    # update_max("Hello, this is Max. How can I help you?")
    # playsound.playsound(os.path.join(cfg.voice_path, 'hello.mp3'))
        text = speech_to_text()
        if len(text) > 0:
            key = [key in text for key in service_franka]
            if (len(key) and (True in key)):
                text = text.replace(service_franka[key.index(True)], 'franka')
            update_user(text)

            if any(key in text for key in service_franka):
                update_max(go_franka[0])
                playsound.playsound(os.path.join(cfg.voice_path, 'call_franka.mp3'))

                # calling franka service =========================================================================
                # connect to franka robot
                conn_flag = con_franka(cfg)

                # for testing
                # conn_flag = True

                if conn_flag == True:
                    franka = FrankaMax()
                    # calling Franka service
                    update_service('franka')
                    update_max(franka_ready[0])
                    playsound.playsound(os.path.join(cfg.voice_path, 'franka_ready.mp3'))
                    # calling Franka service
                    franka_service(cfg, franka)
                else:
                    update_max(no_conn[0])
                    playsound.playsound(os.path.join(cfg.voice_path, "lost_connection.mp3"))
                continue
        # else:
        #     update_user("...")
        #     update_max("Waiting operator's command...")

def con_franka(cfg):

    try:
        franka_open_brakes(cfg.HOSTNAME, cfg.LOGIN, cfg.PASSWORD)
        update_franka("Connected to Franka", "good", "Waiting...")
        flag = True
    except:
        update_franka("Not connected...", "good", "Waiting...")
        flag = False

    return flag


if __name__ == "__main__":

    cfg = Config()

    # calling Max service
    Max_Home(cfg)

