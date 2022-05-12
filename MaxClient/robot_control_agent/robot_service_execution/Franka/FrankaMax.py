from FrankaWebAPI import franka_open_brakes, franka_close_brakes, franka_execute_task, franka_stop_task
from time import sleep
import playsound
from configure import Config
import os
from update_conversation import *
import time

cfg =Config()

class FrankaMax():
    """Interface between robot assistant Max and Franka Emika robot.
    """
    FETCH_OBJECTS = ['bottom_cover', 'pcb', 'fuse_one', 'fuse_two', 'top_cover']

    def __init__(self, hostname='172.27.23.65', login='Panda', password='panda1234'):
        # Credentials to connect to Franka Web interface.
        self.HOSTNAME = hostname
        self.LOGIN = login
        self.PASSWORD = password

    def DoAction(self, intent):
        err_str0 = 'Argument "intent" must be a dictionary, see class "Intents" in intent_genetator.py'
        assert(type(intent) is dict), err_str0

        intent_key = intent['Intent']

        if intent_key == 'BringPart':
            obj_name = intent['Object']
            if 'Color' in intent:
                color = intent['Color']
            else:
                color = None
            self.fetch_one_object(obj_name, color)
        elif intent_key == 'TaskCheck':
            # return function name of task checking
            pass
        elif intent_key == 'PauseTask':
            # return function name of pausing task
            # check if it is even feasible...
            pass
        elif intent_key == 'ContinueTask':
            # same as PauseTask
            pass
        elif intent_key == 'ActionConfirmation':
            # how to make it, open gripper?
            pass
        elif intent_key == 'Stop':
            self.close_brakes()
        elif intent_key == 'Start':
            self.open_brakes()
        elif intent_key == 'FullAssembly':
            if 'Color' in intent:
                color = intent['Color']
            else:
                color = None
            self.fetch_all_objects(color)
        elif intent_key == 'MoveHome':
            self.move_home()
        elif intent_key == 'AutoAssembly':
            if 'Color' in intent:
                color = intent['Color']
            else:
                color = None
            self.auto_assembly(color)

    def stop_task(self):
        franka_stop_task(self.HOSTNAME, self.LOGIN, self.PASSWORD)

    def open_brakes(self):
        franka_open_brakes(self.HOSTNAME, self.LOGIN, self.PASSWORD)

    def close_brakes(self):
        self.stop_task()
        franka_close_brakes(self.HOSTNAME, self.LOGIN, self.PASSWORD)

    def move_home(self):
        task_name = '_move_home'
        franka_execute_task(self.HOSTNAME, self.LOGIN, self.PASSWORD, task_name)

    # NOTE: this function is written only for 1 color objects. (blue)
    def fetch_one_object(self, obj, color):
        if color is not None:
            assert(color == 'blue'), 'TEST MODE: only "blue" color is available for now...'

        if obj == self.FETCH_OBJECTS[0]:
            print('Fetching bottom cover')
            task_name = 'assembly_body_danny'  # this has to match with task name in Franka Web Desk interface
            franka_execute_task(self.HOSTNAME, self.LOGIN, self.PASSWORD, task_name)
        elif obj == self.FETCH_OBJECTS[1]:
            print('Fetching pcb')
            task_name = 'assembly_pcb_danny'  # this has to match with task name in Franka Web Desk interface
            franka_execute_task(self.HOSTNAME, self.LOGIN, self.PASSWORD, task_name)
        elif obj == self.FETCH_OBJECTS[2]:
            print('Fetching fuse 1')
            task_name = 'assembly_fuse_1_danny'  # this has to match with task name in Franka Web Desk interface
            franka_execute_task(self.HOSTNAME, self.LOGIN, self.PASSWORD, task_name)
        elif obj == self.FETCH_OBJECTS[3]:
            print('Fetching fuse 2')
            task_name = 'assembly_fuse_2_danny'  # this has to match with task name in Franka Web Desk interface
            franka_execute_task(self.HOSTNAME, self.LOGIN, self.PASSWORD, task_name)
        elif obj == self.FETCH_OBJECTS[4]:
            print('Fetching top cover')
            task_name = 'assembly_cover_danny'  # this has to match with task name in Franka Web Desk interface
            franka_execute_task(self.HOSTNAME, self.LOGIN, self.PASSWORD, task_name)
        else:
            raise ValueError("Some unexpected object was passed and not caught previously.")

    # NOTE: this function is written only for 1 color objects. (blue)
    # NOTE 2: System currently doesnt check Franka's status -> so there are sleep()
    # breaks to give robot time to execute the tasks.
    def fetch_all_objects(self, color):
        """ A function to fetch all objects required to assembly 1 phone unit.
            Starts with moving releasing the brakes, homing Franka position, and fetching
            1. bottom cover, 2.PCB, 3.left fuse, 4.right fuse, 5.top cover.
        """
        assert(color == 'blue'), 'TEST MODE: only "blue" color is available for now...'

        self.open_brakes()
        self.move_home()
        sleep(5)
        self.fetch_one_object(self.FETCH_OBJECTS[0], color)
        update_max("Ok, hold the house first.")
        playsound.playsound(os.path.join(cfg.voice_path, 'house_one.mp3'))
        playsound.playsound(os.path.join(cfg.voice_path, 'gripper.mp3'))
        sleep(20)
        self.fetch_one_object(self.FETCH_OBJECTS[1], color)
        update_max("PCB is coming...Please put it on the house and check the orientation.")
        playsound.playsound(os.path.join(cfg.voice_path, 'PCB_two.mp3'))
        sleep(30)
        self.fetch_one_object(self.FETCH_OBJECTS[2], color)
        update_max("Now! let's get fuses placed. Here is the first one.")
        playsound.playsound(os.path.join(cfg.voice_path, 'fuse_three.mp3'))
        sleep(25)
        self.fetch_one_object(self.FETCH_OBJECTS[3], color)
        update_max("And I am bring the second one now.")
        playsound.playsound(os.path.join(cfg.voice_path, 'fuse_four.mp3'))
        sleep(25)
        self.fetch_one_object(self.FETCH_OBJECTS[4], color)
        update_max("Ok, we are almost done. let me bring the last part to you, the cover.")
        playsound.playsound(os.path.join(cfg.voice_path, 'cover_five.mp3'))
        sleep(4)
        print('Fetching sequence has been finished.')
        update_max("Well done, now you have learned the process of assembly a phone.")
        playsound.playsound(os.path.join(cfg.voice_path, 'demo_done.mp3'))

        # NOTE: this function is written only for 1 color objects. (blue)
    def auto_assemble_body(self):
        task_name = 'self_assembly_body_danny'
        franka_execute_task(self.HOSTNAME, self.LOGIN, self.PASSWORD, task_name)

    def auto_assemble_pcb(self):
        task_name = 'self_assembly_pcb_danny1'
        franka_execute_task(self.HOSTNAME, self.LOGIN, self.PASSWORD, task_name)

    def auto_assemble_fuse_1(self):
        task_name = 'self_assembly_fuse_1_danny'
        franka_execute_task(self.HOSTNAME, self.LOGIN, self.PASSWORD, task_name)

    def auto_assemble_fuse_2(self):
        task_name = 'self_assembly_fuse_2_danny'
        franka_execute_task(self.HOSTNAME, self.LOGIN, self.PASSWORD, task_name)

    def auto_assemble_cover(self):
        task_name = 'self_assembly_cover_danny'
        franka_execute_task(self.HOSTNAME, self.LOGIN, self.PASSWORD, task_name)


    def auto_assembly(self, color):
        assert (color == 'blue'), 'TEST MODE: only "blue" color is available for now...'
        self.open_brakes()  # self-blocking.
        self.move_home()
        sleep(8)
        self.auto_assemble_body()
        update_max('First, we need to get the bottom cover.')
        sleep(33)
        update_max('Now that the bottom cover is in place, let me fetch the pcb.')
        #sleep(10)
        self.auto_assemble_pcb()
        sleep(20)
        update_max('Perfect. Now the fuses.')
        #sleep(10)
        self.auto_assemble_fuse_1()
        sleep(28)
        update_max('Here is the first fuse.')
        #sleep(10)
        self.auto_assemble_fuse_2()
        sleep(28)
        update_max('Now both fuses are in place.')
        #sleep(10)
        self.move_home()
        sleep(5)
        self.auto_assemble_cover()
        update_max('We are almost done! Just missing the top cover.')
        sleep(26)
