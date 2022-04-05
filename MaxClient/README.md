# Max Client

## Structure
- response_template: It includes the pre-defined the responses of Max.
- robot_control_agent:
  - robot_service_execution: It includes the scripts of control the MiR robot. 
    In **Mir.py** script, you will need to put your MiR robot IP to **"self.host = xxx"** and add your 
    authentication code to **"self.headers['Authorization'] = xxxx"**. 
  - robot_service_management: lists all the skills and robots currently supported by Max Server
  - call_gpt.py: It defines the API calls of GPT2 and GPT3. To use GPT3 API, you need to
    apply OpenAI API keys and set it up in the **"configure.py"**. Add your API key to
    **"self.api_key ="**
  - call_other_service.py: it includes the scripts to start the chit-chat. The trained LSTM model is 
    leveraged. If the user's intent does not fall into the MiR services or chit-chat, it will check if
    user requests service which is defined in Max Server while not synchronized on the Max Client.
  - static and templates: It includes the scripts (e.g., js, html, css) which define the Max web interface.
- configure.py: It defines the parameters for running the Max Client, e.g., project path, trigger words.
- Max.py: It defines the main function of the Max Client, e.g., speech to text, text to speech, call MiR service, Call Small talk service.
- max_interface.py: It defines Max web interface. 
- update_conversation.py: the user's utterance and max's response will be updated by using this script.
- run.py: for running Max Client.

## Max Client interface
Max can verbally respond to the operator’s questions or commands while showing the 
text-based response or related system status simultaneously to enhance
the operator’s experience during training. Using a web-based interface, 
allows the operators to browse Max’s services.

The following figure displays the main interface of Max Client, together with the two 
robot service menus. The Dialogue Panel shows the response from Max according to
the operator’s questions or commands. A picture of the robot and a
list of the robot services are wrapped into the Robot Service
Panel, which provides additional information and indicates
what services are currently supported for the desired robot
platform. The System Status Panel is mainly used in the
Training and Assistance scenarios, where it displays the
real-time information of the tasks or system, e.g., network
connection, which task is running.

<img src="https://github.com/lcroy/Jetson_nano/blob/main/Image/Application_Interface.png" width="1000" />

## Instruction
### Run Max web interface
First, make sure you installed all the packages listed in the homepage of this repository.
Activate the environment you created. Replacing the "tod" with your environment.
```
conda activate tod
```
Second, call the max interface
```
python max_interface.py
```
### Run Max Client
You need to open another terminal, activate the environment and call the Max Client. 
```
python run.py
```
### How to talk to Max
To invoke the Max service, you need to say the trigger word - "Max", for example
"Hey, Max". 

To invoke the MiR service, you need to say the trigger word - "mobile robot", for example,
"Hey Max, please call mobile robot". 

To invoke the LSTM supported chit-chat service, you may ask question, e.g., "Max, how old are you"

To invoke the GPT supported chit-chat service, you need to say the trigger word, for example
you may say "Let's have a talk". 

**Please check the configure.py file for the list of the trigger words**. The trigger words are 
only used for activating the service. BERT and LSTM models are used for interpreting the user's utterance. 
More details will be found on Max Model.   
