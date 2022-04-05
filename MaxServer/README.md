# Max Server

The Max server combines three functionalities, 1) lan-
guage service, devoted to recognizing the operator’s intent
and the keywords of the utterance, 2) grounding, grounds
each phrase tuple extracted by language service to a robot
operation action or entity/relationship, and 3) the response
generation.

## Model
Our language-enabled VA supports a hybrid method, the BERT model and rule-based keyword extraction, to achieve the recognition of the operator’s intent.
we fine-tuned the BERT model to encode all the intents, slots and slot values (annotated with Inside-Outside-Beginning 
(IOB) tags) of the current operator’s utterance into an embedded representation. The model architecture
is illustrated in the following diagram. The output of the BERT model is the predicted intent class
label and the IOB tags of the given input. The BERT model is a masked language model that randomly masks 15% of
input tokens and predicts the next sentence. It is primarily used in NLP tasks, e.g., intent classification, slot filling. After
training on our dataset, the model achieves 0.977 and 0.968 on intent accuracy and slot F1 score respectively
<p align="center">
    <img src="https://github.com/lcroy/Jetson_nano/blob/main/Image/BERT.png" width="500" />
</p>

## Structure
- models/main_intent: includes the trained model for chit-chat service
- models/mir: please put your trained BERT model in this folder. You can also directly download our trained [BERT model](https://drive.google.com/file/d/1u3Vl4JOP4BRReoERKw5qU7cndcBiJB7U/view?usp=sharing) and put it here. 
- services
  - language_services: includes the scripts for intent recognition, state-tracker and template of Max response
  - robot_service: TBC
- static and templates: It includes the scripts (e.g., js, html, css) which define the Max Server web interface.
- configure.py: It defines the parameters for running the Max Server, e.g., project path, model path.
- run.py: the main script for running Max Server.

## Instruction
You need to open a terminal, activate your pre-defined environment and run the following script. 
```
python run.py
```