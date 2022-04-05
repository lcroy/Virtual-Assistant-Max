# Max Server

The Max server combines three functionalities, 1) lan-
guage service, devoted to recognizing the operator’s intent
and the keywords of the utterance, 2) grounding, grounds
each phrase tuple extracted by language service to a robot
operation action or entity/relationship, and 3) the response
generation.

## Model
Our language-enabled VA supports a
hybrid method, the BERT model and rule-based keyword extraction, to achieve the recognition of the operator’s intent.
we fine-tuned the BERT model to encode all the intents, slots and slot values (annotated with Inside-Outside-Beginning 
(IOB) tags) of the current operator’s utterance into an embedded representation. The model architecture
is illustrated in the following diagram.

<img src="https://github.com/lcroy/Jetson_nano/blob/main/Image/BERT.png" width="600" />