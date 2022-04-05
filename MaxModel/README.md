# Max Model

## Dataset for chit-chat
- data/main_intent: includes the dataset and intent labels for chit-chat service. We currently support 10 
  intents, for example, greeting, tell a joke. 
- data/mir: includes the dataset and intent labels for MiR service. We currently support six intents, mission check, 
  battery check, position check, stop mission, continue mission, and package delivery.

## Scripts for training the models
- train/main_intent: training scripts for LSTM and BERT model of chit-chat.
- train/robot_service_intent: training script for MiR service by using BERT model.
- reliability_diagrams.py: generating the reliability diagrams for testing the models.

## Configuration file
- configure.py: defines the parameters for training and predicting, e.g., dataset path, model path.

## Test chit-chat
- predict_BERT.py/predict_LSTM: for testing the performance of chit-chat service.

## Required packages for training model
- requirements.txt: list all the packages required to train the models (We suggest you train the model on a desktop computer and later deploy the 
  model on Jetson Nano)

## Training and Evaluation models
First activate your virtual environment. We build a conda environment, botx, based on the packages
listed on the requirements.txt file. We design and test the scripts in [Pycharm](https://www.jetbrains.com/pycharm/) environment. 
```
conda activate botx
```
Training model for chit-chat by using LSTM. The **"train_LSTM.py"** script is under
path: /train/main-intent/
```
python train_LSTM.py
```
Training model for chit-chat by using BERT. The **"train_BERT.py"** script is under
path: /train/main-intent/
```
python train_BERT.py
```
Training and evaluation model for MiR service by using BERT. The **"run.py"** script is under
path: /train/robot_service_intent/
```
python run.py --task mir
```

**We also provide the trained BERT model for MiR service. You can clicke [here](https://drive.google.com/file/d/1u3Vl4JOP4BRReoERKw5qU7cndcBiJB7U/view?usp=sharing) 
to download the model. The trained chit-chat model is already placed in Max Server folder.**
