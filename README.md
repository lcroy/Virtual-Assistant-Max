# Industrial virtual assistant implemented on Jetson Nano

## Introduction
This repository showcases building industrial oriented virtual assistant via fintuning a pretrained BERT model using Transformer architecture, and contains the dataset, source code and pre-trained model. It is implemented and tested on
Nvidia Jetson Nano. Our research paper is:

[How can I help you? An Intelligent Virtual Assistant for Industrial Robots](https://dl.acm.org/doi/10.1145/3434074.3447163), 
 Chen Li, Jinha Park, Hahyeon Kim, Dimitrios Chrysostomou. *HRI '21 Companion: Companion of the 2021 ACM/IEEE International Conference on Human-Robot Interaction*

<img style="padding: inherit" src="https://github.com/lcroy/Jetson_nano/blob/main/Image/BERT.png" width="400" />

In the light of recent trends toward introducing Artificial Intelligence (AI) 
to enhance Human-Robot Interaction (HRI), intelligent virtual assistants (VA) 
driven by Natural Language Processing (NLP) receives ample attention in the 
manufacturing domain. However, most VAs either tightly bind with a specific 
robotic system or lack efficient human-robot communication. In this work, we 
implement a layer of interaction between the robotic system and the human 
operator. This interaction is achieved using a novel VA, called Max, as an 
intelligent and robust interface. It is ongoing project. It is deployed on the
[Nvidia Jetson Nano](https://developer.nvidia.com/embedded/jetson-nano-developer-kit)

## Architecture Overview
Max is designed as a web-based application based on the [Flask](https://flask.palletsprojects.com/en/2.1.x/) web framework. 
It involves three main actors: 1) the Max Client, devoted to the translation 
of the operator’s verbal commands, showing the robot’s status, displaying the 
response, and controlling the shop floor robots; 2) the Max Server, committed to serve the interpretation of the operator’s requests,
ground verbal commands to robot’s actions and generate the corresponding 
response; 3) the robotic platform. The following figure shows the overview of 
the system architecture.

<img style="padding: inherit" src="https://github.com/lcroy/Jetson_nano/blob/main/Image/system_architecture.png" width="800" />

## Installation
The following steps show how to set up the environment for running Max on Jetson Nano.

### Step 1: Update system
Open the terminal and run the following command to update and upgrad the Jetson Nano
```
$ sudo apt update
$ sudo apt upgrade
```
### Step 2: Install Anaconda
Install some python packages before install Anaconda:
```
$ sudo apt install python3-h5py libhdf5-serial-dev hdf5-tools python3-matplotlib python3-pip libopenblas-base libopenmpi-dev
```
Download the Archiconda package and install
```
$ wget https://github.com/Archiconda/build-tools/releases/download/0.2.3/Archiconda3-0.2.3-Linux-aarch64.sh
$ sudo sh Archiconda3-0.2.3-Linux-aarch64.sh
```
### Step 3: Install PyTorch
Please follow the [Nvidia instruction](https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-10-now-available/72048) to install the PyTorch on the Jetson Nano. You need to double-check the
version of JetPack to install the corresponding PyTorch.

### Step 4: Create Virtual Environment
Use the following command to create an virtual environment. 
```
conda create -n name_of_environment python=3.6
```
Enter the virtual environment. For example, I create my environment named "tod"
```
conda activate tod
```
### Step 5: Install Trnasformers
First, install sentencepiece
```
git clone https://github.com/google/sentencepiece
cd /path/to/sentencepiece
mkdir build
cd build
cmake ..
make -j $(nproc)
sudo make install
sudo ldconfig -v
cd .. 
cd python
python3 setup.py install
```
Second, install tokenizers
```
curl https://sh.rustup.rs -sSf | sh
rustc --version
exit
restart
pip3 install tokenizers
```
Install Transformers
```
pip install transformers
```
### Step 6: Install Other Packages
Flask is a micro web framework written in Python. 
```
pip install flask
```
Install Speech Recognition packages. Since the arm64 architecture does not suppor azure-cognitiveservices-speech,
you may choose the [SpeechRecognition](https://pypi.org/project/SpeechRecognition/) instead.
```
pip install SpeechRecognition
sudo apt-get install flac
sudo apt-get install libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0
sudo apt-get install libav-tools
sudo pip install pyaudio
pip install playsound
pip install pyttsx3
pip install pygame
```
Install natural language processing related packages
```
pip install nltk
pip install spacy==2.3.5
```
Install geopy package. geopy makes it easy for Python developers to locate the coordinates of addresses, cities, countries, 
and landmarks across the globe using third-party geocoders and other data sources. We use this package to
calculate the location of the MiR.
```
pip install geopy
```

The requirements.txt includes the packages for running Max on Jetson Nano.