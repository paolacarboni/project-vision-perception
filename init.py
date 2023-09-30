import os

current_folder = os.getpwd()
os.makedirs(os.path.join(current_folder, r"resrcs\datasets\textAwareMultiGan"), exist_ok=True)
os.makedirs(os.path.join(current_folder, r"resrcs\datasets\textDetection"), exist_ok=True)
os.makedirs(os.path.join(current_folder, r"resrcs\neuralNetworks\textAwareMultiGan"), exist_ok=True)
os.makedirs(os.path.join(current_folder, r"resrcs\neuralNetworks\textDetection"), exist_ok=True)
os.makedirs(os.path.join(current_folder, r"resrcs\results"), exist_ok=True)