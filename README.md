# About the project
This project was developed for the Vision and Perception course for the academic year 2022-2023 at Sapienza University of Rome.
The goal is to remove text from images. To achieve this goal, two different neural networks have been implemented.
* UNet: to obtain masks covering the text in the images
* TextAwareMultiGan: to impainting    
The weights of the networks we obtained are available at the link: https://drive.google.com/drive/folders/10KcpzptTwK6jJJdPxIUwdt1X2hTUj2_Z?usp=drive_link
# Getting Started
## Installation
1. Clone the repo  
    ```git clone https://github.com/paolacarboni/project-vision-perception.git```
3. Install Python packages  
   2.1 Linux/MacOs  
       ```make install```  
   2.2 Windows  
       install all the packages in the requirements file
# Usage
## Training
* TextAwareMultiGan  
    * Run ```make dataset``` by selecting a folder structured as follows:  
       train  
  |-- masks  
  |-- textures  
  test  
  |-- masks  
  |-- textures
    * Run ```make train``` and select gan
* UNet
  Run ```make train``` and select text
## Testing
You can launch an interface that allows text detection, inpainting, or text removal by using ```make exec```. To do this, include the networks weights in the "resources/neuralNetworks" folder.
  
