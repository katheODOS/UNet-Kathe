Hello! Thank you for visiting my Bachelor's capstone project :3 please find information that can help you navigate these files and the setup below. 

_UNet-Kathe is built upon Alexandre Milesi's Pytorch implementation of the UNet (https://github.com/milesial/Pytorch-UNet)._ The idea, as stated in my thesis paper, is basically to build upon Alexandre's awesome work with functionalities that helped me test the feasibility of using the Biodiversity dataset in production. This project would not have been possible without his repository. 

Project Setup:
-install anaconda: https://www.anaconda.com/download 
- create venv (I use python 3.8.20) with 'conda create --name UNet-Kathe python=3.8.20'
- conda activate UNet-Kathe
- on windows: navigate to 'C:\Users\YourUser\anaconda3\envs\UNet-Kathe' and clone repo into this directory
- in anaconda window: (UNet-Kathe) C:\Users\YourUser\anaconda3\envs\UNet-Kathe>pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
- cd UNet-Kathe
- (UNet-Kathe) C:\Users\YourUser\anaconda3\envs\UNet-Kathe\UNet-Kathe>pip install -r requirements.txt
- (UNet-Kathe-Training) C:\Users\YourUser\anaconda3\envs\UNet-Kathe-Training\UNet-Kathe>pip install seaborn scikit-learn imgaug
- install CUDA


In the main development folders (_preprocessing_scripts_ and _model_evaluation_ for example) you can find README.md files which have more information about a general workflow process using different files within those folders. 
