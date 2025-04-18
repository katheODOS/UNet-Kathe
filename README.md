Project Setup
-install anaconda (insert link)
- create venv (I use python 3.8.20) with 'conda create --name UNet-Kathe python=3.8.20'
- conda activate UNet-Kathe
- on windows: navigate to 'C:\Users\YourUser\anaconda3\envs\UNet-Kathe' and clone repo into this directory
- in anaconda window: (UNet-Kathe) C:\Users\YourUser\anaconda3\envs\UNet-Kathe>pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
- cd UNet-Kathe
- (UNet-Kathe) C:\Users\YourUser\anaconda3\envs\UNet-Kathe\UNet-Kathe>pip install -r requirements.txt
- (UNet-Kathe-Training) C:\Users\YourUser\anaconda3\envs\UNet-Kathe-Training\UNet-Kathe>pip install seaborn scikit-learn
- install CUDA
