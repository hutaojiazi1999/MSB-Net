pip install matplotlib scikit-image opencv-python yacs joblib natsort h5py tqdm kornia tensorboard ptflops

Training:
Modify the training set path for line 41 and the test set path for line 42 in train.by
Run: python train.exe | tee logs_Main200H.txt

Test: 
Modify the test set path on line 15, output path on line 16, and weight duplication path on line 17 of test. py

If you want to modify the dim dimension, go to line 300 of model. py

Modify the 20th line of train.by for different ablation experiments from different versions of the model.
For example: import MultiscaleNet as myNet from model-v1
from model_v2 import MultiscaleNet as myNet