English | [简体中文](https://github.com/missuo/XrayR-V2Board/blob/main/README_CN.md)

## Description
Source code for the paper: [B Sun, Z Wu, Y Hu, T Li, Golden subject is everyone: A subject transfer neural network for motor imagery-based brain computer interfaces, Neural Networks 151, 111-120, 2022](https://www.sciencedirect.com/science/article/abs/pii/S0893608022001034).

In the current repository, we provide the data and labels for subject No. 23 (which has been proven by our other algorithms that the subject's effect is indeed bad), in addition to the data and labels for subject No. 17 (the gold subject). The data of other subjects in the paper are not open source at the moment, so please contact the authors if you need them. You can also easily use this code on your own dataset.

We test the CNN and STNN on subject No. 23, whose training process and results were saved in the CNN.log and golden.log files, respectively. The results showed that the test result on CNN for subject No. 23 was about 67.8%, and the test result on STNN was 72.5%.

The code to run STNN is main_pytorch_golden.py, and the code to run CNN is main_pytorch_cnn.py.

## How to run the STNN code:

1. In order to run STNN, a CNN model needs to be trained first.
2. To facilitate your testing, first we provide a pretrained model for subject No.17 using CNN, saved in the model_save folder, so you can load this pretrained model into main_pytorch_golden.py and run the STNN code directly.
3. Alternatively, you can first train your own CNN model with the data from subject No. 17, taking care to save the model and load your trained model into main_pytorch_golden.py to run STNN.

## Description of each file in the repository：
- model_save folder: for saving trained models；
- data folder: the data of subject No. 17 and subject No. 23 are stored;
- cnn.log: save the training process and results on the CNN model for subject No. 23;
- golden.log: save the training process and results of the 23 subjects' data on the STNN model;
- main_pytorch_cnn.py: code to train the CNN model, line 171 is used to save the training model;
- main_pytorch_golden.py: code to train the STNN model, line 89 is used to load the training model for subject No. 17;
- nnModelST_pytorch.py: CNN model architecture;
- G_D.py: STNN model architecture;
- tools_golden_subject.py: dataloaders for STNN.
