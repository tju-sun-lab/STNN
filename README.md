# stnn
Source code for the paper: B Sun, Z Wu, Y Hu, T Li, Golden subject is everyone: A subject transfer neural network for motor imagery-based brain computer interfaces, Neural Networks 151, 111-120, 2022.

在这个测试文件当中，我们提供了23号被试(该被试经过我们其他算法证明该被试的效果的确很糟糕)的数据和标签，此外还提供了17号被试(黄金被试)的数据和标签；

我们在23号被试上重新进行了CNN和STNN的测试，其训练过程和结果分别保存在了CNN.log和golden.log文件中。结果显示，23号被试在CNN上测试结果约为67.8%，在STNN上测试结果为72.5%。

运行STNN代码为main_pytorch_golden.py；运行CNN的代码为main_pytorch_cnn.py；

如何运行STNN代码：

1. 为了方便您的测试，首先我们提供了17号被试在CNN上的一个训练模型，保存在了model_save文件夹下，因此您可以将该训练模型加载到main_pytorch_golden.py中并直接运行STNN的代码。

2. 其次您可以先用CNN训练17号被试的数据和代码，注意要保存其模型，并将您训练的模型加载到main_pytorch_golden.py中并运行。


文件夹中各文件说明：
1.model_save文件夹：用于保存训练模型；
2.data文件夹：存储有17号被试数据和23号被试数据；
3.cnn.log：保存有在2022年10月14日，23号被试数据在CNN模型上的训练过程和结果；
4.golden.log: 保存有在2022年10月14日，23号被试数据在STNN模型上的训练过程和结果；
5.main_pytorch_cnn.py：用于训练CNN代码，第171行用于保存训练模型；
6.main_pytorch_golden.py: 用于训练STNN代码，第89行用于加载17号被试的训练模型；
7.nnModelST_pytorch.py：写有CNN模型；
8.G_D.py：写有STNN的generator部分和CNN部分；
9.tools_golden_subject.py：加载STNN模型的数据。
