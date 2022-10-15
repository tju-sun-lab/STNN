## 代码说明
论文源代码: [B Sun, Z Wu, Y Hu, T Li, Golden subject is everyone: A subject transfer neural network for motor imagery-based brain computer interfaces, Neural Networks 151, 111-120, 2022](https://www.sciencedirect.com/science/article/abs/pii/S0893608022001034).

在当前的仓库中，我们提供了23号被试(该被试经过我们其他算法证明该被试的效果的确很糟糕)的数据和标签，此外还提供了17号被试(黄金被试)的数据和标签。论文中的其他被试数据暂未开源，如有需要请联系作者。你也可以很方便的在自己的数据集上使用本代码。

我们在23号被试上重新进行了CNN和STNN的测试，其训练过程和结果分别保存在了CNN.log和golden.log文件中。结果显示，23号被试在CNN上测试结果约为67.8%，在STNN上测试结果为72.5%。

运行STNN代码为main_pytorch_golden.py，运行CNN的代码为main_pytorch_cnn.py。

## 如何运行STNN代码：
1. 为了运行STNN，需要先训练一个CNN模型。
2. 为了方便您的测试，首先我们提供了17号被试在CNN上的一个预训练模型，保存在了model_save文件夹下，因此您可以将该预训练模型加载到main_pytorch_golden.py中并直接运行STNN的代码。
3. 或者，您可以先用17号被试的数据训练你自己的CNN模型，注意要保存模型，并将您训练好的模型加载到main_pytorch_golden.py中以运行STNN。

## 文件夹中各文件说明：
- model_save文件夹：用于保存训练模型；
- data文件夹：存储有17号被试数据和23号被试数据；
- cnn.log：保存有在2022年10月14日，23号被试数据在CNN模型上的训练过程和结果；
- golden.log: 保存有在2022年10月14日，23号被试数据在STNN模型上的训练过程和结果；
- main_pytorch_cnn.py：用于训练CNN代码，第171行用于保存训练模型；
- main_pytorch_golden.py: 用于训练STNN代码，第89行用于加载17号被试的训练模型；
- nnModelST_pytorch.py：写有CNN模型；
- G_D.py：写有STNN的generator部分和CNN部分；
- tools_golden_subject.py: 加载STNN模型的数据。
