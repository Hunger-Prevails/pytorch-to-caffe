欢迎使用pytorch_converter将你训练好的pytorch模型转换成caffe模型:
	
	1. Dependencies:

		i. pycaffe

		ii. graphviz

		iii. pytorch 0.2.0_3

	2. Invoke:

		i. 将pytorch_converter.py文件拷贝至你的pytorch代码所在目录

		ii. 参考mnist_test.py在你代码的最后调用pc.convert()函数完成转换

		iii. 该程序目前支持仅含有如下层的网络模型的转换:

			pytorch		---------------------------------------- caffe

			Conv2d		---------------------------------------- Convolution

			Linear		---------------------------------------- InnerProduct

			Softmax		---------------------------------------- Softmax

			Relu		---------------------------------------- ReLU

			MaxPool2d	---------------------------------------- Pooling(MAX)

			AvgPool2d	---------------------------------------- Pooling(AVE)

			Split		---------------------------------------- Slice

			View		---------------------------------------- Reshape

			Cat			---------------------------------------- Concat

			Add			---------------------------------------- Eltwise(SUM)

			Mul			---------------------------------------- ELtwise(PROD)

			Max			---------------------------------------- Eltwise(MAX)

			BatchNorm2d	---------------------------------------- BatchNorm + Scale

	3. Input Parameters:

		i. 已经训练好的网络模型对象model

		ii. 网络输入Tensor的形状组成的列表

		iii. 网络名称

	4. Examples:

		i. mnist_convert.py

		ii. resnet_convert.py

		iii. inception_convert.py

		iv. livenet_convert.py

	5. Remarks:

		i. 你的pytorch网络中的pooling层需要设置属性ceil_mode为True

		ii. 你的pytorch网络可以有多个输入和输出
