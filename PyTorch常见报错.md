## 报错： ValueError: num_samples should be a positive integer value, but got num_samples=03

可能的原因：传入的Dataset中的len(self.data_info)==0，即传入该dataloader的dataset里没有数据

解决方法：

1. 检查dataset中的路径，路径不对，读取不到数据。
2. 检查Dataset的__len__()函数为何输出为零



## 报错：TypeError: pic should be PIL Image or ndarray. Got <class 'torch.Tensor'>

可能的原因：当前操作需要PIL Image或ndarray数据类型，但传入了Tensor

解决

1. 检查transform中是否存在两次ToTensor()方法
2. 检查transform中每一个操作的数据类型变化



## 报错：RuntimeError: invalid argument 0: Sizes of tensors must match except in dimension 0. Got 93 and 89 in dimension 1 at /Users/soumith/code/builder/wheel/pytorch-src/aten/src/TH/generic/THTensorMath.cpp:3616

可能的原因：dataloader的__getitem__函数中，返回的图片形状不一致，导致无法stack

解决方法：检查__getitem__函数中的操作



## 报错：conv:  RuntimeError: Given groups=1, weight of size 6 1 5 5, expected input[16, 3, 32, 32] to have 1 channels, but got 3 channels instead

linear: RuntimeError: size mismatch, m1: [16 x 576], m2: [400 x 120] at ../aten/src/TH/generic/THTensorMath.cpp:752

可能的原因：网络层输入数据与网络的参数不匹配

解决方法：

1. 检查对应网络层前后定义是否有误
2. 检查输入数据shape



## 报错：AttributeError: 'DataParallel' object has no attribute 'linear'

可能的原因：并行运算时，模型被dataparallel包装，所有module都增加一个属性 module. 因此需要通过 net.module.linear调用

解决方法：

1. 网络层前加入module.



## 报错：RuntimeError: Attempting to deserialize object on a CUDA device but torch.cuda.is_available() is False. If you are running on a CPU-only machine, please use torch.load with map_location=torch.device('cpu') to map your storages to the CPU.

可能的原因：gpu训练的模型保存后，在无gpu设备上无法直接加载

解决方法：

1. 需要设置map_location="cpu"



## 报错：AttributeError: Can't get attribute 'FooNet2' on <module '__main__' from '

可能的原因：保存的网络模型在当前python脚本中没有定义

解决方法：

1. 提前定义该类



## 报错：RuntimeError: Assertion `cur_target >= 0 && cur_target < n_classes' failed.  at ../aten/src/THNN/generic/ClassNLLCriterion.c:94

可能的原因：

1. 标签数大于等于类别数量，即不满足 cur_target < n_classes，通常是因为标签从1开始而不是从0开始

解决方法：

1. 修改label，从0开始，例如：10分类的标签取值应该是0-9



## 报错：RuntimeError: expected device cuda:0 and dtype Long but got device cpu and dtype Long

Expected object of backend CPU but got backend CUDA for argument #2 'weight'                                         可能的原因：需计算的两个数据不在同一个设备上

解决方法：采用to函数将数据迁移到同一个设备上



## 报错：RuntimeError: DataLoader worker (pid 27) is killed by signal: Killed. Details are lost due to multiprocessing. Rerunning with num_workers=0 may give better error trace.

可能原因：内存不够（不是gpu显存，是内存）

解决方法：申请更大内存



## 报错：RuntimeError: reduce failed to synchronize: device-side assert triggered

可能的原因：采用BCE损失函数的时候，input必须是0-1之间，由于模型最后没有加sigmoid激活函数，导致的。

解决方法：让模型输出的值域在[0, 1]



## 报错：RuntimeError: unexpected EOF. The file might be corrupted.

torch.load加载模型过程报错，因为模型传输过程中有问题，重新传一遍模型即可



## 报错：UnicodeDecodeError: 'utf-8' codec can't decode byte 0xff in position 1: invalid start byte

可能的原因：python2保存，python3加载，会报错

解决方法：把encoding改为encoding='iso-8859-1'

check_p = torch.load(path, map_location="cpu", encoding='iso-8859-1')



## 报错：RuntimeError: Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor) should be the same

问题原因：数据张量已经转换到GPU上，但模型参数还在cpu上，造成计算不匹配问题。

解决方法：通过添加model.cuda()将模型转移到GPU上以解决这个问题。或者通过添加model.to(cuda)解决问题




## 报错：RuntimeError: cuDNN error: CUDNN_STATUS_INTERNAL_ERROR

问题原因：jupyter notebook中调用了cuda，但没有释放

解决方法：把对应的ipynb文件shutdown就可以了



## 报错: RuntimeError: CUDA out of memory. Tried to allocate 46.00 MiB (GPU 0; 2.00 GiB total capacity; 54.79 MiB already allocated; 39.30 MiB free; 74.00 MiB reserved in total by PyTorch)

原因: 可以看出在GPU充足的情况下无法使用,本机有两个GPU,其中一个GPU的内存不可用?

解决办法:在model文件(只有model中使用了cuda)添加下面两句:

import os

os.environ['CUDA_VISIBLE_DEVICES']='2, 3'



## 报错：OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized错误提示

原因：用的是pytorch ==1.5，版本高于1.2

解决办法1：治本的办法是再建一个虚拟环境，装1.2版本；

解决办法2：import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"



## 报错：from torchvision import _C as C 

Importerror: DLL load failed: 找不到指定的模块

原因：torchvision安装不正确，当pytorch安装cpu，torchvision安装是gpu时，会报这样的错误

解决办法：

卸载torchvision，重新安装，重新安装的方式强烈建议下载whl文件的形式进行安装，直接pip install 可能失败。

第一步：查看自己torch匹配的torchvision版本，查看方式：[https://github.com/pytorch/vision](https://github.com/pytorch/vision)

|torch|torchvision|python|
|:----|:----|:----|
|master / nightly|master / nightly|>=3.6|
|1.6.0|0.7.0|>=3.6|
|1.5.1|0.6.1|>=3.5|
|1.5.0|0.6.0|>=3.5|
|1.4.0|0.5.0|==2.7, >=3.5, <=3.8|
|1.3.1|0.4.2|==2.7, >=3.5, <=3.7|
|1.3.0|0.4.1|==2.7, >=3.5, <=3.7|
|1.2.0|0.4.0|==2.7, >=3.5, <=3.7|
|1.1.0|0.3.0|==2.7, >=3.5, <=3.7|
|<=1.0.1|0.2.2|==2.7, >=3.5, <=3.7|

第二步：下载对应版本的torchvision，一定要和你当前torch的匹配，注意是cpu还是gpu！

到这里下载 [https://download.pytorch.org/whl/torch_stable.html](https://download.pytorch.org/whl/torch_stable.html)

时间：2020年9月17日

贡献者：pytorch框架第五期criminal -深圳-机器学习，余老师整理

贡献报错格式如下

编号：20

报错信息或是坑的描述：raise HTTPError(req.full_url, code, msg, hdrs, fp)



## 报错：urllib.error.HTTPError: HTTP Error 500: Internal Server Errorraise 

可能的原因：每个 epoch 都 show 图，可能是图太多了，显示不了了？

解决方法：不每一步都 imshow



## 报错信息或是坑的描述：TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to 

可能的原因：将之前CPU人民币二分类的训练验证程序转到GPU上运行，除了要用.to()将输入的张量，和训练的模型转到GPU上，训练后的预存结果显示还是在CPU上所以会报一类错误

解决方法：按照提示找到报错行在转numpy数据前加是cpu()




## libpng warning: iCCP: cHRM chunk does not match sRGB

Ctrl+shift 将输入法切换，不使用QQ输入法下运行，报错就没了。
