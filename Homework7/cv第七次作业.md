# cv第七次作业

## 1、作业要求

Github或者主页下载运行一个超分算法，获得结果试着训练一两个Epoch，给出超分结果。

## 2、简要介绍

​		所谓超分算法就是图像超分辨率重建算法，指的是将给定的低分辨率图像通过特定的算法恢复成相应的高分辨率图像。实际上情况中，受采集设备与环境、网络传输介质与带宽、图像退化模型本身等诸多因素的约束，我们通常并不能直接得到具有边缘锐化、无成块模糊的理想高分辨率图像。提升图像分辨率的最直接的做法是对采集系统中的光学硬件进行改进，但是由于制造工艺难以大幅改进并且制造成本十分高昂，因此物理上解决图像低分辨率问题往往代价太大。由此，从**软件和算法的角度**着手，实现图像超分辨率重建的技术成为了图像处理和计算机视觉等多个领域的热点研究课题。

​		按照时间和效果进行分类，可以将超分辨率重建算法分为传统算法和深度学习算法两类。

### （1）传统超分辨率重建算法

​		传统的超分辨率重建算法主要依靠基本的数字图像处理技术进行重建，常见的有如下几类：

#### ★基于插值的超分辨率重建

​		基于插值的方法将图像上每个像素都看做是图像平面上的一个点，那么对超分辨率图像的估计可以看做是利用已知的像素信息为平面上未知的像素信息进行拟合的过程，这通常由一个预定义的变换函数或者插值核来完成。基于插值的方法计算简单、易于理解，但是也存在着一些明显的缺陷。

​		首先，它假设像素灰度值的变化是一个连续的、平滑的过程，但实际上这种假设并不完全成立。其次，在重建过程中，仅根据一个事先定义的转换函数来计算超分辨率图像，不考虑图像的降质退化模型，往往会导致复原出的图像出现模糊、锯齿等现象。常见的基于插值的方法包括最近邻插值法、双线性插值法和双立方插值法等。

#### ★基于退化模型的超分辨率重建

​		此类方法从图像的降质退化模型出发，假定高分辨率图像是经过了适当的运动变换、模糊及噪声才得到低分辨率图像。这种方法通过提取低分辨率图像中的关键信息，并结合对未知的超分辨率图像的先验知识来约束超分辨率图像的生成。常见的方法包括迭代反投影法、凸集投影法和最大后验概率法等。

#### ★基于学习的超分辨率重建

​		基于学习的方法则是利用大量的训练数据，从中学习低分辨率图像和高分辨率图像之间某种对应关系，然后根据学习到的映射关系来预测低分辨率图像所对应的高分辨率图像，从而实现图像的超分辨率重建过程。常见的基于学习的方法包括流形学习、稀疏编码方法。

### （2）基于深度学习的超分辨率重建算法

​		2014年，Dong等人首次将深度学习应用到图像超分辨率重建领域，他们使用一个三层的卷积神经网络学习低分辨率图像与高分辨率图像之间映射关系，自此，在超分辨率重建率领域掀起了深度学习的浪潮，他们的设计的网络模型命名为**SRCNN(Super-Resolution Convolutional Neural Network)**。

​		SRCNN采用了插值的方式先将低分辨率图像进行放大，再通过模型进行复原。Shi等人则认为这种预先采用近邻插值的方式本身已经影响了性能，如果从源头出发，应该从样本中去学习如何进行放大，他们基于这个原理提出了**ESPCN (Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network)**算法。该算法在将低分辨率图像送入神经网络之前，无需对给定的低分辨率图像进行一个上采样过程，而是引入一个亚像素卷积层（Sub-pixel convolution layer），来间接实现图像的放大过程。这种做法极大降低了SRCNN的计算量，提高了重建效率。

​		这里需要注意到，不管是SRCNN还是ESPCN，它们均使用了MSE作为目标函数来训练模型。2017年，Christian Ledig等人从照片感知角度出发，通过对抗网络来进行超分重建（论文题目：Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network）。他们认为，大部分深度学习超分算法采用的MSE损失函数会导致重建的图像过于平滑，缺乏感官上的照片真实感。他们改用生成对抗网络（Generative Adversarial Networks, GAN）来进行重建，并且定义了新的感知目标函数，算法被命名为**SRGAN**，由一个生成器和一个判别器组成。生成器负责合成高分辨率图像，判别器用于判断给定的图像是来自生成器还是真实样本。通过一个博弈的对抗过程，使得生成器能够将给定的低分辨率图像重建为高分辨率图像。在SRGAN这篇论文中，作者同时提出了一个对比算法，名为**SRResNet**。SRResNet依然采用了MSE作为最终的损失函数，与以往不同的是，SRResNet采用了足够深的残差卷积网络模型，相比于其它的残差学习重建算法，SRResNet本身也能够取得较好的效果。

## 3、作业分析

根据本次作业要求，我选择SRResNet算法并进行训练测试。

### （1）超分重建基本处理流程

​		最早的采用深度学习进行超分重建的算法是SRCNN算法，其原理很简单，对于输入的一张低分辨率图像，SRCNN首先使用双立方插值将其放大至目标尺寸，然后利用一个三层的卷积神经网络去拟合低分辨率图像与高分辨率图像之间的非线性映射，最后将网络输出的结果作为重建后的高分辨率图像。尽管原理简单，但是依托深度学习模型以及大样本数据的学习，在性能上超过了当时一众传统的图像处理算法，开启了深度学习在超分辨率领域的研究征程。SRCNN的网络结构如图所示。

![img](https://img-blog.csdnimg.cn/20200218142904571.bmp?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FpYW5iaW4zMjAwODk2,size_16,color_FFFFFF,t_70)

其中![f_1](https://private.codecogs.com/gif.latex?f_1)、![f_2](https://private.codecogs.com/gif.latex?f_2)、![f_3](https://private.codecogs.com/gif.latex?f_3)分别表示1、2、3层卷积对应的核大小。

​		SRCNN作为早期开创性的研究论文，也为后面的工作奠定了处理超分问题的基本流程：

- 寻找大量真实场景下图像样本；
- 对每张图像进行下采样处理降低图像分辨率，一般有2倍下采样、3倍下采样、4倍下采样等。如果是2倍下采样，则图像长宽均变成原来的1/2.。下采样前的图像作为高分辨率图像H，下采样后的图像作为低分辨率图像L，L和H构成一个有效的图像对用于后期模型训练；
-  训练模型时，对低分辨率图像L进行放大还原为高分辨率图像SR，然后与原始的高分辨率图像H进行比较，其差异用来调整模型的参数，通过迭代训练，使得差异最小。实际情况下，研究学者提出了多种损失函数用来定义这种差异，不同的定义方式也会直接影响最终的重建效果；
- 训练完的模型可以用来对新的低分辨率图像进行重建，得到高分辨率图像。

​		从实际操作上来看，整个超分重建分为两步：图像放大和修复。所谓放大就是采用某种方式（SRCNN采用了插值上采样）将图像放大到指定倍数，然后再根据图像修复原理，将放大后的图像映射为目标图像。超分辨率重建不仅能够放大图像尺寸，在某种意义上具备了图像修复的作用，可以在一定程度上削弱图像中的噪声、模糊等。因此，超分辨率重建的很多算法也被学者迁移到图像修复领域中，完成一些诸如jpep压缩去燥、去模糊等任务。

### （2）ResNet网络

​		ResNet中文名字叫作深度残差网络，主要作用是图像分类。现在在图像分割、目标检测等领域都有很广泛的运用。ResNet在传统卷积神经网络中加入了残差学习（residual learning），解决了深层网络中梯度弥散和精度下降（训练集）的问题，使网络能够越来越深，既保证了精度，又控制了速度。

​		ResNet可以直观的来理解其背后的意义。以往的神经网络模型每一层学习的是一个 y = f(x) 的映射，可以想象的到，随着层数不断加深，每个函数映射出来的y误差逐渐累计，误差越来越大，梯度在反向传播的过程中越来越发散。这时候，如果改变一下每层的映射关系，改为 y = f(x) + x,也就是在每层的结束加上原始输入，此时输入是x，输出是f(x)+x，那么自然的f(x)趋向于0，或者说f(x)是一个相对较小的值，这样，即便层数不断加大，这个误差f(x)依然控制在一个较小值，整个模型训练时不容易发散。

![img](https://img-blog.csdnimg.cn/20200221122432953.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FpYW5iaW4zMjAwODk2,size_16,color_FFFFFF,t_70)

​		上图为残差网络的原理图，可以看到一根线直接跨越两层网络（跳链），将原始数据x带入到了输出中，此时F(x)预测的是一个差值。有了残差学习这种强大的网络结构，就可以按照SRCNN的思路构建用于超分重建的深度神经网络。SRResNet算法主干部分就采用了这种网络结构，如下图所示：

![img](https://img-blog.csdnimg.cn/20200221123607191.bmp?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FpYW5iaW4zMjAwODk2,size_16,color_FFFFFF,t_70)

​		上述模型采用了多个深度残差模块进行图像的特征抽取，多次运用跳链技术将输入连接到网络输出，这种结构能够保证整个网络的稳定性。由于采用了深度模型，相比浅层模型能够更有效的挖掘图像特征，在性能上可以超越浅层模型算法（SRResNet使用了16个残差模块）。注意到，上述模型每层仅仅改变了图像的通道数，并没有改变图像的尺寸大小，从这个意义上来说这个网络可以认为是前面提到的修复模型。下面会介绍如何在这个模型基础上再增加一个子模块用来放大图像，从而构建一个完整的超分重建模型。

### （3）子像素卷积

​		子像素卷积（Sub-pixel convolution）是一种巧妙的图像及特征图放大方法，又叫做pixel shuffle（像素清洗）。在深度学习超分辨率重建中，常见的扩尺度方法有直接上采样，双线性插值，反卷积等等。ESPCN算法中提出了一种超分辨率扩尺度方法，即为子像素卷积方法，该方法后续也被应用在了SRResNet和SRGAN算法中。因此，这里需要先介绍子像素卷积的原理及实现方式。

​		采用CNN对特征图进行放大一般会采用deconvolution等方法，这种方法通常会带入过多人工因素，而子像素卷积会大大降低这个风险。因为子像素卷积放大使用的参数是需要学习的，相比那些手工设定的方式，这种通过样本学习的方式其放大性能更加准确。

​		具体实现原理如下图所示：

![img](https://img-blog.csdnimg.cn/20200221130035883.bmp?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FpYW5iaW4zMjAwODk2,size_16,color_FFFFFF,t_70)

​		上图很直观得表达了子像素卷积的流程。假设，如果想对原图放大3倍，那么需要生成出3^2=9个同等大小的特征图，也就是通道数扩充了9倍（这个通过普通的卷积操作即可实现）。然后将九个同等大小的特征图拼成一个放大3倍的大图，这就是子像素卷积操作了。

​		实现时先将原始特征图通过卷积扩展其通道数，如果是想放大4倍，那么就需要将通道数扩展为原来的16倍。特征图做完卷积后再按照特定的格式进行排列，即可得到一张大图，这就是所谓的像素清洗。通过像素清洗，特征的通道数重新恢复为原来输入时的大小，但是每个特征图的尺寸变大了。这里注意到每个像素的扩展方式由对应的卷积来决定，此时卷积的参数是需要学习的，因此，相比于手工设计的放大方式，这种基于学习的放大方式能够更好的去拟合像素之间的关系。

​		SRResNet模型也利用子像素卷积来放大图像，具体的，在上述所示模型后面添加两个子像素卷积模块，每个子像素卷积模块使得输入图像放大2倍，因此这个模型最终可以将图像放大4倍，如下图所示：

![img](https://img-blog.csdnimg.cn/20200221131047680.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FpYW5iaW4zMjAwODk2,size_16,color_FFFFFF,t_70)

### （4）SRResNet结构剖析

​		SRResNet使用深度残差网络来构建超分重建模型，主要包含两部分：深度残差模型、子像素卷积模型。深度残差模型用来进行高效的特征提取，可以在一定程度上削弱图像噪点。子像素卷积模型主要用来放大图像尺寸。完整的SRResNet网络结果如下图所示：

![img](https://img-blog.csdnimg.cn/20200221131937219.bmp?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FpYW5iaW4zMjAwODk2,size_16,color_FFFFFF,t_70)

​		上图中，k表示卷积核大小，n表示输出通道数，s表示步长。除了深度残差模块和子像素卷积模块以外，在整个模型输入和输出部分均添加了一个卷积模块用于数据调整和增强。

​		需要注意的是，SRResNet模型使用MSE作为目标函数，也就是通过模型还原出来的高分辨率图像与原始高分辨率图像的均方误差.

​		MSE也是目前大部分超分重建算法采用的目标函数。后面我们会看到，使用该目标函数重建的超分图像并不能很好的符合人眼主观感受，SRGAN算法正是基于此进行的改进。

## 4、Pytorch实现

基于深度学习框架Pytorch来完成所有的编码工作。

### （1）运行环境

| 工具         | 版本                |
| ------------ | ------------------- |
| OS           | Windows10           |
| IDE          | VS Code             |
| python       | 3.19.2              |
| pytorch      | 1.12.1              |
| scikit-image | 0.19.2              |
| tensorflow   | 2.10.0              |
| train data   | COCO2014            |
| test data    | Set5、Set14和BSD100 |

### （2）代码结构

| 目录                 | 含义                                                         |
| -------------------- | ------------------------------------------------------------ |
| data文件夹           | 用于存放训练和测试数据集以及文件列表；                       |
| results文件夹        | 用于存放运行结果，包括训练好的模型以及单张样本测试结果；     |
| create_data_lists.py | 生成数据列表，检查数据集中的图像文件尺寸，并将符合的图像文件名写入JSON文件列表供后续Pytorch调用； |
| datasets.py          | 用于构建数据集加载器，主要沿用Pytorch标准数据加载器格式进行封装； |
| models.py            | 模型结构文件，存储各个模型的结构定义；                       |
| utils.py             | 工具函数文件，所有项目中涉及到的一些自定义函数均放置在该文件中； |
| train_srresnet.py    | 用于训练SRResNet算法；                                       |
| eval.py              | 用于模型评估，主要以计算测试集的PSNR和SSIM为主；             |
| test.py              | 用于单张样本测试，运用训练好的模型为单张图像进行超分重建；   |

### （3）运行顺序

| 运行顺序如下：                                               |
| ------------------------------------------------------------ |
| 1、运行create_data_lists.py文件用于为数据集生成文件列表；    |
| 2、运行train_srresnet.py进行SRResNet算法训练，训练结束后在results文件夹中会生成checkpoint_srresnet.pth模型文件； |
| 3、运行eval.py文件对测试集进行评估，计算每个测试集的平均PSNR、SSIM值； |
| 4、运行test.py文件对results文件夹下名为test.jpg的图像进行超分还原，还原结果存储在results文件夹下面； |

### （4）数据列表

​		在训练前需要先准备好数据集，按照特定的格式生成文件列表供Pytorch的数据加载器torch.utils.data.DataLoader对图像进行高效率并行加载。只有准确生成了数据集列表文件才能进行下面的训练。

​		create_data_lists.py文件内容如下：

```python
from utils import create_data_lists
 
if __name__ == '__main__':
    create_data_lists(train_folders=['./data/COCO2014/train2014',
                                     './data/COCO2014/val2014'],
                      test_folders=['./data/BSD100',
                                    './data/Set5',
                                    './data/Set14'],
                      min_size=100,
                      output_folder='./data/')
```

​		首先从utils中导入create_data_lists函数，该函数用于执行具体的JSON文件创建。在主函数部分设置好训练集train_folders和测试集test_folders文件夹路径，参数min_size=100用于检查训练集和测试集中的每张图像分辨率，无论是图像宽度还是高度，如果小于min_size则该图像路径不写入JSON文件列表中。output_folder用于指明最后的JSON文件列表存放路径。


### （5）构建SRResNet模型

​		模型的定义在models.py文件中给出：

```python
class SRResNet(nn.Module):
    """
    SRResNet模型
    """
    def __init__(self, large_kernel_size=9, small_kernel_size=3, n_channels=64, n_blocks=16, scaling_factor=4):
        """
        :参数 large_kernel_size: 第一层卷积和最后一层卷积核大小
        :参数 small_kernel_size: 中间层卷积核大小
        :参数 n_channels: 中间层通道数
        :参数 n_blocks: 残差模块数
        :参数 scaling_factor: 放大比例
        """
        super(SRResNet, self).__init__()
 
        # 放大比例必须为 2、 4 或 8
        scaling_factor = int(scaling_factor)
        assert scaling_factor in {2, 4, 8}, "放大比例必须为 2、 4 或 8!"
 
        # 第一个卷积块
        self.conv_block1 = ConvolutionalBlock(in_channels=3, out_channels=n_channels, kernel_size=large_kernel_size,
                                              batch_norm=False, activation='PReLu')
 
        # 一系列残差模块, 每个残差模块包含一个跳连接
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(kernel_size=small_kernel_size, n_channels=n_channels) for i in range(n_blocks)])
 
        # 第二个卷积块
        self.conv_block2 = ConvolutionalBlock(in_channels=n_channels, out_channels=n_channels,
                                              kernel_size=small_kernel_size,
                                              batch_norm=True, activation=None)
 
        # 放大通过子像素卷积模块实现, 每个模块放大两倍
        n_subpixel_convolution_blocks = int(math.log2(scaling_factor))
        self.subpixel_convolutional_blocks = nn.Sequential(
            *[SubPixelConvolutionalBlock(kernel_size=small_kernel_size, n_channels=n_channels, scaling_factor=2) for i
              in range(n_subpixel_convolution_blocks)])
 
        # 最后一个卷积模块
        self.conv_block3 = ConvolutionalBlock(in_channels=n_channels, out_channels=3, kernel_size=large_kernel_size,
                                              batch_norm=False, activation='Tanh')
 
    def forward(self, lr_imgs):
        """
        前向传播.
        :参数 lr_imgs: 低分辨率输入图像集, 张量表示，大小为 (N, 3, w, h)
        :返回: 高分辨率输出图像集, 张量表示， 大小为 (N, 3, w * scaling factor, h * scaling factor)
        """
        output = self.conv_block1(lr_imgs)  # (16, 3, 24, 24)
        residual = output  # (16, 64, 24, 24)
        output = self.residual_blocks(output)  # (16, 64, 24, 24)
        output = self.conv_block2(output)  # (16, 64, 24, 24)
        output = output + residual  # (16, 64, 24, 24)
        output = self.subpixel_convolutional_blocks(output)  # (16, 64, 24 * 4, 24 * 4)
        sr_imgs = self.conv_block3(output)  # (16, 3, 24 * 4, 24 * 4)
 
        return sr_imgs
```

​		整个模型完全参照SRResNet的实现方式，组成方式为：1个卷积模块+16个残差模块+1个卷积模块+2个子像素卷积模块+1个卷积模块。

### （6）模型训练

​		train_srresnet.py文件内容如下：

```python
import torch.backends.cudnn as cudnn
import torch
from torch import nn
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from models import SRResNet
from datasets import SRDataset
from utils import *
 
 
# 数据集参数
data_folder = './data/'          # 数据存放路径
crop_size = 96      # 高分辨率图像裁剪尺寸
scaling_factor = 4  # 放大比例
 
# 模型参数
large_kernel_size = 9   # 第一层卷积和最后一层卷积的核大小
small_kernel_size = 3   # 中间层卷积的核大小
n_channels = 64         # 中间层通道数
n_blocks = 16           # 残差模块数量
 
# 学习参数
checkpoint = None   # 预训练模型路径，如果不存在则为None
batch_size = 400    # 批大小
start_epoch = 1     # 轮数起始位置
epochs = 130        # 迭代轮数
workers = 4         # 工作线程数
lr = 1e-4           # 学习率
 
# 设备参数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ngpu = 2           # 用来运行的gpu数量
 
cudnn.benchmark = True # 对卷积进行加速
 
writer = SummaryWriter() # 实时监控     使用命令 tensorboard --logdir runs  进行查看
 
def main():
    """
    训练.
    """
    global checkpoint,start_epoch,writer
 
    # 初始化
    model = SRResNet(large_kernel_size=large_kernel_size,
                        small_kernel_size=small_kernel_size,
                        n_channels=n_channels,
                        n_blocks=n_blocks,
                        scaling_factor=scaling_factor)
    # 初始化优化器
    optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()),lr=lr)
 
    # 迁移至默认设备进行训练
    model = model.to(device)
    criterion = nn.MSELoss().to(device)
 
    # 加载预训练模型
    if checkpoint is not None:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    if torch.cuda.is_available() and ngpu > 1:
        model = nn.DataParallel(model, device_ids=list(range(ngpu)))
 
    # 定制化的dataloaders
    train_dataset = SRDataset(data_folder,split='train',
                              crop_size=crop_size,
                              scaling_factor=scaling_factor,
                              lr_img_type='imagenet-norm',
                              hr_img_type='[-1, 1]')
    train_loader = torch.utils.data.DataLoader(train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True) 
 
    # 开始逐轮训练
    for epoch in range(start_epoch, epochs+1):
 
        model.train()  # 训练模式：允许使用批样本归一化
 
        loss_epoch = AverageMeter()  # 统计损失函数
 
        n_iter = len(train_loader)
 
        # 按批处理
        for i, (lr_imgs, hr_imgs) in enumerate(train_loader):
 
            # 数据移至默认设备进行训练
            lr_imgs = lr_imgs.to(device)  # (batch_size (N), 3, 24, 24), imagenet-normed 格式
            hr_imgs = hr_imgs.to(device)  # (batch_size (N), 3, 96, 96),  [-1, 1]格式
 
            # 前向传播
            sr_imgs = model(lr_imgs)
 
            # 计算损失
            loss = criterion(sr_imgs, hr_imgs)  
 
            # 后向传播
            optimizer.zero_grad()
            loss.backward()
 
            # 更新模型
            optimizer.step()
 
            # 记录损失值
            loss_epoch.update(loss.item(), lr_imgs.size(0))
 
            # 监控图像变化
            if i==(n_iter-2):
                writer.add_image('SRResNet/epoch_'+str(epoch)+'_1', make_grid(lr_imgs[:4,:3,:,:].cpu(), nrow=4, normalize=True),epoch)
                writer.add_image('SRResNet/epoch_'+str(epoch)+'_2', make_grid(sr_imgs[:4,:3,:,:].cpu(), nrow=4, normalize=True),epoch)
                writer.add_image('SRResNet/epoch_'+str(epoch)+'_3', make_grid(hr_imgs[:4,:3,:,:].cpu(), nrow=4, normalize=True),epoch)
 
            # 打印结果
            print("第 "+str(i)+ " 个batch训练结束")
 
        # 手动释放内存              
        del lr_imgs, hr_imgs, sr_imgs
 
        # 监控损失值变化
        writer.add_scalar('SRResNet/MSE_Loss', loss_epoch.val, epoch)    
 
        # 保存预训练模型
        torch.save({
            'epoch': epoch,
            'model': model.module.state_dict(),
            'optimizer': optimizer.state_dict()
        }, 'results/checkpoint_srresnet.pth')
    
    # 训练结束关闭监控
    writer.close()
 
 
if __name__ == '__main__':
    main()
```

### （7）图像变换

图像变换的实现方式在utils.py文件中的ImageTransforms类给出：

```python
class ImageTransforms(object):
    """
    图像变换.
    """
 
    def __init__(self, split, crop_size, scaling_factor, lr_img_type,
                 hr_img_type):
        """
        :参数 split: 'train' 或 'test'
        :参数 crop_size: 高分辨率图像裁剪尺寸
        :参数 scaling_factor: 放大比例
        :参数 lr_img_type: 低分辨率图像预处理方式
        :参数 hr_img_type: 高分辨率图像预处理方式
        """
        self.split = split.lower()
        self.crop_size = crop_size
        self.scaling_factor = scaling_factor
        self.lr_img_type = lr_img_type
        self.hr_img_type = hr_img_type
 
        assert self.split in {'train', 'test'}
 
    def __call__(self, img):
        """
        对图像进行裁剪和下采样形成低分辨率图像
        :参数 img: 由PIL库读取的图像
        :返回: 特定形式的低分辨率和高分辨率图像
        """
 
        # 裁剪
        if self.split == 'train':
            # 从原图中随机裁剪一个子块作为高分辨率图像
            left = random.randint(1, img.width - self.crop_size)
            top = random.randint(1, img.height - self.crop_size)
            right = left + self.crop_size
            bottom = top + self.crop_size
            hr_img = img.crop((left, top, right, bottom))
        else:
            # 从图像中尽可能大的裁剪出能被放大比例整除的图像
            x_remainder = img.width % self.scaling_factor
            y_remainder = img.height % self.scaling_factor
            left = x_remainder // 2
            top = y_remainder // 2
            right = left + (img.width - x_remainder)
            bottom = top + (img.height - y_remainder)
            hr_img = img.crop((left, top, right, bottom))
 
        # 下采样（双三次差值）
        lr_img = hr_img.resize((int(hr_img.width / self.scaling_factor),
                                int(hr_img.height / self.scaling_factor)),
                               Image.BICUBIC)
 
        # 安全性检查
        assert hr_img.width == lr_img.width * self.scaling_factor and hr_img.height == lr_img.height * self.scaling_factor
 
        # 转换图像
        lr_img = convert_image(lr_img, source='pil', target=self.lr_img_type)
        hr_img = convert_image(hr_img, source='pil', target=self.hr_img_type)
 
        return lr_img, hr_img
```

## 5、评估结果

eval.py文件完整代码如下：

```python
#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from utils import *
from torch import nn
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from datasets import SRDataset
from models import SRResNet,Generator
import time

# 模型参数
large_kernel_size = 9   # 第一层卷积和最后一层卷积的核大小
small_kernel_size = 3   # 中间层卷积的核大小
n_channels = 64         # 中间层通道数
n_blocks = 16           # 残差模块数量
scaling_factor = 4      # 放大比例
ngpu = 1                # GP数量
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    
    # 测试集目录
    data_folder = "./data/"
    test_data_names = ["Set5","Set14", "BSD100"]

    # 预训练模型
    srgan_checkpoint = "./results/checkpoint_srgan.pth"
    #srresnet_checkpoint = "./results/checkpoint_srresnet.pth"

    # 加载模型SRResNet 或 SRGAN
    checkpoint = torch.load(srgan_checkpoint,map_location='cpu')
    generator = Generator(large_kernel_size=large_kernel_size,
                        small_kernel_size=small_kernel_size,
                        n_channels=n_channels,
                        n_blocks=n_blocks,
                        scaling_factor=scaling_factor)
    generator = generator.to(device)
    generator.load_state_dict(checkpoint['generator'])

    # 多GPU测试
    if torch.cuda.is_available() and ngpu > 1:
        generator = nn.DataParallel(generator, device_ids=list(range(ngpu)))
   
    generator.eval()
    model = generator
    # srgan_generator = torch.load(srgan_checkpoint)['generator'].to(device)
    # srgan_generator.eval()
    # model = srgan_generator

    for test_data_name in test_data_names:
        print("\n数据集 %s:\n" % test_data_name)

        # 定制化数据加载器
        test_dataset = SRDataset(data_folder,
                                split='test',
                                crop_size=0,
                                scaling_factor=4,
                                lr_img_type='imagenet-norm',
                                hr_img_type='[-1, 1]',
                                test_data_name=test_data_name)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1,
                                                pin_memory=True)

        # 记录每个样本 PSNR 和 SSIM值
        PSNRs = AverageMeter()
        SSIMs = AverageMeter()

        # 记录测试时间
        start = time.time()

        with torch.no_grad():
            # 逐批样本进行推理计算
            for i, (lr_imgs, hr_imgs) in enumerate(test_loader):
                
                # 数据移至默认设备
                lr_imgs = lr_imgs.to(device)  # (batch_size (1), 3, w / 4, h / 4), imagenet-normed
                hr_imgs = hr_imgs.to(device)  # (batch_size (1), 3, w, h), in [-1, 1]

                # 前向传播.
                sr_imgs = model(lr_imgs)  # (1, 3, w, h), in [-1, 1]                

                # 计算 PSNR 和 SSIM
                sr_imgs_y = convert_image(sr_imgs, source='[-1, 1]', target='y-channel').squeeze(
                    0)  # (w, h), in y-channel
                hr_imgs_y = convert_image(hr_imgs, source='[-1, 1]', target='y-channel').squeeze(0)  # (w, h), in y-channel
                psnr = peak_signal_noise_ratio(hr_imgs_y.cpu().numpy(), sr_imgs_y.cpu().numpy(),
                                            data_range=255.)
                ssim = structural_similarity(hr_imgs_y.cpu().numpy(), sr_imgs_y.cpu().numpy(),
                                            data_range=255.)
                PSNRs.update(psnr, lr_imgs.size(0))
                SSIMs.update(ssim, lr_imgs.size(0))


        # 输出平均PSNR和SSIM
        print('PSNR  {psnrs.avg:.3f}'.format(psnrs=PSNRs))
        print('SSIM  {ssims.avg:.3f}'.format(ssims=SSIMs))
        print('平均单张样本用时  {:.3f} 秒'.format((time.time()-start)/len(test_dataset)))

    print("\n")
```

​		最终在三个数据集上的测试结果如下表所示：

|                    | Set5   | Set14  | BSD100 |
| ------------------ | ------ | ------ | ------ |
| PSNR               | 29.021 | 25.652 | 24.652 |
| SSIM               | 0.839  | 0.693  | 0.644  |
| 平均单张样本用时/s | 0.595  | 0.480  | 0.349  |

上述测试值在性能上已经超越了很多算法，例如DRCNN、ESPCN等。这个主要归功于深度残差网络的作用，我们采用了16个残差模块进行学习，其对于图像的特征表示能力更加显著。

## 6、测试结果

test.py文件完整代码如下：

```python
from utils import *
from torch import nn
from models import SRResNet
import time
from PIL import Image
 
# 测试图像
imgPath = './results/test.jpg'
 
# 模型参数
large_kernel_size = 9   # 第一层卷积和最后一层卷积的核大小
small_kernel_size = 3   # 中间层卷积的核大小
n_channels = 64         # 中间层通道数
n_blocks = 16           # 残差模块数量
scaling_factor = 4      # 放大比例
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
 
if __name__ == '__main__':
 
    # 预训练模型
    #srgan_checkpoint = "./results/checkpoint_srgan.pth"
    srresnet_checkpoint = "./results/checkpoint_srresnet.pth"
 
    # 加载模型SRResNet 或 SRGAN
    checkpoint = torch.load(srresnet_checkpoint)
    srresnet = SRResNet(large_kernel_size=large_kernel_size,
                        small_kernel_size=small_kernel_size,
                        n_channels=n_channels,
                        n_blocks=n_blocks,
                        scaling_factor=scaling_factor)
    srresnet = srresnet.to(device)
    srresnet.load_state_dict(checkpoint['model'])
   
    srresnet.eval()
    model = srresnet
    # srgan_generator = torch.load(srgan_checkpoint)['generator'].to(device)
    # srgan_generator.eval()
    # model = srgan_generator
 
    # 加载图像
    img = Image.open(imgPath, mode='r')
    img = img.convert('RGB')
 
    # 双线性上采样
    Bicubic_img = img.resize((int(img.width * scaling_factor),int(img.height * scaling_factor)),Image.BICUBIC)
    Bicubic_img.save('./results/test_bicubic.jpg')
 
    # 图像预处理
    lr_img = convert_image(img, source='pil', target='imagenet-norm')
    lr_img.unsqueeze_(0)
 
    # 记录时间
    start = time.time()
 
    # 转移数据至设备
    lr_img = lr_img.to(device)  # (1, 3, w, h ), imagenet-normed
 
    # 模型推理
    with torch.no_grad():
        sr_img = model(lr_img).squeeze(0).cpu().detach()  # (1, 3, w*scale, h*scale), in [-1, 1]   
        sr_img = convert_image(sr_img, source='[-1, 1]', target='pil')
        sr_img.save('./results/test_srres.jpg')
 
    print('用时  {:.3f} 秒'.format(time.time()-start))
 
```

​		从网上选取一张低分辨率的证件照片进行实验测试。原图像大小为130x93，然后分别对其进行双线性上采样以及超分重建，图像放大4倍，变为520x372。对比效果如下所示：(从左到右依次为：原图、bicubic上采样、超分重建)

![](https://github.com/Perryen/CV/blob/main/Homework7/results/test.jpg?raw=true)

![test_bicubic](C:\Users\pxy\Desktop\SRGAN\results\test_bicubic.jpg)

![test_srgan](C:\Users\pxy\Desktop\SRGAN\results\test_srgan.jpg)

## 7、参考资料

https://blog.csdn.net/qianbin3200896/article/details/104181552

https://github.com/luzhixing12345/image-super-resolution
