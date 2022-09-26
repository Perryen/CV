# 作业要求

1、安装OpenCV, Numpy, Matplotlib 

2、学习Numpy和CV2工具包的使用

3、用OpenCV检测人脸 

- 读取图片 
- 调用opencv的cv2.CascadeClassifier检测出人脸位置 
- 画出方框，写上文本，显示图片

# 作业1：安装相关工具包

使用Anaconda对工具包进行统一管理，通过Anaconda Navigator进行相关工具包的下载：

![image-20220926100954774](https://cdn.jsdelivr.net/gh/Perryen/Typora_Picture//img/202209261010891.png)

![image-20220926101033165](https://cdn.jsdelivr.net/gh/Perryen/Typora_Picture//img/202209261010205.png)

![image-20220926101106341](https://cdn.jsdelivr.net/gh/Perryen/Typora_Picture//img/202209261011377.png)

备注：科学上网不顺利时，可以使用以下国内镜像源（清华源）：

```
https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge 
https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/msys2/
```

# 作业2：学习工具包的使用

## Numpy

NumPy是Python的一个用于科学计算的基础包。它提供了多维数组对象，多种衍生的对象（例如隐藏数组和矩阵）和一个用于数组快速运算的混合的程序，包括数学，逻辑，排序，选择，I/O，离散傅立叶变换，基础线性代数，基础统计操作，随机模拟等等。

### 1、Vectorization

向量化计算(vectorization)，也叫vectorized operation，也叫array programming，说的是一个事情：将多次for循环计算变成一次计算。

![img](https://cdn.jsdelivr.net/gh/Perryen/Typora_Picture//img/202209261054682.jpeg)

将迭代语句转换为基于矢量的运算称为矢量化。它更快，因为现代CPU针对此类操作进行了优化。

### 2、Version

```python
import numpy as np

print(np.__version__)
```



### 3、Ndarray

NumPy包的核心是ndarray对象。它封装了n维同类数组。很多运算是由编译过的代码来执行的，以此来提高效率。NumPy数组和标准Python序列有以下几点重大区别：

- NumPy数组创建的时候有一个固定的大小，不像Python列表可以动态地增加要素。改变一个ndarray的大小会创建一个新的数组并删除原数组。
- NumPy数组中的要素必须是同一个数据类型，从而在内存中是同样的大小。唯一例外是可以由Python（包括NumPy）对象作为要素组成数组，因此允许有要素大小不同的数组的存在。
- NumPy数组更有利于大规模数据的高级数学运算。通常来说，这些运算执行更高效，并且代码量比用Python自带的序列来实现更少。
- 越来越多的科学和数学Python包使用NumPy数组；虽然这些包通常支持Python序列输入，但它们通常在处理前把输入转化为NumPy数组。换据话说，想要高效地使用当今很多（甚至是大部分）基于Python的科学或数学计算软件，只是了解如何使用Python内置的序列类型已经不够了，你必须知道如何使用NumPy数组。

![image-20220926105942971](https://cdn.jsdelivr.net/gh/Perryen/Typora_Picture//img/202209261059009.png)

数组相关重要操作（运行截图略）：Dimensions in Arrays、Check Number of Dimensions、Access Array Elements、Negative Indexing、Slicing Arrays、Copy vs View、Shape、Reshape等

### 4、Data Types in NumPy

NumPy有一些额外的数据类型，并用一个字符引用数据类型，如i表示整数，u表示无符号整数等。

以下是NumPy中所有数据类型的列表以及用于表示它们的字符：

```
i - integer

b - boolean

u - unsigned integer

f - float

c - complex float

m - timedelta

M - datetime

O - object

S - string

U - unicode string

V - fixed chunk of memory for other type ( void )
```

数据类型重要操作（运行截图略）：Check、Convert、Random number等

## OpenCV

OpenCV是Intel®开源计算机视觉库。它由一系列 C 函数和少量 C++ 类构成，实现了图像处理和计算机视觉方面的很多通用算法。

> 重要特性：
>
> OpenCV 拥有包括 300 多个C函数的跨平台的中、高层 API。它不依赖于其它的外部库——尽管也可以使用某些外部库。
>
> OpenCV 对非商业应用和商业应用都是免费（FREE）的。（细节参考 [BSD License](http://en.wikipedia.org/wiki/BSD_license)）。
>
> OpenCV 为Intel® Integrated Performance Primitives (IPP) 提供了透明接口。 这意味着如果有为特定处理器优化的的 IPP 库， OpenCV 将在运行时自动加载这些库。 更多关于 IPP 的信息请参考： http://www.intel.com/software/products/ipp/index.htm

### 1、读取图片并显示

![image-20220926130200398](https://cdn.jsdelivr.net/gh/Perryen/Typora_Picture//img/202209261302478.png)

![image-20220926130255079](https://cdn.jsdelivr.net/gh/Perryen/Typora_Picture//img/202209261302140.png)

### 2、RGB与BGR

![image-20220926130358367](https://cdn.jsdelivr.net/gh/Perryen/Typora_Picture//img/202209261303462.png)

### 3、图像保存

![image-20220926130435801](https://cdn.jsdelivr.net/gh/Perryen/Typora_Picture//img/202209261304837.png)

### 4、颜色空间转换

![image-20220926130514276](https://cdn.jsdelivr.net/gh/Perryen/Typora_Picture//img/202209261305347.png)

![image-20220926130537495](https://cdn.jsdelivr.net/gh/Perryen/Typora_Picture//img/202209261305554.png)

![image-20220926130555457](https://cdn.jsdelivr.net/gh/Perryen/Typora_Picture//img/202209261305519.png)

### 5、在图中画一条蓝线和一个红点

![image-20220926130650208](https://cdn.jsdelivr.net/gh/Perryen/Typora_Picture//img/202209261306276.png)

![image-20220926130753830](https://cdn.jsdelivr.net/gh/Perryen/Typora_Picture//img/202209261307054.png)

# 作业3：用OpenCV检测人脸

实验结果：

![image-20220926132526235](https://cdn.jsdelivr.net/gh/Perryen/Typora_Picture//img/202209261325362.png)

实验代码：

```python
import cv2
import matplotlib.pyplot as plt # 用于绘制图像

# 实例化人脸分类器
face_cascade = cv2.CascadeClassifier('C:/Users/49740/.conda/envs/py37/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
# 读取测试图片
img = cv2.imread('./Lena.png',cv2.IMREAD_COLOR)
text='Lena'
# 将原彩色图转换成灰度图
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 开始在灰度图上检测人脸，输出是人脸区域的外接矩形框
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=1)
# 遍历人脸检测结果
for (x,y,w,h) in faces:
    # 在原彩色图上画人脸矩形框
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
    # 在原彩色图上写上文本
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (128, 128, 0), 2)
# 对图像进行处理，并显示其格式
finalimg = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
plt.figure(figsize=(12,12))
plt.imshow(finalimg)
# 隐藏坐标显示
plt.axis("off")
# 显示图像
plt.show()
```

