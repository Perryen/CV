# cv第八次作业

## 一、作业要求

- https://github.com/eriklindernoren/PyTorch-GAN上任选一个感兴趣的GAN的程序，下载运行成功。 
- 阅读该程序的论文，写出阅读总结，并对应代码标注出论文中的公式以及网络所对应的代码，阐述清楚。
- 不超过两页。

## 二、GAN程序

进入所述网址会发现有很多研究论文中提出的生成对抗网络品种的 PyTorch 实现，先对其进行简单了解然后再决定阅读哪篇论文。

| Achievement                   | Abstract                                                     |
| ----------------------------- | ------------------------------------------------------------ |
| Auxiliary Classifier GAN      | 合成高分辨率逼真的图像，改进用于图像合成的生成对抗网络（GAN）训练的新方法 |
| Adversarial Autoencoder       | 对抗性自动编码器，它是一种概率自动编码器，它使用最近提出的生成对抗网络（GAN）通过将自动编码器的隐藏代码向量的聚合后验与任意先验分布匹配来执行变分推理。 |
| BEGAN                         | 一种新的均衡执行方法，该方法与从Wasserstein距离得出的损失配对，用于训练基于自动编码器的生成对抗网络。 |
| BicycleGAN                    | 许多图像到图像的转换问题都是模棱两可的，因为单个输入图像可能对应于多个可能的输出。在这项工作中，我们的目标是在条件生成建模设置中对可能输出的\emph{分布}进行建模。 |
| Boundary-Seeking GAN          | 介绍了一种使用离散数据训练 GAN 的方法，该方法使用来自鉴别器的估计差异度量来计算生成样本的重要性权重，从而为训练生成器提供策略梯度。 |
| Cluster GAN                   | 提出了ClusterGAN作为使用GAN进行聚类的新机制。通过抽样 来自独热编码变量和连续潜在变量混合的潜在变量，再加上 逆网络（将数据投影到潜在空间）与聚类特定损失联合训练，我们 能够在潜在空间中实现聚类。 |
| Conditional GAN               | 介绍了生成对抗网络的条件版本，它可以通过简单地馈送数据来构建，我们希望同时对生成器和鉴别器进行条件处理。 |
| Context-Conditional GAN       | 引入了一种简单的半监督学习方法，用于使用对抗性损失的基于绘画的图像 |
| Context Encoder               | 提出了一种由基于上下文的像素预测驱动的无监督视觉特征学习算法。通过与自动编码器的类比，我们提出了上下文编码器 - 一种卷积神经网络，经过训练以生成任意图像区域的内容，以周围环境为条件。 |
| Coupled GAN                   | 提出了耦合生成对抗网络（CoGAN）来学习多域图像的联合分布。    |
| CycleGAN                      | 提出了一种在没有配对示例的情况下学习将图像从源域X转换为目标域Y的方法。 |
| Deep Convolutional GAN        | 介绍了一类称为深度卷积生成对抗网络（DCGAN）的CNN，它们具有一定的架构约束，并证明它们是无监督学习的有力候选者。 |
| DiscoGAN                      | 提出了一种基于生成对抗网络的方法，该方法学习发现不同域之间的关系（DiscoGAN）。利用发现的关系，我们提出的网络成功地将风格从一个域转移到另一个域，同时保留了方向和人脸识别等关键属性。 |
| DRAGAN                        | 我们建议将GAN训练动力学研究为遗憾最小化，这与流行的观点相反，即真实分布和生成分布之间的分歧始终最小化。 |
| DualGAN                       | 开发了一种新颖的双GAN机制，使图像翻译人员能够从来自两个域的两组未标记图像中进行训练。 |
| Energy-Based GAN              | 引入了“基于能量的生成对抗网络”模型（EBGAN），该模型将判别器视为一种能量函数，它将低能量归因于数据流形附近的区域，并将较高能量归因于其他区域。 |
| Enhanced Super-Resolution GAN | 超分辨率生成对抗网络（SRGAN）是一项开创性的工作，能够在单图像超分辨率期间生成逼真的纹理。然而，幻觉的细节往往伴随着令人不快的伪影。为了进一步提高视觉质量，我们深入研究了SRGAN的三个关键组成部分 - 网络架构，对抗性损失和感知损失，并对其进行了改进以得出增强型SRGAN（ESRGAN）。 |
| GAN                           | 提出了一个通过对抗过程估计生成模型的新框架                   |
| InfoGAN                       | 本文描述了InfoGAN，这是生成对抗网络的信息理论扩展，能够以完全无监督的方式学习解开的表示 |
| Least Squares GAN             | 提出了最小二乘生成对抗网络（LSGANs），它采用最小二乘损失函数作为判别器。 |
| MUNIT                         | 提出了一个多模态无监督图像到图像转换（MUNIT）框架            |
| Pix2Pix                       | 研究了条件对抗网络作为图像到图像转换问题的通用解决方案       |
| PixelDA                       | 提出了一种新的方法，该方法以无监督的方式学习像素空间从一个域到另一个域的转换。 |
| Relativistic GAN              | 提出了一个变体，其中鉴别器平均估计给定真实数据比虚假数据更真实的概率 |
| Semi-Supervised GAN           | 我们通过强制鉴别器网络输出类标签，将生成对抗网络（GAN）扩展到半监督上下文。 |
| Softmax GAN                   | Softmax GAN是生成对抗网络（GAN）的新颖变体。Softmax GAN的关键思想是将原始GAN中的分类损失替换为单个批次样本空间中的softmax交叉熵损失。 |
| StarGAN                       | 提出了StarGAN，这是一种新颖且可扩展的方法，可以仅使用单个模型对多个域进行图像到图像的转换。 |
| Super-Resolution GAN          | 提出了SRGAN，一种用于图像超分辨率（SR）的生成对抗网络（GAN）。据我们所知，它是第一个能够推断出 4 倍放大因子的照片级逼真自然图像的框架。 |
| UNIT                          | 做了一个共享潜在空间假设，并提出了一个基于耦合GAN的无监督图像到图像转换框架。 |
| Wasserstein GAN               | 引入了一种名为WGAN的新算法，这是传统GAN训练的替代方案。在这个新模型中，我们展示了我们可以提高学习的稳定性，摆脱模式崩溃等问题，并提供对调试和超参数搜索有用的有意义的学习曲线。 |
| Wasserstein GAN GP            | 提出了一种剪切权重的替代方案：惩罚批评家关于其输入的梯度规范。我们提出的方法比标准WGAN性能更好，并且能够稳定训练各种GAN架构， |
| Wasserstein GAN DIV           | 提出了一种新的Wasserstein散度（W-div），它是W-met的宽松版本，不需要k-Lipschitz约束。 |

## 三、选择程序

| Generative Adversarial Networks | 地址                                |
| ------------------------------- | ----------------------------------- |
| GAN论文                         | https://arxiv.org/abs/1406.2661     |
| GAN源码                         | https://github.com/AYLIEN/gan-intro |



## 四、论文阅读

文章的题目为Generative Adversarial Networks，简单明了。[参考译文](https://blog.csdn.net/weixin_41109655/article/details/119906048)

首先Generative，我们知道在机器学习中含有两种模型，生成式模型（Generative Model）和判别式模型（Discriminative Model）。生成式模型研究的是联合分布概率，主要用来生成具有和训练样本分布一致的样本；判别式模型研究的是条件分布概率，主要用来对训练样本分类，两者具体的区别在这里不再赘述。判别式模型因其多方面的优势，在以往的研究和应用中占了很大的比例，尤其是在目标识别和分类等方面；而生成式模型则只有很少的研究和发展，而且模型的计算量也很大很复杂，实际上的应用也就比判别式模型要少很多。而本文就是一篇对生成式模型的研究，并且这个研究一提出来便在整个机器学习界掀起了轩然大波，连深度学习的鼻祖Yann LeCun都说生成对抗网络是让他最激动的深度学习进展，也可见这项研究的伟大，当然也说明了这篇文章做描述的生成式模型是和传统的生成式模型是很不同的，那么不同之处在哪里呢？

然后Adversarial，这边是这篇文章与传统的生成式模型的不同之处；对抗，谁与之对抗呢？当然就是判别式模型，那么，如何对抗呢？这也就是这篇文章主要的研究内容。在原文中，作者说了这么一段话：“The generative model can be thought of as analogous to a team of counterfeiters, trying to produce fake currency and use it without detection, while the discriminative model is analogous to the police, trying to detect the counterfeit currency.”大意就是，生成式模型就像是罪犯，专门造假币，判别式模型就像是警察，专门辨别假币，那生成式模型就要努力提高自己的造假技术使造的假币不被发现，而判别式模型就要努力提高自己辨别假币的能力。这就类似于一个二人博弈问题。最终的结果就是两人达到一个平衡的状态，这就是对抗。
那么这样的对抗在机器里是怎样实现的呢？

在论文中，真实数据x的分布为1维的高斯分布p(data)，生成器G为一个多层感知机，它从随机噪声中随机挑选数据z输入，输出为G(z)，G的分布为p(g)。判别器D也是一个多层感知机，它的输出D(x)代表着判定判别器的输入x属于真实数据而不是来自于生成器的概率。再回到博弈问题，G的目的是让p(g)和p(data)足够像，那么D就无法将来自于G的数据鉴别出来，即D(G(z))足够大；而D的目的是能够正确的将G(z)鉴别出来，即使D(x)足够大，且D(G(z))足够小，即(D(x)+(1-D(G(z))))足够大。
那么模型的目标函数就出来了，对于生成器G，目标函数为（梯度下降）：
$$
\nabla_{\theta_g} \frac{1}{m} \sum_{i=1}^m \log \left(1-D\left(G\left(z^{(i)}\right)\right)\right)
$$
而对于判别器D，目标函数为（梯度上升）：
$$
\nabla_{\theta_d} \frac{1}{m} \sum_{i=1}^m\left[\log D\left(x^{(i)}\right)+\log \left(1-D\left(G\left(z^{(i)}\right)\right)\right)\right] .
$$
于是模型就成了优化这两个目标函数的问题了。这样的话就只需要反向传播来对模型训练就好了，没有像传统的生成式模型的最大似然函数的计算、马尔科夫链或其他推理等运算了。

这就是这篇文章大概的思路，具体的内容细节以及一些数学公式的推导可以仔细阅读论文原文应该都是比较好理解的。

## 五、代码分析

该代码在TensorFlow上利用生成对抗网络来近似1维的高斯分布。

在代码的开头便定义了这个均值为4，方差为0.5的高斯分布：

```python
class DataDistribution(object):
    def __init__(self):
        self.mu = 4
        self.sigma = 0.5
 
    def sample(self, N):
        samples = np.random.normal(self.mu, self.sigma, N)
        samples.sort()
        return samples
```

以及一个初始的生成器分布：

```python
class GeneratorDistribution(object):
    def __init__(self, range):
        self.range = range
 
    def sample(self, N):
        return np.linspace(-self.range, self.range, N) + \
            np.random.random(N) * 0.01
```

然后定义了一个线性运算：

```python
def linear(input, output_dim, scope=None, stddev=1.0):
    norm = tf.random_normal_initializer(stddev=stddev)
    const = tf.constant_initializer(0.0)
    with tf.variable_scope(scope or 'linear'):
        w = tf.get_variable('w', [input.get_shape()[1], output_dim], initializer=norm)    #输入的第二维作为数据维度
        b = tf.get_variable('b', [output_dim], initializer=const)
        return tf.matmul(input, w) + b
```

即简单的y=wx+b的运算，代码中使用了tf.variable_scope()，实际上这是使用了一个名为scope的变量空间，再通过tf.get_variable()定义该空间下的变量，变量的名字为“scope/w”和“scope/b”，这在很复杂的模型中有利于简化代码，并且方便用来共享变量，在后面也用到了共享变量。

接下来，定义了生成器运算：

```python
def generator(input, h_dim):
    h0 = tf.nn.softplus(linear(input, h_dim, 'g0'))
    h1 = linear(h0, 1, 'g1')
    return h1
```

判别器运算：

```python
def discriminator(input, h_dim, minibatch_layer=True):
    h0 = tf.tanh(linear(input, h_dim * 2, 'd0'))
    h1 = tf.tanh(linear(h0, h_dim * 2, 'd1'))
 
    # without the minibatch layer, the discriminator needs an additional layer
    # to have enough capacity to separate the two distributions correctly
    if minibatch_layer:
        h2 = minibatch(h1)
    else:
        h2 = tf.tanh(linear(h1, h_dim * 2, scope='d2'))
 
    h3 = tf.sigmoid(linear(h2, 1, scope='d3'))
    return h3
```

这里有一个minibatch，minibatch的内容在原始论文中并没有提到，在后面我们会说到这个。总的来说生成器和判别器都是很简单的模型。

```python
def optimizer(loss, var_list, initial_learning_rate):
    decay = 0.95
    num_decay_steps = 150
    batch = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(
        initial_learning_rate,
        batch,
        num_decay_steps,
        decay,
        staircase=True
    )
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(
        loss,
        global_step=batch,
        var_list=var_list
    )
    return optimizer
```

定义了模型的优化器，模型的学习率使用指数型衰减，模型使用梯度下降来进行损失函数的优化。

接下来定义了GAN类，在GAN类中主要介绍以下几个部分：

```python
def _create_model(self):
    # In order to make sure that the discriminator is providing useful gradient
    # information to the generator from the start, we're going to pretrain the
    # discriminator using a maximum likelihood objective. We define the network
    # for this pretraining step scoped as D_pre.
    with tf.variable_scope('D_pre'):                
        self.pre_input = tf.placeholder(tf.float32, shape=(self.batch_size, 1))
        self.pre_labels = tf.placeholder(tf.float32, shape=(self.batch_size, 1))
        D_pre = discriminator(self.pre_input, self.mlp_hidden_size, self.minibatch)
        self.pre_loss = tf.reduce_mean(tf.square(D_pre - self.pre_labels))
        self.pre_opt = optimizer(self.pre_loss, None, self.learning_rate)
    # This defines the generator network - it takes samples from a noise
    # distribution as input, and passes them through an MLP.
    with tf.variable_scope('G'):
        self.z = tf.placeholder(tf.float32, shape=(self.batch_size, 1))
        self.G = generator(self.z, self.mlp_hidden_size)
    # The discriminator tries to tell the difference between samples from the
    # true data distribution (self.x) and the generated samples (self.z).
    #
    # Here we create two copies of the discriminator network (that share parameters),
    # as you cannot use the same network with different inputs in TensorFlow.
    with tf.variable_scope('D') as scope:
        self.x = tf.placeholder(tf.float32, shape=(self.batch_size, 1))
        self.D1 = discriminator(self.x, self.mlp_hidden_size, self.minibatch)
        scope.reuse_variables()
        self.D2 = discriminator(self.G, self.mlp_hidden_size, self.minibatch)
    # Define the loss for discriminator and generator networks (see the original
    # paper for details), and create optimizers for both
    self.loss_d = tf.reduce_mean(-tf.log(self.D1) - tf.log(1 - self.D2))
    self.loss_g = tf.reduce_mean(-tf.log(self.D2))
 
    vars = tf.trainable_variables()
    self.d_pre_params = [v for v in vars if v.name.startswith('D_pre/')]
    self.d_params = [v for v in vars if v.name.startswith('D/')]
    self.g_params = [v for v in vars if v.name.startswith('G/')]
 
    self.opt_d = optimizer(self.loss_d, self.d_params, self.learning_rate)
    self.opt_g = optimizer(self.loss_g, self.g_params, self.learning_rate)
```

与文章中不同的是，这里使用了三种模型：D_pre、G和D。
D_pre是在训练G之前，对D先进行一个预训练，这样能够在训练初期为G提供足够的梯度来进行更新。

G是生成器模型，通过将一个噪声数据输入到这个多层感知机，输出一个具有p(g)分布的数据。

D是判别器模型，代码中用到了scope.reuse_variables()，目的是共享变量，因为真实数据和来自生成器的数据均输入到了判别器中，使用同一个变量，如果不共享，那么将会出现严重的问题，模型的输出代表着输入来自于真是数据的概率。

然后是两个损失函数，三个模型的参数集以及两个优化器。

```python
def train(self):
    with tf.Session() as session:
        tf.global_variables_initializer().run()
        # pretraining discriminator
        num_pretrain_steps = 1000
        for step in range(num_pretrain_steps):
            d = (np.random.random(self.batch_size) - 0.5) * 10.0
            labels = norm.pdf(d, loc=self.data.mu, scale=self.data.sigma)
            pretrain_loss, _ = session.run([self.pre_loss, self.pre_opt], {
                self.pre_input: np.reshape(d, (self.batch_size, 1)),
                self.pre_labels: np.reshape(labels, (self.batch_size, 1))
            })
        self.weightsD = session.run(self.d_pre_params)
 
        # copy weights from pre-training over to new D network
        for i, v in enumerate(self.d_params):
            session.run(v.assign(self.weightsD[i]))
 
        for step in range(self.num_steps):
            # update discriminator
            x = self.data.sample(self.batch_size)
            z = self.gen.sample(self.batch_size)
            loss_d, _ = session.run([self.loss_d, self.opt_d], {
                self.x: np.reshape(x, (self.batch_size, 1)),
                self.z: np.reshape(z, (self.batch_size, 1))
            })
 
            # update generator
            z = self.gen.sample(self.batch_size)
            loss_g, _ = session.run([self.loss_g, self.opt_g], {
                self.z: np.reshape(z, (self.batch_size, 1))
            })
 
            if step % self.log_every == 0:
                print('{}: {}\t{}'.format(step, loss_d, loss_g))
         
            self._plot_distributions(session)
```

训练过程包含了先前三个模型的训练，先进行1000步的D_pre预训练，预训练利用随机数作为训练样本，随机数字对应的正态分布的值作为训练标签，损失函数为军方误差，训练完成后，将D_pre的参数传递给D，然后在同时对G和D进行更新。
之后还有一些从训练完成的模型中采样、打印等函数操作，代码也比较简单，这里就不进行解析了。

然后我们可以运行代码来看看效果:

![image-20221120230932862](https://cdn.jsdelivr.net/gh/Perryen/Typora_Picture@master/img/image-20221120230932862.png)



其中绿线代表真实数据分布，红线代表生成数据分布，蓝线代表判别情况，数值上为判别器将真实数据判定为真实数据的概率。从分布上我们可以看到，生成数据在真实数据的平均值附近分布密集，并且比较窄，而且判别器的概率也高于百分之50。并没有达到一个很好的效果。这是因为判别器只能同时对单一的数据进行处理，并不能很好的反应数据集的分布情况。
