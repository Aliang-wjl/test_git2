# ==脉冲神经网络（Spiking Neural Network，SNN）基础知识概述==

[参考链接](https://zhuanlan.zhihu.com/p/687031591)

1. 什么是SNN
   ANN的神经元在 **功能和复杂性上与生物神经元相比还有很大的差距**
   **为了更加贴近生物神经系统的工作方式**，就诞生了脉冲神经网络SNN。

2. 为何会出现SNN?
   ANN在执行大规模数据处理或复杂任务时，因为它依赖于**具体的任务**（例如，一个简单如区分猫狗的任务和复杂如语言理解任务需要的算力不是一个level）、**模型大小**（更大的模型需要更多的内存存储和更多的计算步骤来进行前向传播和反向传播）、**硬件**等因素，**它的功耗被普遍认为相当高且不稳定。**举个栗子，在训练一个大型的深度学习模型时，能耗可以从几千瓦时（kWh）到数十千瓦时不等，甚至更高。其中大型模型可能需要数周时间在具有高性能计算能力的硬件上运行。可是，你ANN不是模仿人脑的吗，要知道人脑的能耗跟你ANN可不是一个level的。你知道人脑的平均功耗是多少吗？科学研究表明：大约为20瓦！在大脑的复杂性和处理信息的能力如此优越的情况下，这样的功耗可是非常惊人的。

3. SNN是怎样工作的？
   实现SNN的方式： **以ANN的框架为基础（比如我们需要考虑到神经元的设计，前向传播和反向传播的过程），再通过模拟生物神经网络中的工作机制，去搭建我们的SNN。**

   * 脉冲神经元
     ANN对数据的处理方式：y = f(wx+b)
     
     **由这个公式，我们便可以推断出很多情况下脉冲神经元的膜电压是如何变化的，请你思考如下几个问题，思考完了再看答案(需要在Python中得到输出，假设现在我们设计的脉冲神经元没有阈值，它的膜电压可以无限高。**
     
     对于上述问题可以通过模拟神经元来查看具体结果，再次只给出答案。
     第一个：呈指数下降的趋势，最终变为0V
     第二个：刚开始是0，接着变为一个很高的值，然后呈指数下降，最终为0V
     第三个：刚开始是0，接着变为一个很高的值并开始下降，接着又变为更高的值，下降，为0
     第四个：到了一定程度就维持不变了，这个取决于R的值。
     
   * 



# ==入门学习：==

在人工智能领域，神经网络已经成为了主导的机器学习模型之一。从第一代到第二代，再到现在的第三代，神经网络的演进和深化对人工智能的发展起到了巨大的推动作用。

首先，让我们理解一下什么是神经网络。神经网络是一种模拟人脑工作机制的算法模型，它由许多神经元（或称为节点）相互连接而成。每个神经元接收输入信号，并通过计算得出输出信号。这些输出信号又会影响其他神经元的输入，形成一种复杂的反馈系统。

第一代神经网络，即静态神经网络，主要应用在模式识别和分类任务上。然而，由于它们无法处理时间序列数据，因此在很多复杂任务上表现不佳。（简单的线性连接，无隐藏层）

第二代神经网络，即前馈神经网络（Feedforward Neural Network, FNN），通过引入隐藏层解决了这个问题。FNN可以处理输入数据的时间依赖关系，因此在语音识别、自然语言处理等任务上表现优秀。然而，尽管FNN有诸多优点，但它的训练过程需要大量的标签数据，这在很多情况下是不可行的。同时，FNN也无法模拟生物神经网络的动态性质和可塑性。

这时，我们便需要引入第三代神经网络——脉冲神经网络（Spiking Neural Network, SNN）。
脉冲神经网络是一种模拟生物神经元行为的神经网络模型。在生物神经元中，信息的传递是通过电信号进行的，这个电信号的强度和持续时间都会影响神经元的输出。SNN模仿了这个过程，它以脉冲（即时间点）的形式传递信息，通过调整脉冲的频率和时间来模拟神经元的电信号。
SNN的优点在于其生物可解释性和计算效率。由于SNN是以脉冲形式传递信息，因此它可以更好地模拟生物神经元的行为，具有更高的生物可解释性。同时，由于SNN中的计算主要集中在脉冲的产生和接收上，因此它的计算效率比传统的神经网络要高得多。

此外，SNN还具有一些其他的特点，例如对噪声和干扰的鲁棒性以及节能性。这些特性使得SNN在很多应用领域中都有着广阔的前景，例如自动驾驶、物联网、机器人等领域。

然而，SNN也存在一些挑战和难点。例如，如何设计和训练有效的SNN需要深入的理解生物神经科学和机器学习的知识；同时，由于SNN的计算方式和传统神经网络不同，因此需要开发新的算法和工具来支持SNN的训练和应用。

总的来说，脉冲神经网络（SNN）作为第三代神经网络，以其独特的生物可解释性和计算效率受到了广泛的关注和研究。虽然仍存在一些挑战和难点需要我们去面对和解决，但是随着技术的不断进步和应用场景的扩大，我们有理由相信SNN将会在未来的智能计算领域中扮演更为重要的角色。

# ==SNN和ANN的区别==

1. 编码方式
   * ANN是纯数字编码，神经元的输入和输出都是数值，例如浮点数、定点数或模拟值。ANN采用传统的连续数值编码和处理方式，信息以输入层节点的数值和网络连接的权重进行传递和处理。
   * SNN是脉冲时间编码，其神经元的输入和输出都是脉冲序列。这是两者在编码方式上的本质区别。SNN使用时间编码来表示信息，通过神经元产生的脉冲事件来传达信息。这意味着SNN的信息传递和处理是基于脉冲信号的离散事件，与ANN的连续数值处理方式不同。
2. 计算模型不同
   * SNN是基于事件驱动的计算模型。只有在接收到输入脉冲时，脉冲神经元才会执行计算。因此，SNN的计算过程更加==稀疏和异步==。
   * ANN是基于连续数值的计算模型，神经元通过激活函数和各种权重进行==连续==的数值计算。
3. 模拟生物神经系统的程度不同
   * SNN在结构和功能上更接近生物神经系统，它模拟了大脑中神经元之间通过脉冲信号进行通信的方式，并且具有更好的生物相似性。
   * ANN是更抽象的数学模型，其设计更多地侧重于数据的数值处理和模式识别，对生物神经系统的模拟程度较低。
4. 应用领域不同
   * SNN在处理实时数据流、感知处理和事件驱动的任务方面表现更出色，例如用于视觉和听觉处理等方面。
   * ANN在传统的监督学习、分类、回归等任务中表现更加突出，例如在各种大型数据集上的图像识别和语音识别任务中有着广泛的应用。
5. 结构和训练方法不同
   * ANN通常采用==BP学习规则==，即反向传播算法，权重更新仅与后级神经元反馈回来的误差有关，与前级神经元无关。
   * SNN则常采用==STDP学习规则==（脉冲时序相关塑性），这种规则需要同时考虑前级和后级神经元的发放情况，共同决定权重的改变量。值得注意的是，学习规则并不是两者的本质区别，因为两者都有多种训练方法，而且由于STDP的表现一直不理想，现在很多SNN也选择用BP来训练，然后再做一次==ANN到SNN的映射==。
6. 应用和硬件效率
   * SNN比ANN更具硬件友好性和能源效率，尤其对便携式设备来说更具吸引力。SNN的脉冲神经元传递函数通常是不可微的，这阻止了反向传播。下面是几个方法：
     * 引入近似可微的激活函数：研究人员尝试寻找近似于原始不可微激活函数的可微函数。这些近似函数可以在实践中替代原始的不可微函数，使得反向传播算法可以应用于SNN。例如，可以使用类似于sigmoid或tanh的函数来替代不可微激活函数。
     * 另类的反向传播算法：研究人员提出了一些特定的反向传播算法，这些算法专门适用于SNN中不可微的传递函数。这些算法在处理不可微函数时有更好的性能，例如，可以通过近似的梯度计算来训练SNN。
     * 基于优化的方法：一些研究者开发了基于优化的方法，通过优化来近似处理不可微函数，以使其适用于反向传播。这些方法通常需要对传递函数进行一定程度的变换或近似，以使得梯度可以被计算。
   * 由于ANN的非线性激活函数存在导数，使得它可以使用基于梯度的优化方法进行训练，堆叠不止一层的网络。这种网络结构在准确性方面仍领先于SNN，但差距正在缩小。

# ==如何系统学习SNN：==

1. 理解基础概念：

   学习人工神经网络（ANN）的基本概念，因为SNN是ANN的一种变体。

   了解生物神经网络的基本工作原理，因为SNN的设计灵感来源于生物学。

2. 学习神经元模型：

   熟悉脉冲神经元的模型，了解神经元是如何响应输入和产生脉冲的。

   掌握SNN中常用的神经元模型，例如脉冲神经元的LIF（Leaky Integrate-and-Fire）模型。
   关于LIF模型，下面是一些说明:

   * LIF模型绘制出来的变化几乎是一条很直的线
   * EIF模型的曲线是先变快，中间变慢，接着变快
   * QIF模型相比EIF模型，区别就在于其后半程变得非常快
   * ADEX模型最大的区别是两个脉冲间距是由窄变宽，逐渐稳定。这就是所谓的自适应，同时这种自适应其实可以加到多种模型，进行形成自己的模型。

3. 了解脉冲编码：

   * 学习脉冲编码的概念，即如何使用脉冲来表示信息。
   * 了解脉冲的时序性如何在SNN中被利用，以及脉冲的编码方式对信息处理的影响。

4. 研究SNN的学习规则：

   * 了解SNN中常用的学习规则，尤其是基于脉冲时序依赖性的突触可塑性规则（STDP）。
     * STDP是一个时序非对称形式的Hebb学习法则，是由突触前和突触后神经元峰值之间的紧密时间相关性影响的。与其他形式的突触可塑性一样，人们普遍认为它是大脑学习和信息存储的基础，也是大脑发育过程中神经元回路的发展和完善的基础。如果突触前脉冲在突触后脉冲前几毫秒内到达，会导致Long-Term Potentiation(LTP)。反之，会引起LTD。突触的变化作为突触前和突触后动作电位的相对时间的函数被称为STDP函数，在不同的突触类型之间变化。
     * LTP（Long-Term Potentiation）和LTD（Long-Term Depression）是生物神经系统中的两种突触可塑性现象。LTP（长时程增强）指的是在神经元之间的突触连接中，持续性的、增强性的刺激会导致该突触传递效率的长期增强。LTP被认为在学习和记忆中发挥着重要作用，它可以加强神经元之间的连接，提高信号传递的效率，并因此加强相应的记忆或行为。LTD（长时程抑制）则是指持续性的、特定的刺激可以导致突触传递效率的长期抑制。LTD可以削弱神经元之间的连接强度，减少信号传递的效率，这对于平衡神经元之间的连接以及避免过度兴奋和过度抑制是非常重要的。这两种现象是突触可塑性的表现，是神经元之间交流和学习的基础。LTP和LTD的发现对于理解神经系统中的学习、记忆和认知功能具有重要意义。
     * ==有意思的一个点：== 单次刺激不能形成长期记忆，但是多次刺激却可以，但SNN经常提到的是关于记忆的说法，却没有提到所谓的**创新性或者泛化性**，但是ANN追求的确是**泛化性**。

   * 研究SNN如何通过调整神经元之间的连接权重来学习和适应输入模式。

5. 阅读相关文献和教材：

   * 阅读关于脉冲神经网络的研究论文、书籍和教程，以深入了解SNN的理论和应用。
   * 探索SNN在神经形态计算、脑机接口等领域的应用案例。

6. 实践编程：

   * 使用编程语言（如Python）和相应的神经网络框架（如Brian2、NEST、或者其他支持SNN的框架）实现简单的SNN模型。
   * 通过实际编程实践加深对SNN概念的理解，尝试不同的学习任务和模型结构。

7. 深入研究应用领域：

   * 探索SNN在特定领域的应用，例如神经形态计算、事件驱动的传感器处理等。

   * 研究相关领域的文献，了解SNN在实际应用中的性能和优势。

8. “Fire together, wire together”

   * “Fire together, wire together”（共同激活，连线加强）这句话经常用来描述神经元之间的突触可塑性。它表达了一种神经科学上的观点，即当两个神经元同时激活时，它们之间的突触连接会得到加强，从而促进信息传递和学习。
   * 这句话主要被用来形容神经元之间的Hebbian学习规则，这是一种描述突触可塑性的基本原理。根据Hebbian学习规则，当突触前神经元的激活与突触后神经元的激活同时发生时，突触连接的强度会增加。这意味着如果两个神经元的活动模式经常同时出现，它们之间的连接将被加强，从而增加它们之间的通信效率，这可以促进学习和记忆的形成。

9. 稳态的维持，产生记忆和学习的效果——功能可塑性，结构可塑性？
   功能可塑性的缺点是：可以将weight从有变成 0 ，但不可能从0开始生成新的权重。

# ==SNN概览==

[B站参考视频](https://www.bilibili.com/video/BV1pu411j7ZJ/?spm_id_from=333.999.0.0&vd_source=3be4a0eefc99c3ec4109b6e4f90586d1)

神经网络最初的样子为：感知机。 目的是为了做图像分类。

ANN 激活函数  Analog Neural Networks.

SNN  Spiking Neural Networks     IF神经元   Integrated and Fire model       Membrane potential 膜电位Thresholding  function (Issue)

SNN每次传递都是一个脉冲，那么脉冲的编码方式应该是什么样的？  code mechanisms
**Rate Coding：** higher input values produce spike more in quantity
						  Pros:easy to use,robust, common
						  Cons:energy  (最常用，值很高会有很多峰值)
**Temporal(Pulse) Coding**：

higher input values cause spike to spike earlier
Pros: energy efficient ( 脉冲的次数相比速率编码少了很多，因为值越高不再是产生的更多的脉冲，而是产生脉冲的时间更早 )
Cons: underdeveloped, lack of applications (scalability)   （当前研究较少，且在实际应用中遇到了一些问题，深层次网络出现峰值丢失的问题）

LIF（Leaky Integrate and Fire Model）：
Bio-inspired  生物启发
Monitors membrance potential  检测膜电位
Integrate or leak based on presence/absence of input spike  根据输入脉冲存在或者不存在进行集成或者泄露

STDP spike time dependent Plasticity


通过input时间和output时间之间的关系去决定不同node之间的信号强度。
如果一个Input与Output距离的时间越近，那他们的关系强度就更大。  在 I1和I3出现后，O1出现，之后I2才出现，故I2与O1的关系为负关系。

衡量输入和输出的关系，关联性，不断寻找合适的参数去解释这种关系
不足：只能表示浅层网络，深层网络会很受限制，最根本原因是脉冲的多次传递可能会消失。

这种训练方式本质上是为了解决上述问题。
ANN预训练，接着转换（转译）成SNN，继续训练。
ANN能源消耗比较大(ANN本质上传递的是连续的值，这些数相比与用 0 和 1 表示有和没有， 显然计算量是大了很多，同时实数的矩阵乘法也是非常消耗算力)

另外还有其余的方式，如 ==spike based back propagation==
“spike based back propagation”是一种基于脉冲神经元网络的反向传播算法。

* 基于脉冲的反向传播是一种用于脉冲神经网络的学习算法，脉冲神经网络是一种模拟生物神经元行为的人工神经网络。在基于脉冲的反向传播中，网络通过根据脉冲的发生时间来调整神经元之间的连接强度，从而学习。
* 该算法通过比较输出神经元产生的脉冲的时间和期望输出，然后将这个误差通过网络向后传播来调整突触权重。这使得网络能够根据脉冲的发生时间学习产生正确的输出。
* 基于脉冲的反向传播不同于传统神经网络中使用的反向传播，后者修改连接强度是基于预测输出和期望输出之间的差异，而不考虑神经元激活的时间。基于脉冲的反向传播特别适用于脉冲神经网络，因为它考虑了基于脉冲通讯的时间特性。

 现有的问题依然是：隐藏层的层数过多，训练效率会下降，且精度仍然小于传统的ANN。常见的数据集MNIST，我们的SNN已经能够做到很高的精度了，应对大数据集SNN依然不太行。

那如何解决：有人提出使用 hybrid network（混合网络，Lee），浅层网络使用SNN，深层网络使用ANN
RMP(Residula Membrance Potential)-SNN (han) ， hard reset(IF模型的解法) and soft reset。  hard 每次都是从1降到0，接着继续下一次脉冲，但是han这个小组发现，这样做会导致数据缺失或者训练不稳定。他们给出的解决方案是soft，不是从1降到0，而是计算一个值，得到下一次需要的激发强度，让值可以回到这个强度，这样的话更高效。


事件相机  event cameras.
事件相机是一种新型的传感器，它与传统的帧式相机不同，它并不按照固定的帧率连续地捕捉图像，而是在场景中发生较大变化时才会产生输出。这种相机能够以微秒级的时间精度来感知快速变化的光强度，从而实现优异的动态范围和快速响应。通过事件相机，我们可以捕捉到真实世界中发生的快速变化，例如高速移动的物体、强烈的光照变化等，而且由于它们只在发生事件时才产生输出，所以能够以更高的效率传输和处理数据。由于这些优势，事件相机在机器人、自动驾驶、虚拟现实等领域具有很大的潜力，并且正在逐渐受到研究和工业界的关注。

Lee 研发了一个 Spike-FlowNet 的model，本质上是一个deep hybrid Neural Network Architecture(深层混合神经网络架构)，由于是混合结构，深层用的是ANN，所以他的性能并没有被牺牲掉。经过一些数据集的测试，发现该网络可以比ANN-base model 做的还要好。

事件相机：需要快速的时间去运行，需要处理大量数据，需要低能耗。
参考文献文字版
Spiking Neural Networks and Their Applications: A Review.
doi: 10.3390/brainsci12070863

ESWEEK 2021 Education - Spiking Neural Networks.
https://www.youtube.com/watch?v=7TybETICsIM

Neuronal Dynamics: From single neurons to networks and models of cognition and beyond.
https://neuronaldynamics.epfl.ch/online/Ch1.S3.html#Ch1.E5

Spike-FlowNet: Event-based Optical Flow Estimation with Energy-Effcient Hybrid NeuraNetworks.
https://www.ecva.net/papers/eccv 2020/papers ECCV/papers/123740358.pdf

RMP-SNN: Residual Membrane Potential Neuron for Enabling Deeper High-Accuracy andLow-Latency Spiking Neural Network.
https://openaccess.thecvf.com/content CVPR 2020/papers/Han_RMP-SNN Residual MembranePotential Neuron for_Enabling Deeper_High-Accuracy_and CVPR 2020_paper.pdf



# ==生物神经网络中的神经可塑性规则==  

Plasticity rules in biological neural network  2021/10/17
[参考链接](https://www.bilibili.com/video/BV1bP4y1b7Ts/?spm_id_from=333.999.0.0&vd_source=3be4a0eefc99c3ec4109b6e4f90586d1)

记忆存在哪儿？ 海马区？小脑等等

神经元的可塑性：neural plasticity   神经元活动：neural activity

齿状回和记忆痕迹-dentate gyrus & memory engram

功能方面：连接的强度 ；  结构方面：突触形状
**功能性的可塑性**：LTP，在所有的大脑区域中，海马体呈现出了最强的LTP幅度

Long-Term potentiation(LTP)

Among all brain regions, hippocampus presents the strongest LTP amplitude  
Fire together, wire together.
Long Term depression(LTD)    homosynaptic同源突触   heterosynaptic 异突触

**结构性的可塑性： **

Modeling neural plasticity : 建模去理解神经可塑性的规则
赫布理论及其缺陷---Hebbian rule and its drawbacks

metaplasticity

* Positive feedback 正反馈
* Runaway excitation - silencing   失控激励-沉默
* 存在的第一个问题是如何稳定大脑的活动（LTP和LTD的协调）？
  * 如果活动过多，模型倾向于把他拉回来，LTD
  * 如果活动过弱。模型倾向于把他提上来，LTP     metaplasticity 再可塑性


从上图得到的结论：STDP也许并不是唯一的控制规则，或者说他不是主要的控制规则，那么有什么东西是被忽略的？
STDP本质上说的是通过==调整LTP/LTD比例==来稳定网络活动，但上述实验表明3天以后大脑呈现出的状态并不符合STDP规则，且实际上可能并不需要如此复杂的调整LTP/LTD比例，大脑本身的调节机制可能更简单。

Homeostatic synaptic scaling  稳态突触缩放
神经元可能需要自己调节自己的状态，而不关注前面神经元的状态。

现在的情况总体来说就是：Hebbin plasticity ---产生记忆和学习， Homeostatic plasticity ---能够维持稳态，看起来挺好，但是这里大部分人用到的Homeostatic plasticity的持续时间都非常短，短到大脑根本无法在此时间内做出反应。

### 实际上 上面说了这么多，表明的意思就是也许功能可塑性并不是主要的调节机制，而结构可塑性可能是主要的调节机制，或者说结构可塑性是维持稳态的必要条件。

让神经元自由地进行连接。活动太强，减少一些突触连接，活动过弱，就增加连接。  

底层是钙离子浓度的变化。

所以最终保存记忆可能是用这个结构性改变的方式。

Homeostatic structual plasticity  每个人都有自己的连接方式





# ==Neuronal Dynamics  神经元动力学==

==书籍学习==
[电子书链接](https://neuronaldynamics.epfl.ch/online/)

1. Foundations of Neuronal Dynamics （神经元动力学基础）
   1. Introduction 介绍
      * Neurons and Mathematics 神经元和数学
   2. The Hodgkin-Huxley Model 霍奇金-赫胥黎模型
   3. Dendrites and Synapses 树突和突触
   4. Dimensionality Reduction and Phase Plane Analysis 降维与平面分析
2. Generalized Integrate-and-Fire Neurons
   1. Nonlinear Integrate-and-Fire Models 非线性积分和触发模型
   2. Adaptation and Firing Patterns  适应和发射模式
   3. Variability of Spike Trains and Neural Codes  脉冲序列和神经编码的多样性
   4. Noisy Input Models: Barrage of Spike Arrivals  噪音输入模型：一连串的脉冲到达
   5. Noisy Output: Escape Rate and Soft Threshold  噪声输出：逃逸率和软阈值
   6. Estimating Models   评估模型
   7. Encoding and Decoding with Stochastic Neuron models  随机神经元模型的编码和解码

3. Networks of Neurons and Population Activity  神经元网络和群体活动
   1. Neuronal Populations 神经元群
   2. Continuity Equation and the Fokker-Planck Approach   连续性方程和福克-普朗克方法
   3. The Integral-equation Approach  积分方程法
   4. Fast Transients and Rate Models   快速瞬态和速率模型

4. Dynamics of Cognition   认知动力学Cortical
   1. Competing Populations and Decision Making  竞争群体与决策
   2. Memory and Attractor Dynamics   记忆和吸引子动力学
   3. Cortical Field Models for Perception   皮层场感知模型 （Cortical 通常与大脑皮层有关）
   4. Synaptic Plasticity and Learning  突触可塑性与学习
   5. Outlook: Dynamics in Plastic Networks   展望：可塑网络的动力学

**学习记录**

1. 第一章

   * **Elements of Neuronal Systems**  神经元系统的元素
     脉冲序列中的动作电位通常分离良好。即使输入非常强大，也不可能在第一次尖峰期间或之后立即激发第二个尖峰。两个尖峰之间的最小距离定义了==神经元的绝对不应期==。绝对不应期之后是相对耐火度阶段，在此阶段很难激发动作电位，但并非不可能。

   * **Elements of Neuronal Dynamics**   神经元动力学的要素
     
     postsynaptic potential (PSP)  突触后电位
     
     postsynaptic current, PSC 突触后电流
     inhibitory postsynaptic potential (IPSP)  抑制性电位
     
     在静止时，细胞膜已经具有约-65 mV的强负极化。兴奋性突触的输入会降低膜的负极化，因此称为去极化。进一步增加膜负极化的输入称为超极化。
     
     单个 EPSP(excitatory postsynaptic potential  兴奋性突触后电位) 的振幅在1mv 范围内。脉冲起始的临界值比静息电位高出约 20 至 30 mV。因此，在大多数神经元中，四个脉冲（如图1.5C所示）不足以触发动作电位。取而代之的是，大约 20-50 个突触前脉冲必须在短时间内到达才能触发突触后动作电位。
     
   * **Integrate-And-Fire Models**  积分-放电模型
     “Integrate-And-Fire Models”翻译为“积分-放电模型”。这是一种生物学或者神经科学中用于模拟神经元行为的简化数学模型。在这个模型中，神经元接收到来自其他神经元的输入信号，这些输入信号会被积分（加总），当积分的值超过一定的阈值时，神经元便会产生一个放电（动作电位），模拟神经元的兴奋和抑制的行为。这种模型能够帮助科学家们理解和研究神经元的行为，同时也被应用在神经网络模型的设计中。

     [本节公式参考](https://neuronaldynamics.epfl.ch/online/Ch1.S3.html)

   * **Limitations of the Leaky Integrate-and-Fire Model**   积分放电模型的局限性

     不同神经元之间引起的电压大小，且突触所在的位置不同也将引起不同的突触反应，但是LIF考虑不到这些。
     [参考链接](https://neuronaldynamics.epfl.ch/online/Ch1.S4.html)
   
   * **What Can We Expect from Integrate-And-Fire Models?**  我们可以从积分放电模型中得到什么启示？
     或许LIF模型是模拟神经元中脉冲产生的有效模型。

   

   ​		总的来说，现在各项工作的思路是没有问题的，就是首先构建简单模型去尝试拟合观察到的现象，接着将这个模型复杂化去模拟更多的实验现象，最终形成一个可以几乎契合大脑的神经网络。

   * **Summary**   总结
   
2. 第二章   Ion Channels and the Hodgkin-Huxley Model    离子通道和霍奇金-赫胥黎模型

   * Equilibrium potential    平衡电位
     
   * Hodgkin-Huxley Model   霍奇金-赫胥黎模型
     
   * The Zoo of Ion Channels    离子通道动物园
     
   * Summary 小结

3. 第三章  Dendrites and Synapses  树突和突触
   神经元具有复杂的形态：细胞的中心部分是体细胞，其中包含遗传信息和大部分分子机制。在soma起源于长线状延伸，有两种不同的形式。首先，树突形成许多或大或小的分支，突触位于这些分支上。突触是来自其他神经元（即“突触前”细胞）的信息到达的接触点。其次，在体细胞上也开始轴突，神经元用它来向目标神经元发送动作电位。传统上，胞体和轴突之间的过渡区域被认为是决定是否发出尖峰的关键区域。

   * Synapses 3.1 突触
     
   * 3.2 Spatial Structure: The Dendritic Tree    空间结构：树突树
   * 3.3 Spatial Structure: Axons   空间结构：轴突
   * 3.4 Compartmental Models    区室模型
   * 3.5 Summary   小结

4. 第四章   Dimensionality Reduction and Phase Plane Analysis    降维和相平面分析

   * 4.1 Threshold effects    阈值效应
     尽管神经元动作电位的放电通常被视为一种类似阈值的行为，但这种阈值在数学上并没有得到很好的定义。然而，出于实际目的，这种转变可以被视为阈值效应。然而，我们发现的阈值取决于刺激方案。
   * 4.2 Reduction to two dimensions   简化为二维
     
   * 

5. 

  

**总结：** 



# ==知乎链接阅读==

主要是关于SNN的实现及在Mnist数据集上的训练

### [目录 ](https://zhuanlan.zhihu.com/p/591416095)

* [使用PyTorch实现脉冲神经网络(SNN)](https://zhuanlan.zhihu.com/p/558272145)

  * ==SNN介绍：==SNN（Spiking Neural Network）也就是脉冲神经网络，通过模拟神经元放电过程进行训练和推理的神经网络。由于近年来深度学习遇到一定的瓶颈，如推理过程需要大量运算资源、模型表现提升遇到天花板等，人们把希望寄托在脉冲神经网络这一新类型模型之上。因此，脉冲神经网络也被称为第三代神经网络。

  * ==实验结果表明：==SNN的PyTorch实现类似RNN。

  * ==编码方案==

    1. 直接编码: 将图像归一化之后拉成一个序列。
       ```python
       def SimpleEncoder(img_batch):
           img_stack = []
           for img in img_batch:
               img_stack.append(img.view(-1)) # 拉成了一个序列放去list中
           return torch.stack(img_stack)
       ```

    2. 泊松编码：泊松过程是一种累计随机事件发生次数的最基本的独立增量过程。例如随着时间增长累计某电话交换台收到的呼唤次数，就构成一个泊松过程。若我们有生成一个脉冲序列的概率，使用泊松过程对脉冲序列建模比较合适。

       将图像数值归一化之后，我们认为每个像素的数值是一个泊松过程的强度参数。根据这一参数，生成一系列服从泊松过程的序列。这就是泊松编码。

       

  * ==激活函数的近似==

    * 所谓激活函数（Activation Function），就是在人工神经网络的神经元上运行的函数，负责将神经元的输入映射到输出端。脉冲神经网络中的激活函数是δ函数。δ函数，是英国物理学家狄拉克(Dirac)在20世纪20年代引人的，用于描述瞬间或空间几何点上的物理量。例如，瞬时的冲击力、脉冲电流或电压等急速变化的物理量，以及质点的质量分布、点电荷的电量分布等在空间或时间上高度集中的物理量。

    * S = $$\begin{cases} 1 & {if U>= U_{rmthr}} \\ 0& {if U< U_{rmthr}}  \end{cases}$$

    * $$\begin{cases} ＆≈\frac{U}{1 + k|U|} \\ \frac{∂S}{∂U}＆= \frac{1}{(1+k|U|)^2} \end{cases}$$

      

  * ==复现论文==  
  
    * 论文题目：Brain-inspired global-local learning incorporated with neuromorphic computing
    * 这篇论文提出了一种训练策略。作者希望自上而下的全局梯度能够在每个神经元处解耦为局部梯度和全局梯度，所以为每一个神经元设置了一个元学习模块学习额外的参数。
    * 光看代码有点不明白，还是需要看原文的。
    
    
    
  
* [概述和图像分类](https://zhuanlan.zhihu.com/p/588958782)

  * 第二代神经网络的推理过程需要大量运算资源，模型表现提升遇到天花板等问题，由此脉冲网络应运而生。其本质为通过模拟神经元放电过程进行训练和推理的神经网络。
  * 直接编码、泊松编码
  * 解码器的作用是将脉冲序列转化为可以被卷积神经网络理解或计算的形式。最简单粗暴的方式是将整个序列相加。虽然这种解码方式比较粗暴，但是有一定原理。

* [编码为脉冲序列](https://zhuanlan.zhihu.com/p/589139289)

  * 对于MNIST数据集的编码，可以使用两种方式：

    1. 在每个时间步传递相同的值，这些值可以归一化到0-1之间。（没有考虑SNN的时间动态性）
  
    2. 将输入转换为序列，逐步输入网络。
  
       
  
  * 数据到脉冲的编码（Encode）

  1. 频率编码（Rate coding）使用输入特征来确定脉冲的频率
     
  2. 延迟编码（Latency coding）使用输入特性来确定脉冲的时间
     
  3. 增量调制（Delta modulation）使用输入特征的时间变化来产生脉冲信号
     
     增量调制编码的原理是：其用于将模拟信号数字化，根据信号样本之间的差值来构建数字信号。使用1表示值增大，0表示值减小。==在脉冲网络中：== 增量调制的意思是差值高于阈值输出脉冲，低于阈值不输出脉冲，还需要考虑使用差值的真实值还是绝对值。使用绝对值可能会导致脉冲输出数目变多。
     
     
  
* [LIF（Leaky Integrate and Fire）神经元](https://zhuanlan.zhihu.com/p/589157902)

  [snntorch : 一种将torch引入到snn中的脉冲神经网络训练框架（P1 如何将数据转化为脉冲序列）](https://blog.csdn.net/cyy0789/article/details/121351527?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522170575793416800226596855%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fblog.%2522%257D&request_id=170575793416800226596855&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~blog~first_rank_ecpm_v1~rank_v31_ecpm-1-121351527-null-null.nonecase&utm_term=snntorch&spm=1018.2226.3001.4450)
  [snntorch：P2—【LIF神经元模型】手撕公式、代码实现与演示](https://blog.csdn.net/cyy0789/article/details/121432756)
  [snntorch_P3: 脉冲神经网络与其他经典算法的对比](https://blog.csdn.net/cyy0789/article/details/127440891?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522170575793416800226596855%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fblog.%2522%257D&request_id=170575793416800226596855&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~blog~first_rank_ecpm_v1~rank_v31_ecpm-3-127440891-null-null.nonecase&utm_term=snntorch&spm=1018.2226.3001.4450)

  * 神经元模型的范围
    
  * LIF神经元模型
  
    * 脉冲神经元
      
    * 被动膜
    
      像所有细胞一样，神经元被一层薄膜包围。这层膜是一个脂质双分子层，它将神经元内的传导盐水溶液与细胞外介质隔离开来。在电的作用下，由绝缘体隔开的两种导电溶液起电容器的作用。
    
      这层膜的另一个功能是控制进出细胞的物质(例如钠离子)。细胞膜通常不能被离子渗透，从而阻止它们进出神经元体。但是细胞膜上有一些特殊的通道可以通过向神经元注入电流而打开。这种电荷运动由电阻器模拟。
    
      
    
    * **LIF神经元模型的动力学模型**   ==看原文==
    
    * **前向欧拉法求解LIF神经元模型**     ==看原文==
    
    * ### Lapicque的LIF神经元模型
    
  
* [前馈脉冲神经网络](https://zhuanlan.zhihu.com/p/589863877)
  之前的文章中，我们知道LIF神经元模型如何接受和发放脉冲的。在这篇教程中，将要介绍如何改进LIF神经元以更加适应深度学习任务；另外，我们将会实现一个简单的前馈脉冲神经网络（Feedforward Spiking Neural Network, SNN）。

  * Leaky Integrate-and-Fire神经元模型简化

    * 衰减率 β
      
    * 加权输入电流
      
    * 脉冲和重置
    
  * snntorch中的泄漏神经元模型
  
  * 前馈脉冲神经网络
    到目前为止，我们只考虑了单个神经元如何响应输入刺激。snntorch可以直接将其扩展到深度神经网络。在本节中，我们将创建一个维度为784-1000-10的3层全连接神经网络。与我们到目前为止的模拟相比，每个神经元现在将整合更多的输入峰值。
  
  * 这个教程涵盖了如何简化leaky integrate-and-fire神经元模型，然后使用它来构建脉冲神经网
    络。在实践中，我们几乎总是更喜欢使用snn.Synaptic和snn.Alpha用于训练网络，因为有一个较
    小的超参数搜索空间。
  
    
  
* [二阶脉冲神经元模型](https://zhuanlan.zhihu.com/p/590609738)
  在本教程中，你将了解更先进的LIF神经元模型: Synaptic和Alpha

  * 基于突触电导的LIF神经元模型
    在之前的教程中探索的神经元模型假设输入电压尖峰导致突触电流的瞬时跳跃，然后有助于膜电位。实际上，脉冲会导致神经递质从突触前神经元**逐渐**释放到突触后神经元。基于突触电导的LIF模型考虑了输入电流的时序渐进动态特性。

    * 突触电流建模

      如果突触前神经元放电，电压尖峰就会传递到神经元的轴突。它触发小泡释放神经递质到突触间隙。这些激活突触后受体，直接影响流入突触后神经元的有效电流。
    
    * snntorch中的突触神经元模型
      
      <img src="https://pic3.zhimg.com/80/v2-967bfc13c375c980e8cb2bf0b9934a0e_1440w.webp" alt="img" style="zoom: 50%;" />
      
      突触电流整合了输入的脉冲和以α衰减的电位
      
      膜电位整合了突触电流和以β衰减的电位
      
      输出脉冲由每一个时间点上越过阈值的膜电位所产生。
      
    * 一阶神经元与二阶神经元的对比
    
      1. 二阶神经元对于保持长期关系是有益的——对应的一种类似描述是 输入脉冲模式是稀疏的
      2. 二阶神经元对于控制脉冲的精确时间更容易。 输出峰值对于输入峰值有延迟
      3. 一阶神经元更适用于简单的数据，随着数据增加，二阶神经元具有更好的拟合性。
    
  * #### α神经元模型(hacked Spike Response Model)
  
    * 建立Alpha神经元模型
  
  * 结论
  
    
  
* [全连接脉冲神经网络（SNN for FCN）](https://zhuanlan.zhihu.com/p/591326540)

  * SNN的循环表示
    <img src="https://pic2.zhimg.com/80/v2-18ef2a774b53ce7d2946d1d2338679a5_720w.webp" alt="img" style="zoom:50%;" />
  * 脉冲的不可微性
    
  * 克服死神经元问题
    
  * BPTT(Backprop Through Time)
    
  * 设置损耗/输出解码
    
  
* [卷积脉冲神经网络中的替代梯度下降](https://zhuanlan.zhihu.com/p/591401049)

  * 代理梯度下降(Surrogate Gradient Descent)

    在泄露的Integrate-and-Fire函数中，fast sigmoid函数的梯度可以覆盖Dirac-Delta函数

* [基于 Tonic + snnTorch的神经形态数据集](https://zhuanlan.zhihu.com/p/592067786)
  
* [脉冲神经网络中的种群编码（Population Coding）](https://zhuanlan.zhihu.com/p/592219068)

  有人认为，速率编码本身不能成为初级皮层的主要编码机制。其中一个原因是，神经元的平均放电速率大约为0.1-1赫兹，远远慢于动物和人类的反应反应时间。

  但是，如果我们将多个神经元聚集在一起，并一起计算它们的峰值，那么就有可能在非常短的时间窗口内测量一群神经元的放电率。种群编码增加了速率编码机制的可信性。
  
  在本教程中，你将:
  学习如何训练种群编码网络。我们将扩展到每个类别的多个神经元，并将它们的峰值聚集在一起，而不是每个类别分配一个神经元。
  
  <img src="https://pic1.zhimg.com/80/v2-fb8502ace5507e7ccb3403e7c0259d28_1440w.webp" alt="img" style="zoom:50%;" />
  
* 即使我们只在一个时间步长上进行训练，引入额外的输出神经元也可以立即实现更好的性能。
  
* 随着时间步数的增加，群体编码带来的性能提升可能会开始减弱。但它也可能比增加时间步长更可取，因为PyTorch是为处理矩阵-向量乘积而优化的，而不是随着时间的推移顺序、一步一步的操作。
  
  

# ==csdn链接阅读==

[那些年与SNN的爱恨情仇](https://blog.csdn.net/weixin_46186672/article/details/130901924)
[某博主关于脉冲神经网络的代码](https://blog.csdn.net/cyy0789?type=blog)   ——这里面有三部分，自己在jupyter中运行完了。



关于步长这个概念：有时间步长，可以得到脉冲发放次数这个概述，而这个值除以总的时间步长可以用来表示当前的脉冲发放概率，进而也就转换成了ANN中经过激活函数的值。



# ==神经元内信号传递的计算模型-HH模型==

[参考链接](https://blog.csdn.net/weixin_46263718/article/details/121884683)

神经元动力学可以被设想为一个总和过程(有时也称为“集成”过程)，并结合一种触发动作电位高于临界电压的机制。

##### 膜作为电路的生物物理学

**1.电刺激达到阈值，膜电势出现改变，各离子通道允许与不允许状态的概率发生改变：**

**2.离子通道的状态改变就对应着相应的门控开关开放概率的改变：**

**3.各离子通道允许离子通过的能力可以用电导$g$来进行衡量：**

**4.由g = 1/R, I = U /R 我们可以计算出流经各离子通道的电流：**

**5.根据电学公式 $C = q/U, I_c = C* du/dt$ 以及并联电流公式我们可以计算出此时的膜电势：**

由于HH模型需要使用4个常微分方程进行表示模型，故计算较为复杂，很难进行大规模仿真。**但其精确地描绘出膜电压的生物特性，能够很好地与生物神经元的电生理实验结果相吻合。**
