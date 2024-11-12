# [2024年顶会、顶刊SNN相关论文](https://blog.csdn.net/qq_43622216/article/details/135167498)

1. AAAI 2024（共15篇）
   * **论文1: [Shrinking Your TimeStep: Towards Low-Latency Neuromorphic Object Recognition with Spiking Neural Networks](https://arxiv.org/abs/2401.01912)**
     * 由电子科技大学左琳教授团队发表于AAAI 2024。
     * 提出了Shrinking SNN (SSNN)，将SNN划分为多个阶段，每个阶段的时间步长逐渐收缩，实现低时间步长神经形态目标识别。（一个异质性时间步长的SNN）
     * 在SNN每个阶段后额外加入分类器，与标签计算损失，缓解代理梯度和真实梯度的不匹配、梯度爆炸/消失问题，从而提升SNN的性能。
   *  **论文2: [Ternary Spike: Learning Ternary Spikes for Spiking Neural Networks](https://arxiv.org/abs/2312.06372)**
     * 由中国航天科工集团公司智能科学技术研究院研究人员发表于AAAI 2024。
     * 提出了ternary spike neuron产生0/1/-1脉冲，并在三元脉冲神经元中嵌入了一个可训练因子来学习合适的脉冲幅值，这样的SNN会逐层采用不同的脉冲幅值α，从而更好地适应膜电位逐层分布不同的现象。
     * 在推理时，通过重参数化将可训练的三元脉冲SNN再次转换为标准SNN。
   * **论文3: [Memory-Efficient Reversible Spiking Neural Networks](https://arxiv.org/abs/2312.06372)**
     * 由浙江大学研究人员发表于AAAI 2024。
     * 提出了reversible SNN以降低直接训练SNN的内存开销，每一层的输入变量和膜电位可以通过其输出变量重新计算而无需在内存中存储。
     * 设计了Spiking reversible block用于构建Reversible spiking residual neural network和Reversible spiking transformer。
   * **论文4: [Gated Attention Coding for Training High-performance and Efficient Spiking Neural Networks](https://arxiv.org/abs/2308.06582)**
     * 由电子科技大学大学、加利福尼亚大学、中科院自动化所（李国齐团队）研究人员发表于AAAI 2024。
     * 提出了Gated Attention Coding (GAC)对输入应用注意力机制进行编码。
     * Gated Attention Unit (GAU)：使用CBAM提取时间维度的注意力分数；使用共享的2D卷积提取每个时刻的通道-空间注意力分数。
   * **论文5: [DeblurSR: Event-Based Motion Deblurring Under the Spiking Representation](https://arxiv.org/abs/2303.08977)**
     - 由德克萨斯大学奥斯汀分校研究人员发表于AAAI 2024。
   * **论文6: [Dynamic Reactive Spiking Graph Neural Network](https://ojs.aaai.org/index.php/AAAI/article/view/29640)**
     - 由西安电子科技大学、上海交通大学等研究人员发表于AAAI2024。
   * **论文7: [An Efficient Knowledge Transfer Strategy for Spiking Neural Networks from Static to Event Domain](https://ojs.aaai.org/index.php/AAAI/article/view/27806)**
     * 由中科院自动化所等研究人员发表于AAAI2024。
     * 同时学习静态数据和神经形态数据，在中间层对二者的表征进行蒸馏学习。
     * sliding training strategy：训练时静态图像输入一定概率地被替换为事件数据，并且这种替换概率随着时间步长（应该是训练的epoch和batch）而增大，直到学习阶段结束，此时事件数据将替换所有的静态图像。
   *  **论文8:[Enhancing Representation of Spiking Neural Networks via Similarity-Sensitive Contrastive Learning](https://ojs.aaai.org/index.php/AAAI/article/view/29635)**

     * 由中国航天科工集团公司智能科学技术研究院研究人员发表于AAAI 2024。
     * 提出了similarity-sensitive contrastive learning以及一个逐层基于CKA加权的策略，最大化ANN和SNN中间表征的互信息，减少SNN表示的不确定性。
     * 对逐层样本特征使用Noise-Contrastive Estimation(NCE)进行对比学习，拉近SNN和预训练的ANN对同一个样本的表征距离，推开不同样本的表征距离，使用逐层的CKA对NCE进行loss加权。
   * **论文9:[Efficient Spiking Neural Networks with Sparse Selective Activation for Continual Learning](https://ojs.aaai.org/index.php/AAAI/article/view/27817)**
     * 由浙江大学等研究人员发表于AAAI2024。
     * 利用SNN中脉冲的稀疏性和权重更新的稀疏性来降低内存开销、缓和灾难性遗忘问题。针对连续学习任务。
     * 提出了trace-based K-Winner-Take-All (KWTA)和可变阈值机制的selective activation SNNs (SA-SNN)持续学习模型，通过增强SNN的神经动力学特性来减轻灾难性遗忘，并且不需要任务标记或记忆重播。
   * **论文10:[Spiking NeRF: Representing the Real-World Geometry by a Discontinuous Representation](https://arxiv.org/abs/2311.09077)**
     - 由浙江大学潘纲教授团队发表于AAAI 2024。
   * **论文11: [SpikingBERT: Distilling BERT to Train Spiking Language Models Using Implicit Differentiation](https://arxiv.org/abs/2308.10873)**
     * 由宾夕法尼亚州立大学研究人员发表于AAAI 2024。
     * 提出了一个spiking language model (LM)。利用模型的稳态收敛性，引入了一种可操作的脉冲注意机制，提出了一种新的基于ANN-SNN的KD，以实现更快、更有效的学习，并探索了使用基于隐式微分的技术训练spiking LMs，从而克服了影响基于SNN架构训练的多个问题。
   * **论文12: [TC-LIF: A Two-Compartment Spiking Neuron Model for Long-term Sequential Modelling](https://arxiv.org/abs/2308.13250)**
     * 由香港理工大学（Kay Chen Tan团队）、新加坡国立大学、香港中文大学(深圳)研究人员发表于AAAI 2024。
     * 给出了P-R锥体神经元的泛化公式：两房室神经元。
     * 基于P-R椎体神经元，设计了Two-Compartment(TC)-LIF神经元以促进长期的序列建模。
   * **论文13: [Enhancing Training of Spiking Neural Network with Stochastic Latency](https://ojs.aaai.org/index.php/AAAI/article/view/28964)**
     * 由穆罕默德·本·扎耶德人工智能大学等研究人员发表于AAAI2024.
     * 提出了Stochastic Latency Training (SLT)，训练期间的每个batch随机采样延迟对SNN进行训练。
   * **论文14: [Enhancing the robustness of spiking neural networks with stochastic gating mechanisms](https://ojs.aaai.org/index.php/AAAI/article/view/27804)**
     * 由北京大学（黄铁军、于肇飞组）研究人员发表于AAAI 2024。
     * 在大脑中，神经元反应通常具有离子通道和突触诱导的随机性，而随机性在计算任务中的作用尚不清楚。
     * 将随机性引入SNN，构建了stochastic gating (StoG) spiking neural model，可以被视为用于防止攻击下误差放大的正则化器。
     * 每一层、每一个时间步长从特定的伯努利分布中采用门控因子G的值，然后对前一层产生的脉冲进行门控相乘，随机接收信息。提升对抗攻击鲁棒性。
   * **论文15: [Dynamic Spiking Graph Neural Networks](https://arxiv.org/abs/2401.05373)**
     - 由穆罕默德·本·扎耶德人工智能大学、河北工业大学、北京大学、吉林大学、哈尔滨工业大学等研究人员发表于AAAI 2024
   * **论文16:[Point-to-Spike Residual Learning for EnergyEfficient 3D Point Cloud Classification](https://ojs.aaai.org/index.php/AAAI/article/view/28425/28830)**
     * 由安徽大学等研究人员发表于AAAI 2024，使用SNN用于点云识别。
     * 提出了一个point-to-spike residual learning network：设计了一个spatial-aware kernel point spiking(KPS) neuron，以及3D spiking residual block。
     * KPS神经元：连接IF神经元和kernel point convolution操作。
   * 
2. IJCAI 2024
   * 
3. ICLR 2024
   * 
4. ICML 2024（共13篇）
   * **论文1:[High-Performance Temporal Reversible Spiking Neural Networks with O ( L ) O(L)O(L) Training Memory and O ( 1 ) O(1)O(1) Inference Cost](https://arxiv.org/abs/2405.16466)**
     * 由北京大学田永鸿、自动化所徐波、李国齐等研究人员发表于ICML2024。
     * 实验发现：SNN（ResNet）中每个阶段的最后一层的时间维度梯度比较重要，其他层倒不重要。
     * 设计了Temporal Reversible SNN（T-RevSNN），仅保留每个阶段最后一层脉冲神经元的时间维度。
     * 将第 $l$ 层后面所有层的神经元在前一个时间步长的膜电势向前传递。
   * **论文2: [Robust Stable Spiking Neural Networks](https://arxiv.org/abs/2405.20694)**
     * 由北京大学黄铁军组发表于ICML2024。
     * 旨在通过非线性系统的稳定性来揭示SNN的鲁棒性。
     * 灵感来自于：寻找参数来改变leaky integrate-and-fire动力学可以增强它们的鲁棒性。
     * 指出：膜电位扰动动力学可以可靠地反映扰动的强度；简化的扰动动力学能够满足输入输出稳定性。
     * 基于改进的脉冲神经元提出了新的训练框架，减小了膜电位扰动的均方，以增强SNN的鲁棒性。
   
5. ECCV 2024
6. CVPR 2024
7. ACM MM 2024
8. NeurIPS 2024
9. ICASSP 2024

