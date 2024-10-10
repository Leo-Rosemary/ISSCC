# Lightweight and Efficient Spiking Neural Network

## I Introduction

在本文中，我们旨在回答以下问题：模型稀疏化能否释放GNN的加速潜力？为此，我们首先提出一个关键的观察，即现有文献缺乏对GNN的不规则和结构化剪枝方法的全面评估。因此，我们评估了GNN的不规则和结构化剪枝方法，以量化它们的性能潜力和挑战。我们表明不规则剪枝不适合GNN计算管道，利用稀疏性加速GNN的可行方法是结构化剪枝。然而，与不规则剪枝相比，结构化剪枝使用的维度级压缩策略导致GNN模型稀疏率降低。因此，需要新的剪枝方法来在不损失精度的情况下暴露最大的稀疏性。此外，需要高效的矩阵乘法内核来充分利用高度向量化硬件与结构化剪枝产生的低维模型提供的并行性。

为了应对上述挑战，我们提出了PruneGNN，一个algorithm-architecture结构化的模型剪枝框架，它采用（1）稀疏训练来实现GNN模型中的高结构化稀疏性，以及（2）Prune-SpMM，一种高效的矩阵乘法内核，通过使用独立的线程调度来利用现代GPU中的并行性来释放稀疏模型的性能潜力。与基于LASSO的方法相比，PruneGNN实现了更高的稀疏性[56]，以及其他为GNN剪枝调整和采用的最先进的优化技术，如彩票票假设（LTH）[6]和乘法器的交替方向方法（ADMM）[25]，[36]。PruneGNN的结构化模型剪枝使用NVIDIA A100 GPU在具有代表性的GNN模型和数据集上进行评估，表明与先前基于LASSO的工作相比平均加速了2倍[56]。



变形金刚 [2]、[39] 最近以视觉变形金刚 （ViTs） [11] 的形式重新崛起，在 NLP [3]、[26]、[48]、计算机视觉（例如图像分类 [11]、对象检测 [5]、[66]、语义分割 [62]、图像处理 [7] 和视频理解 [64] 中显示出强大的多功能性）和具有多模态数据的复杂场景。此外，ViT 可以用作有效的骨干网络 [4]、[11]、[19]、[50]，通过微小的微调，具有优异的可转移性。看起来，ViT 和 Transformers 具有巨大的潜力，可以通过通用架构统一不同的应用领域，并解决对稀缺领域数据的依赖，最终解决深度学习中的两个基本问题：（i） 对领域数据的强烈依赖，以及 （ii） 不断改进模型以满足不断变化的需求。ViT 和 Transformers 在很大程度上被认为是未来占主导地位的深度学习技术之一。

然而，要充分发挥 transformer 架构的优势，我们需要在 ViT 和 transformer 成为未来 AI 计算不可或缺的主打产品之前解决以下*挑战*。（i） 尽管自我注意机制是 transformer 架构的一个关键定义特征，但一个众所周知的问题是它与输入标记数量相关的二次时间和内存复杂性。这阻碍了许多设置中的可扩展性，更不用说在资源受限的边缘设备上部署了。（ii） 大多数关于高效 ViT 和 transformer 技术的现有工作都遵循了卷积神经网络 （CNN） 所做的工作，即使用传统的权重修剪 [10]、[56]、[57]、[65]、量化 [35]、[40]、[61] 和紧凑的架构设计 [6]、[8]、[9]、[16]、[18]、[31]、[43]、[49]、 [51]，精度和速度性能有限。（iii） 在探索标记删除以解决二次复杂性的努力中，静态方法 [10]、[13]、[33]、[42]、[45] 以与输入无关的方式删除具有固定比率的标记，忽略了与输入样本相关的冗余;而大多数现有的图像自适应方法 [38]、[54]、[57] 只是简单地丢弃了非信息性的标记，并没有充分探索来自不同注意力头的标记级冗余。这两种方法都实现了相对较低的修剪率（以保持准确性）或被破坏的精度（在高修剪率下）。此外，这些研究都不支持在边缘设备上实现高效的硬件。（iv） Transformer 架构倾向于使用更多对硬件不友好的计算，例如，比 CNN 更多的非线性运算，以提高准确性。因此，我们需要解决此类计算的硬件效率低的问题，同时享受多头自注意力提供的额外优化维度。

在本文中，我们提出了 HeatViT，一种硬件高效的图像自适应令牌修剪框架，结合 8 位量化，用于在嵌入式 FPGA 平台上实现高效而准确的 ViT 推理加速。为了在保持模型准确性的同时提高修剪率，我们通过分析 ViT 中的计算工作流程进行了两个观察，类似于 [28] 中报道的内容：（i） ViT 中不同注意力头的输入标记中的信息冗余不同;（ii） 在早期 transformer 块中识别的非信息性令牌在传播到后面的块时可能仍会编码重要信息。基于这些，我们采用了类似于 [28] 的有效 token selector 模块，但考虑其硬件效率的设计，它可以插入到 transformer 块之前以减少 token 数量（即 token 数量），而计算开销可以忽略不计。如图 1 所示，我们合并了来自多个注意力头的不同 token 级冗余，以便在 token 评分中更加准确。此外，我们不是像 [28] 那样完全丢弃非信息性的，而是将它们打包成一个信息性令牌，为以后的 transformer 块保留信息。

对于边缘 FPGA 设备上的硬件实现，我们设计了具有线性层的令牌选择器，即全连接 （FC） 层，而不是卷积 （CONV） 层，以重用为主干 ViT（即无令牌选择器）执行构建的 GEMM（通用矩阵乘法）硬件组件。此外，我们始终将已识别的（和稀疏的）信息性令牌和打包的信息性令牌连接在一起，以形成密集的令牌输入，以避免硬件上的稀疏计算。为了提高硬件效率，我们进一步对权重和激活应用了 8 位量化，并为 ViT 中常用的非线性函数提出了多项式近似，包括 GELU [21]、Softmax [17] 和 Sigmoid [37]。此外，我们将正则化对量化误差的影响引入多项式近似的设计中，以支持更雄心勃勃的量化。我们基于我们提出的 HeatViT 为 ViT 设计了一个概念验证 FPGA 加速器。我们实现了受 [14]、[32] 启发的 GEMM 引擎，以在主干 ViT 中执行计算最密集的多头自我注意模块和前馈网络，并在我们的令牌选择器中执行分类网络。值得注意的是，仅添加了轻量级控制逻辑，通过重用为主干 ViT 执行构建的相同硬件组件来支持我们的自适应令牌修剪。

为了减少硬件上的推理延迟，同时保持模型的准确性（通常在 0.5% 或 1% 的精度损失范围内），我们采用了与 [28] 类似的延迟感知多阶段训练策略，该策略 （i） 确定用于插入令牌选择器的 transformer 块，以及 （ii） 优化这些令牌选择器所需的（平均）修剪率。为了减少训练时间，我们还调整了训练策略，以插入更少数量的 token 选择器并使用更少的训练 epoch：我们的训练工作大约是没有 token selector 的从头开始训练 ViT 的 90%。

**背景介绍**

1.介绍SNN和Spiking Transformer

**研究动机**

2.介绍遇到的问题，软件时间由于T的引入，训练加速，充分利用SNN的稀疏的脉冲活动

3.介绍现有的算法的缺点（非结构化剪枝算法），跟DNN相比，关于SNN的硬件加速器研究要少很多，现有的硬件加速器的运算缺点（非SNN）



最近，人们研究了希望联合优化硬件架构和软件映射的整体方法[33]、[56]、[64]、[66]。虽然这是一个更有吸引力的范例，但主要挑战是硬件-软件联合优化的空间可能是巨大的。例如，据估计，通过联合优化来解决效率网[58]的瓶颈需要探索O（102300）的大搜索空间。为了应对这一挑战，已经提出了不同的解决方案来减小搜索空间的大小，主要集中在1）设计空间剪枝或2）设计空间近似。例如，HASCO[64]使用统一IR的概念来剪枝搜索空间；FAST[66]引入了几种近似技术，使得设计空间探索只需要在更小的和近似的空间中进行。尽管做出了这些努力，硬件和软件设计选择仍然从算法的角度孤立地探索。



4.介绍本项工作的特点。我们提出了两种关键技术来实现稀疏行。首先，其次。在xxx的基础上，我们进一步提出了xxx技术来xxxx。这项工作提供了一种新颖的解决方案，以解决xx的局限性，实现xxx，从而实现SNN加速器。



在本文中，我们提出了一个用于人工智能加速器协同设计的统一协同优化框架。UNICO以共生的方式处理这种双层大设计空间探索，这样它就更专注于对有前途的候选硬件进行软件探索，同时逐步丢弃不利的硬件配置。与此同时，UNICO旨在找到强大的硬件配置，这些配置可以更好地推广到协同优化中看不到的新工作负载。我们表明，通过在软件探索中采取额外的定量措施，UNICO可以像以前的方法一样减轻硬件过度拟合输入工作负载的影响。具体来说，我们的贡献可以总结如下：

**贡献概述**

5.我们的贡献总结如下：

- 算法和硬件协同设计，以实现有效、硬件高效的SpikingTransomer，从而在 ViTs 中实现高效的图像自适应令牌修剪。
- ViTs 中非线性函数的多项式近似，用于更雄心勃勃的量化和基于 FPGA 的高效实现。
- 一个端到端加速框架，具有图像自适应修剪和 8 位量化，用于嵌入式 FPGA 上的 ViT 推理。
- 实验证明 HeatViT 优于最先进的 ViT 修剪研究的修剪率和推理准确性，以及微不足道的硬件资源开销。

## II Background and Motivation

Model pruning is a well-known technique for reducing the size and complexity of machine learning models by removing less important model parameters while maintaining or improving accuracy [57]. It has been widely explored in traditional DNNs [6], [39], [49], [51], but only recently have works emerged that explore pruning in GNNs. However, GNN s demonstrate unique computational patterns different from those of traditional DNNs due to their extremely unstructured, large, and sparse graph inputs that necessitate the use of challenging sparse matrix operations. This becomes even more challenging with the additional sparsity introduced to the computational pipeline via model pruning.

### A. Spiking Neural Networks  

### B. Spiking Transformer

spikformer

spike-driven

**C. Existing Transformer Accelerators**

D.  Opportunity of  Sparsity in SNNs

时间稀疏性



​    例举目前几个神经网络的稀疏编码后的

空间稀疏性

虽然之前的工作 [6]、[36]、[39]、[49]、[51] 通过在 GNN 中应用不规则修剪来减少 FLOPS 的数量，但 FLOPS 性能估计模型没有考虑用于不规则稀疏矩阵的矩阵乘法核的计算挑战。换句话说，虽然以前的工作表明使用不规则权重修剪的整体计算较少，但通过使用具有计算挑战性的 SpGEMM 内核而减少的并联度导致整体处理时间更长。SpGEMM 要求矩阵乘法的输入和输出保持非常高的稀疏性，以便进行有效处理 [32]、[34]。然而，在不规则稀疏性的情况下，乘法输入中非零的随机分布会累积成更密集的输出，这对 SpGEMM 内核构成了挑战。SpGEMM 中用于不规则修剪 GNN 的部分积不适用于现代 GPU 中的高度矢量化并行性，原因如下：

1. 不规则的非零模式会导致乘法过程中的间接索引。
2. 由于在部分乘积计算期间不规则输入的累积，乘法输出变得密集;因此，乘法输出在其完整维度大小上是最新的。
3. 由于上述几点，矩阵乘法的部分乘积需要在共享 （scratchpad） 内存或 registers 中累加。但是，考虑到图形的大小，硬件中的可用共享内存或 registers 可能不足。即使可用，warp 级别的同步也会带来通信开销。这需要在更新输出之前在全局内存中累积部分乘积，从而进一步增加总体内存开销。



由于这些原因，现代 GPU 的 SIMD 范式的全部性能潜力不能通过不规则修剪来利用，尽管它具有很高的压缩潜力。因此，我们认为计算的减少量（换句话说，压缩率）必须非常高，才能获得不规则的稀疏性，以补偿这些开销。

## III Prune gnn for Efficient Structured Pruning ALGORITHM

![image-20241003172133817](C:\Users\10418\AppData\Roaming\Typora\typora-user-images\image-20241003172133817.png)

![image-20241001223518745](C:\Users\10418\AppData\Roaming\Typora\typora-user-images\image-20241001223518745.png)



 A. The ViTCoD Algorithm Overview 整体介绍算法架构



### B. 稀疏编码

伪代码：Algorithm for the 

稀疏编码图片

稀疏编码可视化（热力图）

### C. 结构化剪枝

剪枝伪代码：Algorithm for the pruning

图片（未剪枝+剪枝+稀疏存储）

![image-20241001223406371](C:\Users\10418\AppData\Roaming\Typora\typora-user-images\image-20241001223406371.png)

![image-20241001223445281](C:\Users\10418\AppData\Roaming\Typora\typora-user-images\image-20241001223445281.png)

**D.  Encoder-Decoder编码模块**

图片（C-Transomformer图片）

Loss损失+Accuracy



## IV VITCOD ACCELERATOR

A. 挑战

为了通过我们的动态令牌修剪来实施 ViT FPGA 加速器，我们解决了以下挑战。（i） 令牌选择器模块应通过添加微型控制逻辑并将现有硬件重新用于主干 ViT ，以最小的硬件开销实现。（ii） 由于多头并行性，GEMM 循环平铺应容纳额外的平铺维度。（iii） ViT 比 CNN 使用更多的非线性运算，我们需要改进这些运算，以便在不损失准确性的情况下实现更激进的量化和高效的硬件实现。

B ViTCoD Accelerator’s Micro-architecture 整体架构

1.Architecture Overview

2.Encoder/Decoder Engines

C LIF资源利用图+LIF跳0计算





## VI Evaluation

A. Experimental  setup

B. Accuracy and GMACs Results

### C. Hardware Results

### 1. Latency

### 2. Throughput

### 3. Memory utilization

### 4. Power utilization

### 5. GPU hardware variations



![image-20241007143946709](C:\Users\10418\AppData\Roaming\Typora\typora-user-images\image-20241007143946709.png)

横坐标 参数量 

纵坐标 Accuracy

spikformer

TIM

Res-SNN



Peak power consumption and area breakdown for Mirage

![image-20241002222613770](C:\Users\10418\AppData\Roaming\Typora\typora-user-images\image-20241002222613770.png)



 Chip layout and its implementation result

![image-20241002175312961](C:\Users\10418\AppData\Roaming\Typora\typora-user-images\image-20241002175312961.png)

## VII Conclusion





In this paper, we have proposed a hardware-efficient image-adaptive token pruning framework called HeatViT for ViT inference acceleration on resource-constraint edge devices. To improve the pruning rate and accuracy, we first adopted an effective and hardware-efficient token selector that can more accurately classify tokens and consolidates non-informative tokens. We also implemented a proof-of-concept ViT hardware accelerator on FPGAs by heavily reusing the hardware components built for the backbone ViT to support the adaptive token pruning module. Besides, we propose a polynomial approximation of nonlinear functions for ambitious (8-bit) quantization and efficient hardware implementation. Finally, to meet both the target inference latency and model accuracy, we applied a latency-aware multi-stage training strategy to learn the number of token selectors to insert into the backbone ViT, and the location and pruning rate of each token selector. Experimental results show that HeatViT achieves superior pruning rate and accuracy compared to state-of-the-art pruning studies while incurring a trivial amount of hardware resource overhead.