---
title: "InfoGAN"
headline: "InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets"
visibility: "public"
status: "complete"
description: "Paper note on InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets."
---

> [!abstract]

> [!info] Contributions

## 1. Introduction



<!-- 
### 1 引言 (Introduction)

无监督学习可以被描述为一个从海量无标签数据中提取价值的一般性问题。在无监督学习中，一个流行的框架是表征学习（representation learning）[1, 2]，它的目标是利用无标签数据来学习一种表征，该表征能将重要的语义特征展现为易于解码的因子。研究表明，能够学习到这种表征的方法是可能存在的 [2]，并且这种表征对于包括分类、回归、可视化以及强化学习中的策略学习在内的许多下游任务都非常有用。

由于在训练时相关的下游任务是未知的，这导致无监督学习通常被视为一个不适定（ill-posed）问题。尽管如此，**解耦表征**（disentangled representation，即显式表示数据实例中显著属性的表征）应当会对这些相关但未知的任务有所帮助。例如，对于一个人脸数据集，一个有用的解耦表征可能会为以下每个属性分配一组独立的维度：面部表情、眼睛颜色、发型、是否戴眼镜，以及对应人物的身份。解耦表征对于需要依赖数据显著属性知识的自然任务（如人脸识别和目标识别）是非常有用的。然而，对于非自然的监督任务来说并非如此，例如，在这类任务中，目标可能是判断图像中红色像素的数量是奇数还是偶数。因此，为了具备实用性，无监督学习算法实际上必须在不直接接触下游分类任务的情况下，正确地预测出可能的下游任务集合。

生成式建模（generative modelling）推动了很大一部分无监督学习的研究。它的动机源于这样一种信念：合成或“创造”观测数据的能力本身就蕴含了某种形式的理解；因此人们希望一个优秀的生成模型能够自动学习到解耦表征，尽管我们很容易构建出能完美生成数据但表征却极其糟糕的生成模型。目前最著名的生成模型是变分自编码器（Variational Autoencoder, VAE）[3] 和生成对抗网络（Generative Adversarial Network, GAN）[4]。

在本文中，我们对生成对抗网络的目标函数提出了一个简单的修改，旨在鼓励其学习可解释且有意义的表征。为此，我们最大化了 GAN 的一小部分固定噪声变量子集与观测数据之间的互信息（mutual information），事实证明这种方法相对直观且简单。尽管该方法非常简单，但我们发现其出奇地有效：它能够在多个图像数据集——如手写数字（MNIST）、人脸（CelebA）和门牌号（SVHN）上，发现具有高度语义且意义明确的隐藏表征。我们通过无监督方式学习到的解耦表征质量，完全可以媲美以往利用监督标签信息进行研究的成果 [5–9]。这些结果表明，结合互信息代价（mutual information cost）的生成式建模，可能是学习解耦表征的一条富有成效的途径。

在本文的其余部分，我们首先回顾相关工作，特别指出了以往学习解耦表征的方法所依赖的监督信息。接着，我们回顾作为 InfoGAN 基础的 GAN。随后，我们详细阐述最大化互信息如何催生出可解释的表征，并推导出一个简单而高效的算法来实现这一目标。最后，在实验部分，我们首先在相对纯净的数据集上将 InfoGAN 与以前的方法进行比较，然后证明 InfoGAN 能够在复杂的数据集上学习到可解释的表征，而在这些复杂数据集上，目前尚未发现有其他无监督方法能够学习到同等质量的表征。
 -->


## 2. Related Work

## 3. Preliminaries


