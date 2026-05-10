---
title: "Universal Steering & Monitoring"
headline: "Toward universal steering and monitoring of AI models"
description: "论文笔记：用 Recursive Feature Machines 从多层激活中抽取概念向量，并把它们统一用于 steering、monitoring 与能力增强。"
status: "complete"
visibility: "public"
topic: "safety_alignment"
venue: "Science 2026"
year: "2026"
paper_url: "https://arxiv.org/abs/2502.03708"
source_pdf: "raw/papers/universal-steering-monitoring.pdf"
tags: ["activation_steering", "monitoring", "concept_vectors"]
related: ["Representation_Engineering", "ActAdd", "Sparse_Autoencoders", "AxBench"]
created: "2026-04-15"
updated: "2026-05-09"
draft: "false"
---

> [!abstract] Contributions
>
> 这篇论文试图把 **概念向量/concept vectors** 从零散技巧推进成一套通用接口。作者的核心判断是：很多安全相关或能力相关的语义概念，在大模型的多层激活里可以被压缩成 **低维线性结构/low-dimensional linear structure**。围绕这个判断，论文提出了一套统一方法：先在每一层上用 **递归特征机/Recursive Feature Machines/RFM** 学出面向目标概念的非线性预测器，再从预测器的 **平均梯度外积/Average Gradient Outer Product/AGOP** 中抽出最重要的线性方向，最后把这些跨层方向同时用于 **激活 steering** 和 **内部状态 monitoring**。论文最强的实验支撑不在单个 demo，而在三类系统证据：跨 512 个概念的自动化 steering 评测、六个 hallucination/toxicity benchmark 上监测优于 judge model、以及 Python→C++ 代码生成这类高精度任务上 steering 优于直接 prompting。
>
> 这篇论文也有很明确的边界。它最强的“universal”证据，其实主要是 **同一模型内部跨概念、跨语言、跨模态** 的通用性，而不是“一个模型学到的向量可以无损迁到另一个模型”。此外，512 概念的自动评测高度依赖 GPT-4o 生成概念、训练语料与 judge，因此大规模结果更适合支持“这条路线可扩展”，而不是支持“每个 concept vector 都已经被严格验证”。作者自己在讨论部分也很诚实地指出：**为什么分类得到的方向可以拿来 steering，以及为什么这么多复杂概念会呈现线性表示，本身仍然是开放问题。**
>
> 它是 `ActAdd` 和 `Representation_Engineering` 的规模化后续：ActAdd 给出最小 steering 操作，RepE 给出 top-down reading/control 框架，本文则用 RFM/AGOP 把概念方向估计做成自动化 pipeline。它也为后面的 `AxBench` 提供被检验对象：如果 concept vectors 真能成为通用 steering/monitoring 接口，就必须在更系统的 benchmark 上面对 prompt、finetuning、SAE 和 ReFT 等基线。

## 1. Introduction

这篇论文真正想解决的，不是某一个具体的安全问题，而是一个更基础的表征问题：**如果模型内部确实“知道”很多东西，我们怎样才能稳定地把这些知识从激活里读出来，并进一步把这种读取接口变成控制接口。** 过去这条线大致有两类思路。一类是 **稀疏自编码器/Sparse Autoencoder/SAE** 这类无监督方法，优点是能在不预设概念的情况下发现可解释 feature，缺点是你并不能保证它会正好把你关心的 harmfulness、hallucination 或 dishonesty 找出来。另一类是监督式 probing，也就是直接拿概念标签去训练线性或逻辑回归探针。作者认为，后一类路线在“定向提取指定概念”上更贴近实际需求，但现有方法对复杂概念的表达能力不够，也没有很好利用跨层信息。

因此，论文的切入点非常明确：它不再把 **steering** 和 **monitoring** 当成两套分离技术，而是把二者都看成“从内部表征中提取概念方向”的两个下游任务。只要你能在各层激活里找到与目标概念最相关的方向，那么同一组方向既可以被加到激活上改变输出分布，也可以被投影成特征来判断某种概念当前是否活跃。这就解释了为什么这篇论文在 `safety_alignment` 里非常关键。它不是简单又发明了一个 steering trick，而是在试图把 **安全监测/monitoring**、**行为控制/steering** 和 **能力改善/capability improvement** 统一成同一套表示级接口。

更有意思的是，作者并没有只拿“anti-refusal”或“harmfulness”这种传统安全概念做展示，而是有意把范围拉宽到 **人类语言、编程语言、persona、政治立场、诗歌风格、商品评分**，甚至 **多概念线性混合**。这背后的论断非常激进：如果这些概念都能以近似线性的形式被抽出，那么大模型里存在的就不只是零散的 feature，而可能是一套更广义的 **概念几何/concept geometry**。这也正是标题里 “universal steering and monitoring” 的野心所在。

## 2. Problem Setup

论文把对象设定为标准的 decoder-only 大模型，也允许输入里包含图像，因此 LLM 和 VLM 都被纳入同一套符号系统。对任意输入 $X$，模型被分解成 embedding、逐层 block 和最终 readout：

$$
\begin{aligned}
A_1(X) &= E(X), \\
A_\ell(X) &= B_\ell(A_{\ell-1}(X)), \quad \ell \in \{2, \dots, L\}, \\
f(X) &= R(A_L(X)).
\end{aligned}
\tag{1}
$$

这里 $A_\ell(X)$ 表示第 $\ell$ 个 block 的激活，$B_\ell$ 是对应 block 的变换，$R$ 是最终读出层。作者后面的全部方法都只依赖这些 block 输出，而不需要改动模型参数本身。这一点很关键，因为它意味着这篇论文的目标不是再训练出一个新模型，而是直接把 **已有模型的激活空间** 当成可操作对象。

在这个框架下，作者考虑两类任务。第一类是 **steering/引导生成**：给定一个概念标签数据集 $\{(X^{(i)}, y^{(i)})\}_{i=1}^n$，其中 $y^{(i)}$ 可以是二值标签，也可以是实数评分，学习一组每层概念向量 $\{v_\ell\}$，使得往激活里加入这些向量后，模型输出会更靠近目标概念。第二类是 **monitoring/监测概念活跃度**：同样给定带标签样本，但这次不是修改生成，而是从每层激活中提取对目标概念最敏感的方向，再把这些方向上的投影拼成一个判别特征，用来判断某个输入或输出是否含有 hallucination、toxicity 之类的失配内容。

> [!note] Problem Decomposition
> 这篇论文最干净的建模动作，是把 steering 和 monitoring 都归约成“先抽概念方向，再做不同下游”的问题。这样一来，安全相关行为不再只是输出层的字符串现象，而是被重写成 **激活空间中的概念表示问题**。

## 3. Algorithm / Methods / Model

### 3.1 Method Overview

方法主线其实可以压成一句话：**每层训练一个面向目标概念的非线性预测器，用预测器的输入敏感方向来定义 concept vector，再把这些跨层 vectors 拿去做 steering 或 monitoring。** 这里真正的技术选择是作者没有满足于线性 probe，而是用了 **Recursive Feature Machines/RFM**。RFM 的作用不是替代分类器本身，而是在非线性预测器上面再做一层 feature extraction，找出“哪些输入方向最影响这个概念预测”。

论文先把每个样本在每层的最后一个 token 激活记成 $a_\ell^{(i)}$，然后在第 $\ell$ 层单独训练一个 RFM 预测器。与普通 probe 最大的区别在于，RFM 不只是输出一个分类结果，它还会计算预测器关于输入的 **AGOP**，也就是平均梯度外积矩阵。作者随后取这个矩阵的主特征向量作为该层的 concept vector。直觉上，这个向量刻画的是：**沿着哪个方向移动激活，会对“这个样本是否属于目标概念”产生最大影响。**

### 3.2 RFM Concept Extraction

作者用的是带 Mahalanobis 度量的 Laplace kernel：

$$
K_M(x, z) = \exp\left(
-\frac{1}{L}\sqrt{(x-z)^\top M (x-z)}
\right),
\tag{2}
$$

其中 $L$ 是 bandwidth，$M$ 是正半定矩阵。RFM 在训练中迭代两步：先用 kernel ridge regression 拟合概念标签，再根据当前预测器对输入的梯度计算 AGOP：

$$
M_{t+1}
=
\frac{1}{n}\sum_{i=1}^n
\nabla_z \hat f_t(a^{(i)})
\nabla_z \hat f_t(a^{(i)})^\top.
\tag{3}
$$

然后取最终 $M_T$ 的主特征向量作为候选 concept vector，再用它与标签的 Pearson 相关系数来定向。这个设计比线性 probe 更强的地方在于，它允许“概念是激活的非线性函数”，但最后抽出来的仍然是 **可加到激活上的线性方向**。这就把“复杂概念的非线性学习”与“部署时的简单线性干预”接到了一起。

### 3.3 Steering

有了每层 concept vector 之后，steering 本身非常直接。对每层每个 token 的激活，作者都做同样的加性扰动：

$$
\tilde A_{\ell,i}(X)
=
A_{\ell,i}(X) + \epsilon v_\ell, \qquad i \in [t].
\tag{4}
$$

这里 $\epsilon$ 是控制强度的 **control coefficient**。如果要做多概念 steering，只需要把多个概念的向量做线性组合后再加进去。这个设计的一个关键含义是：**作者假设许多概念在每层都存在一个可以“整体推一把”的方向。** 这和后面关于 concept mixing、语言迁移、persona 叠加的实验是完全对应的。

### 3.4 Monitoring

monitoring 版方法与 steering 的差别不在于前半段，而在于后半段如何组织特征。作者不再只取每层一个主方向，而是取前 $p$ 个特征向量 $\{v_{\ell,j}\}_{j=1}^p$，然后把样本在这些方向上的投影拼接成特征向量：

$$
h^{(i)}
=
\Big[\langle a_\ell^{(i)}, v_{\ell,j} \rangle\Big]_{\ell=2,\dots,L;\, j=1,\dots,p}.
\tag{5}
$$

后续可以在单层特征上训练判别器，也可以把所有层的特征拼起来训练聚合判别器。论文最终报告的最好结果，会在“最佳单层 probe”和“跨层聚合 probe”之间择优。这个设计解释了为什么标题里是 **steering and monitoring** 而不是单写 steering：作者真正想强调的是，这套方法不是一个只会制造 demo 的方向，而是一套同时支持干预和检测的公共表示接口。

> [!tip]
> 这篇论文最值得记住的技术点，不是 “RFM 比 logistic 更强” 这么平面，而是它把 **非线性学习** 和 **线性控制** 放到了同一个管道里。概念可以是复杂的，但部署接口依然是简单的向量加法和投影分类。

> [!note] Claim Structure
>
> 论文的主张链条是：许多安全和能力相关概念可以在多层激活中被表示为可提取的低维方向；RFM/AGOP 能比简单线性 probe 更稳定地从复杂概念标签中抽出这些方向，并同时用于 steering 与 monitoring。证据来自跨 512 概念的自动 steering 评测、Python→C++ 代码生成、跨语言/多模态 demos 以及六个 hallucination/toxicity benchmark 上 probe 对 judge 的优势。限制是 “universal” 主要指同一方法可扩展，而不是单个向量跨模型通用；训练数据和 judge 依赖 GPT-4o，classification direction 为什么可转成 steering direction 仍缺理论解释。

## 4. Experiments

实验部分基本是在回答三个问题：这套方法能不能大规模工作，能不能在真正需要精度的任务上工作，以及它作为监测器到底是不是比直接让模型当 judge 更靠谱。

第一组结果是 **跨模型、跨模态、跨语言的 steering demos**。作者在 Llama-3.3-70b-4-bit 上用 anti-refusal 向量把原本拒答的模型推到给出制作 thermite 的具体步骤；在 Llama-vision-3.2-90b-4-bit 上用 liberal / conservative 向量让模型对堕胎立场明确站队；在蒸馏到 Llama 的 DeepSeek 模型上，用 deception 向量的负方向让模型从“撒谎规避监管”转向直接承认 insider trading。更重要的是，英文训练得到的 influencer 或 conspiracy vectors，可以直接拿去 steer 中文和法文输出；而把两个概念向量线性相加之后，还能得到 “social media influencer + conservative” 这类混合 persona。作者把这些结果解释为 **概念表示的跨语言迁移性和可组合性**。

真正把 “universal” 这件事撑起来的，是第二组 **512 概念自动化评测**。作者先让 GPT-4o 生成五类概念：fears、experts、moods、topophiles、personas，总计 512 个；再用 400 条通用 statements 加 prefix 的方式自动构造正负训练样本；最后对每个概念设计五个测试问题，并让 GPT-4o 判断被 steer 后的回答是否成功体现了目标概念。这里最重要的结果有两个。第一，**RFM 在总体 steerability 上稳定优于 PCA、difference-in-means 和 logistic regression**。第二，**新一些、更大的模型明显更可 steer**：RFM 的 overall steerability 从 Llama-3.1-8b 的 **49.0%**，升到 Llama-3.1-70b-4-bit 的 **63.0%**，再升到 Llama-3.3-70b-4-bit 的 **79.8%**。其中 topophile 这一类概念最夸张，从 **21.3%** 直接升到 **90.8%**。这组结果很醒目，因为它说明 steering 不是越大模型越难控，相反，模型越新越强，越可能在内部形成可线性调用的概念结构。

第三组结果是 **高精度任务上的能力提升**。作者不是只做风格或 persona 迁移，而是让模型从 Python 代码转到 C++，并拿 50 道 HackerRank 题、共 732 个测试用例去评估。对 Llama-3.3-70b-4-bit 而言，steered C++ 的平均通过率达到 **73.2%**，高于直接 prompt “Answer in C++” 的 **66.6%**，也高于模型原本默认输出 Python 时的 **69.5%**。这组结果意义很大，因为它说明 steering 不只是把输出风格推歪，而是真的可能把模型推到另一个更优的求解轨道上。作者还在附录里展示了 **chain-of-thought steering** 可以提升某些定量推理任务表现，但同时也指出 steering 过强会让模型 “overthink”，最后反而答不出来。

第四组结果是 **monitoring**。作者在六个 benchmark 上比较了两类方案：直接让 LLM judge 输出，以及基于 concept vectors 的 probe。数据集覆盖 FAVABENCH、HaluEval General、HaluEval-Wild、PubMedQA、RAGTruth 和 ToxicChat。最强的 probe 在六个数据集上的 AUROC 分别达到 **0.791、0.728、0.924、0.964、0.894、0.969**，而对应最强 judge model 的最佳值分别是 **0.713、0.647、0.801、0.962、0.817、0.961**。也就是说，除了 PubMedQA 这种本来 judge 已经很强的任务，两者差距不大之外，其余 benchmark 上 probe 都有实质优势。这一点很硬，因为它直接支持了论文的核心主张：**看内部激活，有时比让模型自己“口头判断”更靠谱。**

这些实验的说服力确实很强，但边界也非常清楚。首先，512 概念的训练语料、评测问题和 judge 都高度依赖 GPT-4o，因此这部分更像是一套可扩展 pipeline 验证，而不是每个 concept 都经过人工严审。其次，跨语言迁移是很强的证据，但 **跨模型迁移** 并没有被同样强度地证明；作者的“universal”更多是指同一方法能在不同模型上重新学出向量，而不是“一组向量处处通用”。再次，许多安全 demo 的效果极其戏剧化，但主文里关于副作用、局部退化和分布外失控的系统定量分析还不够。最后，作者自己也承认：**classification vector 为什么能兼做 steering vector**，从几何上仍然没有被解释清楚。

## 5. Related Work & Future Work

在 Related Work 上，这篇论文最自然的邻居有两支。一支是 **无监督 mechanistic interpretability**，例如 SAE、dictionary learning、monosemanticity 这条线。论文对这一支的态度很明确：它承认这些工作证明了模型里存在可解释 feature，但认为它们不保证能定向找到用户想要的概念。另一支是 **监督式 probing 与 activation intervention**，例如线性 probe、PCA、difference-in-means、Inference-Time Intervention/ITI 以及后来的 refusal direction。本文真正的推进，在于它把这些方法从“在单层里找一个向量”的局面，推进到 **非线性 feature learning + 多层聚合**，并同时把 steering 与 monitoring 放进同一个框架。

放回 `safety_alignment` 语境里看，这篇论文的价值尤其体现在它为后续的 **persona features、persona vectors、EM directions** 提供了一套更一般的母框架。后来那些直接围绕 misalignment 或 character drift 的工作，很多都可以被理解成在这篇 paper 打开的表示级接口上继续 specializing：有的更重 SAE，有的更重轻量 mean-diff，有的更重 cross-run convergence。但它们共同继承的前提，就是 **安全相关行为确实可能在激活空间里留下可测、可控的低维结构**。

作者在 Future Work 里明确留下了两个开放问题，我觉得这两个问题都非常硬。第一个是 **为什么这么多复杂概念会呈现线性表示**。翻译、人设、政治倾向、编程语言这些对象，按常识看都不该天然对应一个固定方向，但实验却反复显示向量加法确实有效。第二个是 **为什么分类能导出 steering**。理论上，把英文和印地文区分开来的超平面，与把英文“平移”成印地文的变换方向，完全可以是两回事；可论文偏偏观察到二者在实践中存在强联系。作者用一张几何示意图说明：就算 classification 和 steering 都是线性的，它们对应的向量也未必相同。这正是本文最深的地方。它不只是给了一个好用方法，也把一个更难的表示理论问题摆到了桌面上。

<!-- TODO: insert Figure 1 from paper: unified pipeline for concept extraction, steering, and monitoring -->
<!-- TODO: insert Figure 4 or Figure 6 from paper: 512-concept steerability scaling and monitoring AUROC comparison -->
