---
title: "Probing Classifiers"
headline: "Probing Classifiers: Promises, Shortcomings, and Advances"
description: "论文笔记：系统梳理 probing classifier 框架，指出 probe 更适合回答 representation 中信息的 extractability，而不是直接证明模型真的在使用该信息。"
status: "complete"
visibility: "public"
topic: "safety_alignment"
venue: "Computational Linguistics 2021"
year: "2021"
paper_url: "https://arxiv.org/abs/2102.12452"
source_pdf: "raw/papers/2102.12452.pdf"
tags: ["probing", "representation_reading", "methodology"]
related: ["Discovering_Latent_Knowledge", "AxBench"]
created: "2026-04-18"
updated: "2026-05-09"
draft: "false"
---

> [!abstract] Contributions
>
> 这篇文章不是再提出一个新的 probe，而是对 **probing classifier/probe classifier** 这一整套方法做了一次非常必要的清算。作者把 probing 的基本框架写得很清楚：有一个执行原始任务的模型 $f$，它在中间层产生表示 $f_l(x)$；再训练一个额外分类器 $g$，用这些表示去预测某个外部属性 $z$。论文最重要的贡献，不在于给出更高 probing accuracy，而在于把这套框架拆成几个独立但常被混淆的问题：**要和什么 baseline 比、probe 应该多复杂、probe 的结果到底说明相关性还是因果性、数据集和任务是否被混在了一起、以及被 probe 的属性本身是不是先验限定死了**。沿着这条线，作者系统整理了后续改进，包括 random baselines、control tasks、selectivity、control functions、control datasets、MDL 指标以及表示干预类因果 probing。
>
> 这篇文章最重要的结论非常克制也非常硬：**高 probing performance 主要说明某个属性在表示中是可提取/extractable 的，而不自动说明原模型真的在用这个属性做决策**。如果不做控制和干预，probe 很容易更多反映 probe 自己的能力、数据集捷径，或者标签分布，而不是原模型内部真正依赖的机制。论文的边界也很明确，它是一个方法论综述，不提供统一大实验去最终裁决各派观点；因此它更像一份“你以后再做 representation reading 时必须时刻带着的审稿人清单”。
>
> 在阅读顺序上，它应该放在 `Discovering_Latent_Knowledge` 和 `Representation_Engineering` 之前，作为所有 readout / probe 结果的审稿人框架。后续 `AxBench` 几乎继承了这篇的批判精神，只是把问题从“读出一个属性是否说明模型用了它”推进到“读出的 concept direction 是否真的能稳定 steering”。

## 1. Introduction

probe 会在 NLP interpretability 里变得这么流行，原因很简单：深度模型太黑箱，而我们又想知道它的中间表示到底编码了什么。于是一个看起来近乎显然的办法出现了：如果某个语言学属性能从 hidden state 里被轻松预测出来，那模型应该“学到了”这个属性。这个逻辑在早期工作里非常有吸引力，因为它比只看下游任务分数要细得多，也比直接猜测模型内部结构来得便宜。

但这篇文章的价值恰好在于，它提醒你这个逻辑其实跳了很多步。probe 预测得准，可能说明表示里有相关信息；也可能只是 probe 太强，自己把数据集背下来了；还可能是原始模型和 probing 数据集之间存在偶然共线性。更尖锐的一点是，就算某个属性在表示里“可读”，也不代表原模型在做原始任务时真的依赖它。**readable/可读**、**extractable/可提取**、**used/被模型实际使用**，这三个概念在 probe 文献里经常被混说，而本文就是来把它们强行拆开的。

这也是为什么这篇 survey 对 `safety_alignment` 很有价值。后面你读 `Discovering Latent Knowledge`、`Representation Engineering`、`Sparse Autoencoders`，再到 `AxBench`，核心问题其实都和这里一样：**我们从表示里读到的东西，到底是真实机制，还是一种与任务相关但未必被模型依赖的相关物**。Belinkov 这篇文章相当于把后面一整条 reading / steering 线的 methodological debt 先记出来了。

## 2. Problem Setup

作者把 probing classifier 框架形式化成四个对象。第一是原始模型

$$
f: x \mapsto \hat{y},
$$

它在原始数据集 $D_O=\{(x^{(i)},y^{(i)})\}$ 上训练，并用某个原始任务性能指标 $PERF(f,D_O)$ 评价。第二是中间表示 $f_l(x)$，也就是我们真正拿来做 probe 的隐藏层输出。第三是 probing classifier

$$
g: f_l(x) \mapsto \hat{z},
$$

它在单独的 probing 数据集 $D_P=\{(x^{(i)},z^{(i)})\}$ 上训练，用来预测某个属性 $z$，通常是词法、句法或语义性质。第四是 probing performance

$$
PERF(g,f,D_O,D_P),
$$

它不仅依赖 probe 本身，还依赖原模型、原始数据集和 probing 数据集。

这个写法的好处是，它把 probe 从一个“简单实验”还原成了一个多组件系统。你不能只看最终 accuracy，而不问这些 accuracy 究竟在比较什么。作者还指出，从信息论视角看，probe 的训练可以理解为在估计属性 $z$ 与表示 $h=f_l(x)$ 之间的互信息

$$
I(z;h),
$$

但这仍然只是“相关信息量”的视角，不自动提供因果解释。

更重要的是，这个 formalization 直接暴露了 probing 常被忽略的混杂源。比如，$D_P$ 的标签结构可能本身就容易被背；$g$ 的复杂度可能远高于你以为的“轻量读出”；$D_O$ 和 $D_P$ 的分布可能把任务和属性捆死。后面的所有批评，基本都是沿着这几个变量展开。

## 3. Algorithm / Methods / Model

### 3.1 Probing 能承诺什么

作者先回顾了 probing 的“Promises”。最早的 probing 工作把它当作一种**细粒度评估表示质量**的方法：不再只问 embedding 或 hidden state 对下游任务有没有用，而是问里面有没有 part-of-speech、tense、syntax、semantic role 之类的结构。于是 probing 常被用来支持几类说法：表示质量更好、某种语言信息更可读、某属性更容易被抽取，甚至原模型可能依赖该属性完成原任务。

问题在于，这些说法强度完全不同。本文的一个核心贡献，就是明确指出 probing 最稳妥的语义其实是 **extractability**。也就是说，probe 准确率高，最多说明“存在某个从表示到属性的读出函数”，而不该自动上升成“这个属性是原模型的工作变量”。

### 3.2 Comparisons and Controls

第一个大问题是：一个 probe 分数到底算高还是低。只报一个 accuracy 几乎没有意义，所以文献里开始出现各种 baseline 和 control。

最朴素的是 majority baseline、static embedding baseline 和 random-feature baseline。random baseline 特别重要，因为很多工作发现，就算原模型是随机初始化的，probe 也能在某些任务上读出不少信息。这说明 probe 的成功不能简单归功于“训练出的好表示”。

接着，Hewitt and Liang 提出 **control task** 和 **selectivity**。他们把 probing 数据集的标签随机打乱，构造一个只能靠记忆而不是靠语义结构完成的任务，然后定义

$$
SEL(g,f,D_O,D_P,D_{P,\mathrm{Rand}})
= PERF(g,f,D_O,D_P) - PERF(g,f,D_O,D_{P,\mathrm{Rand}}).
\tag{1}
$$

如果一个 probe accuracy 很高，但 selectivity 很低，那么这通常意味着 probe 主要在背数据，而不是在衡量表示里真实有用的信息。线性 probe 往往 selectivity 更高，而复杂非线性 probe 更容易在这里露怯。

Pimentel 等人则从信息论出发，用 **control function** 替代 control task。设 $c$ 是作用在表示上的某个控制函数，他们定义信息增益

$$
G(z,h,c)=I(z;h)-I(z;c(h)).
\tag{2}
$$

直觉上，你在比较“原始表示关于属性的信息量”和“被控制后剩下的信息量”。这条线把 probing 往更 principled 的统计解释推了一步。

再往前，Ravichander 等人提出 **control dataset**：构造一个原始任务仍然可做、但属性 $z$ 对原任务不再有判别力的数据集 $D_{O,z}$。如果模型在这样的数据上训练后，probe 仍能把 $z$ 读出来，就说明“representation 中能读出 z”并不自动等于“z 对原任务重要”。

> [!note] 这篇文章最值钱的一刀
>
> 它直接把 probing 从“看 accuracy 就够了”的幼稚阶段，推进到了“你必须先控制 probe、控制数据、控制标签，再谈 representation” 的阶段。

### 3.3 Which Probe Should You Use?

第二个问题是 probe 本身应该有多强。早期很多文章偏爱线性 probe，因为它们更“保守”：如果线性读得出来，大家更愿意相信这是表示里相对直接、低复杂度的信息。复杂 probe 虽然可能分数更高，但也更可能自己额外计算出属性，而不是单纯读取。

不过这件事没有这么简单。Pimentel 等人从“估计表示里到底有多少信息”这个目标出发，反而主张应该用**尽可能强的 probe**，因为弱 probe 可能低估表示中的信息量。Voita 和 Titov 则试图两边兼顾，提出 **minimum description length/MDL** 作为联合衡量 probe 性能与复杂度的指标：

$$
MDL(g,f,D_O,D_P),
\tag{3}
$$

它衡量的是：已知表示以后，还需要多长代码才能传输属性 $z$。这个量本质上是在问，probe 到底用了多复杂的“解码程序”。

作者对这条线的态度很清楚：不存在一个绝对正确的 probe 复杂度，关键是你到底想回答哪个问题。若你关心“信息是否存在”，更强 probe 有意义；若你关心“信息是否以简单形式可用”，简单 probe 更有解释力。因此，probe 选择本身就是研究问题的一部分，而不是实验细节。

### 3.4 Correlation vs. Causation

这是整篇 survey 最重要的一节。标准 probing 把 $f$ 冻住、单独训练 $g$，所以它天然只能告诉你 **correlation**：属性 $z$ 与表示 $f_l(x)$ 之间有关联。它并不告诉你，原模型在完成 $x \mapsto y$ 时是否真的使用了这部分表示。

为了逼近因果结论，后续工作开始做**表示干预**。Giulianelli 等人用 probe 的梯度去修改原模型表示，再看原任务性能如何变化；Elazar 等人则尝试把某些属性从表示里投影掉，构造去除了该属性的新表示 $\tilde{f}_l(x)$，然后观察原始任务性能 $PERF(\tilde{f},D_O)$ 是否下降。结果并不一致：有时 probe 读得很准，但把对应方向拿掉，对原任务影响并不大。这恰恰说明了一个残酷事实：**representation 中“有某信息”和模型“靠该信息工作”不是一回事**。

因此，作者最后给出的态度相当明确：如果想知道某个属性是不是原模型真正依赖的中介变量，probe 本身不够，必须结合 intervention、ablation、causal mediation 或 continued training 之类的因果方法。

> [!note] Claim Structure
>
> 这篇 survey 的主张链条是：probing classifier 能回答的最低限度问题是某属性是否可从表示中提取，而不是模型是否真正使用该属性；要把结论从 extractability 推向 mechanism，需要 baseline、control task、probe-complexity control、dataset control 和因果干预。它的证据不是单一新实验，而是对 random baselines、selectivity、MDL、control datasets 和 intervention probing 等文献的综合。作为 safety 前置工作，它的影响是给 truth probe、deception detector、SAE label 和 concept direction 都加上一条审稿规则：读得出来必须和用得上、干预得动分开证明。

## 4. Experiments

严格说，这篇文章没有自己的统一实验表，而是在综述中串联了多个代表性结果。就笔记写作来说，这一点反而应该被保留下来，因为它本身就是一个重要边界：**这篇 paper 的贡献不是用一个新 benchmark 一锤定音，而是搭起一个评价 probing 文献的框架。**

作者归纳出来的实证共识大致有几条。第一，random 或 shallow baselines 往往比直觉里强得多，因此绝不能把 probing accuracy 直接读成“模型学会了语言学结构”。第二，非线性 probe 往往会带来更高 accuracy，但 selectivity 更低，说明它们更容易混入 probe 自身的记忆和计算。第三，probe 与原任务之间的关系不是稳定单调的；某些属性可被高度提取，但并不显著影响原模型原任务表现。第四，dataset choice 的作用远比文献里常被承认的更大，很多关于“模型比模型更懂某属性”的结论，其实掺杂了“在哪个数据集上训练、在哪个数据集上 probe”。

这组梳理最重要的价值，是把 probing 从一种“默认可信的镜子”降级成一种**需要精心校准的测量仪器**。如果你之后还要读 concept detector、honesty probe、SAE labeling 或 steering benchmark，这个结论几乎是必须先记住的。

从 safety 的角度看，这个结论尤其锋利。很多安全论文会训练一个 probe 去检测 truthfulness、deception、harmfulness、sycophancy 或 power-seeking，然后把 probe 分数当成内部状态证据。本文提醒我们，这一步必须非常小心：probe 高分可能只说明这些属性在某个 prompt distribution 下可被线性分开，不能自动说明模型在真实部署时会按照这个属性决策，也不能说明干预这个方向就能稳定改变行为。因此，后续安全 readout 至少需要三类补强证据：跨数据分布泛化、与输出行为的因果干预关系、以及排除 probe 自身过拟合的 control task / control dataset。

这也解释了为什么“简单线性 probe”有时反而比复杂 probe 更适合解释。复杂分类器可能把表示中分散的弱线索重新组合成一个很强的预测器，但这部分计算可能发生在 probe 里，而不是原模型里。线性 probe 的能力更弱，结论也更窄：如果线性可分，说明表示中至少存在一个较直接的方向；如果只有深层非线性 probe 能读出，研究者就需要更谨慎地说明，这到底是模型表示本身的结构，还是 probe 额外做了大量任务求解。这个 distinction 对后面的 concept vector、truth direction、refusal direction 都非常关键。

## 5. Related Work & Future Work

这篇 survey 和后来的几条主线都有直接关系。它和 `Discovering Latent Knowledge` 的关系最紧，因为 `CCS` 本质上也在做 representation reading，只是它换成了逻辑一致性而不是监督 probe；和 `Representation Engineering` 的关系也很直接，因为 `LAT`、lie detector、harmfulness detector 这些方法，都面临本文提出的同一批 methodological objections；至于 `AxBench`，几乎可以看成是把本文的批评精神继续推进到 steering benchmark 上。

未来工作方向在本文里说得很清楚。第一，要更系统地分离 **task** 与 **dataset**，避免把数据集特定模式误当成任务相关结构。第二，要发展不依赖预定义属性的 reading 方法，因为传统 probe 强依赖现成标注和研究者先验。第三，也是最关键的，要把 probing 和因果干预更紧地绑在一起。只要这一点不做到位，probe 最稳妥的定位就仍然只是：**衡量某属性在表示中的可提取程度**，而不是模型真实机制的终局解释。

这也解释了为什么它虽然是 2021 年的 survey，仍然应该保留在当前 safety_alignment 队列里。随着领域从 linguistic probing 转向 honesty probes、representation steering、SAE feature labels 和 model diffing，旧问题并没有消失，只是换了更安全相关的名字。读这篇的价值，就是给后续所有“我们在激活里找到了一个方向”的论文加上一组检查项：这个方向是读出来的，还是因果有用的；是数据集属性，还是模型机制；是 probe 学到的，还是模型本来在用的。

对当前阅读路径来说，这篇论文还提供了一种批判姿势：不要被“内部表示”四个字自动说服。内部测量当然比只看输出更有信息，但它仍然是测量，不是机制本身。只有当读出结果能跨数据保持、能通过干预改变行为、能在控制任务下保持选择性时，我们才有资格把它往机制解释上推进。
