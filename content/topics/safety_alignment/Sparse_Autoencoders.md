---
title: "Sparse Autoencoders"
headline: "Sparse Autoencoders Find Highly Interpretable Features in Language Models"
description: "论文笔记：用 sparse autoencoder 从语言模型激活中学习稀疏特征字典，试图把 feature 从 superposition 中分离出来，并展示更强的可解释性与更细粒度的 IOI 因果定位。"
status: "complete"
visibility: "public"
topic: "safety_alignment"
venue: "arXiv 2023"
year: "2023"
paper_url: "https://arxiv.org/abs/2309.08600"
source_pdf: "raw/papers/2309.08600.pdf"
tags: ["sparse_autoencoders", "superposition", "monosemanticity"]
related: ["Representation_Engineering", "Universal_Steering_Monitoring", "AxBench"]
created: "2026-04-18"
updated: "2026-05-09"
draft: "false"
---

> [!abstract] Contributions
>
> 这篇论文的中心问题非常清楚：如果单个 neuron 往往是 **polysemantic/多义** 的，那我们是不是一开始就切错了解释单位。作者沿着 **superposition/叠加表示** 假说往前走，认为模型真正的 feature 并不一定和单个 neuron 对齐，而更可能是激活空间里一组**过完备/overcomplete**、非正交、稀疏激活的方向。为此，论文训练 **Sparse Autoencoder/SAE** 来重构语言模型内部激活，让重构系数保持稀疏，从而学习出一套 feature dictionary。实验表明，这些 learned features 在自动可解释性评估上明显优于默认 basis、随机方向、PCA 和 ICA；在 **Indirect Object Identification/IOI** 任务上，它们也能比 PCA 更精确地定位造成 counterfactual 行为变化的因果特征；案例分析还显示，一些 feature 的确接近 **monosemantic**，例如 apostrophe、closing parenthesis 等非常具体的模式。
>
> 这篇论文真正奠定的是 SAE 作为 mechanistic interpretability 基础设施的可行性，而不是“SAE 已经彻底解决可解释性”。边界同样必须记住。第一，最稳定的结果主要来自 **Pythia-70M/410M 的 residual stream**，而不是更复杂、更强模型的全域结论。第二，autoencoder 的重构并不完美，用重构替换真实激活会明显伤害 perplexity，说明字典还没有把全部计算保住。第三，自动可解释性评估本身也有很强局限，它更擅长识别 token-level 或局部模式，不一定能公正评估高层 feature。换句话说，这篇论文更像是在说：**从 superposition 中抽 feature 这条路是通的**，而不是已经到站。
>
> 它和 `Representation_Engineering` 的分工很清楚：RepE 直接把高层 safety concepts 当作 reading/control 对象，SAE 则先问激活空间能不能被拆成更可解释的 feature basis。后续 `Universal_Steering_Monitoring` 和 `AxBench` 都依赖这层背景：前者试图绕过纯无监督 SAE，定向学习用户关心的概念方向；后者则检验 SAE-style features 在 steering benchmark 上到底有没有工程优势。

## 1. Introduction

mechanistic interpretability 的一个老问题是：你到底该拿什么当基本分析单位。早期很多工作会盯住单个 neuron，但很快大家发现，neuron 往往是 polysemantic 的，也就是同一个 neuron 会在几类彼此无关的语境里一起亮。这会让“这个 neuron 代表了什么”这种问题变得异常含糊。

这篇论文沿着 Anthropic 的 **toy models of superposition** 直觉往前走，提出一个更硬的解释：也许问题不在于我们还没找对 neuron，而在于模型压根就没把 feature 对齐到 neuron basis 上。尤其在 Transformer 的 **residual stream** 里，本来就没有理由期待标准坐标轴是语义自然基。模型更可能是把比维度数更多的 feature 挤在若干非正交方向里，只要这些 feature 足够稀疏地共现，干扰就还能被控制在可接受范围内。

如果这个判断成立，那么 interpretability 的任务就变了。我们不该再问“哪个 neuron 对应这个 feature”，而应该问“能不能从激活里恢复出那组真正的 feature directions”。作者给出的回答是：可以把这件事写成 **sparse dictionary learning**。这一步非常关键，因为它把 superposition 从一个解释性故事，变成了一个可训练、可验证的技术目标。

从 `safety_alignment` 的角度看，这篇论文的意义也非常直接。后面很多监测和控制工作，无论是 persona vector、SAE feature diffing、GemmaScope 还是 AxBench 对 SAE steering 的检验，都默认接受了一个前提：**activation space 里存在比 neuron 更自然的 feature basis**。这篇 SAE 论文就是这个前提最早的系统实证之一。

这里需要把 **superposition** 和 **polysemanticity** 的关系说清楚。Polysemanticity 是观测现象：一个 neuron 或方向看起来同时响应多个不相关概念。Superposition 是解释机制：模型可能因为维度有限、特征稀疏，而把多个 feature 编码在同一个表示空间的非正交方向里。SAE 的目标不是给每个 neuron 起更好的名字，而是换一个坐标系，让原本叠在一起的 feature 在稀疏 code 里分开。这个区别很重要，因为后续安全工作真正想监测的往往不是 neuron，而是更抽象的 persona、refusal、deception 或 harmfulness feature。

换句话说，SAE 不是在寻找“模型里天然存在的标签”，而是在学习一个对研究者更友好的坐标变换。这个坐标变换是否有意义，要看它能否同时满足三件事：重构原激活时损失不能太大；激活系数必须足够稀疏，避免每个输入都由大量 feature 混合解释；学出的方向还要能被人类或自动解释器稳定命名。三者缺一都会出问题。重构差，说明字典丢了计算；不稀疏，说明没有真正拆开叠加；不可解释，说明换了坐标也没有获得分析收益。

## 2. Problem Setup

作者把问题形式化成一个标准的 sparse dictionary learning 任务。设一组语言模型内部激活向量为

$$
\{x_i\}_{i=1}^{n} \subset \mathbb{R}^{d},
$$

它们来自模型的 residual stream、MLP sublayer 或 attention sublayer。作者假设每个激活都可以由一小部分“真实 feature”线性叠加得到：

$$
x_i = \sum_{j} a_{i,j} g_j,
\tag{1}
$$

其中 $g_j$ 是未知的 ground-truth feature directions，而系数向量 $a_i$ 是稀疏的。目标不是精确恢复唯一正确的 $g_j$，而是学出一套 dictionary features $f_k$，使得每个真实 feature 至少被某个 learned feature 很好地近似。

论文主要在 Pythia-70M 和 Pythia-410M 上做实验，特别关注 residual stream。这里的一个关键直觉是：如果 superposition 真的存在，那么单个 activation $x_i$ 虽然高维，但应该能由一个**很稀疏**的 feature 组合重建。于是，学习目标就变成：找一套过完备字典，让每个激活都能被少量 feature 重建。

这个形式化也解释了为什么字典通常要过完备。真实 feature 的数量可能远大于 residual stream 的维度，但任意一个 token/context 中真正活跃的 feature 只占很小一部分。过完备字典给了模型足够多的候选方向，稀疏惩罚则防止每个输入都用一大堆方向糊过去。换句话说，SAE 同时利用两个假设：feature 多，但每次只激活少数；feature 不一定正交，但稀疏共现能让它们被分离。

## 3. Algorithm / Methods / Model

### 3.1 Sparse Autoencoder as Dictionary Learner

作者用的是一个非常干净的 SAE。给定输入激活 $x$，编码器先产生稀疏 hidden code：

$$
c = \mathrm{ReLU}(Mx+b),
\tag{2}
$$

其中 $M \in \mathbb{R}^{d_{\mathrm{hid}}\times d_{\mathrm{in}}}$，hidden size 设成

$$
d_{\mathrm{hid}} = R \cdot d_{\mathrm{in}},
\tag{3}
$$

$R$ 是控制字典过完备程度的超参数。解码时使用 tied weights，于是重构向量为

$$
\hat{x} = M^{\top} c = \sum_{i=1}^{d_{\mathrm{hid}}} c_i f_i,
\tag{4}
$$

这里 $M$ 的每一行都可以看成一个 dictionary feature $f_i$。由于作者对行做归一化，模型不能简单通过把 feature 向量放大来逃避稀疏惩罚。

训练目标是

$$
L(x)=\|x-\hat{x}\|_2^2+\alpha\|c\|_1,
\tag{5}
$$

其中第一项要求重构准确，第二项用 $\ell_1$ 惩罚鼓励 hidden code 稀疏。直觉上，这个目标在逼模型学一组“只有少量会同时开火”的方向，于是更可能把 feature 从 superposition 中解缠出来。

> [!note] 为什么这里的稀疏性是关键
>
> 如果 feature 不稀疏，很多非正交方向会频繁同时激活，彼此干扰就会变大，superposition 也难以维持。正因为作者假设真实 feature 是稀疏激活的，SAE 才有可能把它们拆出来。

### 3.2 Measuring Interpretability

论文没有满足于“重构得不错”，而是直接追问 learned features 是否更可解释。作者采用 OpenAI 的 **autointerpretability** 流程：先收集某个 feature 的高激活文本片段，让 LLM 写出自然语言解释，再让 LLM 根据这个解释去预测其他文本上的 feature activation，最后用预测与真实激活的相关性当作 interpretability score。

这套指标当然不完美，但在当时很实用，因为作者要比较的是成千上万个 feature。实验里，他们把 SAE 学出的 dictionary features 和默认 basis、随机方向、PCA、ICA 放在一起比，核心问题不是哪种分解更“优美”，而是哪种分解更接近**人类可命名的 feature**。

自动可解释性指标还有一个隐藏假设：如果一个 feature 的高激活样本能被自然语言规则概括，并且这个规则能预测其他样本激活，那么这个 feature 就更接近 monosemantic。但这类评估会偏爱局部、词面、容易描述的 feature；对“模型正在保持某个角色”“回答风格变得更欺骗”“正在执行某个抽象任务策略”这类高层 feature，LLM 解释器可能写不出好规则。因此，本文的自动分数更适合证明 SAE 比传统 basis 更可解释，不适合证明 SAE 已经能稳定覆盖所有安全相关概念。

这点直接影响后续安全读法。假如某个 SAE feature 被解释成“有害建议”或“拒绝回答”，我们不能只看解释文本就相信它代表真正的安全机制；还要看它在不同提示、不同任务、不同上下文下是否稳定激活，以及增强或抑制它是否真的改变对应行为。本文的贡献是证明 feature-level 分解比 neuron basis 更有希望，但它没有取消因果验证的需求。

### 3.3 Causal Localization with IOI

只做 interpretability score 还不够，因为一个方向即使“看上去像有意义”，也不一定是模型真正用到的因果部件。作者因此在 **Indirect Object Identification/IOI** 任务上做 activation patching。具体地说，他们先在 counterfactual target sentence 上记录 feature activations，再把这些 activations 注入 base sentence 对应层的 residual stream，看输出 logits 与 target 的 KL divergence 能被多快推过去。

如果某套 decomposition 更贴近真实 feature，那么理论上应该能用**更少 patch、或更小 edit magnitude** 达到同样的输出改写效果。论文正是用这个标准去比较 SAE dictionary、PCA 和非稀疏字典。

> [!note] Claim Structure
>
> 论文的主张链条是：polysemantic neurons 说明标准 basis 不是合适解释单位；如果模型用 superposition 存储稀疏 feature，那么 overcomplete sparse dictionary 可以恢复更接近人类可命名、也更适合局部干预的 feature basis。SAE 方法用重构损失与 L1 稀疏惩罚学习字典，证据来自自动可解释性优于 neuron/PCA/ICA 以及 IOI patching 中用更少 feature 达到更强 counterfactual effect。限制是模型规模和层位较窄、重构仍损伤 perplexity、自动解释偏向局部 token-level feature，不能直接推出高层 safety trait 已经可靠可控。

## 4. Experiments

解释性结果相当亮眼。论文报告，SAE 学到的 feature 在自动可解释性上显著优于默认 basis、随机方向、PCA 和 ICA，尤其在较早的 residual stream layers 上优势最明显。作者也很诚实地指出，这种优势会在更后层逐渐缩小，可能说明后层 feature 本身更复杂，也可能说明 autointerpretability 对“靠输出效果定义”的高层 feature 评估能力还不够。

更硬的结果来自 IOI patching。作者发现，相比 PCA decomposition，SAE dictionary 能用**更少的 feature patch** 达到同样程度的 target-output KL divergence，也能在**更小 edit magnitude** 下完成相同强度的行为改写。与此同时，如果把稀疏惩罚关掉，变成非稀疏字典，这种优势就消失了。这一点非常关键，因为它说明真正带来更细粒度因果定位的，不只是“多学了一个字典”，而是**稀疏特征分解本身**。

案例分析也非常有说服力。作者展示了一些几乎可以称作 monosemantic 的 feature，比如主要在 apostrophe 上激活，并且压低这个 feature 会最明显地下调下一 token 是 `'s'` 的 logit；还有能追踪 closing parenthesis 的 feature，并能沿前后层构造出一棵相对可读的 causal tree。你可以把这些例子理解成：SAE 不只是给你更高的 interpretability score，而是真开始让“feature-level circuit tracing”变得自动化。

不过实验边界也非常清楚。第一，最稳结果主要在 residual stream；作者自己承认 current pipeline 对 MLP layers 只有 mixed success。第二，重构并不完美。论文直接报告，用 layer-2 reconstruction 替换真实 residual stream，会把 Pythia-70M 在 Pile 上的 perplexity 从 **25** 拉高到 **40**，这说明字典丢掉了不少信息。第三，很多最漂亮的单 feature 例子都是 token-level 模式，比如 apostrophe、period、newline，这说明方法的“高层语义抽象能力”在当时还没有被充分证明。

因此，这篇论文的实验结论应该被拆成两个层级。稳的层级是：SAE 在小到中等规模 LM 的 residual stream 上，确实能学出比 neuron basis / PCA / ICA 更可解释、更细粒度的稀疏方向，并且这些方向在 IOI patching 中有更好的因果定位效果。更激进、仍需后续证明的层级是：同样方法能否在更大模型、更深层、更抽象行为和安全 trait 上稳定恢复 feature。后来的 Scaling Monosemanticity、Gemma Scope、Persona Features、AxBench 都是在追这个第二层问题。

从方法论上看，IOI 实验是全文最能支撑“SAE 不只是漂亮解释”的部分。因为 patching 不是让解释器猜标签，而是把某些 feature activation 真的替换进模型计算里，再观察输出分布是否朝 counterfactual target 移动。这个实验把 SAE 从相关性向因果性推进了一步。尽管 IOI 本身是一个相对干净的小任务，它仍然说明：如果字典分解真的更接近模型使用的计算单位，那么在干预时应该比 PCA 这种全局正交分解更省、更准、更局部。论文的结果正好支持这个判断。

## 5. Related Work & Future Work

这篇论文可以看成 SAE 研究线真正起势的起点。它和早期 superposition toy model、Sharkey 等人的中期报告直接相连，往后则几乎无缝衔接到 `Scaling Monosemanticity`、`GemmaScope`、各种 feature steering 工作，以及后来对 SAE 的批判性 benchmark，比如 `AxBench`。如果说 `Discovering Latent Knowledge` 和 `Representation Engineering` 更像是在问“有没有 truth / honesty / utility 这种高层表示可以读”，这篇论文则在更底层回答：**表示空间能不能先被拆成一套比较像 feature 的坐标系**。

未来方向在文中说得很坦白。首先，要减少 reconstruction loss，最好直接优化“替换重构后模型输出变化最小”，而不是只优化向量级重构误差。其次，要把方法更稳定地扩展到 MLP 和 attention layers。再次，自动可解释性流程本身也需要升级，否则 SAE 学到的高层 feature 可能被低估。最后，如果 SAE 真要成为安全接口，它就不能只在小模型和 token-level pattern 上好看，而要在更强模型、更抽象 feature、以及真正的 monitoring/steering 任务里站住脚。这也正是后来 `AxBench` 会拿它开刀的原因。

放在当前 topic 里，SAE 的角色不是直接解决 alignment，而是提供一种可能的 **feature coordinate system**。如果 emergent misalignment、refusal、sycophancy 或 deception 真对应某些可定位 feature，那么 SAE 或类似字典学习方法就可能让这些 feature 从 dense activation 里显出来；如果 SAE 只能找到漂亮但低层的 token feature，那么它对安全控制的帮助就会非常有限。这个二分正是后续读 persona feature 与 steering benchmark 时要一直追问的核心。

所以这篇笔记的落点应当非常清楚：SAE 是机制解释的基础设施，不是安全控制的最终答案。它给了我们从 dense activation 走向 feature-level analysis 的工具，也给后续 persona features、model diffing、feature suppression 提供了技术语言；但真正把它变成 alignment 工具，还需要证明这些 feature 能覆盖高层安全行为，能跨模型和数据稳定复现，并且能在干预时改变目标行为而不严重伤害正常能力。

这个边界必须记住，否则很容易把“可解释特征”误当成“可控安全机制”。
