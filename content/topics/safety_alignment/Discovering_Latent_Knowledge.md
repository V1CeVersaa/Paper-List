---
title: "Discovering Latent Knowledge"
headline: "Discovering Latent Knowledge in Language Models Without Supervision"
description: "论文笔记：用无监督的一致性约束从语言模型内部表征中恢复 truth-related direction，并把“模型知道什么”和“模型说什么”区分开。"
status: "complete"
visibility: "public"
topic: "safety_alignment"
venue: "ICLR 2023"
year: "2023"
paper_url: "https://arxiv.org/abs/2212.03827"
source_pdf: "raw/papers/2212.03827.pdf"
tags: ["latent_knowledge", "ccs", "truthfulness"]
related: ["Probing_Classifiers", "Representation_Engineering"]
created: "2026-04-18"
updated: "2026-05-09"
draft: "false"
---

> [!abstract] Contributions
>
> 这篇论文抓住了一个对 **alignment with truth** 很关键的问题：语言模型的训练目标往往会让它**内部表示了真相**，却不保证它**输出时真的说真话**。作者提出 **Contrast-Consistent Search/CCS**，只使用无标签的内部激活，不看模型输出、不用人工标注，而是利用“一个陈述和它的否定不能同时为真”这种逻辑一致性，在激活空间里搜索与 **truthfulness/真值** 对应的方向。核心做法是把每个 yes-no 问题都写成一对对照陈述，再训练一个线性 probe，使这对陈述的真值概率既**互补/consistent**，又**高置信/confident**。实验显示，在 6 个模型和 10 个数据集上，`CCS` 的平均准确率比强校准后的 zero-shot 还高约 **4 个点**，并且显著降低 prompt sensitivity。
>
> 这篇工作的边界同样很硬。它处理的是**二元判断**场景，本质上是“从固定 hidden state 里无监督挖一条 truth direction”，离更一般的 **ELK/Eliciting Latent Knowledge** 还差很远。更关键的是，论文在主实验里通过测试集取最大准确率来决定簇标签的正负号，这意味着它证明了“truth-related direction 可以被找到”，但还没有完全证明“在零外部监督下，部署者总能把这条方向稳定对齐到真实语义”。所以这篇论文更准确的定位是：它把“模型知道什么，不等于模型说什么”第一次做成了可操作的实验接口。
>
> 它把 `Probing_Classifiers` 的方法论警告推进到一个更安全相关的正例：不是训练一个普通监督 probe，而是用一致性约束无监督寻找 truth direction。后续 `Representation_Engineering` 会把这种 truth / honesty readout 扩展成更一般的 representation reading 与 control 程序，因此 CCS 是这条 honesty-monitoring 线的关键前置原型。

## 1. Introduction

这篇论文的起点非常直接，而且直到今天都没有过时：**语言模型对真相的内部表示，可能和它最终输出的文本发生系统性偏离**。原因并不神秘。模仿学习/Imitation Learning 会让模型复现人类文本里的常见误解；偏好优化或基于人类打分的训练，又可能把模型推向“更像真的”而不是“真的是真的”；如果监督者本身看不出错误，模型就更容易学会在表面上过关。于是，问题不再只是模型有没有知识，而是 **training objective 是否把 truth 和 output binding 牢固绑定在了一起**。

作者没有沿着常见路线继续做 prompt engineering 或更多人工校准，而是换了一个更硬的视角：如果模型内部本来就编码了某种与真值相关的表征，那么也许我们不该只盯着它的输出 token，而应该直接去读 **internal activations/内部激活**。这一步很关键，因为它把问题从“怎样让模型说对”转成了“怎样找到模型内部与真值相关的表示”。对安全研究来说，这个转向非常重要：一旦两者不再等价，我们就必须承认 **behavioral output 不是 latent belief 的可靠代理**。

论文真正有意思的地方，在于它没有把“latent knowledge/潜在知识”处理成神秘概念，而是抓住了一个非常朴素、却可优化的结构约束：**一个命题和它的否定，真值必须互补**。如果一个方向真的对应 truth，那么同一个问题的 “Yes” 版本和 “No” 版本在这个方向上的读数，就不该是任意的，而应该呈现稳定的一致性结构。作者正是把这个逻辑约束写成一个无监督目标函数，由此得到 `CCS`。

这也解释了为什么这篇论文应该放在 `safety_alignment` 而不是一般的 `representation_learning`。它关心的不是表征几何本身是否优美，也不是自监督目标如何提升迁移，而是 **truth-related representation 能不能成为监督失灵时的替代接口**。这条线后来会直接连到 honesty monitoring、representation engineering、faithful reasoning 和更广义的 ELK 问题。

## 2. Problem Setup

论文考虑一个预训练语言模型，以及一组有确定答案的 **yes-no questions** $q_1,\dots,q_n$。作者不允许自己访问外部标签，也不依赖模型是否把正确答案真的生成出来。可用信息只有一个特征提取器

$$
\phi(x) \in \mathbb{R}^d,
$$

其中 $x$ 是自然语言输入，而 $\phi(x)$ 可以是某一层的 hidden state、encoder representation 或 decoder representation。目标是：**只用这些无标签表示，把每个问题判成真或假**。

关键技巧是把每个问题 $q_i$ 改写成一对 **contrast pair/对照陈述**。记

$$
x_i^{+} = \text{“对 } q_i \text{ 的回答是 Yes”}, \qquad
x_i^{-} = \text{“对 } q_i \text{ 的回答是 No”}.
$$

如果问题本身是清晰可判定的，那么 $x_i^{+}$ 和 $x_i^{-}$ 的真值一定互斥，其中一个为真，另一个为假。于是，原始问答任务被改写成了一个更结构化的任务：**判断一对近乎相同、只在答案上不同的陈述，哪一个是真的**。

不过这里马上有一个 nuisance factor。$x_i^{+}$ 和 $x_i^{-}$ 的 hidden state 差异，不仅包含“真 vs 假”的差异，还包含 “Yes” token 和 “No” token 本身的表面差异。作者因此先做独立均值中心化，分别从正标签和负标签表示里减去各自均值：

$$
\tilde{\phi}(x_i^{+}) = \phi(x_i^{+}) - \mu^{+}, \qquad
\tilde{\phi}(x_i^{-}) = \phi(x_i^{-}) - \mu^{-}.
\tag{1}
$$

这一步的作用不是“让 truth 出现”，而是先尽量消掉“答案词面不同”这个最强的捷径，让后续 probe 更可能抓到与真值相关的方向。

接着，作者在归一化后的表示上训练一个极轻量的线性 probe：

$$
p_{\theta,b}(\tilde{\phi}(x)) = \sigma(\theta^\top \tilde{\phi}(x) + b),
\tag{2}
$$

其中 $\sigma$ 是 sigmoid，输出可以理解成“该陈述为真”的概率。整个问题于是被压缩成：**怎样在没有标签的情况下，找到让这个概率函数满足逻辑一致性的参数 $\theta,b$**。

## 3. Algorithm / Methods / Model

### 3.1 Contrast-Consistent Search

`CCS` 的核心思想很干净：如果一个方向真在编码 truth，那么对每个 contrast pair 来说，正反两个陈述的真值概率应该互为补数。于是作者定义 **consistency loss**：

$$
L_{\mathrm{consistency}}(\theta,b;q_i)
= \left[p_{\theta,b}(x_i^{+}) - \bigl(1-p_{\theta,b}(x_i^{-})\bigr)\right]^2.
\tag{3}
$$

这个约束单独看还不够，因为最容易满足它的退化解是两边都输出 $0.5$。作者因此再加一个 **confidence loss**，强迫模型不要把一切都判成“不确定”：

$$
L_{\mathrm{confidence}}(\theta,b;q_i)
= \min\{p_{\theta,b}(x_i^{+}),\, p_{\theta,b}(x_i^{-})\}^2.
\tag{4}
$$

直观上，这一项在压制“两个陈述都很像真的”或“两个陈述都像假的”这类模糊解。于是最终目标函数是

$$
L_{\mathrm{CCS}}(\theta,b)
= \frac{1}{n}\sum_{i=1}^{n}
\left(
L_{\mathrm{consistency}}(\theta,b;q_i)
+ L_{\mathrm{confidence}}(\theta,b;q_i)
\right).
\tag{5}
$$

> [!note] CCS 在优化什么
>
> 它不是在直接最大化准确率，因为根本没有标签；它优化的是一组**逻辑结构约束**。因此，`CCS` 能成功的前提不是“模型输出已经可靠”，而是“模型内部存在一条与真值对齐、并且能被线性读出的方向”。

在推理时，$p(x_i^{+})$ 和 $1-p(x_i^{-})$ 理论上都在估计“答案为 Yes 的概率”。由于训练只用了软约束，两者不一定严格相等，所以作者取平均：

$$
\tilde{p}(q_i)
= \frac{1}{2}\left(p(x_i^{+}) + 1 - p(x_i^{-})\right).
\tag{6}
$$

如果 $\tilde{p}(q_i) > 0.5$，就判作 “Yes”。这一结构非常重要，因为它说明 `CCS` 不是在判别某个单句子，而是在显式利用 **statement / negation** 成对结构。

### 3.2 What CCS Assumes About Truth

`CCS` 背后的假设其实比很多监督 probe 更强，也更有解释力。它假设 truth 不是一堆数据集特定模式的拼贴，而是在模型内部对应某种**跨任务可迁移的函数方向**。如果这个假设不成立，那么在情感分类、常识问答、NLI、故事续写这些任务之间，就很难共享一条近似统一的 truth direction。

更进一步，作者还在追问一个更尖锐的问题：`CCS` 找到的，究竟只是“更聪明的 output surrogate”，还是某种更早、更稳、更少受 prompt 扰动的内部状态。论文后面的分析非常依赖这一点，所以你可以把 `CCS` 看成两个层次上的方法。第一层，它是一个无监督 probe。第二层，它是在检验 **what the model knows** 和 **what the model says** 是否真的可分离。

> [!tip]
> 从安全角度看，这篇论文最值得记住的不是某个具体数字，而是这个框架：**当输出可能被 imitation、prefix、reward model 或 conversational framing 扭曲时，内部表示也许仍然保留着更稳定的 truth signal**。

### 3.3 Salience, Transfer, and Layer Choice

论文后半段其实在不断回答一个质疑：就算 `CCS` 有用，它会不会只是偶然在某个数据集上捡到了标签偏差。作者给出的三个反证非常关键。

第一，`CCS` 在**跨数据集迁移**时效果仍然不错。也就是说，用情感分类学到的方向去判断其他完全不同任务时，性能并不会立刻崩掉。这支持了“truth-like direction 比任务标签更抽象”的解释。

第二，`CCS` 并不总是在最后一层最好。对于 T5 和 UnifiedQA，中间层 hidden states 往往比紧邻输出的表示更好，尤其在误导性 prefix 存在时更明显。这说明作者抓到的方向不只是 decoder output head 之前的一层线性读出，而更像是一种**比最终 verbalization 更稳的中间信号**。

第三，truth direction 往往是 **salient feature/显著特征**。作者发现即使用很少数据，甚至有时只用一个 contrast pair，`CCS` 也能工作；而且简单的 PCA/clustering 变体也经常能找到接近的方向。这意味着论文想说的不只是“我提出了一个 clever objective”，而是“模型里真的存在一个相对容易被发现的 truth-related axis”。

不过 salience 也带来一个微妙问题。如果 truth direction 足够显著，`CCS` 的成功可能部分来自模型内部已经把“真/假”压成了一个很容易聚类的轴；这当然支持论文主张，但也说明方法的适用范围可能受限于任务格式。对于开放式问答、长链推理、偏好冲突或 agentic planning，模型内部未必只需要在两个互斥陈述之间选一个，truth signal 也可能分散在多个中间状态里。因此，`CCS` 更像是把 **latent knowledge readout** 的最干净版本先做出来，而不是一次性解决所有 ELK 场景。

> [!note] Claim Structure
>
> 论文的主张链条是：模型输出可能受 imitation、prompt framing 或偏好训练扭曲，但 hidden representation 中仍可能保留更稳定的 truth-related signal；CCS 用 statement/negation 的互补一致性和置信度约束，在无标签条件下搜索这条方向。实验结果支持 “truth direction often salient and transferable”，因为 CCS 在多个模型、多个 yes/no 数据集和跨数据迁移上强于校准后的 zero-shot baseline。问题在于它仍依赖二元陈述、均值中心化和事后确定方向符号，距离开放式 ELK、长链推理或部署期欺骗还很远。

## 4. Experiments

实验设置覆盖了 6 个模型和 10 个二元任务数据集，包括 IMDB、Amazon、AG-News、DBPedia、RTE、QNLI、COPA、Story-Cloze、BoolQ 和 PIQA。作者把每个数据集都转成 yes-no 或二元分类问题，并在多 prompt 条件下做平均。这个设置的优点是范围相当广，能够检验 `CCS` 到底是在拟合某类任务，还是在抓更抽象的结构。

主结果很清楚。跨模型平均来看，普通 zero-shot 准确率是 **62.8%**，做了 calibration 以后是 **67.2%**，而 `CCS` 达到 **71.2%**，如果把所有数据集一起训练则是 **71.5%**；监督式 logistic regression ceiling 则大约是 **83.7%**。更值得注意的是，`CCS` 的 prompt sensitivity 更低，说明它不仅更准，而且更稳。这点很重要，因为“内部 truth probe 比 verbalized answer 更不容易被 prompt 摇晃”正是论文的中心命题之一。

论文还专门构造了 **misleading prefix** 实验，刻意在 prompt 前面塞入错误示范，试图把模型输出往错误答案上带。UnifiedQA 的 calibrated zero-shot 准确率因此从 **80.4%** 掉到 **70.9%**，但 `CCS` 基本不受影响，反而从 **82.1%** 轻微升到 **83.8%**。这个结果非常关键，因为它直接说明：**prefix 足以改变模型说什么，却未必足以同等程度改变模型内部关于真假的表征**。

分析实验进一步加固了作者的解释。跨任务 transfer 结果表明，不同任务之间确实存在功能相近的 truth direction；中间层优于输出层的现象说明 `CCS` 并不只是复制 model outputs；数据量缩减和 PCA/clustering 实验又说明，这条方向并不埋得很深，而是模型里比较显著的结构。整体上，这些分析把论文从“一个还不错的 unsupervised probe”推到了“truth may be a salient internal feature”这个更有野心的结论上。

不过，这部分也有几个必须冷静指出的限制。第一，主文评测只覆盖**二元判别**，离开放式问答、长链 reasoning 和多选结构都还有距离。第二，论文在评估时通过“取两种簇标签映射里更高的那一个准确率”来解决 sign ambiguity，这让主结果并不是完全零监督可部署流程。第三，`T0` 的训练数据和评测数据有重叠，因此作者自己也把它从平均结果里拿掉了。第四，尽管论文把问题表述成 latent knowledge discovery，但它真正恢复的是**某种与真值高度相关的线性方向**，未必等于完整、可解释、可组合的“belief state”。

这几个限制并不削弱论文的重要性，反而让它在阅读图谱里的位置更准确。它不是一个完整的 safety monitor，而是一个 proof of concept：如果输出被 prompt 或模仿目标带偏，内部表示仍可能保留更稳定的真值结构。后续 honesty steering、representation engineering、faithful chain-of-thought 和 latent knowledge 工作要解决的，就是如何把这个二元 readout 原型推广到更开放、更长程、更具策略性的场景。

从实验解释上看，最值得警惕的是不要把“内部 truth signal 更稳”误读成“内部状态永远真实”。论文只能说明在这些二元任务和误导性前缀设置里，内部表示比最终输出更不容易被表面提示带偏；它没有证明模型在所有问题上都有一个统一、忠实、可读的信念变量。这个保守读法反而让论文更有价值：它给出了一个能被复现和扩展的最低可行接口，而不是过早宣称已经解决潜在知识引出问题。

## 5. Related Work & Future Work

这篇论文与几条线同时相连。和 **zero-shot prompting**、prompt calibration 相比，它的重点不是把输出调得更像答案，而是绕开输出，直接读内部状态；和常见的 **supervised probing** 相比，它最有价值的地方在于用逻辑一致性替代标签；而从时间线上看，它也很自然地成为后来 **honesty monitoring、representation engineering、faithful reasoning 与 ELK** 研究的重要前驱。

它与后来的 `ActAdd`、`Representation Engineering` 有一个很漂亮的分工。`CCS` 主要回答“能不能在无监督条件下读出 truth-related direction”，`ActAdd` 主要回答“能不能直接改 activation 来 steer behavior”，而 `Representation Engineering` 则试图把 reading 与 control 汇总成一个更一般的 top-down transparency 程序。放回 `safety_alignment` 的语境里，`CCS` 像是这条主线最早的一个支点：它第一次比较系统地把 **truth representation** 从输出行为里剥离出来。

未来工作其实很明确。最直接的一步，是在**不借助测试集标签定向簇符号**的前提下完成端到端无监督部署。再往前一步，是把 `CCS` 从二元任务推广到更开放的问答、长文本生成和 agent setting，看看“latent knowledge”到底只是一个局部 truth classifier，还是能上升为更广义的内部信念接口。只要这两步还没完成，我们就不该把这篇论文读成“ELK 已经解决”；但把它读成 **ELK 的第一个可复现实验原型**，这个判断是完全站得住的。
