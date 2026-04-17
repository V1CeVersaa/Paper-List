---
title: "ActAdd"
headline: "Steering Language Models With Activation Engineering"
description: "论文笔记：用 contrast pair 的激活差直接构造 steering vector，在不训练模型的情况下做 inference-time activation steering。"
status: "complete"
visibility: "public"
topic: "safety_alignment"
venue: "arXiv 2023"
year: "2023"
paper_url: "https://arxiv.org/abs/2308.10248"
source_pdf: "raw/papers/steering-language-models-with-activation-engineering.pdf"
tags: ["activation_steering", "actadd", "linear_representations"]
related: []
created: "2026-04-17"
updated: "2026-04-17"
draft: "false"
---

> [!abstract] Contributions
>
> 这篇论文想解决的是 **冻结大语言模型后，能不能不用再训练，就在推理时直接改它当前的行为倾向**。作者把这条路线命名为 **激活工程/activation engineering**，并提出最核心的方法 **Activation Addition/ActAdd**：先选一对自然语言对照 prompt，例如“Love”与“Hate”或“我总在谈论婚礼”与“我不谈论婚礼”，再取它们在某一层 **残差流/residual stream** 上的激活差，把这个差向量按系数放大后加到用户 prompt 的前向传播里。这样做完全不需要标签、不需要反向传播，也不改模型权重。论文在 GPT-2-XL 上展示了对 **情感、话题、风格乃至局部事实倾向** 的控制，并给出三类定量证据：与目标话题更相关的文本上困惑度下降、最佳注入层可把话题 steering 成功率从约 **2% 提到 90%**、以及 ConceptNet 上的离题常识性能几乎不受影响；同时，额外计算开销随模型规模并不会恶化。
>
> 这篇论文的边界也非常鲜明。首先，它最硬的量化实验几乎都围绕 **GPT-2-XL + wedding vector** 展开，Llama-13B 与 GPT-J-6B 更多只是定性复现。其次，方法仍然依赖两个超参数：**注入层/layer** 和 **注入强度/injection coefficient**，论文只给了不完整的网格搜索。更关键的是，作者虽然把结果解释成对 **线性表征/linear representation** 的因果证据，但论文并没有真正解释清楚：为什么一对 prompt 的激活差会稳定对应到一个可以跨上下文复用的语义方向。

## 1. Introduction

这篇论文的出发点很直接：**我们已经很会训练 LLM，却还不够会在运行时精细控制它。** 传统做法要么是 prompt engineering，要么是 supervised finetuning、RLHF、guided decoding。这些方法当然有用，但它们都有明显代价。prompt 受上下文窗口限制，而且控制强度是离散的；finetuning 与 RLHF 则需要训练算力、数据和实现成本，还会把你对原模型权重的机制理解一起打乱。作者想追问的，是不是存在一条更轻的路径，直接修改 **当前前向传播中的内部状态**，而不是重新塑造整个模型。

因此，论文把焦点放到 **残差流/residual stream** 上。残差流可以理解为 Transformer 每一层持续读写的中间表示缓存，后续注意力头和前馈层都在这个空间里继续计算。作者的核心判断是：如果“愤怒”“婚礼”“阴谋论”“更有害”这类高层属性，真的在模型内部对应某种相对稳定的方向，那么我们就不一定非要通过再训练来激活它，只需要在合适层把这个方向加进去就够了。这个判断在今天看已经非常熟悉了，但在 2023 年它其实很激进，因为它把 **行为控制** 直接改写成了 **激活空间中的向量操作**。

更重要的是，这篇论文不只是发明了一个便宜技巧。它真正打开的是一个后面会不断被放大的问题：**模型当前“正在扮演什么角色、朝什么目标走”这件事，究竟更像权重里的长期能力，还是更像激活里的短时目标配置。** 作者明显押注后者，所以才会强调 activation engineering 比 finetuning 更像一种 **online steering**。这也是它在 `safety_alignment` 里有位置的原因：它还没有解决安全问题，但它把安全控制接口第一次非常明确地钉在了激活空间上。

## 2. Problem Setup

论文处理的是标准的 **decoder-only Transformer**。对输入 prompt $x$，模型在第 $\ell$ 层之前的残差流记作 $A_\ell(x) \in \mathbb{R}^{T \times d}$，其中 $T$ 是序列长度，$d$ 是残差维度。作者不改动任何权重，只在某一层的残差流上做加法干预。

给定一对对照 prompt $c^{+}$ 与 $c^{-}$，其中 $c^{+}$ 强调想要放大的属性，$c^{-}$ 提供反向或中和的参照，作者先把较短的 prompt 用空白字符右填充到相同 token 长度，然后记录它们在第 $\ell$ 层前的激活。由此得到 **steering tensor**：

$$
\Delta_\ell = \alpha \bigl(A_\ell(c^{+}) - A_\ell(c^{-})\bigr),
\tag{1}
$$

这里 $\alpha$ 是 **注入系数/injection coefficient**，控制 steering 强度。随后，在用户 prompt $x$ 的前向传播到第 $\ell$ 层之前，把这个张量加到残差流上，再让模型继续生成。结合 Algorithm 1 与 Appendix B，这个过程可以理解为对用户输入前缀位置做 **front activation addition**，也就是从第一个 token 的残差流开始注入，而不是插到最后一个位置。

> [!note] Why A Contrast Pair Matters
> 论文反复强调，**单独使用正向 prompt 不如使用对照差分**。这点很关键，因为它说明 ActAdd 不是简单把某个词的 embedding 强行塞进上下文。比如 “I love talking about weddings” 和 “I hate talking about weddings” 都含有 wedding 相关 token，但它们的差分仍然能把生成显著推向婚礼话题。这意味着真正起作用的并不只是词面共现，而是更高层的属性差异。

这个设定的目标有两层。表面上，它要实现 **inference-time steering**：在不训练的前提下，把模型输出推向某种情感、话题或风格。更深一层，它其实是在测试一个更强的假说：**模型是否真的把某些高层特征编码成了可加可减的线性方向。** 如果答案是否定的，那么这种简单的差向量加法大概率只会制造噪声，而不会得到成体系的行为改变。

## 3. Algorithm / Methods / Model

### 3.1 Method Overview

ActAdd 的方法主线极其干净，几乎可以压缩成“三次前向传播”。先对 $c^{+}$ 跑一次前向，拿到目标层激活；再对 $c^{-}$ 跑一次，做差并乘上系数；最后对用户 prompt 跑前向，并在指定层把这个差张量加进去。整个过程只有前向传播，没有优化、没有 probe、没有 classifier，也没有任何额外训练信号。和后来的很多 steering 方法相比，它最大的特点不是“更强”，而是 **极简**。

作者把这条路线叫作 **activation engineering**，并有意把它和 prompt engineering、finetuning、weight editing 区分开来。它既不是改 prompt 文本，也不是改参数，而是把模型中途“推”到另一个激活区域里继续计算。由于这个推力来自一对自然语言 prompt，方法还拥有一个很重要的用户界面优势：**用户可以直接用自然语言指定想要的方向，而不是先去构造标注数据或训练控制器。**

### 3.2 Why It Is Not Merely Prompting

论文很敏锐地意识到，一个最自然的质疑是：ActAdd 会不会只是某种伪装的 prompt injection。作者专门在附录里检验了这个假说。对于 wedding steering，他们比较了两种干预：一种是正常的 ActAdd，另一种是直接把 `' weddings'` 这个 token 前置到输入里。结果很有意思。对婚礼相关文本，两者都能稍微改善 perplexity；但对无关文本，**prompting 的 perplexity ratio 是 1.132，明显坏于 ActAdd 的 0.994**。这说明 ActAdd 至少不完全等价于“在上下文里多塞了一个词”。

作者还做了另一个更硬的检验：如果愤怒 steering 只是 embedding 在作祟，那么把早期的 embedding 信息直接搬到后面层，也该得到同样强的效果。但实验发现，这样做的作用很弱，远不如正常的 Anger-Calm 激活差。于是论文给出一个重要判断：**真正起作用的，不只是 token embedding，而是前面若干 Transformer block 已经计算出来的中间表征。**

> [!tip]
> 从机制上看，ActAdd 最值得记住的一点不是“向量加法很神奇”，而是 **它在要求模型自己完成剩下的解释工作**。你并没有直接指定输出句子，而只是把残差流往某个方向推了一下；后面的层会把这个方向重新整合进注意力与下一个 token 的分布里。

### 3.3 Hyperparameters and Implementation Choices

方法虽然简单，但并不是完全免调参。作者承认有两个关键自由度：**注入层 $\ell$** 和 **注入强度 $\alpha$**。经验上，中间层更有效，系数通常落在 3 到 15 的量级，但不同属性并不完全一致。比如较具体的 love vector 更适合较早层，而较抽象的 conspiracy vector 则更适合较晚层。论文没有给出完整统计，只说这些值可通过简单网格搜索获得，而且同一冻结模型里往往可以复用。

还有一些实现细节非常影响效果。首先，contrast pair 可以不同长，但最好把较短 prompt 用 whitespace 右填充；如果一定要用“中性” prompt，重复空白比 end-of-text token 更稳。其次，steering vector 通常比用户 prompt 更短，所以必须决定把它加在什么位置。作者发现越靠后的位置往往 steering 越强，但如果直接改最后一个 residual stream position，语法很容易崩。因此主文统一使用 **front addition**。这些细节说明：ActAdd 远不是“任意差向量都能工作”的万能按键，它依然高度依赖对残差流接口的工程把握。

### 3.4 What The Authors Think This Means

论文最终想支持的，不只是一个工程 recipe，而是 **线性表征假说/linear representation hypothesis**。作者认为，像 love-hate、anger-calm、weddingness 这样的属性，在固定层上可能对应相对稳定的方向；把这个方向加进残差流，后续层就会据此调整生成。附录中的几组实验都是围绕这个解释展开的。随机向量即便匹配相似范数，也很难产生同样强的定向效果；只加部分维度时，steering 强度还会随维度比例平滑变化；而且某些 steering vector 的范数甚至能接近乃至超过原始残差流的一大部分，这说明它不是微小扰动，而是相当强的重新定向。

但这里也要保持克制。论文提供的是 **因果迹象**，不是完整理论。它确实说明“沿某个差向量去推，行为会变”，却还没有证明这个方向就是唯一、最小或最稳定的语义轴。更进一步，为什么一个比较型 prompt pair 的差分会比直接指定目标输出更有效，论文也没有真正解释。这些都被作者留给了后续工作。

<!-- TODO: insert Figure 1 from paper: ActAdd pipeline from contrast pair to activation difference to steered generation -->

## 4. Experiments

实验部分一开始先用大量定性样例证明，这个方法的作用范围不止一个话题词。主文与附录里的例子覆盖了 **婚礼话题、愤怒情绪、阴谋论风格、局部事实倾向**，甚至还有较弱的 harmfulness steering。比如在 GPT-2-XL 上，婚礼向量可以把一个普通对话前缀推向不停讨论 wedding；“Bush did 9/11 because” 之类的对照对可以把后续文本推向阴谋论口吻；而 “The Eiffel Tower is in Rome” 与 “The Eiffel Tower is in France” 这样的 prompt pair，则展示了一种很粗糙的事实编辑效果。这里最重要的信息其实不是单个 demo 是否惊艳，而是 **同一接口确实可以触碰到多种高层属性**。

更硬的量化实验围绕 wedding vector 展开。作者先在 OpenWebText 上抽取 30 万篇文档，并按 wedding 相关词的频率分桶。结果显示，随着文本本身与婚礼话题更相关，ActAdd 相对未修改模型的 **perplexity ratio 会持续改善**；换句话说，steering 并不是盲目把所有文本都推歪，而是更像在“婚礼语境里更愿意说婚礼相关的话”。紧接着，作者又看 token 级别的 log-probability 变化，发现正向尾部最明显被抬高的，主要正是 wedding 相关 token。这一步非常关键，因为它说明方法不是通过某种无关方式“碰巧降低了 perplexity”，而是确实在定向改动目标词族的概率。

再往后，论文直接测生成层面的 steering 成功率。对一个固定 prompt，作者在 48 个层上逐层扫描注入位置，并用 200 个 completion 统计是否出现 wedding 相关词。结果显示，steering 在很早的层就已经可见，在中间层最强，最佳注入点可把相关 completion 的比例从约 **2% 提升到 90%**。这个结果很醒目，因为它说明 ActAdd 并不需要精确命中极少数神经元，只要在合适深度推对方向，整个后续计算就会顺着这个方向继续展开。

更有价值的是，作者没有只展示“能 steer”，还检验了 **副作用**。他们用 ConceptNet/LAMA 常识填空基准比较加婚礼向量前后的 top-$k$ 正确答案概率，发现离题常识性能几乎不变。这一点不能被夸大成“完全无害”，但至少支持一个更谨慎的结论：**ActAdd 的主要作用不是全面破坏模型，而是比较局部地重分配输出倾向。**

论文还专门讨论了可扩展性。由于额外成本主要来自对 contrast pair 的少量前向传播，而不是训练或逐 token 的复杂优化，所以随着模型变大，推理时的相对开销并不会同步恶化。图 6 给出的结果是，在 GPT 系列里，这个 inference premium 甚至随参数规模增大而下降；在 OPT 系列里，也大致保持平坦。附录里的 GPT-J-6B 重复了婚礼向量在 token 分布与 perplexity 上的主要动态，Llama-13B 也能复现部分定性样例，不过像 Eiffel/Rome、anger、harm 这些例子出现了明显失败。

如果要对实验部分冷一点看，问题也很明显。**第一，主量化几乎被 wedding vector 一统天下。** 这当然足以证明“这种接口存在”，但不足以证明它对广泛安全属性同样稳定。**第二，更强模型上的复现还不够系统。** GPT-J 有一些定量图，Llama-13B 基本还是 qualitative table，而失败样例本身已经说明鲁棒性远没有主文最亮眼的 demo 那么整齐。**第三，超参数报告不完整。** 对一个需要调层和调系数的方法来说，这会让复现实验边界显得有些模糊。最后，论文明确承认 GPT-2-XL 太弱，无法认真讨论 reasoning steering；因此它在今天更应该被看成 **activation steering 的开山接口论文**，而不是成熟的通用控制方案。

<!-- TODO: insert Figure 2 from paper: perplexity ratio improves as target topic becomes more relevant -->
<!-- TODO: insert Figure 6 from paper: inference-time premium stays flat or decreases with model size -->

## 5. Related Work & Future Work

作者把这篇论文放在三条已有脉络之间。第一条是 **latent space arithmetic**，也就是图像生成和词向量里早就出现过的“方向操作语义”现象；ActAdd 延续了这个直觉，但把对象从 embedding 或 latent variable 换成了 **Transformer 的中间激活**。第二条是更常见的 **prompt engineering / finetuning / decoding control**。论文的态度很明确：这些方法都能控模型，但代价、粒度和在线性都不一样，而 activation engineering 的独特之处在于它把控制点放到了运行中的残差流。第三条是更接近的 **activation intervention** 工作，例如需要训练 steering vector 或依赖标注 truthfulness 数据的方案。ActAdd 相比它们最突出的差别，就是 **零优化、零标签、只靠一对自然语言 contrast prompts**。

未来工作方面，作者点了几条很值得追的线。首先是 **更高层、更安全敏感的 steering vector**，例如某种更普适的 “be helpful” 方向，或者故意诱发泄露机密的方向，从而把这类风险提前测出来。其次是继续扩展到 **更大模型**，看看这种 activation-level 控制到底会变得更稳，还是更不可预测。再次，作者对 **非自然语言 contrast pair** 也持开放态度，说明他们并不认为自然语言一定是唯一接口。最后，也是最根本的一点，论文承认必须解释清楚：**为什么把两次前向传播的中间结果相减再相加，会如此稳定地改写模型行为。**

如果把这篇论文放回今天的语境里，更准确的评价不是“它已经解决 steering”，而是 **它把 steering 变成了一个可以被系统研究的问题**。从这个意义上说，ActAdd 的真正价值不只是效果，而是它把一个后来会不断扩张的研究接口第一次压到了足够简单的形式。
