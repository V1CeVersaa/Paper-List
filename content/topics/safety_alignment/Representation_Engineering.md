---
title: "Representation Engineering"
headline: "Representation Engineering: A Top-Down Approach to AI Transparency"
description: "论文笔记：把 representations 而非 neurons/circuits 作为透明性研究的主对象，提出 LAT、Contrast Vector 与 LoRRA 等 reading/control 基线，并在 honesty、harmlessness、power、bias 等任务上做验证。"
status: "complete"
visibility: "public"
topic: "safety_alignment"
venue: "arXiv 2023"
year: "2023"
paper_url: "https://arxiv.org/abs/2310.01405"
source_pdf: "raw/papers/2310.01405.pdf"
tags: ["representation_engineering", "lat", "honesty_control"]
related: ["Discovering_Latent_Knowledge", "ActAdd", "Universal_Steering_Monitoring"]
created: "2026-04-18"
updated: "2026-05-09"
draft: "false"
---

> [!abstract] Contributions
>
> 这篇论文想推动的是一条比 mechanistic interpretability 更“上层”的透明性路线。作者把它命名为 **Representation Engineering/RepE**，核心主张是：如果我们真正关心的是 **honesty、harmfulness、power-seeking、bias、memorization** 这类高层认知现象，那么研究单位不该只盯着单个神经元或局部 circuit，而应该直接研究 **population-level representation/群体表征**。围绕这个立场，论文提出了两类基础工具。第一类是 **Representation Reading**，其中最核心的基线是 **Linear Artificial Tomography/LAT**：通过设计 stimulus 和 task template、收集层内激活、再对配对差分做 PCA，提取 concept/function 的 **reading vector**。第二类是 **Representation Control**，作者系统比较了 reading vector、contrast vector 和 **Low-Rank Representation Adaptation/LoRRA** 三种控制器，以及线性加法、piece-wise 放大和 projection 三类操作。随后，论文把这套接口投到 honesty、utility、power、emotion、harmlessness、bias、knowledge editing 和 memorization 等任务上，展示了 reading 与 control 的广泛可用性。
>
> 这篇论文的分量主要来自**方法论汇总和安全广度**，而不是某个单一 benchmark 的极限分数。它最亮眼的具体结果集中在 honesty：`LAT` 从模型内部读 truthfulness，在 TruthfulQA 上显著优于标准 zero-shot 评估；进一步用 contrast vector 或 LoRRA 做 honesty control，又能把 LLaMA-2-Chat-13B 的 TruthfulQA MC1 从 **35.9** 提高到 **54.0** 或 **47.5**。不过它的边界也非常明显。论文是一个高度 exploratory 的 programmatic work，任务极多、指标异质、很多实验依赖小规模或合成刺激集，证据类型以 correlation 与 manipulation 为主，离“高层认知已经被稳定、统一地测量和控制”还差很远。因此，这篇论文更适合被读成：**top-down transparency 的研究纲领 + 一组 surprisingly strong baselines**，而不是一套已经闭环的安全工程方案。
>
> 它在阅读链里承接 `Discovering_Latent_Knowledge` 和 `ActAdd`：前者说明 truth-related direction 可以从内部状态中被读出，后者说明 activation direction 可以被直接拿来 steering；RepE 把 reading 和 control 合并成一个更一般的安全透明性程序。后续 `Universal_Steering_Monitoring` 基本是在这个程序上继续自动化和规模化，把 concept direction 估计从手工刺激和 PCA 推向 RFM/AGOP 管线。

## 1. Introduction

这篇论文的野心并不是再做一个新的 probe，而是试图重写“透明性研究该以什么为中心”这个问题。作者认为，现有很多 interpretability 工作默认采用一种偏 bottom-up 的思路，也就是把神经元/neuron、注意力头/head、局部电路/circuit 当成主要解释单位。这条路在解释较简单机制时当然有用，但一旦问题上升到 **truthfulness、honesty、utility、power、emotion** 这类更抽象的对象，仅靠局部组件往往很难组织出清晰的分析框架。

于是论文借用了认知神经科学里更偏 **Hopfieldian** 的视角：认知现象未必最自然地存在于单个神经元层面，而可能更自然地表现为**群体活动构成的表示空间结构**。把这个想法移植到大模型以后，作者提出：AI transparency 不一定要先把每条 circuit 全部拆清楚，才有资格碰高层安全问题；相反，我们也可以先在 **representation space** 上直接定义、读出、干预这些高层变量。

这一步非常重要，因为它把“透明性”从纯解释任务推向了**监测与控制接口**。如果 honesty、harmlessness 或 power-seeking 真能在激活空间里形成相对稳定的方向或子空间，那么我们就不只能做 post hoc 解释，还能把这些方向拿来做 lie detection、jailbreak defense、行为 steering，甚至做训练时的低秩控制。也正因为如此，这篇论文天然属于 `safety_alignment`：它虽然大量讨论表示，但最终落点始终是 **safety-relevant cognition 是否能被直接测量和操控**。

从仓库已有主线看，这篇论文也站在一个很清楚的位置上。它往前承接 `Discovering Latent Knowledge in Language Models Without Supervision` 这类 truth representation 读出工作，往旁边连接 `Steering Language Models With Activation Engineering` 这类 activation steering 接口，往后则会自然通向更系统的 persona vectors、universal monitoring、refusal direction 和 emergent misalignment 表示机制。换句话说，RepE 不是孤立 paper，而是把 reading 和 control 明确绑成一条研究程序的关键节点。

## 2. Problem Setup

论文把 **Representation Engineering** 分成两个主问题。第一是 **Representation Reading**：给定一个模型和一类高层概念或功能，能否从内部激活中定位出与它稳定相关的方向。第二是 **Representation Control**：一旦找到了这种方向，能否通过修改表示，把模型行为往目标方向推。

对 concept 类对象，例如 truthfulness、utility、probability、emotion，作者先设计一个 task template $T_c$ 来显式唤起模型对概念 $c$ 的判断，再从模型在该模板下的激活中抽取表示。若 $s_i$ 是 stimulus，$Rep(M,\cdot)$ 返回某层所有 token 的表示，则一个默认的概念激活集合写成

$$
A_c = \{Rep(M, T_c(s_i))[-1] \mid s_i \in S\}.
\tag{1}
$$

这里默认取最后一个 token 的表示，因为在 decoder 模型里，这个位置往往最接近模型对“当前问题要输出什么判断”的内部状态。对 function 类对象，例如 honesty、instruction-following、power-seeking，作者不再只看一个静态概念，而是让模型在 **experimental prompt** 与 **reference prompt** 下完成生成，再沿着回复 token 序列收集激活：

$$
A_f^{\pm}
= \left\{
Rep\!\left(M, T_f^{\pm}(q_i, a_i^k)\right)[-1]
\;\middle|\;
(q_i,a_i)\in S,\; 0<k\le |a_i|
\right\}.
\tag{2}
$$

这样做的意义在于，honesty 这类对象本来就不是一个静态标签，而更像一种**生成过程中的程序性倾向/procedural tendency**。论文因此明确区分 declarative concept 和 procedural function。

找到激活以后，作者用一个很朴素但 surprisingly effective 的线性模型来构造 **reading vector**。对 concepts，若能形成只在目标概念上不同的配对刺激，就对配对差分做 PCA；对 functions，则对实验态和参考态的激活差分做 PCA。记第一主成分为 $v$，则一个最基本的读取分数就是

$$
score(x) = Rep(M,x)^\top v.
\tag{3}
$$

这就是 `LAT` 的核心。它不要求标签监督，也不要求知道电路细节，而是通过**任务模板 + 成对差分 + 线性方向提取**，把高层现象压成一个可读向量。

## 3. Algorithm / Methods / Model

### 3.1 LAT: Reading Concepts and Functions

`LAT` 的流程可以压成三步。第一步是 **Designing Stimulus and Task**。这一步几乎决定成败，因为你必须把想要的 concept 或 function 真正唤起来。对 concept，模板往往是“Consider the amount of <concept> in the following...”；对 function，则是同一条 instruction 下切换 experimental/reference prompt，迫使模型在“执行该功能”和“不执行该功能”之间切换。

第二步是 **Collecting Neural Activity**。作者强调 token position 和 layer 的选择并不是细节。对 concepts，概念词所在位置和最后一个 token 往往都可能有信号；对 functions，则需要沿着回复 token 序列持续扫描，因为功能性倾向会在生成过程中逐步展开。论文多数实验默认取最后一个 token 或 response token 的层内表示，但它也明确承认，对更复杂概念，激活收集策略可能必须更精细。

第三步是 **Constructing a Linear Model**。作者主要用 PCA，对差分向量集合求第一主成分，把它当作 reading vector。写得更具体一点，如果 concept 的成对激活是 $A_c^{(i)}$ 和 $A_c^{(j)}$，function 的实验/参考激活是 $A_f^{+}$ 和 $A_f^{-}$，那么 PCA 的输入大致是

$$
\{A_c^{(i)} - A_c^{(j)}\}
\quad \text{或} \quad
\{(-1)^i (A_f^{+} - A_f^{-})\}.
\tag{4}
$$

> [!note] LAT 的真正主张
>
> `LAT` 不是在说 PCA 神奇，而是在说：如果刺激模板设计得足够好，那么**高层概念本身就会在表示空间里留下线性、成对、可分的痕迹**。PCA 只是把这个痕迹捞出来的最便宜方法。

更有意思的是，论文没有把“读到一个方向”直接当作终点，而是提出了四级证据框架：**correlation、manipulation、termination、recovery**。这比常见 probe 工作更严谨，因为它明确承认：相关性本身不等于因果，真正有说服力的是你能不能顺着这个方向把模型行为推走、拿掉以后有没有损失、再放回去能不能恢复。

### 3.2 Representation Control

在控制侧，作者先区分三种 controller。第一种是 **reading vector** 本身，优点是便宜，缺点是 stimulus-independent，对所有输入施加同方向扰动。第二种是 **contrast vector**，即在同一个输入上跑 experimental/reference 两个 prompt，把两种表示之差当作控制向量。这通常更强，因为它是 input-conditional，但代价是推理时要额外多次前向。第三种是 **LoRRA**，也就是 Low-Rank Representation Adaptation：不再在线注入向量，而是在训练时学一个低秩 adapter，让模型表示逼近由 contrast vector 指定的 target representation，最后把 adapter 并进模型，推理几乎不增成本。

若把当前表示记为 $R$，控制元素记为 $v$，论文主要讨论三类操作。最基础的是 **linear combination**：

$$
R' = R \pm v.
\tag{5}
$$

这对应最直接的刺激或抑制。第二类是 **piece-wise operation**：

$$
R' = R + \mathrm{sign}(R^\top v) v,
\tag{6}
$$

它的目标不是简单把模型全局推向某个方向，而是**条件性地放大当前已存在的相关激活**。第三类是 **projection**：

$$
R' = R - \frac{R^\top v}{\|v\|^2} v.
\tag{7}
$$

这相当于显式拿掉与该方向对齐的分量，用来测试某个概念是否必要。

`LoRRA` 则把控制进一步做成低秩训练。它在若干目标层收集 base representation $r_l$，再构造 target representation

$$
r_l^{t} = R(M,l,x_i) + \alpha v_l^{c} + \beta v_l^{r},
\tag{8}
$$

其中 $v_l^{c}$ 是 contrast vector，$v_l^{r}$ 是可选 reading vector。训练目标是让带 LoRA adapter 的模型表示逼近这个 target。直觉上，LoRRA 把“一次性向量注入”变成了“把目标表征写入一个可复用的低秩更新”。

### 3.3 Why This Is Not Just Another Probe Paper

这篇论文和普通 probing 工作的一个根本区别在于，它一直坚持 reading 和 control 的闭环。一个方向如果只能提高分类准确率，却对 manipulation、termination 或 recovery 几乎没有作用，那么它更可能只是 correlational marker，而不是模型真的在用的表征。论文在 utility 例子里就明确展示了这一点：logistic regression 在 correlation 上最好，但行为操纵能力并不一定最强；反而某些更朴素的方向在多种实验设置下更稳。

这也是 RepE 相对 mechanistic interpretability 最鲜明的姿态。它不试图先把底层计算图还原完整，而是先问：**有没有一个高层表示对象，能同时承担解释、监测和控制的职责**。从安全工程角度看，这个问题其实更务实，因为部署场景里你往往来不及等一个完整电路图。

这里也能看到 RepE 的风险：如果只追求高层接口，它很容易把不同机制压成同一根向量。比如 honesty direction 可能混入“回答风格更谨慎”“更像 TruthfulQA 正例”“拒绝不确定问题”等多个因素；power-seeking 或 immorality direction 也可能同时捕捉文本语气、角色设定和真实目标倾向。因此，RepE 的正确读法不是“线性方向已经解释了高层认知”，而是“高层认知可以先被表示级接口捕捉，再用更严格的因果实验继续拆解”。

> [!note] Claim Structure
>
> 论文的主张链条是：许多 safety-relevant cognition 不适合只从 neuron/circuit 层面自底向上解释，而应该在 population-level representation 中直接读取和控制；LAT、contrast vector 与 LoRRA 分别给出 reading、online control 和 trainable adaptation 三类接口。最强证据来自 honesty reading/control、jailbreak harmlessness recovery、MACHIAVELLI power/immorality 调节，以及 bias、memorization、knowledge editing 的广覆盖案例。它的影响是把 `Discovering_Latent_Knowledge` 的 truth direction 和 `ActAdd` 的 activation steering 组织成统一研究程序；问题是任务模板、小规模刺激集和异质指标太多，很多方向仍需要更严格的因果拆解与 benchmark 化验证。

## 4. Experiments

论文的实验部分不是一个统一 benchmark，而是一系列“reading → manipulation”的案例研究。其中最核心的是 **honesty**。作者先在标准 QA benchmark 上用 `LAT` 读取 truthfulness，发现它经常比 few-shot prompting 更准；再到 TruthfulQA 上比较标准评估、启发式 truthfulness prompting 和 `LAT`。以 LLaMA-2-Chat-13B 为例，标准评估只有 **35.9**，启发式能到 **50.3**，而 `LAT` 用不同 stimulus set 时可到 **53.1-54.2**。这个差距的含义非常尖锐：**模型内部关于 truth 的概念，往往比它在生成目标下实际说出来的答案更可靠**。

更进一步，作者把 honesty reading vector 拿去做 lie detection 和 honesty control。对 TruthfulQA MC1，13B 模型在无控制时是 **35.9**；用 ActAdd 风格向量加法后到 **38.8**；直接加 reading vector 到 **42.4**；用 stimulus-dependent contrast vector 则到 **54.0**；LoRRA 也有 **47.5**。这组结果几乎把 RepE 的 reading-control 闭环压成了一个最有说服力的例子：先从表示里读出 honesty，再反向把它写回去，行为就真的会变。

论文在其他方向上也给了不少有价值的证据。对 harmlessness，作者先读出 harmfulness direction，再用 piece-wise operator 提升模型对有害指令的内部敏感度；在 adversarial suffix 攻击下，Vicuna 基线的 harmless rate 很低，而控制后能大幅回升。对 utility、morality 和 power，论文展示了 reading vector 在相关性、操纵和“去除后性能下降”上的不同表现，并把 LoRRA 用到 MACHIAVELLI 这类交互环境里调节 power-seeking 与 immorality。对 bias，作者声称从 race-only 刺激中抽出的方向还能泛化到 gender/profession bias，暗示模型里可能存在更统一的 bias representation。对 knowledge editing 和 memorization，论文进一步展示了 representation-level interventions 可以修改具体事实、放大或抑制某个非数值概念，或者降低对热门文本的复述倾向。

实验的真正亮点不是“每个任务都做到了最好”，而是它们共同支持了一个更强的命题：**很多安全相关行为确实能在表示空间中找到低维接口**。这比单纯的 probe 更进一步，因为作者持续在做 counterfactual manipulation。

但如果要冷一点看，问题也不少。首先，论文极度依赖**任务模板设计**，而模板本身可能就是最强 inductive bias 来源。其次，许多实验使用的 stimulus set 很小，甚至包含模型自生成数据；这对探索性工作没问题，但对稳健性与泛化的结论必须保守。再次，任务之间指标不统一，breadth 很强，depth 却不总是均匀。最后，作者虽然提出了 correlation / manipulation / termination / recovery 的评价框架，真正系统做全四类证据的案例仍然有限，很多 section 依然主要停留在“读得到，而且加上去会变”的层面。

这恰恰是 RepE 对后续工作的价值所在。它给出了一个很强的研究纲领，也暴露了纲领必须补上的证据缺口：读到方向以后，要证明这个方向不是 surface cue；操纵行为以后，要证明不是破坏模型分布造成的副作用；跨任务泛化以后，还要证明不是评估模板共享造成的虚假迁移。后面的 Universal Steering、AxBench、refusal direction 和 persona vectors，本质上都在不同角度补这些洞。

## 5. Related Work & Future Work

这篇论文最适合放在三条线的交叉点上读。第一条线是 **truth / honesty reading**，前面有 `Discovering Latent Knowledge in Language Models Without Supervision` 这种用逻辑一致性找 truth direction 的工作；RepE 在 honesty 上显然继承了这一脉络，但把范围从 truth 扩到更广的 safety-related cognition。第二条线是 **activation steering**，最直接的邻居就是 `Steering Language Models With Activation Engineering`。ActAdd 证明“加一个方向能改行为”，RepE 则进一步系统化了 controller 类型、操作类型和评估框架。第三条线是 **mechanistic interpretability**。RepE 并不否认电路分析的重要性，但它明确主张：如果目标是高层安全接口，那么 representation-level 研究本身就值得作为一条独立路线。

从仓库主线看，RepE 的价值恰好在于它把后面很多工作提前组织好了。后来不管是 refusal direction、persona vectors、universal steering and monitoring，还是 emergent misalignment 的 feature-level 解释，本质上都在重复一个信念：**高层安全属性可以在 activation space 中被低维近似、读取和控制**。RepE 是最早把这个信念写成系统研究程序的论文之一。

未来工作也很清楚。首先，需要把 reading/control 从“靠任务模板和少量刺激做 demo”推进到更标准化、跨模型的 benchmark。其次，需要更严格地区分“检测到了相关词汇或情境联想”和“真的检测到了目标认知状态”，特别是在 honesty、immorality、harmfulness 这类很容易被表面 cue 污染的方向上。最后，如果 RepE 真想成为安全接口，而不只是研究 demo，那么 termination 和 recovery 证据必须更扎实，跨架构迁移也必须更系统。只有这样，**representation engineering** 才能从一套很有启发性的实验策略，变成真正可靠的 transparency methodology。
