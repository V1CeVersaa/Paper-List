---
title: "LVLMs-Saliency"
headline: "Hallucination Begins Where Saliency Drops"
description: "ICLR 2026 Oral note on LVLMs-Saliency, SGRS, and LocoRE for diagnosing and mitigating LVLM hallucination through gradient-aware output-token saliency."
status: complete
visibility: public
topic: "safety_alignment"
venue: "ICLR 2026 Oral"
year: 2026
paper_url: "https://arxiv.org/abs/2601.20279"
openreview_url: "https://openreview.net/forum?id=sjnErRHXf3"
source_pdf: "raw/papers/arxiv-2601.20279.pdf"
tags: ["hallucination", "multimodal_monitoring", "saliency"]
created: "2026-05-01"
updated: "2026-05-01"
---

> [!abstract] Contributions
>
> 这篇论文处理的是 **large vision-language model/LVLM hallucination/大视觉语言模型幻觉**，但切入点不是传统的 image attention sink，而是 **output-token saliency/输出 token 显著性**。作者提出 **LVLMs-Saliency**，把 attention weights 与对应 loss gradients 做 Hadamard product，用来衡量先前输出 token 对下一 token prediction 的真实影响强度。核心观察是：正确 token 生成时，近期输出 token 对当前预测有较高 saliency；幻觉 token 生成时，这种 saliency 会塌缩，说明 autoregressive generation 失去了局部上下文依赖。
>
> 基于这个观察，论文提出两个 inference-time interventions。**Saliency-Guided Rejection Sampling/SGRS** 在候选 token commit 前计算 saliency，拒绝低于自适应阈值的候选；**Local Coherence Reinforcement/LocoRE** 在 token accepted 后增强下一步 query 对近期输出 token 的 attention，从而抵消“忘记前文”的趋势。实验显示 SGRS+LocoRE 在 CHAIR、POPE、MME 和若干 VQA benchmark 上能降低 LVLM hallucination。边界也很明显：SGRS 需要 backward pass，部署成本高；低 saliency 与 hallucination 的关系在 rebuttal 后补了统计验证，但 causal claim 仍要谨慎；这篇更像 multimodal reliability / hallucination mitigation 论文，而不是狭义 LLM alignment 论文。

## 1. Introduction

LVLM hallucination 的常见解释是模型没有正确使用图像信息，或者被 attention sink、语言先验、解码策略带偏。很多训练-free 方法因此干预视觉 token attention、anchor token、EOS logit 或 contrastive decoding。作者认为这些方法抓住了一部分现象，但 attention map 本身只描述 forward pass 中的权重分配，不告诉我们一个 token 对最终 loss 或下一 token 预测到底有多大影响。attention 高不等于 causal influence 高，attention 图像看起来相似也不代表信息流相似。

论文的关键直觉是，LVLM 的幻觉不只来自图像 grounding 失败，也来自 **output stream 内部的 contextual memory failure**。在 autoregressive generation 中，模型每一步都依赖前面已经生成的文本。如果近期输出 token 对下一 token 的 saliency 下降，模型就更容易脱离前文，生成与已建立语境不一致的词。作者用可视化展示：attention maps 对正确 token 和 hallucinated token 看起来差异不明显，但 saliency maps 会出现明显分化。

这个视角有意思，因为它把 hallucination 从“有没有看图”扩展到“有没有持续使用自己刚刚说过的话”。例如 caption 里前面已经建立了 object relation，后续 token 如果不再受前文约束，就可能突然生成错误颜色、错误物体或错误状态。论文因此不直接去强化所有 visual attention，而是专门针对 output-token dependency 设计两个机制：一个在 token 进入序列前过滤，一个在 token 进入序列后强化局部连贯。

放在 Safety & Alignment 的会议列表里，这篇的相关性来自 **monitoring and reliability**，而不是 deception 或 RLHF。幻觉是安全问题的一部分，尤其在医疗、自动化决策和多模态 agent 中；但这篇主要还是 multimodal hallucination mechanism and mitigation，不是监督失灵或对齐目标设计。

## 2. Problem Setup

给定输入序列 $x\in\mathcal{V}^n$，模型 $\mathcal{M}$ 输出下一 token distribution $y$、各层各头 attention matrix $\mathbf{A}^{(l,h)}$ 和目标 token logits $s$。普通 attention analysis 只看 $\mathbf{A}^{(l,h)}$，但作者引入 loss $L(y,s)$ 对 attention 的梯度：

$$
\nabla \mathbf{A}^{(l,h)} = \frac{\partial L}{\partial \mathbf{A}^{(l,h)}}.
$$

然后定义每个 head 的 saliency matrix：

$$
\mathbf{S}^{(l,h)}
= \operatorname{tril}\left(\mathbf{A}^{(l,h)} \odot \nabla \mathbf{A}^{(l,h)}\right).
$$

这里 $\operatorname{tril}$ 保留 causal lower-triangular part，$\odot$ 是 element-wise product。直觉上，attention weight 表示信息路由强度，gradient 表示 loss 对这条路由的敏感度，二者相乘才更接近“这条 attention connection 对当前预测有多重要”。随后作者对 heads 求平均并做 $\ell_2$ normalization，得到 layer-wise saliency $\bar{\mathbf{S}}^{(l)}$。

论文关心的不是所有 token 的 saliency，而是 **previous output tokens to current token** 的 saliency。系统 token、图像 token、prompt token 都可以被分析，但作者的主要发现是：prompt saliency 对正确性有影响但不是主因，真正区分 hallucinated tokens 的是 prior output token saliency collapse。这个定义让方法专注于 autoregressive text stream 内部的局部一致性。

需要注意，这里的 hallucination labels 来自 POPE、CHAIR 等 benchmark 以及人工标注的 token-level analysis。论文不是从模型内部直接读出“幻觉原因”，而是先观察正确/幻觉 token 的 saliency 差异，再设计 intervention 去验证这种差异是否可用于缓解幻觉。

## 3. Methods

**SGRS** 是生成前的 gating。对于位置 $P$，模型基于上下文 $\mathbf{x}_{<P}$ 和图像 $\mathcal{I}$ 产生 logits，并从 top-$K$ distribution 中采样候选集合 $\mathcal{C}^{(P)}$。对每个候选 token $c_i$，作者计算它对 prior output positions 的平均 saliency：

$$
\mathcal{S}(c_i)
= \frac{1}{|\mathcal{L}_{\mathrm{target}}|\cdot|\mathcal{J}|}
\sum_{l\in\mathcal{L}_{\mathrm{target}}}\sum_{j\in\mathcal{J}}\bar{\mathbf{S}}^{(l)}_{P,j}.
$$

其中 $\mathcal{L}_{\mathrm{target}}$ 是中深层目标层集合，$\mathcal{J}=\{j\mid \mathrm{Sys}_L+\mathrm{Img}_L\le j<P\}$ 是已经生成的输出 token positions。候选必须超过自适应阈值才会被接受：

$$
\tau^{(P)}
= \alpha \cdot \frac{1}{|\mathcal{H}|}\sum_{j\in\mathcal{H}}\mathcal{S}(x_j),
\quad
\mathcal{H}=\{j\in\mathcal{J}\mid (P-1)-j\le W\}.
$$

$\alpha$ 控制严格程度，$W$ 是近期历史窗口。如果所有候选都被拒绝，算法选择 saliency 最高的候选作为 fallback。这个设计直接把论文的 pattern 操作化：低 saliency token 被认为更可能破坏上下文连贯，因此在 commit 前拦掉。

**LocoRE** 是生成后的 stabilization。它不计算梯度，也不改模型参数，而是在下一步 forward 中增强 query token 对近期输出 token 的 attention。设当前位置为 $P$，下一步预测 token $P+1$，先前输出位置集合为 $\mathcal{J}_P$。对每个 $j\in\mathcal{J}_P$，定义局部窗口增益：

$$
\gamma^{(P)}_j
= 1+\beta\cdot \mathbb{I}((P-j)\le w_s).
$$

然后把 attention matrix 中从 query $P+1$ 指向 key $j$ 的权重乘以 $\gamma_j^{(P)}$：

$$
\mathbf{A}^{(P+1)}_{P+1,\mathcal{J}_P}
\leftarrow
\mathbf{A}^{(P+1)}_{P+1,\mathcal{J}_P}\odot \gamma^{(P)}.
$$

这一步的作用是主动强化近期输出 token 对下一步预测的影响，避免模型在长输出中“忘记”刚刚生成的内容。SGRS 和 LocoRE 的组合因此是 closed loop：SGRS 确保进入序列的 token 有足够上下文 grounding，LocoRE 确保这些 token 在后续步骤里继续被使用。

从工程角度看，两者差别很大。SGRS 需要对候选 token 做 backward-pass saliency computation，因此 memory 和 latency 成本高；LocoRE 是 forward-only attention manipulation，成本低得多。论文 appendix 也承认 full SGRS+LocoRE 不是实时场景的最佳选择，而 LocoRE-only 是更轻的部署折中。

这个方法还有一个容易忽略的结构选择：作者没有把 saliency 聚合到图像 token，而是聚合到已生成输出 token。这样做会把问题从 **visual grounding** 转成 **local textual coherence**。如果幻觉来自模型完全没看对图像，强化前文可能只能让错误叙述更连贯；如果幻觉来自生成后半段逐渐脱离前文，LocoRE 就更可能有效。因此这套方法最适合描述性输出、长 caption 和连续陈述中的 drift，不一定同样适合单步 VQA、知识型错误或图像中根本不可见实体的猜测。

SGRS 的 rejection 机制也有一个内在风险。低 saliency token 不一定是错 token，可能只是一个语义转折、罕见物体、关系词或需要引入新信息的 token；高 saliency token 也不一定正确，可能只是模型强烈依赖前文错误。作者通过 adaptive threshold 和 fallback 减少这种风险，但真正部署时需要看 rejection rate、false rejection rate 和不同 token type 的分布。reviewer 要求这些统计是合理的，因为它们决定 SGRS 是可靠过滤器，还是一个会让生成变保守的启发式。

## 4. Experiments

实验覆盖 LLaVA-1.5-7B/13B、Qwen2-VL-7B、Qwen2.5-VL 系列和 Intern-VL-7B/13B。评估分三类：综合多模态能力 benchmark，如 LLaVAW、MM-Vet、MME；通用 VQA benchmark，如 VizWiz、ScienceQA；以及 hallucination benchmark，如 POPE 和 CHAIR。

主表以 LLaVA-1.5-7B 为 baseline。LocoRE 把 POPE F1/Acc 提到 86.9/87.3，把 CHAIR-S/CHAIR-I 降到 38.4/11.2；SGRS+LocoRE 进一步达到 POPE 87.0/87.5，CHAIR-S/CHAIR-I 35.6/8.2，MME total 668.33。和 VCD、OPERA、DOPRA、EAH、TAME、MemVR 等方法相比，它在 hallucination 指标上有竞争力，同时在 MME 的 existence、count、position、color 上也保持或改善。

跨模型结果大体支持 plug-and-play claim。LLaVA-1.5、Qwen2-VL、Qwen2.5-VL、Intern-VL 上，LocoRE 和 SGRS+LocoRE 通常降低 CHAIR hallucination，并在 POPE 或通用 VQA 上给出小到中等增益。最明显的效果出现在 LLaVA-1.5-7B 和 Intern-VL-7B 等 hallucination baseline 较高的模型上；对更强的 Qwen2.5-VL-32B，改进较小，说明方法收益和模型原始错误形态相关。

Ablation 显示 SGRS 是主要 hallucination reduction 来源，LocoRE 提供额外增益和轻量版本。论文报告 full method 在 LLaVA-1.5 上将 CHAIR hallucination rate 降低约 28.3%，在 Qwen2-VL 上降低约 22.8%。$\alpha$ 越大，SGRS 越严格，hallucination 可继续下降，但 latency 上升且可能拒绝正确但中等 saliency 的 token。论文推荐 $\alpha=0.6$ 作为折中。这里有一个小问题：正文说推荐 $\alpha=0.6,\beta=1.2$，但表格中合理 $\beta$ 是 0.15 或 0.20，$\beta=1.0$ 反而明显退化；这很可能是写作错误，读者不能照抄 1.2。

Reviewer 在初评中集中质疑 saliency-hallucination relation 的统计支撑。最终 arXiv 版本加入了 rebuttal 实验：约 12,000 个 token 的统计分析显示，正确 token 的平均 saliency 明显高于幻觉 token，例如 LLaVA-v1.5-7B 为 0.472 vs 0.193，Qwen2-VL-7B 为 0.664 vs 0.355，InternVL-7B 为 0.508 vs 0.224。按 saliency bin 看，最低区间的 hallucination rate 达到约 68%-76%，最高区间降到约 18%-28%。人工降低 correct tokens 的 saliency 也会让 CHAIR 变差、POPE F1/Acc 降低。这些补充显著增强了论文，但仍然不能把“全部幻觉都由 saliency drop 引起”说死。

实验最主要的弱点是成本和适用范围。SGRS 要 backward pass，很多生产推理栈不暴露或不允许这种操作；对于 32B、72B 级别 LVLM，memory pressure 会很大。论文声称 LocoRE-only 可低成本部署，但最强指标通常来自 SGRS+LocoRE。另一个弱点是 failure analysis 不足：它主要报告 aggregate benchmark，不够清楚方法在哪些 hallucination type 上失败，比如关系错误、知识错误、医学图像、稀有物体或噪声输入。

表 2 也提醒我们，方法效果不是无条件单调。对一些更强或更不同架构的模型，SGRS+LocoRE 在个别指标上的收益变小，甚至 CHAIR 指标不总是同时改善。这说明 output-token saliency collapse 是重要信号，但不是唯一幻觉机制。LVLM hallucination 至少还包括视觉编码不足、object detector-like bias、语言先验压倒图像证据、训练数据共现偏差、prompt ambiguity 和 decoding temperature 等因素。LocoRE 主要处理其中的局部上下文遗忘，不应该被包装成通用幻觉根治方法。

另外，论文中的 saliency intervention 还需要更严格的因果设计。人工 decay saliency 后指标变差，说明 saliency 与输出质量有因果相关性，但这并不自动证明自然发生的所有 hallucination 都是 saliency drop 导致的。更强的实验应该反过来做：在保持其他 attention pattern 尽量不变的前提下，只提升特定 prior-output saliency，看 hallucinated token 是否系统性减少；再按对象、属性、关系、计数、位置等错误类型分层。这样才能知道 LocoRE 是广泛机制，还是对某些 caption-style hallucination 特别有效。

还有一个读法上的细节：论文把 saliency collapse 称为 “contextual memory failure”，这句话有解释力，但也要避免过度脑补。这里的 memory 不是显式外部记忆模块，也不是模型真的把前文存丢了；它指的是在当前 token 预测中，先前输出 token 对 loss-sensitive attention pathway 的贡献变弱。这个表述更精确，也更能解释为什么 LocoRE 通过改 attention weights 就能部分缓解问题：它不是恢复一个消失的记忆，而是在下一步预测时提高近期输出 token 的有效影响。

## 5. Reviewer Discussion

OpenReview 最终决定是 **Accept (Oral)**。四位 reviewer 的分数是 4、6、8、6。正面评价集中在 novelty：用 attention-gradient saliency 而不是纯 attention 来解释和缓解 LVLM hallucination，是一个清楚且有潜力的方向；SGRS 和 LocoRE 都是 inference-time 方法，不需要重新训练；实验覆盖多个模型和多个 benchmark，指标上有一致改善。

批评同样集中而有效。最低分 reviewer 认为，论文最核心的 pattern 主要靠 case study 支撑，缺少 statistically significant validation；表 1 的改进幅度可能落在实验波动内；方法依赖多个 hyperparameters，实用性受限。其他 reviewer 也提出类似问题：低 saliency 与 hallucination 是 correlation 还是 causation；SGRS 的 backward pass 限制大模型部署；缺少 rejection frequency、false-rejection rate、layer/head sensitivity、failure mode breakdown 和完整内存成本分析。

作者在 rebuttal 后补了 token-level saliency distribution、saliency bin hallucination probability、人工 saliency decay intervention 和 latency discussion，这解释了为什么最终能 oral。我的客观评述是：这篇论文从初稿角度看确实有“机制 claim 大于证据”的风险，但最终版本把最关键的统计缺口补上了一部分。它现在最稳的表述应该是：**low output-token saliency is a strong diagnostic and useful intervention signal for LVLM hallucination**。至于“decisive causal origin”，还需要更多受控实验。

这篇对 safety reading 的启发在于，它展示了一种可解释 reliability signal：不是只看模型有没有答错，而是看生成过程中某类信息依赖是否塌缩。这个想法和 alignment monitoring 有邻接价值；但它目前主要服务于 multimodal hallucination mitigation，不应被强行解释成 deception、scheming 或 RLHF alignment。

我会把 reviewer 争议总结成一句话：大家认可 **attention-gradient saliency** 是新鲜且有用的观察窗口，但要求作者证明这个窗口看到的不是漂亮 case。最终版本补了统计分布和 intervention，这让论文从“有趣可视化”上升到“有实证支撑的诊断信号”。不过 reviewer 对成本、hyperparameter portability 和 failure analysis 的担忧仍然成立。尤其 SGRS 在大模型上昂贵，LocoRE 又没有 SGRS 那么强，这会让实际系统更可能采用 LocoRE-only 或近似 saliency，而不是照搬完整算法。

对读者来说，最重要的不是记住某个 CHAIR 数字，而是记住这条机制假设：**生成过程中的局部上下文依赖可以被显著性信号测量，并且这种依赖塌缩与幻觉风险相关**。如果未来做 multimodal agent monitoring，这个思想可以迁移到工具轨迹、视觉状态记忆和 action history 中：不是只问 agent 最终说了什么，而是问它当前 action 是否还被近期有效证据约束。

这也解释了它和前面两篇 monitoring 论文的差异。TRACE 和 MRT 关注的是模型是否利用监督或监控漏洞；这篇关注的是生成过程是否失去局部证据约束。二者都属于广义可靠性，但风险类型不同：前者更接近监督接口和对抗性利用，后者更接近生成过程中的证据约束与 hallucination 机制。

## 6. Related Work & Future Work

这篇与 OPERA、DOPRA、VCD、EAH、TAME、Farsight、MemVR 等 LVLM hallucination mitigation 方法相邻。区别在于，很多方法直接调视觉 attention、anchor token 或 decoding logits；LVLMs-Saliency 先用 gradient-aware signal 诊断 output-token dependency，再针对这个 dependency 做 filtering 和 attention reinforcement。

它也和 mechanistic interpretability 有弱连接。Saliency matrix 不是完整 circuit，不提供 feature-level semantic interpretation，但它把“attention map 看起来像不像”推进到“哪条 attention connection 对当前 loss 敏感”。这种 attention-gradient product 在工程上比训练 SAE 或 transcoder 轻，但解释深度也更浅。

后续最重要的方向是把因果性做硬。可以系统 boost 或 suppress prior-output saliency，观察不同 hallucination type 是否单调变化；也可以把 SGRS 的 rejection logs 做成 error analysis，看被拒 token 是真 hallucination、语义合理但低 saliency，还是低频正确 token。另一个方向是更低成本 saliency approximation，让大模型和生产推理栈不必每个候选都 backward。最后，方法需要在更真实的 multimodal agents 中测试，因为多轮视觉工具使用里的 hallucination 不只来自 caption generation，还来自状态记忆、工具返回、行动计划和外部环境不一致。
