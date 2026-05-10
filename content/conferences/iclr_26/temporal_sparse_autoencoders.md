---
title: "Temporal SAEs"
headline: "Temporal Sparse Autoencoders: Leveraging the Sequential Nature of Language for Interpretability"
description: "ICLR 2026 Oral note on Temporal Sparse Autoencoders, a temporally regularized SAE objective for separating smoother semantic features from local syntactic features."
status: complete
visibility: public
topic: "safety_alignment"
venue: "ICLR 2026 Oral"
year: 2026
paper_url: "https://arxiv.org/abs/2511.05541"
openreview_url: "https://openreview.net/forum?id=bojVI4l9Kn"
source_pdf: "raw/papers/arxiv-2511.05541.pdf"
tags: ["sparse_autoencoders", "mechanistic_interpretability", "activation_steering"]
created: "2026-05-01"
updated: "2026-05-01"
---

> [!abstract] Contributions
>
> 这篇论文提出 **Temporal Sparse Autoencoder/T-SAE/时间稀疏自编码器**，核心判断是普通 SAE 把每个 token activation 当作独立样本训练，因此更容易学到局部、噪声化、偏语法的 features，而很难学到跨 token 保持稳定的语义或上下文 features。作者把 SAE feature space 切成 high-level 和 low-level 两部分，在 Matryoshka-style reconstruction loss 外，对 high-level features 加入 temporal contrastive loss，使同一序列中相邻 token 的 high-level activation 更接近，同时让不同序列的 high-level activation 可分离。
>
> 这篇 oral 的价值在于它把 SAE 的一个常见失败模式说得很清楚：很多 SAE feature 不是“不够可解释”，而是**可解释但粒度太低**，只能解释某个词、标点、格式或短程语法，不能提供 sequence-level 的语义把握。论文实验显示，T-SAE 在 Pythia-160M 和 Gemma-2-2B 上保持接近普通 SAE 的 reconstruction/SAEBench 指标，同时更好地区分 MMLU 语义类别、上下文来源和词性，并在 HH-RLHF 数据分析与 steering case study 中显示出安全相关用途。边界也很明确：核心实验规模偏小，temporal smoothness assumption 对突变文本和多主题上下文不总成立，且方法仍依赖 LLM judge 的自动解释和评估。

## 1. Introduction

SAE 在 mechanistic interpretability 中很有吸引力，因为它把 dense residual stream activation 分解成稀疏 features，让研究者可以给 feature 命名、统计、干预或 steering。问题在于，很多 SAE features 虽然能被解释，却经常落在很局部的表面模式上，例如句首的特定词、句末标点、代码符号、格式边界。这类 feature 对 reconstruction 有用，也可能在 token-level 很清楚，但如果研究者想审计模型是否在表达有害内容、是否处在某种任务语义状态、是否可以被语义 steering，它们就显得太碎。

论文的关键直觉是，语言不是一堆独立 token。**Syntax/语法** 往往在短程局部变化，例如下一个词的词性、括号是否闭合、句号位置；**semantics/语义** 和 **context/上下文** 往往在一段文字内部持续存在，例如“植物生物学讨论”“宗教文本”“数学题解”。普通 SAE 的训练目标没有告诉模型这种时间结构，只要求每个 activation 被稀疏重构，于是更容易恢复对重构最直接的 token-local signals。

T-SAE 因此不是重新发明 SAE，而是在 SAE 的训练目标里加入一个自然语言先验：高层语义 features 应该在相邻 token 间相对平滑，低层语法 features 可以快速变化。这个先验并不需要人工语义标签，所以它仍是 self-supervised；它只利用同一序列中 token 的相邻关系，把“来自同一上下文的 high-level activations 应该更像”写入 objective。

这对 alignment 和 safety 的意义在于，许多安全审计关心的对象本来就是 sequence-level concepts。比如 harmful compliance、sexual content、crime instruction、refusal style、length bias、polite-but-unhelpful response，这些不是某一个 token 的属性，而是跨多个 token 的语义状态。如果 SAE 只给出局部 syntactic features，研究者很难用它做数据审计或稳定 steering；如果 SAE 能产生更平滑的 semantic features，它就更接近一个可用的安全分析工具。

## 2. Problem Setup

论文把语言生成写成一个简化的 latent-variable process。设一段 token 序列为 $\tau_1,\ldots,\tau_T$，第 $t$ 个 token 由历史上下文、high-level latent variables 和 low-level latent variables 共同决定：

$$
\tau_t=\phi(\tau^{t-1},\mathbf{h}_t,\mathbf{l}_t).
$$

这里 $\mathbf{h}_t$ 表示语义、意图、主题、上下文等高层信息，$\mathbf{l}_t$ 表示词性、局部语法、具体词形等低层信息。LLM 在第 $L$ 层产生 activation $\mathbf{x}_t^L\in\mathbb{R}^d$，论文省略层标记后写成 $\mathbf{x}_t$，并假设模型通过某个可逆映射 $g$ 编码这些潜变量：

$$
g(\mathbf{h}_t,\mathbf{l}_t)=\mathbf{x}_t.
$$

目标是从 $\mathbf{x}_t$ 中恢复对应 $\mathbf{h}_t$ 和 $\mathbf{l}_t$ 的可解释 features。论文的两个假设很重要。第一个是 **Temporal Consistency**：同一序列中的 high-level latent 在时间上近似不变，也就是 $\mathbf{h}_t\approx\mathbf{h}_{t'}$。这不是说一整篇文章永远只有一个主题，而是说相邻 token 或短窗口内的语义状态通常比随机跨序列 token 更接近。

第二个假设是 **Hierarchical Representation**：只用 high-level part 也能粗略重构 activation，而 low-level part 负责 high-level 无法解释的 residual。形式上，作者写成 high-level reconstruction 的误差不超过 $\epsilon$，full reconstruction 再用 low-level residual 补齐。这对应 Matryoshka SAE 的思想：先让粗粒度 features 承担主要结构，再让细粒度 features 解释剩余局部差异。

这个 setup 的强处在于它把 SAE feature learning 和语言结构联系起来；弱处在于它把语义 smoothness 当作比较普遍的先验。真实文本中主题切换、讽刺、代码混合自然语言、多轮对话中的角色变化都会让 $\mathbf{h}_t$ 突然变化。T-SAE 的 temporal prior 因此更适合“语义在局部窗口内连续”的场景，不适合把所有序列都当成单一平稳主题。

## 3. Algorithm / Methods / Model

T-SAE 从普通 SAE 出发。给定 activation $\mathbf{x}_t$，encoder 产生稀疏 feature activation：

$$
\mathbf{f}(\mathbf{x}_t)=\sigma(\mathbf{W}^{\mathrm{enc}}\mathbf{x}_t+\mathbf{b}^{\mathrm{enc}}),
$$

decoder 用 features 重构原 activation：

$$
\hat{\mathbf{x}}(\mathbf{f})=\mathbf{W}^{\mathrm{dec}}\mathbf{f}(\mathbf{x}_t)+\mathbf{b}^{\mathrm{dec}}.
$$

作者把 $m$ 个 SAE features 切成两段，前 $h$ 个是 high-level features，后 $m-h$ 个是 low-level features。训练时使用 Matryoshka-style loss：high-level split 先尝试重构 $\mathbf{x}_t$，full feature set 再重构完整 activation。这样 high-level features 不只是附属解释，而是被迫承载足够多的全局信息；low-level features 则学习 high-level 重构剩下的 residual。

真正的新东西是 temporal contrastive loss。设 $\mathbf{z}_t=\mathbf{f}_{0:h}(\mathbf{x}_t)$ 是 high-level feature activation，$s(\cdot,\cdot)$ 是 cosine similarity。对一个 batch 中的配对 activation $(\mathbf{x}_t^{(i)},\mathbf{x}_{t-1}^{(i)})$，作者要求同一序列相邻 token 的 $\mathbf{z}_t$ 与 $\mathbf{z}_{t-1}$ 更相似，同时把不同样本的 high-level activation 作为 negatives：

$$
\mathcal{L}=\sum_{i=1}^N\mathcal{L}_{\mathrm{matr}}(\mathbf{x}_t^{(i)})+\alpha\mathcal{L}_{\mathrm{contr}}.
$$

这里 $\mathcal{L}_{\mathrm{contr}}$ 是双向 contrastive loss：一项从 $\mathbf{z}_t^{(i)}$ 找对应的 $\mathbf{z}_{t-1}^{(i)}$，另一项从 $\mathbf{z}_{t-1}^{(j)}$ 找对应的 $\mathbf{z}_t^{(j)}$。这个设计比单纯最小化 $\|\mathbf{z}_t-\mathbf{z}_{t-1}\|$ 更重要，因为单纯 smoothness 会鼓励所有 high-level features 塌缩成常数；contrastive negatives 迫使不同序列仍然可分，避免“所有东西都很平滑但没有语义区分度”的失败。

Low-level features 不加 temporal constraint。它们只通过 residual reconstruction 学习，因此自然倾向于吸收 high-level features 没解释掉的局部变化。这个设计给出一个很清楚的解释：T-SAE 不是让所有 feature 都平滑，而是让一部分 feature 表示慢变量，另一部分 feature 表示快变量。对后续使用者来说，这种 split 很有价值，因为你可以在安全审计中优先看 high-level semantic features，在语法、代码或格式研究中再看 low-level features。

论文还测试了不同 temporal pairing 方式。默认使用相邻 token $t-1$；ablation 中也随机采样过去 25 个 token 内的某个位置，得到更长程的 context signal。这带来一个实用启发：T-SAE 的“语义粒度”不是固定的，它会受 contrastive pair 的时间跨度影响。短跨度更保留 syntax，长跨度更偏 context；如果任务是安全监测，可能更愿意牺牲一点局部语法，换取更稳定的高层语义 features。

## 4. Experiments

实验使用 Pythia-160M 和 Gemma-2-2B，比较 BatchTopK SAE、Matryoshka SAE 和 T-SAE。训练数据是 The Pile，probe evaluation 用 MMLU、Wikipedia 和 FineWeb。作者选择 Pythia layer 8 和 Gemma layer 12，主要为了和 Neuronpedia 上已有 SAE 资源可比。这个选择合理但也形成限制：实验没有覆盖更大模型的系统训练，只在 rebuttal 中补了 Llama-3.1-8B-Instruct 的额外结果。

第一组实验看语义、上下文和语法 probe。T-SAE high-level features 在 MMLU 题目类别、Wikipedia 主题、FineWeb 上下文来源等任务中更能预测语义和上下文；low-level features 则更能预测 POS/词性。更关键的是，Matryoshka SAE 虽然也有 high/low split，但任务表现主要来自 high-level split，low-level split 不明显承担独立语法角色。这说明 T-SAE 的 contrastive constraint 真的推动了功能分工，而不只是把 feature 编号切成两段。

第二组实验看标准 SAE 指标。Pythia 上 T-SAE 的 FVE 为 0.94，Matryoshka 和 BatchTopK 为 0.95；Gemma 上 T-SAE FVE 为 0.75，Matryoshka 0.75，BatchTopK 0.76。Cosine similarity、fraction alive 和 autointerp score 也都接近。这个结果支撑论文最重要的工程 claim：加入 temporal contrastive loss 没有明显牺牲重构质量。也就是说，T-SAE 的收益不是拿 reconstruction 换 semantic coherence，而是在接近同等 reconstruction 下改变 feature 的组织方式。

第三组实验看 temporal consistency。作者把多段文本拼接在一起，例如 MMLU 生物题、达尔文信件、Animal Farm 文章、数学题，然后观察 top active features 随 token 的变化。T-SAE features 在段落边界有清楚 phase transition，在段内相对平滑，而且自动解释标签和真实文本语义相符；Matryoshka SAE 的 top features 则更频繁跨 token 抖动，难以形成 sequence-level view。这个可视化是论文最有说服力的部分，因为它直接展示了“普通 SAE 可解释但太局部”的问题。

第四组是 safety case study。作者用 T-SAE 分析 Anthropic HH-RLHF 数据集中 chosen/rejected response 的 feature activation 差异。T-SAE 找到了和 rejected response 相关的安全语义 features，例如身体接触、犯罪、暴力、社交规范等；同时也发现了 spurious correlations，例如 rejected response 往往更长、更正式、包含引用或在线资源。这来自数据中的一个结构：当两个回答都不好时，labeler 常常偏好不相关但无害的短回答，而拒绝更长、更服从但有害的回答。T-SAE 把这种 length/compliance confound 显出来，这对 preference data audit 很有价值。

第五组是 steering case study。作者用 30 个 features 测试 steering success 和 coherence，并用 Llama-3.3-70B 评分、人工验证。结果显示 high-level T-SAE features 在 steering 上 Pareto dominate baseline SAEs：更能改变生成语义，同时更少产生 token repetition 或不连贯输出。这个结果和方法直觉一致：如果一个 feature 本来就在整段生成中保持稳定，那么在每个 token 上加 steering vector 更接近训练分布；如果 feature 只是局部 token signal，强 steering 容易把模型推向重复词或格式异常。

Ablation 支持 temporal contrastive loss 的必要性。去掉 contrastive term 后，semantic 和 context probe 明显下降；改变 high/low split 会在语义、上下文、语法之间移动性能；把 contrastive pair 改成随机历史 token 会提升 context、降低 syntax。这说明 T-SAE 不是一个单点 trick，而是一组可以调节的 inductive biases。真正使用时，研究者要先想清楚自己想要的 feature 时间尺度。

还有一个实验解释值得单独拎出来：T-SAE 的优势主要体现在**聚合解释**而不是单个 feature 的神秘新能力。普通 SAE 的某些 token-level features 也可能很准，但当你把几百个 token、几千条 preference samples 或一段长回答放在一起看时，它们会变成高频噪声。T-SAE 通过让 high-level features 在序列内持续激活，使同一个 semantic state 能跨 token 积累统计量。这正是 HH-RLHF case study 能看到长度、顺从性和有害语义 confound 的原因。换句话说，T-SAE 改善的是从 **feature inspection** 到 **dataset auditing** 的可用性。

不过，论文对自动解释的依赖仍然需要谨慎。Autointerp score 由 Llama-3.3-70B-Instruct 生成和判分，这类指标容易受解释模板、激活样本选择、judge 偏好影响。一个 feature 被解释为“crime and malicious activities”，并不等于它只编码犯罪语义；它也可能同时编码长回答、教程口吻、特定数据源或拒答前的上下文。实际做安全审计时，T-SAE feature 最好被当作候选变量：先用它定位数据模式，再用反事实样本、人工标注和 intervention 检查因果性。

## 5. Reviewer Discussion

OpenReview 最终决定是 **Accept (Oral)**。四位 reviewer 的初始分数是 10、6、6、4，作者 rebuttal 后，三个非 10 分 reviewer 都表示愿意提高分数，最终讨论口径大致变成 10、8、8、6。AC 的 meta-review 明确认为 temporal characteristics integrated into SAE 是对 SAE 的重要推进，实验和复现细节也受到认可。

正面评价集中在三个方面。首先，reviewer 普遍认为论文抓住了 SAE 的真实痛点：很多 feature 过于 token-local，导致 downstream monitoring、dataset understanding 和 steering 中可用性差。其次，方法本身简单，和 Matryoshka SAE 兼容，没有引入复杂模型结构。最后，case studies 比单纯 probe 更有说服力，尤其是 HH-RLHF 数据审计和 steering，直接回应了“语义 feature 到底有什么用”这个问题。

批评也很具体。最早的低分 reviewer 质疑实验模型太小，Pythia-160M 和 Gemma-2-2B 未必能代表更大的 instruction-tuned LLM；另一个 reviewer 认为 related work 没有充分讨论已有 temporal SAE 或 feature absorption work；还有 reviewer 指出 contrastive loss 的形式解释不够清楚、smoothness metric 可能被 outliers 支配、high/low split 的 20/80 比例偏经验化。作者在 rebuttal 中补了 Llama-3.1-8B-Instruct 结果、feature absorption 讨论、更多 smoothness metrics、baseline comparison 和 steering experiments，这些补充让评审整体转正。

我的客观评述是：这篇论文的贡献不是“证明 T-SAE 已经解决 SAE interpretability”，而是**把 SAE 的时间结构缺失变成一个可训练、可评估、可消融的目标**。这个贡献很实在，因为它没有空泛地说“需要更语义的 features”，而是给出 temporal contrastive prior，并用 probe、smoothness、数据审计、steering 四条证据链支撑。

不过 reviewer 对规模和 metric 的担忧很合理。T-SAE 的 smoothness 不是天然等于 semantic faithfulness；一个 feature 很平滑，也可能只是捕捉 prompt template、文档来源、长度、topic mixture 或数据集 artifact。HH-RLHF case study 恰好说明这一点：T-SAE 能发现 spurious correlations，但它也会把这些 correlations 作为 high-level features 学出来。因此使用 T-SAE 做 safety audit 时，不能把 feature explanation 直接当成人类偏好或风险因果变量，只能当作需要进一步验证的语义相关信号。

另一个需要冷静看的点是 high/low split 的解释权。论文把 high-level features 解释为 semantic/context，low-level features 解释为 syntax/residual，但这个命名来自训练先验和 probe 行为，不是来自数学保证。20/80 split、contrastive pair 选择、batch negatives 的构成都会影响两组 features 的含义。最稳妥的读法是：T-SAE 创造了一个更偏慢变量的 feature 子空间，以及一个更偏快变量的 residual 子空间；至于“慢变量等于语义”在每个任务里是否成立，需要单独验证。

我会把这篇定位成 **SAE utility paper**，而不是纯 mechanistic theory paper。它最强的地方是把 SAE 从 token-level microscope 推向 sequence-level audit tool；最弱的地方是它还没有给出 feature causal validity 的强保证。对 safety alignment 来说，它提供的是更好用的 dictionary learning substrate，后续仍需要因果干预、数据反事实和人工审计来确认这些 features 真的对应安全机制。

## 6. Related Work & Future Work

这篇论文和 standard SAE、BatchTopK SAE、Matryoshka SAE、feature absorption、transcoder、SAEBench、activation steering 和 preference-data interpretability 放在同一条线上。它和 *What's In My Human Feedback?* 的关系尤其近：后者用 SAE 分析 preference data 中的可解释偏好相关 features，T-SAE 则试图提供更平滑、更高层的 features，让这种数据审计更不容易被 token-local noise 淹没。

未来最重要的方向是扩大模型和任务规模。T-SAE 需要在更大的 instruction-tuned 模型、reasoning models、多轮对话、代码和 agent trajectories 上验证。尤其是 safety 场景中，高层语义状态可能跨越工具调用和多轮上下文，而不是单段文本；temporal contrastive pair 也许要从相邻 token 扩展到 sentence、turn 或 trajectory segment。

第二个方向是更强的 causal evaluation。现在 T-SAE 证明了 features 更平滑、更能预测语义类别、更适合 steering，但这还不等于 feature 是模型自然机制中的因果变量。后续可以系统比较 T-SAE feature ablation、activation patching、causal scrubbing、counterfactual prompt edits，看看 high-level features 是否真的控制 safety-relevant behavior，而不仅仅是预测它。

第三个方向是把 temporal prior 做成可调接口。不同任务需要不同时间尺度：toxicity monitoring 可能希望 feature 在整段回答中稳定；代码括号或数学符号分析可能希望保留局部变化；agent 状态监控可能需要跨 action step 的慢变量。T-SAE 的 random previous-token ablation 已经暗示这个方向，下一步可以设计 multi-timescale SAEs，让 features 自己分布到多个 temporal bands。

最后还需要更系统地评估失败样本。哪些语义转折、混合语言、引用文本、代码块或对话轮次会让 high-level features 错误延续？这些失败模式决定 T-SAE 能否进入真实安全审计流程。
