---
title: "CRV"
headline: "Verifying Chain-of-Thought Reasoning via Its Computational Graph"
description: "ICLR 2026 Oral note on CRV, a white-box CoT verification method based on attribution graph fingerprints."
status: complete
visibility: public
topic: "safety_alignment"
venue: "ICLR 2026 Oral"
year: 2026
paper_url: "https://arxiv.org/abs/2510.09312"
openreview_url: "https://openreview.net/forum?id=CxiNICq0Rr"
source_pdf: "raw/papers/arxiv-2510.09312.pdf"
tags: ["chain_of_thought", "mechanistic_interpretability", "reasoning_verification"]
created: "2026-05-01"
updated: "2026-05-01"
---

> [!abstract] Contributions
>
> **CRV/Circuit-based Reasoning Verification** 把 CoT verification 从“看最终答案或隐藏状态是否像错了”推进到“检查生成某个 reasoning step 的计算图结构是否像错了”。论文先用 per-layer transcoders 替换 LLM 的 MLP 模块，使中间计算经过稀疏可解释 feature；再为每个 CoT step 构建 attribution graph，把 tokens、transcoder features 和 output logits 之间的高影响路径连成一张有向加权图；最后抽取 global stats、node influence stats 和 topological features，用 gradient boosting classifier 判断这个 step 是否正确。
>
> 实验显示，CRV 在 Boolean、Arithmetic、GSM8K 三类 step-level verification 上都超过 black-box logit baselines 和 gray-box hidden-state baselines，尤其在 Synthetic Arithmetic 上 AUROC 达到 92.47，远高于最强 baseline 的 76.45。论文更有意思的贡献是分析层面：错误的 attribution graph fingerprint 高度 domain-specific，node influence/activation features 比 topology 更关键，并且一个 case study 显示抑制单个 multiplication transcoder feature 可以纠正错误推理。边界也很硬：方法计算昂贵、需要替换 MLP 并训练 transcoders，baseline probe 的弱表现被 reviewer 质疑，causal intervention 目前主要还是少量 case study。

## 1. Introduction

CoT verification 的目标是判断一段 chain-of-thought 中每个 reasoning step 是否正确。现有方法大致分两类。Black-box 方法看最终文本、答案概率、perplexity、entropy 或 final logits；gray-box 方法看隐藏状态轨迹、activation probe 或 hidden-state dynamics。这些方法可以发现“模型现在可能错了”，但通常解释不了“为什么这个计算过程走错了”。对于需要 debugging 的 model developer 来说，这个解释缺口很大。

CRV 的基本直觉来自 mechanistic interpretability：如果 transformer 在执行某种 latent algorithm，那么正确推理和错误推理应该不只是输出不同，也会在内部计算路径上留下不同结构。论文把 attribution graph 当成近似的 **execution trace/执行轨迹**。在普通软件里，debugger 会看程序执行路径；在 LLM 中，CRV 试图看 token、feature、logit 之间的高影响信息流，判断这条计算路径是否具有错误 fingerprint。

这个视角对 alignment and monitoring 有意义，因为它绕开了纯文本 CoT 的一个问题：CoT 文本可能只是表面解释，不一定忠实呈现模型内部计算。CRV 不直接相信文本内容，而是追踪产生该 step 的计算图结构。和 TRACE 通过截断 CoT effort profile 检测 reward hacking 不同，CRV 处理的是 reasoning correctness 本身：一个 step 是否在模型内部走了像“正确计算”的路径。

论文也很克制地承认，它不是一个即插即用 verifier。CRV 需要 full model access，需要为每层训练 transcoder，需要构建 attribution graph，成本比普通 probe 或 reward model 高得多。因此它更像科学仪器：用来研究 reasoning failure 的内部结构，探索 white-box verification 和 targeted intervention 的可能性，而不是马上部署到每个 inference request 上。

## 2. Problem Setup

设 LLM 生成一段 CoT：

$$
S=(s_1,s_2,\ldots,s_m),
$$

其中 $s_i$ 是第 $i$ 个 reasoning step。对每个 step，模型内部有一个计算状态 $M_i$。CRV 从这个状态构建 attribution graph：

$$
G_i=(V,E),
$$

其中 $V$ 包含输入 tokens、active transcoder features 和 output logits，$E$ 是高 attribution 的有向边，表示组件之间的 causal influence。然后 CRV 用一个 feature extractor $\phi$ 把图 $G_i$ 转成固定长度结构指纹：

$$
x_i=\phi(G_i).
$$

最终训练一个 classifier：

$$
\hat{y}_i=f_\theta(x_i),
$$

预测 step $s_i$ 是 correct 还是 incorrect。注意这里的 label 是 step-level correctness，而不是整段 CoT 的最终答案对错。这个粒度很关键，因为很多 CoT 最终错，是由中间某一步的局部错误传播而来；也有一些后续步骤在错误上下文里局部计算正确，但全局路径已经损坏。

为了得到可靠 step labels，作者自己生成和标注数据，而不是直接使用 PRM800K 或 REVEAL 这类 text-only datasets。原因很直接：CRV 需要捕获特定目标模型在生成时的 internal computational trace，静态的 $(text,label)$ pair 不够。数据集因此必须是 $(text,label,computational trace)$ 三元组。

论文构造了 Boolean、Arithmetic 两个 synthetic datasets 和一个 GSM8K step-level dataset。Synthetic tasks 的好处是 ground truth 清楚，step correctness 可以通过 parser、programmatic verifier 和 LLM judge 共同确认。GSM8K 更接近真实数学推理，但不能完全 programmatically verify，所以作者使用 Llama 3.3 70B Instruct 作为 expert judge，并做人类验证。对 error propagation，作者采用 strict truncation policy：保留到第一个 incorrect step 为止，后续步骤丢弃，避免“在错误前提下局部正确”的标签歧义。

这个 setup 的优点是标签相对干净，缺点是任务分布受控且窄。Boolean、Arithmetic 和 GSM8K 都是数学或符号 reasoning，离真实 agentic reasoning、多轮工具调用和开放域解释还很远。CRV 的实验结果应当读成“计算图 fingerprint 在这些受控 reasoning tasks 中可行”，不能直接外推为“所有 CoT 错误都有通用 graph signature”。

严格截断到第一个错误 step 也会改变任务定义。它让 label 更清楚，因为后续 steps 不再混入“基于错误前提的局部正确计算”；但它也让 CRV 主要学习 first-error detection，而不是完整错误恢复过程。真实 CoT 监督经常需要判断一条已经偏离的 reasoning path 是否能自我修正，或者后续步骤是否暴露了 earlier mistake。CRV 当前的干净设定适合验证 computation fingerprint，但还没有覆盖这种 error recovery 和 self-correction 场景。

## 3. Algorithm / Methods / Model

CRV 的第一步是把目标 LLM 变成可解释 surrogate model。作者对每个 MLP module 训练一个 per-layer transcoder/PLT。Transcoder 和普通 sparse autoencoder 类似，都会产生稀疏 feature activation；区别是 transcoder 学的是近似 MLP 的 input-output function，而不是只重构输入 activation。换句话说，transcoder 是原 MLP 的功能替代品：输入 residual stream 后，它用稀疏 feature basis 近似 MLP 输出。

训练好 transcoders 后，作者用它们替换原模型的 MLP modules。这样，模型 forward pass 中原本 dense、难解释的 MLP computation 会经过 sparse interpretable features。后续 attribution graph 的节点就不只是模糊的 hidden dimension，而是 active transcoder features、tokens 和 logits。这一步是 CRV 的 white-box 基础。

第二步是构建 step-level attribution graph。作者采用 recent circuit analysis 方法，从 final logits 往回追踪高 attribution paths，得到 sparse weighted directed graph。图里的边表示一个 token 或 feature 对后续 feature 或 output logit 的高影响连接。它并不是完整 transformer computation graph，而是剪枝后的核心信息流子图，目标是近似这一 step 的执行轨迹。

第三步是抽取 graph features。论文把 features 分成三类。**Global Graph Statistics** 记录 active feature nodes 数量、final logit probability、entropy 等全局复杂度和不确定性。**Node Influence and Activation Statistics** 统计 feature activation values、influence scores、layer histogram 等，区分“少数强 feature 主导”的计算和“许多弱 feature 扩散”的计算。**Topological and Path-Based Features** 包含 density、centrality、connectivity 等图结构指标，用来描述信息流组织方式。

第四步是 diagnostic classifier。作者使用 Gradient Boosting Classifier/GBC 处理这些异质 tabular features，并用 feature importance 分析哪些 graph properties 最能预测错误。这个选择比较务实：CRV 的核心不是发明复杂 classifier，而是看 attribution graph features 是否已经包含足够强的 correctness signal。

这里有一个非常重要的 interpretability caveat。论文的叙事强调 computational graph structure，但实验里最关键的 feature family 是 Node Influence & Activation Stats，而不是 topological features。也就是说，CRV 的强信号可能很大程度来自 transcoder activation/influence 本身，而不一定来自复杂图拓扑。Reviewer 抓住这一点质疑：如果不用 attribution graph，只用 transcoder activation statistics，是否也能接近 CRV？这正是后续必须补的 baseline。

从机制解释角度看，这个 caveat 不会让 CRV 失效，但会改变我们对它的理解。如果 node influence 足够强，CRV 更像是在问“错误步骤激活了哪些稀疏计算特征，以及这些特征对输出 logit 的影响有多异常”；如果 topology 也强，CRV 才更像是在问“信息流路径的组织结构哪里坏了”。当前结果更支持前者。因此，论文的标题说 computational graph 没有错，但最稳的贡献表述应该是 **transcoder-mediated attribution fingerprints**，而不是纯 graph topology verification。

## 4. Experiments

实验使用 Llama 3.1 8B Instruct，因为它能稳定产出 CoT。作者将其替换为 transcoder-infused model 后，在 Synthetic Boolean、Synthetic Arithmetic 和 GSM8K 上做 step correctness verification。指标包括 AUROC、AUPR 和 FPR@95，其中 incorrect label 被当作 positive class。FPR@95 表示在检测 95% 错误 step 时误报正确 step 的比例，越低越好。

主表结果支持 CRV 的核心假设。在 Boolean 上，CRV AUROC 为 75.87，高于 black-box 和 gray-box baselines；在 Arithmetic 上，CRV AUROC 达到 92.47，而最强 baseline Energy 是 76.45，CoE-C 是 69.39；在 GSM8K 上，CRV AUROC 为 70.17，也高于 Energy 的 62.55 和其他 hidden-state methods。AUPR 在 Boolean 上很低，是因为 incorrect label 只有 0.2%，极端类别不平衡会压低 precision-recall 表现；但在 Arithmetic 上，CRV AUPR 28.92，明显高于最强 baseline 的 5.59。

FPR@95 也说明 CRV 的可靠性优势。Arithmetic 上 CRV 把 FPR@95 降到 37.09%，而最强 baseline 仍在 63.33% 以上。这个指标比 AUROC 更贴近 verifier 的使用场景：如果你希望抓住绝大多数错误，又不想把大量正确步骤误判为错误，CRV 的结构信号确实更有价值。

跨域泛化结果更复杂。CRV trained on Arithmetic 在 GSM8K 上只有 57.04 AUROC，低于 Energy baseline；trained on GSM8K 在 Boolean 和 Arithmetic 上也表现差。作者把这解释为 error fingerprints 高度 task-specific：Boolean logic、synthetic arithmetic 和自然语言数学 word problems 的错误在计算图上有不同结构。这个结论很重要，因为它限制了 CRV 作为通用 verifier 的直接使用。

不过 combined training 有积极信号。把多个任务训练集合并后，CRV 在 Arithmetic 上达到 90.51 AUROC，接近 in-domain 92.47；在 GSM8K 上达到 70.62，略高于 in-domain specialist 的 70.17。这说明不同 domain 的 error geometry 虽然不同，但不是互斥的；一个 classifier 可以学到多个 failure signatures 的 union。未来如果要做更通用 verifier，可能需要覆盖足够多 reasoning domains，而不是期待零样本迁移。

Feature ablation 揭示了 CRV 真正依赖什么。在 Arithmetic 上，去掉 Node Influence & Activation features 后性能下降最大，FPR@95 从 37.09 升到 49.07；去掉 Global Stats 也有明显影响；去掉 Topological Stats 影响最小，FPR@95 只到 39.19。这说明错误检测主要来自局部 feature influence 和 activation pattern，而不是高阶图拓扑。这个结果削弱了“graph structure 是关键”的强叙事，但增强了论文可信度，因为作者没有隐藏不利于叙事的 ablation。

最有意思的是 causal intervention case。给定表达式 $(7*((5+9)+7))$，模型先正确算出 $5+9=14$，但下一步错误地直接做 $7*14=98$，跳过了 $14+7$。CRV 标出该 step 错误，feature importance 指向一个 late-layer transcoder feature 91814，作者解释它与 multiplication 相关。手动把这个 feature activation clamp 到零后，模型改为生成 $14+7=21$，再算 $7*21=147$，得到正确答案。这个例子展示了从检测到定位再到修复的闭环。

但必须强调，intervention 仍是 proof-of-concept。一个成功 case 不说明 CRV 已能系统修复推理错误；如果每次都需要人工解释 feature、选 hook、调 clamp，成本会很高。真正有说服力的下一步是批量 intervention：自动选出 top failure features，在多个 error categories 上评估修复率、误伤率和是否引入新错误。

此外，intervention 的成功也带来一个反向问题：替换 MLP 为 transcoders 后的 surrogate model 是否和原始模型足够等价？如果 surrogate model 的计算已经被稀疏瓶颈改变，那么在 surrogate 上发现并干预的 feature 未必能无损迁移回原模型。论文把 transcoders 作为 functional substitute，但任何替代都会引入 approximation error。对 verification 来说，只要 surrogate 保留足够多 correctness signal 就有用；对 model repair 来说，迁移性要求更高，因为最终要修的是原始模型或同架构部署模型。

## 5. Reviewer Discussion

OpenReview 最终决定是 **Accept (Oral)**。四个 reviewer 的分数是 8、4、8、6。AC 总结认为 reviewer 认可核心概念和贡献：把 mechanistic interpretability 技术用于 reasoning verification 和 model debugging，构建 attribution graphs，提取 structural features，并通过 intervention 展示潜在 causal discovery 能力。最终 AC 认为作者回应了主要担忧，适合 oral。

正面评价集中在 CRV 的问题设定。Reviewer 认为把 computational graph 当作 debuggable execution trace 是一个很好的 framing，也认可论文没有只停留在“解释模型很有趣”，而是把 attribution graph 接到一个具体任务：判断 CoT step 是否正确。Cross-domain experiment 也被正面评价，因为它暴露了方法泛化差这一不利结果，却提供了有用信息。

最尖锐的批评来自 baseline fairness。低分 reviewer 指出，已有工作显示 hidden representations 中常有 reasoning correctness signal，但本文 LR probe 和 MLP probe 表现弱，甚至被简单 black-box methods 超过。这很反常，可能说明 probe 设计不够强，例如平均所有 token hidden states 过于粗糙、layer/position 选择不足、数据量不够、没有 attention pooling 或 sliding window。这个批评非常有效，因为如果 gray-box baselines 被低估，CRV 的相对优势会被放大。

第二个关键担忧是 attribution graph 是否真的必要。CRV 同时做了两件事：用 transcoders 替换 MLP，构建 attribution graph。实验 ablation 显示 Node Influence & Activation Stats 最重要，Topological Stats 最不重要，这就打开了一个可能性：也许主要信号来自 sparse transcoder representation，而不是图结构本身。Reviewer 要求一个更简单 baseline：直接用 transcoder activation statistics 分类，不构图。如果这个 baseline 接近 CRV，那么论文的“computational graph”叙事就需要收缩。

我的客观评述是：CRV 是一篇强设定、强方向、但还没有完全封口的论文。它最值得读的地方是把 CoT verification 和 mechanistic interpretability 接起来，让“验证推理”不再只是外部打分，而是内部计算路径审计。它最脆弱的地方也很清楚：当前证据还不能区分“图结构真的提供关键新信号”和“稀疏 transcoder features 已经足够”。Reviewer 的批评不是挑刺，而是击中论文中心因果链。

因此我会把 CRV 读成 **white-box reasoning audit prototype**。它已经证明 attribution-graph-derived fingerprints 在受控任务中有强信号，也展示了 causal intervention 的漂亮案例；但它还不是成熟 verifier，也不是通用 mechanistic debugging pipeline。后续如果补上更强 probes、更简单 transcoder-only baselines、批量 intervention 和真实 agent/reasoning domains，CRV 的贡献会更扎实。

还有一个评审没有充分展开、但我认为很关键的问题是成本模型。CRV 要训练 transcoders、替换模型、为每个 step 构图并抽取特征，这比 PRM 或 hidden-state probe 重很多。这个成本在研究环境中可以接受，因为它换来可解释 diagnosis；但在生产环境中，CRV 更可能作为 offline audit 和 failure analysis 工具，而不是 online verifier。换句话说，它适合回答“为什么这类推理错了、内部哪些 feature 异常”，不适合直接回答“每个用户请求实时要不要放行”。

## 6. Related Work & Future Work

CRV 和 Process Reward Modeling、CoT verification、hidden-state self-verification probes、transcoder circuits、attribution graph interpretability 放在同一组。和 PRM 相比，它不是训练一个文本 verifier，而是分析产生 step 的内部计算。和 hidden-state probes 相比，它用稀疏 feature 和 attribution paths 提供更可解释的结构。和 TRACE 相比，它不是检测 reward hacking 的 effort anomaly，而是检测 reasoning step correctness 的 computation fingerprint。

未来最关键的方向是 **baseline hardening**。CRV 必须和更强的 gray-box methods 比较：layer-wise probes、token-position-aware probes、attention pooling probes、transcoder-activation-only classifiers、mean difference of hidden states、以及不替换 MLP 但使用 SAE activations 的 variants。只有这些比较补齐，才能知道 CRV 的优势到底来自 white-box graph、transcoder sparsity，还是当前 baselines 太弱。

第二个方向是 **scalable intervention**。论文展示了抑制 multiplication feature 可以纠正一个 arithmetic error，这非常有启发性。下一步应该把错误分成 categories，例如 premature operation、operator confusion、number binding error、entity mismatch、unit conversion error，然后自动寻找对应 transcoder features，批量测试 clamping、amplification 或 steering 是否能提高 CoT correctness。这样 CRV 才能从 diagnosis 走向 repair。

第三个方向是 **alignment-oriented monitoring**。CRV 当前验证的是数学和符号推理正确性，不是 deception、reward hacking 或 harmfulness。但如果把 attribution graph fingerprint 用于监督模型是否在利用 verifier loophole、是否在隐藏不忠实 CoT、是否在 agent tool trajectory 中走异常路径，它就会更直接进入 safety alignment。这个方向需要新的数据：真实工具使用、代码环境、多轮 agent、以及可标注的 reasoning failure 或 exploit traces。

这也解释了它和 safety alignment 的关系为什么不是自动成立的。CoT step 错误本身属于 reasoning reliability；只有当错误验证被用于监督欺骗、奖励投机、不可审计推理或高风险决策时，它才成为 safety mechanism。CRV 提供的是底层 white-box verification primitive，未来能不能变成 safety tool，取决于是否能把这种 primitive 接到真实风险场景。
