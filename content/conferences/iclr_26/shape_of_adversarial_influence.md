---
title: "Adversarial Influence PH"
headline: "The Shape of Adversarial Influence: Characterizing LLM Latent Spaces with Persistent Homology"
description: "ICLR 2026 Oral note on using persistent homology to characterize topological compression in LLM latent spaces under prompt-injection and backdoor-style adversarial influence."
status: complete
visibility: public
topic: "safety_alignment"
venue: "ICLR 2026 Oral"
year: 2026
paper_url: "https://arxiv.org/abs/2505.20435"
openreview_url: "https://openreview.net/forum?id=v2PglvLLKT"
source_pdf: "raw/papers/arxiv-2505.20435.pdf"
tags: ["persistent_homology", "adversarial_monitoring", "representation_geometry"]
created: "2026-05-01"
updated: "2026-05-01"
---

> [!abstract] Contributions
>
> 这篇论文把 **Persistent Homology/PH/持久同调** 用到 LLM adversarial influence 的表示空间分析中。作者不再只问 clean 和 poisoned activations 能否被线性 probe 分开，而是把每层 final-token activations 当作高维 point cloud，计算 Vietoris--Rips filtration 的 persistence barcodes，再把 barcode summary 作为几何拓扑特征，分析 adversarial inputs 如何改变 latent space 的整体形状。
>
> 论文的核心发现是 **topological compression/拓扑压缩**：在 prompt injection 和 backdoor fine-tuning/sandbagging 两类攻击下，adversarial representations 往往表现为 connected components 的 mean death time 增大、$H_1$ loops 数量减少、剩余 loops 的 lifetime 变长。这个签名在 3.8B 到 70B 的六个模型上总体成立，并且 PH summaries 在 clean/poisoned separation 上可达到很高精度。边界也很清楚：这种 compression 主要是描述性和相关性的，还没有证明拓扑变化导致攻击成功；线性 baseline 已经很强，因此 PH 的价值更偏几何解释和跨攻击表征，而不是单纯检测。

## 1. Introduction

很多 LLM safety 和 interpretability 方法默认 representation 中存在可线性读取的方向，例如 refusal direction、truthfulness direction、harmfulness feature 或 adversarial state probe。这类方法有用，但它们主要看局部或线性结构。论文指出，adversarial influence 可能改变的是整个 representation cloud 的关系结构：点之间如何聚类、是否形成环、局部小结构是否塌缩成更少的大尺度结构。这类非线性、关系型几何很难被单个 probe direction 解释。

Persistent homology 正好处理这类问题。PH 不关心原始坐标的具体方向，而是看 point cloud 在不同距离阈值下如何逐渐连边、成团、形成环、填掉环。它输出的 barcode 描述了拓扑特征在尺度变化中的 birth 和 death。对 LLM activation cloud 来说，$H_0$ bars 反映 connected components 如何合并，$H_1$ bars 反映 loops/cycles 何时形成和消失。作者把这些 barcodes 压缩成 41 维 summary statistics，用来比较 clean 和 adversarial states。

论文研究两种攻击。第一种是 **Indirect Prompt Injection/XPIA**：攻击者把隐藏指令塞进 retrieved data，让模型混淆用户任务和外部内容中的恶意指令。第二种是 **sandbagging via backdoor fine-tuning**：模型经过后门式微调，默认 locked 状态下压低能力，只有特定触发词出现时才进入 elicited 状态。两者的机制不同，一个发生在 inference-time context，一个来自 training-time model modification；如果两者产生相似的 latent geometry signature，就说明 adversarial influence 可能有跨机制的表示效应。

这篇论文对 safety 的吸引力在于，它尝试找一种比单个线性 probe 更结构化的监测信号。线性 probe 可以告诉你“clean 和 poisoned 可分”，但很难告诉你 poisoned 状态的表示空间发生了什么形变。PH 的承诺是给出一种 coordinate-free、multiscale 的描述：攻击不是只沿某个方向移动 activation，而是把 representation cloud 从多样、局部丰富的小结构，压成较少、较大尺度、低熵的结构。

## 2. Problem Setup

给定一个模型、一层 $\ell$ 和一批输入，作者取每个输入的 final-token activation，形成 point cloud：

$$
X_\ell=\{\mathbf{x}_1^\ell,\ldots,\mathbf{x}_n^\ell\}\subset\mathbb{R}^{D}.
$$

这里 $D$ 是 hidden dimension。使用 final token 的理由是，它通常聚合了当前输入上下文的信息；这也构成一个限制，因为 prompt injection 或多 token 轨迹中的其他 token 可能携带不同结构。

PH 的第一步是构建 Vietoris--Rips complex。对距离阈值 $\epsilon$，若两个点距离小于 $\epsilon$，就连边；若多个点两两相连，就加入更高阶 simplex。随着 $\epsilon$ 从小到大变化，complex 逐渐长大，connected components 合并，loops 出现再被 triangles 填掉。这一整套随尺度变化的结构叫 filtration。

Persistence barcode 记录拓扑特征的 birth 和 death。$H_0$ bars 对应 connected components：一开始每个点都是独立 component，随着 $\epsilon$ 增大逐渐合并，death time 越大意味着点云中组件要到更大尺度才连上。$H_1$ bars 对应 loops：bar 数量反映循环结构的多样性，bar 长度反映某个 loop 在尺度变化中持续多久。

由于 barcodes 不能直接作为欧氏向量输入普通 classifier，作者计算 41 维 barcode summaries，包括 birth/death/persistence 的均值、方差、中位数、四分位数、birth-death ratio、bar count、total persistence 和 persistent entropy。这个 summary 是论文连接 topology 和 machine learning 的接口。

实验中，global layer-wise analysis 对每层分别抽样 clean 和 adversarial activations。主设置中每层取 $K=64$ 个 subsamples，每个 subsample 大小 $k=4096$，分别计算 clean/adversarial barcodes，再做 PCA、CCA、logistic regression 和 SHAP interpretation。local analysis 则把相邻层或非相邻层的同一 neuron activation 组成二维点云 $(v_i^\ell,v_i^{\ell'})$，分析 layer-to-layer information flow 的拓扑变化。

## 3. Algorithm / Methods / Model

Global analysis 的流程是：先从 clean 和 poisoned activation distributions 中抽取多个 point clouds；然后用 Ripser++ 计算 $H_0$ 和 $H_1$ barcodes；接着把每个 barcode 转成 41 维 summary；最后去除高度相关的 summary features，得到 pruned barcode summaries。作者发现，barcode summaries 中有一组高度相关特征，其中 $H_0$ mean death 是最突出的代表，同时 $H_1$ bar count 也保留重要信息。

在可视化层面，作者对 pruned barcode summaries 做 PCA。以 Mistral 7B 的 XPIA setting 为例，clean 和 poisoned summaries 在 layer 1、8、16、24、32 的前两个 principal components 上清楚分离。作者再用 CCA 分析 clean/adversarial 两组 summary 的关系，发现 $H_0$ mean deaths 在所有层中贡献最高，$H_1$ bar count 也显著。

在检测层面，作者训练 logistic regression 区分 clean 和 poisoned barcode summaries。Mistral 7B 上 PH summary 在所有报告层 test accuracy 和 AUC 都达到 1.0。对比线性 baseline 时，LDA、SVM、LR 直接在 raw activations 或 SAE-compressed activations 上也很强，很多层接近或达到 1.0；PH 的优势主要体现在 early layer 或 SAE-reduced setting 中更稳。作者因此强调，PH 的真正价值不是“比线性 probe 更能分类”，而是它的 summary features 可解释：它告诉我们 separation 对应什么几何变化。

这个几何变化就是 topological compression。作者的 SHAP 和 histogram 分析显示，poisoned activations 通常有更高的 $H_0$ mean death，也就是 components 在更大尺度才连通；同时 $H_1$ loops 数量更少，但剩余 loops 更大、更持久。直观地说，clean states 有更多 compact small-scale structures，poisoned states 则把许多小结构压成较少的大尺度结构。这里的“compression”不是普通 PCA 维度压缩，而是拓扑结构从多样局部特征变成少量主导结构。

Local analysis 进一步看 layer-to-layer information flow。对两个层 $\ell,\ell'$，作者把每个 neuron index $i$ 的 activation pair $(v_i^\ell,v_i^{\ell'})$ 作为二维点，得到 neuron-level point cloud。相邻层 activations 通常高度相关，点会聚在 $y=x$ 附近；若某些 neurons 在层间发生结构化变换，PH 会捕捉到 loops 和 topology complexity 的变化。

在 Mistral 7B 上，local analysis 发现 clean inputs 的 topology complexity 在浅层较高、深层变简单；poisoned activations 则从较简单开始，在中后层 complexity 增加，并在大约 layer 12 附近明显偏离 clean。作者称这是 adversarial influence reconfigures information flow 的 phase transition。Permutation control 会抹掉这种 signal，说明它不是单纯 activation distribution 的统计 artifact，而依赖特定 neuron-to-neuron pairing。

论文还提出一个 unsupervised layer selection signal：如果不知道 clean/poisoned labels，可以看某个 topological feature 在所有样本中的 overall variance。实验中，这个 variance 和 clean-vs-poisoned difference 的绝对值相关，尤其 total persistence of 1-bars 的 Spearman correlation 达到 0.78。这给出一个实用想法：高 variance topology layers 可能是 adversarial effects 最集中的层。

## 4. Experiments

数据覆盖六个 instruction-tuned models：Phi-3-mini 3.8B、Mistral 7B、LLaMA3 8B、Phi-3-medium 14B、Mixtral-8x7B、LLaMA3 70B。XPIA 使用 TaskTracker 数据，其中 poisoned samples 把用户任务和外部注入任务混合，攻击内容来自 BeaverTails、HarmBench、JailbreakBench、Do-Not-Answer 等安全数据源。Sandbagging/backdoor setting 则在 Mistral 7B 和 LLaMA3 8B 上用 LoRA 构造 locked/elicited 两种模式。

主结果显示，topological compression 在大多数模型和层上成立。summary table 中，所有模型的 minimum logistic-regression accuracy 都是 1.00；$H_0$ mean death 增大在六个模型中都打勾；$H_1$ loops 数量减少也基本成立，但 LLaMA3 70B 的方向跨层不稳定；$H_1$ lifetime 增长多数成立，少数深层有反转。这个 pattern 支持“跨架构、跨攻击机制存在共同几何签名”的 claim，但也说明它不是完全无例外的定律。

作者还比较了线性方法。Table 1 中，Mistral 7B 的 LDA 在原始 activation 上几乎全层接近 1.0，SVM 和 LR 在深层也接近完美。这个结果一方面说明 clean/poisoned 的分离非常强，另一方面削弱了“PH 是必要 detector”的表述。论文比较谨慎地把 PH 定位成 complementary interpretability：线性 probe 证明可分，PH 解释可分背后的拓扑结构。

Adaptive attack 方面，作者在 LLMail-Inject red-teaming dataset 上报告 PH signature 仍然存在。这个补充很重要，因为如果 PH 只在 TaskTracker 固定注入模板上有效，它可能只是捕捉 prompt format artifact；adaptive examples 能稍微缓解这个担忧。不过，论文仍没有系统覆盖 benign OOD shifts，例如主题变化、prompt length、retrieval style 或无害 dummy injection，这些都可能产生类似 geometry changes。

Local analysis 的结果支持“攻击影响不是单层线性位移”。Mistral 7B 上 clean/poisoned 的 total persistence of 1-bars 在相邻层之间出现不同趋势，permuted neuron-index control 消除 signal，非相邻层间隔增大后 signal 变弱。这说明 local PH 至少捕捉到了层间 activation interaction，而不只是全局 scale shift。

论文还引入 Local Dispersion Ratio/LDR，把 poisoned samples 分成 executed、refused、ignored。附录结果显示，executed 和 ignored attacks 在中层相对 clean 有更高 dispersion，而 refused attacks 被映射到更 compressed、低 dispersion 区域。这个结果把几何变化和行为类别稍微接上了：不同攻击处理结果在表示空间中不是完全一样的。可惜这部分主要在附录，主文对 task-level outcome 的连接仍偏弱。

总体实验支撑了三个强结论：PH summary 确实能稳定区分 clean/adversarial activation clouds；adversarial states 在作者设定中常表现为 topological compression；local neuron-level PH 能定位 layer-wise reconfiguration。实验没有完全支撑的结论是：topological compression 本身导致攻击成功，或者 PH 比简单线性方法更适合作为生产 detector。这两个更强 claim 需要干预实验和更复杂 baseline。

这里还要注意一个统计层面的解释风险。PH summary 是对一批 activations 的 point cloud 做统计，而不是对单条 prompt 直接读出“这个样本被攻击了”。因此它天然更适合 batch-level audit、layer diagnostics 和 dataset comparison。如果要把它变成单样本 detector，需要定义单样本附近的局部 point cloud 或在线滑动窗口，这会引入新的邻域选择、采样方差和延迟问题。论文在 rebuttal 中讨论了单 prompt barcode 的可行性，但主文证据主要还是分布级。

另外，topological compression 这个名字容易让人误解成“表示变得更集中”。论文实际说的是 adversarial states 的 connected components mean death 变大、loops 数量变少、剩余 loops 更大更持久；这更像是小尺度结构减少、主导大尺度结构增强。换句话说，它不是普通意义上的点云收缩，而是多尺度拓扑复杂度重分配。读这篇时要把 compression 理解成 **topological simplification with larger-scale dominant features**，否则会和 LDR 中某些 poisoned states 更 dispersed 的结果混在一起。

## 5. Reviewer Discussion

OpenReview 最终决定是 **Accept (Oral)**。四个 reviewer 的初始分数是 8、6、6、4；AC 的 meta-review 记录，4 分 reviewer 有较大概率提高到 6，两个 6 分 reviewer 可能提高到 8，8 分 reviewer 维持 8。AC 总结的主要担忧包括：topological compression 是相关而非因果；如果线性模型已经能检测 adversarial representations，PH barcode 的增量不显然；metrics/baselines 需要加强；结果和 task-level outcome 的连接不足；以及 subsampling 的可扩展性和稳定性。

正面评价集中在方法新颖性和实验覆盖。Reviewer 认可把 persistent homology 系统性应用到 LLM adversarial representations 是有创意的，也认可论文覆盖多个模型、两类攻击机制、global 和 local 两种分析。尤其高分 reviewer 认为 PH barcode summary 不只是分类器，而是给出 architecture-agnostic 的几何描述，这对理解 adversarial state 很有价值。

最尖锐的批评是 causality。论文说 adversarial inputs induce topological compression，但当前证据主要是观察 clean/poisoned conditions 的 barcode differences。Reviewer 要求更清楚地区分“拓扑压缩是攻击机制的一部分”还是“攻击样本、prompt length、模板、数据分布差异带来的伴随现象”。作者在 rebuttal 中补了 probabilistic toy model、perturbative analysis 和 ablations，但也承认真正的 interventional topology-aware defense 超出本文范围。

第二个批评是 baseline value。若 raw activations 上的 LDA 或 LR 已经近乎完美，PH 的检测价值就不在 AUC，而在解释。这个批评非常有效，也迫使论文把 contribution 收缩到“geometry-aware understanding”。我的判断是，这种收缩反而让论文更可信：PH 不应该被包装成比线性 probe 更神奇的 detector，而应该被视为揭示 representation cloud shape 的工具。

第三个批评是 scalability 和 subsampling。Vietoris--Rips PH 计算成本高，作者必须 subsample。Reviewer 担心 subsampling 会丢掉 rare but high-impact topological features，也担心不同 distance metric、filtration、token pooling 会改变结果。作者补了 subsampling ablations 和资源说明，表示单 prompt barcode 可毫秒级、full layer 可在多 GPU 上处理，但这仍然说明 PH 更适合 offline audit 和 research analysis，而不是直接替换轻量 online monitor。

我的客观评述是：这篇 oral 的强项是**把 adversarial representation shift 从“可分”推进到“形状如何改变”**。它给出的 topological compression 叙事很有启发性，尤其适合和 linear probes、SAE features、activation steering 互补。它的弱项是 causal status 不够强，且 detection baseline 已经很高。读这篇时不要被 1.00 accuracy 吸引，真正该读的是 $H_0$ death、$H_1$ loops、local layer phase shift 这些几何解释。

对 safety alignment 来说，它是一个很好的 monitoring-method seed，但不是成熟 defense。它能帮助我们问：“模型在 adversarial input 下是否进入一种拓扑更简单、更大尺度主导的 latent mode？” 但它还不能回答：“把这个 topology 改回来是否能减少 attack success？” 这个差距正是后续最值得做的地方。

我还会把 reviewer 的线性 baseline 批评看成一条建设性路线，而不是对论文的否定。如果线性 probe 已经很强，PH 仍然可以回答线性 probe 回答不了的问题：哪一类拓扑统计最能解释 separation？不同攻击机制是否共享相同 barcode pattern？同一模型不同层的几何变化是否同步？这些问题服务的是机制理解和审计报告，而不是最小成本分类。因此，后续论文最好不要把 PH 的卖点写成“检测更准”，而应写成“检测之外的几何诊断”。

还有一个安全实践上的限制：PH 的解释粒度仍然偏抽象。说某层 $H_1$ loops 减少，研究者知道 representation geometry 变了，但还不知道是哪类语义、策略或工具状态导致变化。它需要和更可命名的表征工具结合，例如 SAE features、linear concept directions、prompt metadata、behavioral outcome labels。否则 PH 可能成为一种漂亮但难以行动化的诊断图。

## 6. Related Work & Future Work

这篇论文和 persistent homology for neural networks、representation geometry、linear probes、SAE interpretability、representation engineering、prompt injection detection、backdoor/sandbagging analysis 放在同一条线上。和普通 probe 相比，它不只看 label separability；和 SAE 相比，它不分解单个 activation 的可解释 features，而是分析一批 activations 的关系形状；和 activation steering 相比，它更偏监测和诊断，不直接提供干预方向。

未来第一步是更严格的 benign OOD control。Prompt injection 样本往往在文本长度、格式、主题、安全语料来源上都和 clean 样本不同。必须加入无害检索块、无害长文本、格式相同但无恶意指令的 dummy injection、跨主题 clean samples，才能判断 topological compression 是否真是 adversarial influence，而不是普通 distribution shift。

第二步是 topology-aware intervention。论文现在展示 compression，但没有测试改变 compression 是否改变攻击结果。后续可以训练带 PH regularization 的模型、对 activations 做 topology-preserving projection、或设计 steering 让 poisoned activations 回到 clean barcode regime，然后测 attack success、refusal、task completion 是否改变。这一步才能把 PH 从 descriptive monitor 推向 causal safety tool。

第三步是 token-trajectory analysis。Final-token activation 很方便，但 agentic prompt injection、tool use 和 multi-turn deception 往往沿时间展开。更自然的 PH 对象可能是整段 trajectory 的 token activations、action-state pairs 或 memory states，而不是单个 final token 的 sample cloud。这个方向会让 PH 更接近真实 agent monitoring。

第四步是和 SAE/feature methods 联合。PH 告诉我们 representation cloud 的形状变化，SAE 告诉我们哪些可解释 features 活跃。把两者合起来，可能得到更可读的解释：topological compression 是否对应某类 safety features dominance、某些 refusal/attack features 合并、或某些 semantic dimensions 被压缩。这会把“形状”接回人类可理解概念。

第五步是建立跨攻击的 topology signature library。论文已经比较了 XPIA 和 sandbagging，但真实安全风险还包括 jailbreak、reward hacking、tool-use sabotage、data exfiltration、deceptive alignment probes、benign distribution shift 等。若不同风险有可区分的 barcode profiles，PH 可以成为模型安全体检的一部分；若很多 benign shifts 也产生同样 compression，它就更适合作为泛化 OOD 警报，而不是 adversarial-specific detector。
