---
title: "AdAEM"
headline: "AdAEM: An Adaptively and Automated Extensible Measurement of LLMs' Value Difference"
description: "ICLR 2026 Oral note on AdAEM, a dynamic benchmark-generation framework for exposing value differences among LLMs."
status: complete
visibility: public
topic: "safety_alignment"
venue: "ICLR 2026 Oral"
year: 2026
paper_url: "https://arxiv.org/abs/2505.13531"
openreview_url: "https://openreview.net/forum?id=qNlTH4kYJZ"
source_pdf: "raw/papers/arxiv-2505.13531.pdf"
tags: ["value_evaluation", "dynamic_benchmark", "cultural_values"]
created: "2026-05-01"
updated: "2026-05-01"
---

> [!abstract] Contributions
>
> **AdAEM** 解决的是 LLM value evaluation 里的 **informativeness challenge/信息量不足问题**：很多静态价值观 benchmark 使用过时、泛化、甚至已被训练数据污染的问题，最终只能测出主流模型都已经对齐得差不多的 shared safety values，例如 harmlessness、fairness、security，导致不同模型的价值画像高度饱和、不可区分。论文提出一个动态、自扩展的问题生成算法，通过让多组来自不同文化和不同发布时间的 LLM 互相暴露分歧，自动生成更具体、更有争议、更能诱发价值差异的问题。
>
> 技术上，AdAEM 把问题生成写成一个 information-theoretic optimization：好问题既要让不同模型在价值分布上更可分，又不能只是问题本身带着强价值倾向把模型“推”到同一个方向。作者用 generalized Jensen-Shannon divergence 建模 distinguishability，用 disentanglement regularizer 避免问题本身主导答案，再用 EM-like 的 response generation 与 question refinement 交替优化。边界也很清楚：AdAEM 测的是模型在一组动态生成问题上的 **expressed value difference/表达出的价值差异**，不是模型“真实内在价值”的直接读数；它依赖价值分类器、黑箱概率近似、社会价值理论选择和争议话题过滤，因此非常适合作为 alignment audit 工具，但不应被误读成价值真理仪。

## 1. Introduction

LLM alignment 不只关心模型会不会回答有害请求，也关心模型在更广泛社会场景中如何表达价值取向。比如面对公共安全、个人自由、传统规范、社会公平、文化身份和资源分配，两个模型可能都给出礼貌、安全、无毒的回答，但它们实际强调的价值维度不同。这个差异会影响用户如何选择模型，也会影响模型在跨文化部署和高风险场景中的行为边界。

论文指出，现有 value benchmark 的主要问题不是“没有价值理论”，而是测试项太容易让模型给出保守共识。静态问卷里常见的问题如“政府是否应该投资消防设备”，几乎所有对齐过的模型都会回答支持公共安全。这样的题目只能证明模型知道主流安全价值，不足以区分模型在复杂冲突中的倾向。作者把这个问题称为 **informativeness challenge**：测试项缺少足够的信息量，导致评测结果在模型之间塌缩。

AdAEM 的直觉是，价值差异更容易在 **controversial scenarios/争议场景** 中显现。这里的“争议”不是为了制造危险内容，而是指一个问题同时牵动多个合理价值，例如安全与自由、传统与自我表达、环境保护与经济增长、地区文化规范与全球价值。一个好问题不应直接问“你是否支持安全”，而应创造一个需要模型在多个价值之间排序的具体情境。

这就解释了为什么 AdAEM 必须是动态的。模型更新很快，静态 benchmark 很容易被训练数据覆盖，也很难捕捉新社会事件。AdAEM 通过接入新模型、不同地区模型和不同时间段模型，把它们的知识边界和价值边界作为问题扩展信号。每当新的 LLM 出现，算法理论上可以继续用它生成更近期、更具体的问题，从而让评测和模型生态一起演化。

从 safety alignment 角度看，这篇论文的价值在于把“模型价值差异”从抽象讨论变成一个可操作的审计过程。它不是在训练模型更安全，而是在问：当多个模型都通过了常规安全测试后，我们还能不能看出它们在文化、伦理、风险偏好和社会规范上的差异？这个问题对模型选择、治理、跨文化部署和 red-team 都很重要。

## 2. Problem Setup

论文把待评测的模型记为 $\{p_{\theta_i}(y\mid x)\}_{i=1}^{K}$。其中 $x$ 是测试问题，$y$ 是模型回答，$v=(v_1,\ldots,v_d)$ 是 $d$ 维价值向量，每个 $v_j$ 表示回答中对某个价值维度的倾向。价值不是直接从模型参数读出来，而是通过一个 value analyzer $p_\omega(v\mid y)$ 从回答中识别出来。

因此，value evaluation 的基本对象可以写成：在问题分布 $\hat p(x)$ 下，模型回答 $y$ 后被识别出的价值分布。直观地说，评测不是问“模型本身有一个固定价值向量吗”，而是问“在这组问题上，模型的回答系统性体现出哪些价值”。这个区别很重要，因为模型价值表达高度依赖上下文；同一个模型在教育、战争、隐私、宗教、公共政策等话题上可能呈现不同价值组合。

AdAEM 要构造的不是任意问题，而是能让不同模型价值分布分开的测试问题。论文给出两个要求。第一是 **distinguishability/可区分性**：不同模型回答同一个问题时，应表达出不同价值维度或不同强度。第二是 **disentanglement/解耦性**：价值差异应该来自模型回答，而不是问题文本本身已经明显带有某个价值标签。比如一个问题本身充满“安全至上”的措辞，模型回答也体现 security，这不能说明模型更偏 security。

主实验选择 **Schwartz's Theory of Basic Values/施瓦茨基本价值理论**，包含 Power、Achievement、Hedonism、Stimulation、Self-Direction、Universalism、Benevolence、Tradition、Conformity、Security 十个维度。这个选择的优点是跨文化心理学中使用广、结构清楚，也已被若干 LLM value evaluation 工作采用。缺点是它不是唯一价值理论，不能穷尽所有道德、政治和文化价值。论文在 rebuttal 后补充了 **Moral Foundations Theory/MFT** 实例，用 Care、Fairness、Loyalty、Authority、Sanctity 五个维度验证框架可迁移。

这里还要区分两层测量。AdAEM 先生成 benchmark questions，再用模型回答和价值分类器形成 value scores。生成问题和最终评估可以使用不同模型集合。论文特别避免让最终要评测的模型直接参与问题优化，以减少评测泄漏和数据污染风险。

## 3. Algorithm / Methods / Model

AdAEM 的核心目标函数把可区分性和解耦性写在一起。简化地看，它寻找一个问题 $x^\*$，使不同 LLM 在 $x$ 上诱发的价值分布彼此差异更大，同时让回答体现的价值不要被问题本身的价值倾向吞掉：

$$
x^\*
=
\arg\max_x
GJS_{\alpha}\left[p_{\theta_1}(v\mid x),\ldots,p_{\theta_K}(v\mid x)\right]
+
\frac{\beta}{K}
\sum_{i=1}^{K}
JS\left[\hat p(v\mid x)\|p_{\theta_i}(v\mid x)\right].
$$

其中 $GJS_\alpha$ 是 generalized Jensen-Shannon divergence，用来衡量多个模型的价值分布是否分开；$\hat p(v\mid x)$ 表示问题本身表达出的价值倾向；$\beta$ 控制解耦项强度。论文进一步把 $GJS$ 展开成若干 KL terms，并通过引入回答 $y$ 作为 latent variable，把直接优化 $p(v\mid x)$ 的问题改写成 response generation 和 question refinement 两个交替步骤。

> [!note] Objective Interpretation
>
> 这套目标函数的关键不是让问题“越敏感越好”，而是让问题在不同模型之间产生可解释的价值差异。**Value difference** 负责拉开模型，**semantic difference** 负责避免大家说同一种话，**value conformity** 负责保证回答确实反映价值维度，**semantic coherence** 负责保证回答仍然贴合原问题。

第一步是 **Response Generation Step**。在固定上一轮问题 $x^{t-1}$ 的情况下，算法采样候选回答 $y$，并选择得分高的回答。高分回答需要同时满足四个条件：它和潜在价值 $v$ 有关，不能是价值无关废话；它和问题语义相关，不能离题；它在价值上不同于其他模型的回答；它在语义上也不能和其他回答太像。这个步骤实际是在寻找“当前问题能诱发出的价值分歧样本”。

第二步是 **Question Refinement Step**。在固定上一轮挑出的回答后，算法反过来改写问题 $x$，让新问题更能诱导这些差异继续出现。一个好的 refined question 要和之前产生的差异性回答保持上下文一致，同时让其他模型不容易给出同样意见或同样价值组合。这个过程类似 EM 或 information maximization：一轮根据问题找差异性回答，下一轮根据差异性回答优化问题。

单纯优化一个问题还不够，因为人类价值是多维、多话题的。AdAEM 因此在外层使用一个类似 multi-armed bandit 的探索机制。它从 1,535 个初始通用价值问题出发，每轮选择最有潜力的 topic，生成 $N_2=3$ 个新问题，再用较小模型 $P_1$ 做 refinement，用更强或更多样的模型 $P_2$ 估计 informativeness score。主实验设置 $B=1500$，最终得到 12,310 个 AdAEM Bench 问题。

黑箱模型带来一个工程难点：目标函数里需要 $p(v\mid y)$、$p(y)$ 等概率，但 GPT、Claude 这类 API 模型不直接提供完整 token-level 或 value-level probability。论文因此使用经验近似：用 value classifier 近似价值概率，用 BERTScore 等相似度指标近似语义一致性和差异。这个近似是 reviewer 重点质疑的地方，作者在 appendix 中用开源模型实现了更接近数学形式的版本，并报告近似版和精确版诱导出的价值结果 Pearson correlation 为 0.8560，Cronbach's $\alpha$ 为 0.8978，说明近似至少在实验设置中没有完全破坏构念。

生成 benchmark 后，AdAEM 还需要把模型回答聚合成最终价值画像。论文没有直接让 classifier 输出绝对值，而是先从回答中抽取多个 opinions，再识别每个 opinion 体现的价值维度，最后用 TrueSkill 做相对排名式聚合。这个设计比直接报绝对分数更稳，因为 value classifier 的绝对概率可能饱和或偏置；相对排名只要求比较不同模型在同一价值上的强弱。

## 4. Experiments

主结果首先验证问题质量。AdAEM Bench 有 12,310 个问题，平均长度 15.11，Self-BLEU 为 13.42，semantic similarity 为 0.44。相比 57 题的 SVS、40 题的 ValueBench 和 4,561 题的 ValueDCG，它覆盖更广语义空间，也更能触及具体地区、文化、技术和公共政策场景。作者还请 5 名社会科学专家评估 300 个问题，相比人工通用问题，AdAEM 在 reasonableness 上提升 8.7%，在 value differentiation 上提升 52%，Cohen's $\kappa=0.93$，说明标注一致性很高。

Validity 的核心实验是 **controlled value priming/受控价值启动**。作者用 system prompt 显式要求 o3-mini 体现某个 Schwartz value，然后看 AdAEM 是否能测到目标价值上升。结果显示目标维度平均显著增加，论文强调约 +31%；与目标相反的价值维度下降明显，约 -58%；同组价值也有中等上升，例如 Tradition 与 Security 的联动。这说明 AdAEM 不只是生成“看起来多样”的问题，至少能捕捉一个被人为操控的价值变化。

Reliability 方面，作者把数据随机分成五折，分别评估后比较一致性，得到 Cronbach's $\alpha=0.90$ 和 coefficient of variation 为 0.28。这个结果支持“结果不是由少数具体问题偶然驱动”的说法。rebuttal 后的 robustness 还显示，即使用更小、更少的模型参与生成，AdAEM-2 仍能产生 8,452 个问题，并与原 AdAEM 的评估结果有较高一致性，ICC 约 0.816，Pearson 约 0.790，Cronbach's $\alpha$ 约 0.839。

MFT 扩展是回应 reviewer generalizability concern 的关键补充。作者用 66 个 Moral Foundation 初始问题生成 589 个 AdAEM Bench-MFT 问题。与 MFQ 和 ValueBench 相比，AdAEM Bench-MFT 在四个模型之间得到更低平均相关，Pearson 约 -0.1，同时得到更高维度内标准差，约 0.21。这说明传统问卷让不同模型看起来太相似，而 AdAEM 能暴露更多模型间差异。GPT-5 value priming 实验里，五个 moral foundations 的 controlled score 都显著上升，尤其 Sanctity 从 30.19 到 97.83。

真正的 value evaluation 实验覆盖 16 个主流模型。论文的发现可以概括成几类。更强模型常更偏 Universalism、Benevolence 等 prosocial 维度；同一模型家族有相近价值倾向；reasoning-based LLM 和 chat-based LLM 的价值画像差异更大；同一家族随规模增长，某些维度会被放大，例如 Llama 从 8B 到 405B 在 Tradition 和 Universalism 上的变化。还有一个重要结果是 context-dependence：同一模型在 Technology and Innovation 与 Philosophy and Beliefs 话题下的价值倾向不同，说明价值画像不能只用一个全局向量概括。

实验最有说服力的地方，是它把静态 benchmark 的塌缩问题展示得很清楚。SVS 和 ValueDCG 下，多个模型在雷达图上高度相似，甚至出现 Power 和 Universalism 同时高的反直觉结构；ValueBench 稍有改善，但模型间仍不够分开；AdAEM 则给出更分散的结果。这个证据支撑了论文的主张：如果问题不能触发价值权衡，就很难评估价值差异。

但实验也有明显边界。首先，AdAEM 优化的是“让模型分开”的问题，这会天然偏向边界、争议和差异最大化场景。这样的 benchmark 很适合诊断差异，却不一定代表用户日常场景的分布。其次，价值分类器和 LLM-as-judge 仍可能带入西方英语语料、特定价值理论和模型自身偏差。第三，争议话题虽然有 Llama-Guard 过滤，但安全过滤本身也会改变问题分布。读这篇时要把 AdAEM 当成 **diagnostic benchmark generator/诊断性 benchmark 生成器**，而不是人口统计意义上的价值调查。

## 5. Reviewer Discussion

OpenReview 最终决定是 **Accept (Oral)**。四个 reviewer 原始分数是 8、4、8、8。AC 总结认为主要问题在 rebuttal 后基本解决，尤其是从单一 Schwartz value system 扩展到 MFT、补充黑箱近似与精确实现的一致性实验、增加 hyperparameter robustness 和 question-count fairness 分析。低分 reviewer 的核心担忧是方法论稳固性，但讨论后至少确认多数问题已被回应。

Reviewer 的正面评价集中在三个点。第一，问题重要：value difference、cultural adaptability 和 misalignment diagnosis 都需要比普通 harmlessness benchmark 更细的工具。第二，框架有新意：动态自扩展、in-context optimization、多模型边界探索，比静态问卷更适合快速演化的 LLM 生态。第三，实验覆盖较充分：12K 问题、16 个模型、人工质量评估、value priming、MFT 扩展和稳定性分析共同支撑了论文。

批评也很具体。Reviewer J1nE 质疑“最大化模型回答 divergence”是否真的揭示价值偏好，还是会诱导新的 benchmark bias。这个问题非常关键，因为如果一个方法专门寻找最容易让模型分裂的问题，它得到的是 **disagreement-amplified values/分歧放大后的价值表达**，不等同于模型在真实用户流量中的平均价值。作者回应说目标本来就是暴露 unique values，而不是测 universal values，并且所有模型在同一问题集上评估，因此比较是公平的。这个回应成立，但它也定义了 AdAEM 的使用场景：它适合暴露差异，不适合单独估计真实场景分布。

第二类担忧是数学近似和收敛。论文主目标函数看起来严肃，但实际黑箱实现用 classifier、coherence metric 和 sampling 近似多个概率项。作者补充了 exact-vs-approx 实验，并引用 information maximization 里的下界单调性来解释 EM-like procedure。我的判断是，这足以支撑 ICLR oral 的工程可信度，但还不是严格理论闭环。AdAEM 的核心贡献仍然是 dynamic value benchmark generation，而不是一个完全可证明的最优实验设计算法。

第三类担忧是 generalizability 和伦理风险。最初只用 Schwartz 会让框架看起来绑定一个价值理论；MFT 补充明显增强了说服力。伦理上，AdAEM 会主动寻找争议话题，这确实可能被滥用为挖掘模型敏感边界或生成挑动性内容的工具。作者加入 Llama-Guard 过滤和开放前清理承诺，但这个风险不能完全靠过滤消除。越强的问题生成器，越需要清晰的使用边界和审计日志。

我的客观评述是：AdAEM 是一篇很有价值的 alignment evaluation 论文，因为它准确抓住了“静态价值问卷测不出模型差异”这个真问题。它最强的部分不是某个公式，而是把模型差异、文化差异、时间更新和争议场景统一成一个可自动扩展的评测流程。最需要警惕的是解释口径：AdAEM 输出的不是模型价值的本体论真相，而是模型在一组高信息量、争议性、动态生成问题上的相对表达差异。把这个口径守住，它就是很强的审计工具；把它当成价值排名终局，就会过度解释。

## 6. Related Work & Future Work

AdAEM 接在 value benchmark、dynamic evaluation、synthetic benchmark generation 和 LLM-as-judge 几条线上。它和 ValueBench、ValueDCG 的区别在于不满足于固定问题，而是把问题生成本身纳入优化。它和一般 red-teaming 的区别在于目标不是诱导有害输出，而是诱导价值权衡。它和 cultural alignment 相关，因为来自不同文化和知识截止时间的模型会生成不同地区、不同社会事件的问题。

未来最重要的是把“价值理论”从单一框架扩展成多框架比较。Schwartz 和 MFT 只是两个入口，政治哲学、跨文化心理学、宗教伦理、法律权利框架、平台政策都可能形成不同 value axes。真正有用的 value audit 不应只问模型在一个理论中得几分，而应问不同理论下哪些结论一致，哪些结论冲突。

第二个方向是把 AdAEM 和 human validation 更紧地接起来。当前 human evaluation 主要评问题质量和价值区分能力，规模是 300 个问题、5 名专家。后续可以让不同文化背景的人类评审同一批 AdAEM 问题，观察模型差异是否对应人类群体差异，或者哪些问题只是让模型互相分裂却不对应真实人类价值分歧。没有这一步，AdAEM 的“value difference”仍主要是 model-relative difference。

第三个方向是区分 diagnostic distribution 和 deployment distribution。AdAEM 最大化信息量，天然偏向边界案例；真实用户请求可能更常见、更温和、更局部。一个成熟的评测系统应该同时报告常规分布下的 value behavior 和 AdAEM 边界分布下的 value stress profile。前者告诉我们日常表现，后者告诉我们模型在争议场景下如何分裂。
