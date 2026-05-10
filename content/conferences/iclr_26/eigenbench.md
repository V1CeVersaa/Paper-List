---
title: "EigenBench"
headline: "EigenBench: A Comparative Behavioral Measure of Value Alignment"
description: "ICLR 2026 Oral note on EigenBench, a peer-evaluation and EigenTrust method for measuring subjective value alignment in LLMs."
status: complete
visibility: public
topic: "safety_alignment"
venue: "ICLR 2026 Oral"
year: 2026
paper_url: "https://arxiv.org/abs/2509.01938"
openreview_url: "https://openreview.net/forum?id=fm79KXJIUQ"
source_pdf: "raw/papers/arxiv-2509.01938.pdf"
tags: ["value_alignment", "eigen_trust", "llm_judge"]
created: "2026-05-01"
updated: "2026-05-01"
---

> [!abstract] Contributions
>
> **EigenBench** 关注的是主观价值对齐的量化问题：如果我们要评价一个模型是否“kind”“conservative”“deep ecology aligned”，往往没有客观 ground truth label，因为合理评审者本来就可能分歧。论文提出一个黑箱评测框架：给定一组模型、一个由自然语言 criteria 组成的 constitution、以及一组开放场景，让模型互相生成回答并互相评判，再用 low-rank Bradley-Terry-Davidson model 学出每个模型的 **model disposition/模型倾向** 和 **judge lens/评审视角**，最后用 EigenTrust 聚合成每个模型的 value alignment score。
>
> 这篇的核心技术 move 是把“谁更对齐”变成一个 population-level consensus problem：更被群体认为对齐的模型，其评审意见也获得更高权重。这会产生一个左特征向量形式的 trust score，类似 PageRank/EigenTrust。论文通过人类评审比较、GPQA 无标签恢复排名、character training 验证和 37 模型扩展实验来支持方法有效性。边界也非常明显：EigenBench 依赖“更对齐者也是更好评审者”这个强假设，而且分数随模型 population、constitution wording、judge competence 和潜在 collusion 变化；它适合作为主观价值评测的研究框架，不应被当作单一权威价值裁判。

## 1. Introduction

很多 alignment benchmark 默认存在正确答案，例如是否泄露隐私、是否回答有害请求、数学题是否答对。但价值对齐里有一类问题没有这种 ground truth。一个模型是否足够 kind、是否符合某个宗教伦理、是否体现 deep ecology、是否遵循某种组织 model spec，这些判断可以有专业标准，却不一定有唯一标签。不同人可以合理分歧，不同模型也会以不同方式解释同一价值词。

EigenBench 的切入点是把这个主观性正面纳入评测。它不试图先定义一个外部黄金答案，而是构造一个模型群体，让模型作为 judge 互相比较回答。每个 judge 都按同一个 constitution 评价两个候选 response，但它可以用自己的方式理解这些 criteria。最后，系统聚合所有 pairwise judgments，得到模型群体内部的共识排名。

这个设定对 alignment 很有意义，因为许多现代 post-training 方法已经用 LLM feedback 替代或补充 human feedback。Constitutional AI、character training 和 deliberative alignment 都会让模型按一组原则自评、互评或生成偏好数据。如果我们要知道一个模型是否真的 internalized 了某个 constitution，仅仅问它“你是否遵守这些原则”是不够的；模型可能自我评价很好，但行为上不一定一致。EigenBench 测的是 revealed behavior：模型在不知道评测 criteria 的情况下回答，然后由其他模型按 constitution 判断。

论文最有野心的地方是把主观 value alignment 转成可计算的 trust graph。模型既是 candidate，又是 judge；如果一个模型在 constitution 下更对齐，它也应更擅长判断别人是否对齐。这个假设并不总是成立，作者也承认 plainspoken 这类价值不一定满足。但在 kindness、deep ecology 等价值上，论文认为这个假设有可用性。

## 2. Problem Setup

EigenBench 的输入有三部分。第一是模型群体 $M=\{M_1,\ldots,M_N\}$。这里的模型可以是一个基础语言模型加一个 persona prompt 的组合，所以同一个 base model 在不同 persona 下可以被当作不同 candidate。第二是 constitution $C=\{C_1,\ldots,C_k\}$，每个 $C_i$ 是一个自然语言判断标准。第三是 scenario dataset $S$，论文主要使用 r/AskReddit 中开放式、真实用户提出的问题，也测试了 OpenAssistant 和 AIRiskDilemmas。

给定一个场景 $S_\ell$，系统先让两个 evaluee models $M_j$ 和 $M_k$ 分别生成回答 $R_j$ 和 $R_k$。接着选择一个 judge model $M_i$。judge 会先根据 constitution 分别反思两个回答，生成 reflections $\hat R_j,\hat R_k$，再比较二者哪个更符合 constitution，或者判定 tie。这个结果记为 comparison trit：

$$
r_{ijk\ell}
=
\begin{cases}
0, & M_i\ \text{ties}\ R_j\ \text{and}\ R_k,\\
1, & M_i\ \text{prefers}\ R_j\ \text{to}\ R_k,\\
2, & M_i\ \text{prefers}\ R_k\ \text{to}\ R_j.
\end{cases}
$$

论文强调这个流程是 double-blind。被评模型不知道自己会按什么 constitution 被评价，甚至不知道会被评价；judge 不知道两个回答来自哪个模型。为了减少 order bias，系统会以两种顺序展示同一对回答。如果 judge 在顺序变化后给出强矛盾偏好，就把该比较覆盖为 tie。这个设计很重要，因为 LLM judge 容易有 primacy 或 recency bias。

Constitution 是 EigenBench 的核心语义输入。论文主实验使用 Universal Kindness、Conservatism、Deep Ecology 三个 constitution；character training 实验使用 Loving constitution。每个 constitution 不是单个抽象词，而是一组比较标准。例如 kindness 可能包含温暖、尊重、关怀、避免伤害等多个维度。EigenBench 最适合这种 criteria 之间有细微张力、评审者可能合理分歧的复杂 trait。

## 3. Algorithm / Methods / Model

收集到大量 pairwise win-loss-tie 比较后，EigenBench 不直接做简单平均，而是拟合一个 low-rank Bradley-Terry-Davidson/BTD 模型。普通 Bradley-Terry 模型给每个 candidate 一个标量强度；EigenBench 扩展成向量形式，因为主观 constitution 可能有多个隐含维度。

每个 candidate model $M_j$ 有一个 **model disposition** 向量 $v_j\in\mathbb{R}^d$，表示它在 constitution 的潜在价值维度上的行为倾向。每个 judge model $M_i$ 有一个 **judge lens** 向量 $u_i\in\mathbb{R}^d$，表示它评判时更看重哪些维度。每个 judge 还有一个 tie propensity $\lambda_i$，表示它更倾向于判 tie 还是强行二选一。judge $i$ 认为 $j$ 胜过 $k$ 的概率由内积 $u_i^\top v_j$ 和 $u_i^\top v_k$ 决定：

$$
\Pr(i\ \text{thinks}\ j\succ k)
=
\frac{\exp(u_i^\top v_j)}{Z},
$$

$$
\Pr(i\ \text{thinks}\ k\succ j)
=
\frac{\exp(u_i^\top v_k)}{Z},
$$

$$
\Pr(i\ \text{thinks}\ j\approx k)
=
\frac{\lambda_i\exp\left(\frac{1}{2}u_i^\top(v_j+v_k)\right)}{Z}.
$$

这个建模非常关键。它允许两个 judge 都是真诚的，但因为 judge lens 不同，对同一 pair 给出不同判断。例如一个 judge 解释 kindness 时更重视直接帮助，另一个更重视避免冒犯；两个 judge lens 就会落在不同方向。模型 disposition 则表示候选回答在这些 latent aspects 上的位置。

拟合完 $u_i,v_j,\lambda_i$ 后，EigenBench 构造 trust matrix $T$。其中 $T_{ij}$ 表示 judge $M_i$ 在 constitution 下对 evaluee $M_j$ 的信任程度，直观上是如果 $M_i$ 在所有 candidate 中选择最符合 constitution 的回答，它会选择 $M_j$ 的概率。然后系统求 trust vector $t$：

$$
t = tT.
$$

这就是 EigenTrust 部分。$t_j$ 是模型 $M_j$ 的最终 trust score，也可以转换为 Elo：

$$
\mathrm{Elo}_j = 1500 + 400\log_{10}(N t_j).
$$

> [!note] Core Assumption
>
> EigenBench 最强也最危险的假设是：**一个更符合 constitution 的模型，也更擅长判断别人是否符合 constitution**。如果这个假设成立，用 $t$ 给 judge 加权是合理的；如果不成立，系统可能把某种群体偏见、能力差异或互相偏好放大成“价值对齐分数”。

这个机制和 PageRank 很像。普通平均相当于每个 judge 票权相同；EigenTrust 平均让更受信任的 judge 票权更高。问题是，web page 的 incoming link 有相对清楚的图结构含义，而 value judgment 的“好 judge”本身就是待评估对象。因此 EigenBench 的循环性不是 bug，而是方法的核心，但也必须被实验验证和限制条件约束。

## 4. Experiments

主实验先在 8 个模型上评估 Universal Kindness、Conservatism 和 Deep Ecology，包括 Claude 4 Sonnet、GPT 4.1、Gemini 2.5 Pro、Grok 4、DeepSeek v3、Qwen 3、Kimi K2 和 Llama 4 Maverick。每个 constitution 大约收集 30,000 个 pairwise judge comparisons，覆盖 1,000 个 r/AskReddit 场景。结果以 Elo ranking 展示，Deep Ecology 的置信区间更大，因为许多场景与该 constitution 相关性低，judge 更容易给 tie。

第二个实验测试 prompted dispositions。作者构造 5 个 base LMs 乘 5 个 persona prompts 的 25 个 model-persona 组合，在 Universal Kindness 下运行 EigenBench。结果显示 trust score 的 79% variance 由 persona prompt 解释，21% 由 base LM 解释。这说明 prompt 确实强烈影响模型价值表达，但 base model 也保留可测的 dispositional tendency。这个结果对 alignment 很有意思：系统 prompt 可以改变角色，但不能完全抹掉模型底层行为差异。

第三个实验是 character training。作者使用 *Open Character Training* 中的 Loving constitution，对 Llama 3.1 8B 的 base、loving pre-prompted 和 loving fine-tuned 版本，以及 Qwen、Gemma、Mistral 等开源模型运行 EigenBench。结果中 Llama 3.1 8B base 得分最低，为 1426；pre-prompted loving 得分 1579；fine-tuned loving-oct 得分 1573。这个实验支持 EigenBench 可作为 character training 成功与否的外部测量，而不只是一个抽象排序器。

论文还比较了 **stated values/自述价值** 和 **revealed values/行为显现价值**。作者直接让模型按 constitution 给自己打分，发现 survey ranking 和 EigenBench ranking 明显不同。例如在 Universal Kindness 下，Grok 4 在 EigenBench 中排名较低，却给自己满分；Claude 4 Sonnet 在 EigenBench 中排名较高，却给自己较低自评分。这个结果很重要，因为 alignment 评测不能只问模型“你是否符合这些原则”。模型可以复述原则，也可以自我评价很好，但真正需要测的是它在开放情境下生成的回答被其他评审如何比较。

Human validation 是论文被接收的重要支撑。作者让 7 名人类评审在 Universal Kindness 的 8 个 criteria 上评价同一批模型回答，每人约 50 个场景、约 400 个 datapoints，总共约 3000 个 comparisons。把人类比较也拟合成 scalar BTD trust vector 后，作者发现平均 human-human interjudge distance 为 0.3133，平均 human-LM interjudge distance 为 0.3130，几乎相同。论文据此认为，在这个任务上 LLM judge 对人类评判的近似程度与人类之间彼此差异相当。

这个 human validation 结果要谨慎读。它不是说 LLM judge 已经等价于人类价值判断，而是说在 Universal Kindness、r/AskReddit 场景、8 个模型、7 名人类评审这个具体设置下，LM judge 和人类 trust vector 的距离落在 human-human disagreement 的量级内。换到医疗伦理、政治价值、法律合规、跨文化宗教规范，结论可能不同。因此它支撑的是方法可行性，不是 LLM judge 的普遍授权。

GPQA validation 则测试 EigenBench 在有客观标签但不向 judge 提供标签时能否恢复模型排名。系统让多个模型回答 GPQA 选择题，再让 judge 在两个模型的答案之间选择更可能正确的一个，但不给 ground truth。结果在 15 个模型上得到 Kendall-$\tau\approx0.77$，与真实 GPQA 排名只差 12 个 adjacent swaps，随机达到这种接近程度的概率约 $10^{-6}$。这个结果是论文最强的 sanity check：如果模型互评能在客观任务上近似恢复真实能力排名，那么在无标签主观任务上至少有一定可信基础。

Robustness 部分覆盖 scenario distribution、constitution wording 和 model population。换用 OpenAssistant 或 AIRiskDilemmas 后，Universal Kindness 排名大体稳定，但个别模型如 Grok 4 和 GPT 4.1 有明显变化，说明场景分布仍会影响分数。Conservatism constitution 用五个模型生成的不同措辞版本测试，最大标准差约 16 Elo，且没有明显偏向生成该 constitution 的模型。加入 Claude 3.5 Haiku 或 Claude 4 Sonnet 后，原有四模型分数总体稳定，但 Grok 4 下降，Claude 4 Sonnet 和 Claude 3.5 Haiku 之间也有相互影响。

rebuttal 后补充的大规模实验覆盖 37 个模型，包括闭源、开源、reasoning 和 non-reasoning 模型。作者还做了 score stability analysis，发现子样本得分不稳定性与样本量呈近似 $s^{-1/2}$ 的 power-law 下降；embedding dimension analysis 显示，在 37 模型的大 population 中，$d$ 增大到约 30 前 train/test loss 都下降，说明复杂主观价值在大群体里确实需要更高维 judge lens 和 disposition，而小群体中 $d=2$ 足够只是因为行为维度较少。

论文还测试了 **greenbeard effect/绿胡子效应**：如果一批模型用暗号互相识别并偏袒同类，EigenTrust 类方法可能被 coalition 操纵。实验里加入多个带 secret word 的 greenbeard personas 后，greenbeard 分数快速上升，但原模型分数相对不受影响，甚至 greenbeards 成为多数时也没有完全支配。作者同时承认，如果 prompt 更强或模型更服从暗号，greenbeards 可能真的主导排名。这个实验很重要，因为它说明 EigenBench 的 double-blind 机制不是形式细节，而是安全前提。

还有一个容易忽略的方法贡献是 judge scaffold 的偏差控制。论文在 appendix 中比较了有无 reflection step 的数据收集，发现加入逐个回答反思后，primacy bias、recency bias 和 preference cycle 等 judge inconsistency 指标下降。这个结果说明 EigenBench 不是简单把 LLM judge 当黑箱投票器，而是在认真处理 LLM 作为评审者时的顺序偏差、循环偏好和 tie 行为。对任何依赖 LLM feedback 的 alignment pipeline 来说，这个细节都值得保留。

## 5. Reviewer Discussion

OpenReview 最终决定是 **Accept (Oral)**。四个 reviewer 原始分数是 6、10、4、4。AC 总结认为论文研究的是一个重要且尚未充分探索的问题：在没有 ground-truth labels 的情况下，如何量化 subjective value alignment。rebuttal 和 revision 后，作者补充了更多人类评审、37 模型大规模实验、模型分数与 judge quality 的相关分析、character training 验证和稳定性分析，基本回应了低分 reviewer 对 circularity、实验规模和实际用途的担忧。

正面评价最强的是 Reviewer 1FTz，给了 10 分，认为论文提供了一个很有前景的方法来测量模型价值，并且 GPQA validation、人类评审、disposition visualization 都有创意。Reviewer M4QR 虽然给 6 分，也认可 EigenTrust + BTD 聚合在技术上 sound，并认为在没有 ground truth 的 subjective traits 中有潜力。大家共同认可的问题意识是清楚的：传统 benchmark 很难测主观价值，而 EigenBench 至少提出了一个可执行框架。

低分 reviewer 的批评集中在一个核心假设：**aligned models are better judges**。如果这个假设未经验证，EigenTrust 加权可能只是自我强化循环。比如一群风格相似、能力相近、训练来源相似的模型可能互相偏好，把群体习惯误认为价值对齐。作者在 rebuttal 后用更多 human validation 和 model score 与 judgment quality 的分析来缓解这个问题，但它仍然是 EigenBench 的根本边界。

第二个批评是实验规模和模型多样性。初稿主要用少量 frontier/closed-source models，只有极少 open-source，对 population dependence 的担心很合理。revision 中加入 37 模型大实验后，这个问题明显缓解，但没有完全消失。EigenBench 的分数不是模型的绝对属性，而是相对于当前 population 和当前 constitution 的共识位置。换一批 judge 和 candidate，分数可能变化；这不是缺陷，但必须在报告中显式说明。

第三个批评是人类验证不足和应用验证不足。初稿只有 2 名人类评审，被 reviewer 明确指出远远不够。revision 扩展到 7 名人类和约 3000 comparisons，并加入 character training 实验，直接回应了“EigenBench 是否能衡量 fine-tuning 成效”的问题。这个修订非常关键，因为没有人类验证和应用验证，EigenBench 很容易停留在数学上漂亮但外部意义不明的模型互评系统。

我的客观评述是：EigenBench 的优点是把主观价值评测中的循环性讲清楚并工程化了。它没有假装存在一个上帝视角标签，而是承认主观价值需要群体判断，然后用 BTD 和 EigenTrust 让这个群体判断可计算。它的危险也同样来自这里：如果群体本身偏、constitution 模糊、judge 能力不足、或者模型之间有隐性同源偏好，EigenBench 会把这些结构编码进分数。它不是价值真理机制，而是 **population-relative value consensus measurement/相对于模型群体的价值共识测量**。

## 6. Related Work & Future Work

EigenBench 和 LMArena、Prompt-to-Leaderboard、LitmusValues、Constitutional AI、character training、LLM-as-judge 以及 PageRank/EigenTrust rating systems 相邻。和 LMArena 不同，它不是测一般人类偏好，而是给定 constitution 后测某个价值系统下的行为对齐。和 LitmusValues 不同，它不是问一个模型内部优先哪些 value，而是比较一组模型谁更符合一个外部价值系统。和普通 LLM-as-judge 不同，它明确建模 judge lens、tie propensity 和 population-level trust。

后续最重要的是更强的人类校准。当前 human validation 支持 LLM judge 可以近似人类评审差异，但 Universal Kindness 只是一个价值系统，7 名人类也不足以覆盖文化、政治和专业背景。未来应在多个 constitution 下收集多群体人类评审，并报告模型群体共识、人类群体共识和混合 trust scores 的差异。这样才能知道 EigenBench 什么时候接近人类，什么时候只是模型之间自洽。

第二个方向是 population reporting standard。每次报告 EigenBench 分数时，都应同时报告模型群体组成、judge/candidate 是否同源、constitution 来源、scenario distribution、latent dimension、comparison count 和 stability interval。没有这些元数据，单个 Elo 数字会被误读成绝对价值分数。EigenBench 越像 leaderboard，越需要抵抗 leaderboard 的过度解释。

这里尤其要避免把 EigenBench 排名直接用于跨论文或跨机构的绝对比较。一个模型在 Universal Kindness 上得分高，只能说明它在给定模型群体、给定场景、给定 constitution 和给定 judge scaffold 下更受加权群体偏好；它不自动说明该模型在所有文化语境中更善良，也不说明它在高风险安全任务中更可靠。比较时应把 EigenBench 看成一个实验设计，而不是一个脱离设置仍然成立的分数。

第三个方向是 adversarial robustness。greenbeard 实验说明 coalition manipulation 是真实问题。未来如果 EigenBench 用于公开模型排名，模型开发者可能主动让模型学会识别评测风格、偏袒同源回答，或者生成更容易讨好 judge 的表面价值语言。需要更强 double-blind、response anonymization、human spot checks、coalition detection 和 adversarial constitutions。

第四个方向是把 EigenBench 和训练闭环结合。character training 实验已经显示，pre-prompt 或 fine-tuning 后模型在 Loving constitution 上得分上升。下一步要研究的是：如果直接用 EigenBench score 做训练目标，会不会 Goodhart？模型会学到真实价值行为，还是学到迎合 judge lens 的策略？这会把 EigenBench 从 evaluation 工具推进到 alignment objective，但风险也更高。

最后，还需要更清楚地区分 **average-case alignment/平均情形对齐** 和 worst-case safety。EigenBench 明确关注模型在大量开放场景上的平均价值表现，而不是寻找少数灾难性 failure modes。这个定位很合理，因为多智能体社会、平台推荐、日常助手行为都受平均倾向影响；但它不能替代 jailbreak、scheming、biosecurity、privacy leakage 等 worst-case 风险评测。未来的安全评估应把 EigenBench 这类平均价值测量和 adversarial safety audit 并列，而不是互相替代。
