---
title: "SSPO"
headline: "Semi-Supervised Preference Optimization with Limited Feedback"
description: "ICLR 2026 Oral note on SSPO, a semi-supervised preference optimization method using thresholded pseudo-labeling on unpaired data."
status: complete
visibility: public
topic: "preference_learning"
venue: "ICLR 2026 Oral"
year: 2026
paper_url: "https://arxiv.org/abs/2511.00040"
openreview_url: "https://openreview.net/forum?id=ghwxbTx7do"
source_pdf: "raw/papers/arxiv-2511.00040.pdf"
tags: ["preference_data", "semi_supervised_learning", "pseudo_labeling"]
created: "2026-05-01"
updated: "2026-05-01"
---

> [!abstract] Contributions
>
> **SSPO/Semi-Supervised Preference Optimization** 研究的是 preference labels 很少、但 unpaired SFT-style responses 很多的 alignment 场景。它把 preference optimization 改写成一个 probabilistic classification problem：先用少量 paired preference data 学到 reward-like score，再估计 winning responses 和 losing responses 的 reward density，寻找一个 Bayes-risk-minimizing threshold $\hat\delta$，用这个 threshold 给 unpaired responses 打 pseudo-label，最后把真实 paired loss 和 pseudo-labeled unpaired loss 联合训练。
>
> 论文的核心价值是把“用未标注 SFT 数据辅助 preference optimization”做成了一个有理论解释、可实现、实验上很强的数据效率方法。它在 UltraFeedback、UltraMedical-Preference 和 DSP Business 上显示，1% paired labels 加大量 unpaired data 可以超过不少 10% paired-label baselines。边界也很明显：threshold theorem 依赖 reward distribution 有可分离结构，而实际训练中 reward 是随 policy 更新而移动的；SSPO 的效果也高度依赖 unpaired data 的质量、相关性和 pseudo-label 噪声控制。

## 1. Introduction

Preference optimization 的瓶颈经常不是 loss 不够新，而是高质量 pairwise preference data 太贵。一个有用的 pair 不只是“两个回答谁更好”，还需要 prompt、多个 response、标注者标准、质量控制，有时还需要领域专家。普通 DPO、SimPO、ORPO、KTO 这类方法在训练阶段主要消费 paired comparisons；如果 paired data 只有 1%，模型很容易只学到局部偏好，泛化不足。

与此同时，很多领域已经有大量 unpaired data，例如 SFT 数据中的 prompt-response pairs。这些 response 没有显式比较标签，但可能包含隐式偏好信号：回答是否有结构、是否符合专业语气、是否提供步骤、是否覆盖关键点。SSPO 的直觉是，如果少量 paired data 已经让模型学到“winning response 的 reward 通常更高”，那么可以用一个 reward threshold 把 unpaired response 分成 pseudo-winning 和 pseudo-losing，再把它们转化为弱监督信号。

这篇论文的 alignment 意义在于，它把数据效率问题从“让更强 LLM 帮忙标注”转成了 **semi-supervised learning** 问题。相比直接让 judge model 生成 preference pairs，SSPO 更依赖当前 policy/reward 的统计结构：winning 和 losing response 的 reward distribution 是否已经分开，以及 threshold 是否能稳定跟踪这个分界。这个设定很适合低标注预算场景，但也带来一个风险：如果早期 reward 本身有偏，pseudo-labeling 可能把偏差放大。

## 2. Problem Setup

论文把 paired preference data 记为 $D_L=\{(x^{(i)},y_w^{(i)},y_l^{(i)})\}_{i=1}^{n_L}$，其中 $y_w$ 是 winner，$y_l$ 是 loser。unpaired data 记为 $D_U=\{(x_u^{(j)},y_u^{(j)})\}_{j=1}^{n_U}$，通常 $n_L\ll n_U$。目标是在少量 $D_L$ 和大量 $D_U$ 同时存在时训练 policy。

作者从 preference classifier 出发。给定 prompt $x$ 和两个 response $y,y'$，分类器 $f_\theta(x,y,y')$ 估计 $y$ 是否优于 $y'$。在 paired data 上，因为样本总是写成 $(x,y_w,y_l)$，label 恒为 $1$，于是风险退化成普通 pairwise preference loss。论文默认 reward-like score 使用 SimPO 风格：

$$
r_\theta(x,y)=\frac{\beta}{|y|}\log\pi_\theta(y|x).
$$

这里 $|y|$ 是 response 长度，长度归一化是为了避免模型只通过生成更长文本来抬高 log-probability。这个定义也说明 SSPO 没有训练独立 reward model，而是把 policy 自己当作 implicit reward model。这一点让方法轻量，但也正是 reviewer 担心 confirmation bias 的来源。

对 unpaired response $y_u$，没有另一个 response 可比。作者引入一个假想 boundary response $\bar y$，其 reward 是阈值 $\delta$。如果 $r_\theta(x_u,y_u)>\delta$，就给 $y_u$ 赋 pseudo-winning label；否则赋 pseudo-losing label。关键问题变成：$\delta$ 怎么选？

这个 setup 和普通 semi-supervised classification 有相似处，但也有一个 alignment 特有的麻烦。分类任务里的未标注样本通常有固定输入分布，模型只是在估计 label；SSPO 里的 reward score 来自正在训练的 policy，policy 更新会改变 response 的 log-probability，也会改变 winning/lossing reward density 的形状。因此，threshold 既是 classifier boundary，也是训练过程中会被模型自身移动的对象。论文后面的 adaptive scheduler 和 EMA threshold，本质上都是为了处理这个 non-stationarity。

还要注意，unpaired data 不是“没有信息”的数据。SFT response 往往来自人写答案、较强模型答案或领域材料，它们已经携带了风格、结构、知识密度等隐式偏好。SSPO 的假设是，少量 paired data 能学出一个足够好的 reward direction，把这些隐式偏好从 unpaired corpus 中筛出来。如果 unpaired data 本身质量低、领域错位，或者包含大量模型幻觉，threshold pseudo-label 就会把噪声变成训练信号。

## 3. Algorithm / Methods / Model

SSPO 用 Bayes risk 定义 threshold。设 $p(r|s=1)$ 是 winning response 的 reward density，$p(r|s=0)$ 是 losing response 的 reward density。给定 threshold $\delta$，错分风险是：

$$
R(\delta)
=
P(s=1)\int_{-\infty}^{\delta}p(r|s=1)\,dr
+
P(s=0)\int_{\delta}^{\infty}p(r|s=0)\,dr.
$$

第一项是把 true winner 错判成 loser 的概率，第二项是把 true loser 错判成 winner 的概率。因此最优 threshold 是：

$$
\delta^*=\arg\min_{\delta\in\mathbb{R}} R(\delta).
$$

论文的 Theorem 1 在 sub-Gaussian 假设下证明，如果 winning reward distribution 的均值 $\mu_w$ 高于 losing distribution 的均值 $\mu_l$，并且两者有足够间隔，那么存在一个 threshold 以高概率落在所有 losing rewards 和 winning rewards 之间。这里的理论重点不是声称真实 reward 永远完美可分，而是给 threshold pseudo-labeling 一个条件化保证：当 preference model 已经学出 margin 时，threshold 是合理的弱标注边界。

论文中特别澄清，$\delta^*=\mu_l+t_1=\mu_w-t_2$ 不应该被机械理解成任意分布下都严格相等，而是表示在高概率分离区间 $[\mu_l+t_1,\mu_w-t_2]$ 中选一个代表 threshold。这个澄清很重要，因为 reviewer 正是质疑真实 reward distributions 往往重叠，无法满足漂亮的完全分离。正确读法是：Theorem 1 给出 threshold pseudo-labeling 成立的理想条件；真实算法必须靠 KDE、EMA、scheduler 和置信度效果来承受偏离理想条件的情况。

实际训练中 $\mu_w,\mu_l$ 和 density 都未知，所以作者用 labeled minibatch 上的 winning/lossing rewards 做 **Kernel Density Estimation/KDE**，估计 $\hat p_w(r)$ 和 $\hat p_l(r)$，再数值搜索使 $\hat R(\delta)$ 最小的 $\hat\delta$。pseudo-label 规则写成：

$$
\tilde s_k
=
\mathbf{1}\{r_\theta(x_u^{(k)},y_u^{(k)})>\hat\delta\}.
$$

最终 loss 是 paired risk 和 pseudo-labeled unpaired risk 的加权和：

$$
\mathcal{L}(f_\theta)
=
\gamma' R_{D_L}(f_\theta)
+
(1-\gamma')R_{D_U}(f_\theta),
\quad
\gamma'=\max\{\gamma_{\min},\gamma_0\exp(-\lambda\tau)\}.
$$

这个 adaptive scheduler 是 SSPO 防止自我强化的关键。训练初期 $\gamma_0=1$，模型主要依赖 clean paired labels；随着训练推进，$\gamma'$ 下降，unpaired pseudo-labels 的权重增加；最低权重 $\gamma_{\min}=n_L/(n_L+n_U)$，对应 paired data 在总数据中的比例。直觉上，它先让 reward distribution 变得可分，再逐步放大 pseudo-label 的作用。

论文还用 EMA stabilizes KDE threshold，缓解 training 中 reward distribution 移动的问题。这一点很重要，因为 SSPO 的 reward 是 $\pi_\theta$ 自己给出的，policy 更新会改变 $p_\theta(r|s)$。如果 threshold 固定不动，pseudo-label 很快失真；如果 threshold 完全跟着 minibatch 抖动，训练会噪声很大。KDE + EMA 是一个实际折中：用当前 batch 估计边界，但让边界变化更平滑。

从梯度角度看，pseudo-winning response 会收到提高 $r_\theta(x,y)$ 的信号，pseudo-losing response 会收到降低 $r_\theta(x,y)$ 的信号。若 pseudo-label 正确，unpaired data 就扩展了 preference supervision；若 pseudo-label 错误，这个梯度会把模型往错误方向推。Adaptive scheduler 的意义就在于减少训练早期的错误放大：先用真实 paired labels 建立 reward separation，再让 unpaired loss 接管更多权重。这个机制比“直接把所有高分 unpaired response 当 winner”要稳一些。

SSPO 和 SSRM、SPA 的差异也在这里。SSRM 更关注 semi-supervised reward modeling，SPA 通过迭代 self-annotation 扩展 preference pairs；SSPO 不需要为 unpaired response 生成另一个候选，也不需要反复构造 pair，而是把每个单独 response 相对于 threshold 的位置变成标签。这个接口更便宜，但信息也更粗，因为它只知道“高于或低于当前边界”，不知道两个 unpaired responses 之间的细粒度排序。

## 4. Experiments

Toy experiment 先验证机制。作者构造一个 preference task，winner/loser 由规则决定，并加入 label noise。SSPO 在 $n_L=10,50,100$ 的低标注条件下超过 DPO、ORPO、SimPO；即使 50% noisy setting 下也保持一定优势。这个实验的作用是证明 threshold pseudo-labeling 在可控分布里确实能利用 unpaired samples，不只是大模型 benchmark 上的偶然相关。

真实实验覆盖 UltraFeedback、UltraMedical-Preference 和 DSP Business。通用设置里 paired preference data 只取 1% 或 10%，SSPO 额外使用 10% UltraChat-200k 作为 unpaired data。模型包括 Phi-2、Mistral-7B-Instruct、Llama3-8B-Instruct；领域实验使用 medical 和 business 对应的 unpaired corpora。评测采用 AlpacaEval2.0 的 raw win rate 和 length-controlled win rate，以及 MT-Bench。Length-controlled win rate 很关键，因为 LLM judge 常偏好更长回答；控制长度后更接近语义质量和指令遵循。

数据规模上，UltraFeedback 的 1% 和 10% 分别约为 611 和 6,113 条 paired samples，unpaired UltraChat 则约 20,786 条。Medical domain 使用 UltraMedical-Preference 的 1,093/10,935 paired samples，以及约 20,479 条 UltraMedical unpaired samples。Business domain 的 paired data 更少，1% 只有约 125 条，10% 约 1,255 条，unpaired Business Book 则有 17,480 条。这个设置很适合展示 SSPO 的优势，因为它正是为“标注 pair 极少、领域文本不少”的场景设计的。

主结果很强。UltraFeedback 上，Mistral-7B 只用 1% paired data 的 SSPO 达到 26.7 LC，超过同设置 SPA 的 18.2，也超过所有 10% paired-data baseline 的最高 19.1。10% paired data 下，Mistral SSPO 达到 30.0 LC。Llama3 上，10% SSPO 达到 20.7 LC，也高于 baselines。领域任务里，SSPO 在 UltraMedical-Preference 和 DSP Business 上通常也能超过对应强 baseline，说明它不是只在通用聊天数据上有效。

Ablation 说明两个组件都有作用。prior $P_{D_U}(s=1)$ 取 0.5 最稳，偏到 0.1 或 0.9 会下降，因为过强先验会让模型过度相信 pseudo-label 的一侧。但即使 prior 不完美，SSPO 仍能保持较强表现。Adaptive scheduler 的作用更明显：固定 $\gamma'$ 的版本仍能超过不少 baseline，但完整 scheduler 表现最好。论文可视化显示，训练早期 paired loss 主导，后期 unpaired loss 逐渐接管；reward distributions 也从重叠变得更分离，$\hat\delta$ 随之移动。

Case study 也支持论文的 intuition。比如“如何缝纽扣”这类 AlpacaEval prompt，unpaired UltraChat 中存在高质量 step-by-step answer。SSPO 通过 pseudo-labeling 把这种结构化回答当成有价值信号，生成更完整的材料清单和步骤，而 baseline 可能只给出笼统回答。这个例子说明 unpaired data 的价值不只在规模，而在它携带了模型可学习的格式、细节和领域表达。

论文在 rebuttal 和 appendix 中还补了公平性相关实验。Reviewer 担心 SSPO 只是因为用了更多 unpaired data，于是作者加入 DPO+SFT、SimPO+SFT 这类同样消费 unpaired data 的 baseline，并做 epoch-matched comparison。结果仍显示 SSPO 更强，说明简单把 unpaired data 当 SFT 数据并不足以复现收益；threshold pseudo-labeling 的筛选和区分确实提供了额外偏好信号。不过这个结论仍然依赖 unpaired data 与目标评测的关系，不能自动推广到任意 corpus。

计算开销方面，SSPO 的 KDE thresholding 是额外常数项。Appendix 分析其复杂度主要来自 minibatch 上的 KDE 和 grid search，相比 LLM forward/backward 很小；训练吞吐下降有限。这个点对实践很重要，因为如果半监督方法需要大量额外生成或 judge 调用，就会抵消省标注的好处。SSPO 的优势在于它不反复调用外部 LLM 标注 unpaired data，而是在训练 loop 内部用当前 score 做轻量 pseudo-label。

## 5. Reviewer Discussion

OpenReview 最终决定是 **Accept (Oral)**。四个 reviewer 的原始分数是 8、2、6、8；rebuttal 后，低分 reviewer 表示愿意提高到 6，另一个 6 分 reviewer 表示会提高到 8，AC 认为这是该 batch 中最高整体评价并推荐 oral。这个 review 过程本身很有信息量，因为它正好围绕 SSPO 最脆弱的地方展开：理论和实践是否一致、pseudo-label 是否自我强化、比较是否公平。

正面评价集中在数据效率和实现简洁。Reviewer 认可 SSPO 把 scarce preference labels 和 abundant unpaired data 结合起来，且不需要额外生成 pairwise synthetic feedback。相比 SSRM 和 SPA，SSPO 直接给单个 unpaired response 赋 pseudo-winning/pseudo-losing label，不需要迭代生成大量新 response pair，训练流程更轻。

最尖锐的批评来自原始 2 分 reviewer。其观点是：SSPO 的 theorem 依赖 winning/lossing rewards 高概率分离，但真实 reward distributions 很可能重叠；实践中 policy 自己的 likelihood 同时充当 reward 和训练对象，容易产生 confirmation bias；baselines 没有使用同样的 unpaired data，所以优势可能来自更多数据或更多训练预算。这个批评很有效，因为它直接打到 semi-supervised alignment 的核心风险。

作者 rebuttal 后补了几类证据：加入 DPO+SFT、SimPO+SFT 这类同样使用 unpaired data 的 baseline；做 epoch-matched comparison，缓解“只是训练更久”的质疑；用 KDE+EMA 和 adaptive scheduler 解释 non-stationary threshold；并报告 computational overhead 很小。AC 认为这些回应基本解决主要问题。但我的判断是，批评没有完全消失，只是从“论文不能接收”变成“方法适用条件需要说清”。SSPO 依赖高质量、相关且不污染评测的 unpaired data；如果 unpaired corpus 与 evaluation prompt 过近，结果可能混入 data exposure effect。

我的客观评述是：SSPO 是一篇很好的 data-centric alignment 论文。它不靠复杂 RL，也不靠更强 judge model，而是把 unpaired response 里的 latent preference signal 系统利用起来。它的最大贡献是把“未标注 SFT 数据也有偏好信息”这件事变成了可训练 objective。它的最大风险也来自同一个地方：如果当前 policy 的 reward-like score 方向错了，pseudo-labeling 会把错误方向放大。因此 SSPO 最适合 paired labels 虽少但质量高、unpaired data 与目标任务相关、并且模型已有基本 alignment 表征的场景。

更冷静地说，SSPO 的理论不应该被读成“任何未标注数据都能自动提升 alignment”。它依赖三个条件：paired labels 能学出初始 reward separation；unpaired data 中确实存在可迁移的高质量 response；scheduler 能在 pseudo-label 噪声变大前控制住训练。如果这三个条件不满足，SSPO 可能会把模型原有偏见、verbosity bias、领域误差或 hallucination 风格固化下来。Reviewer 原始低分的价值就在于把这些风险逼出来，rebuttal 后论文更完整。

对 safety alignment 来说，这个风险更高。Helpfulness pseudo-label 错了，可能只是回答啰嗦或格式差；safety pseudo-label 错了，可能把危险回答当成高质量 response 强化。若未来把 SSPO 用到 safety 数据，threshold 不能只基于 policy likelihood，还需要和安全分类器、人工审核或高置信拒绝边界结合。否则半监督数据效率会和安全风险同时放大。

## 6. Related Work & Future Work

SSPO 位于 DPO/SimPO/ORPO/KTO 之后，也接近 semi-supervised reward modeling、self-training 和 synthetic preference generation。它和 SSRM、SPA 的区别在于数据接口：SSRM 更像 reward model self-training，SPA 倾向迭代扩展 preference annotation，而 SSPO 直接把单条 unpaired response 通过 threshold 转成 pseudo-label。这个接口更简单，也更依赖 threshold 质量。

后续最值得追的是三件事。第一，pseudo-label accuracy 要单独评估，最好用 held-out human-labeled pairs 比较 1%、10%、50%、100% label budget 下 threshold 的真实 precision/recall。第二，要更严格地区分 unpaired data relevance 和 evaluation contamination：如果 unpaired data 与 AlpacaEval prompt 高度相似，提升并不全是 preference learning，也可能是风格和内容暴露。第三，要研究 safety alignment 场景中的 SSPO，因为 safety pseudo-label 的错误代价比 helpfulness 风格错误更高；这里可能需要 confidence gating 或 human-in-the-loop correction。

和同组的 SafeDPO、DPO misspecification 放在一起看，SSPO 负责回答“偏好数据太少怎么办”。SafeDPO 处理安全约束如何进入 pairwise ordering；DPO misspecification 处理参数化 policy 下 direct objective 是否统计可靠；SSPO 则处理 preference supervision 的数据稀缺。三篇共同说明，alignment 的瓶颈已经从单一 loss function 扩展成数据构造、目标几何、约束语义和评测可信度的组合问题。

这篇最值得带走的不是 KDE 这个具体工具，而是 **unpaired response can be preference-bearing** 这个判断。很多 SFT corpus 在表面上没有 preference label，但它们包含人类或强模型已经筛过的表达模式。SSPO 用 threshold 把这种隐性质量差异转成训练信号。这个思路对领域 alignment 很实用，因为医疗、法律、金融、科研写作这类领域往往有大量专家文本，却缺少大规模成对偏好标注。

但这个思路也要求更严格的数据审计。Unpaired data 如果来自 benchmark 相似分布，模型提升可能同时来自半监督偏好学习和 test-style exposure；如果来自强模型生成，pseudo-labeling 可能只是把强模型风格蒸馏给当前模型；如果来自网页或历史日志，里面可能混有过时、错误、带偏见的回答。SSPO 本身不能区分这些来源，它只看到 reward score 和 threshold。因此，真正可靠的使用方式应当把 unpaired corpus 的来源、去重、相似度、质量分布和潜在污染作为实验报告的一部分。
