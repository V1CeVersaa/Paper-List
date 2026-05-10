---
title: "DP Domain Discovery"
headline: "Missing Mass for Differentially Private Domain Discovery"
description: "ICLR 2026 Oral note on differentially private domain discovery, missing mass guarantees, WGM, unknown-domain top-k, and k-hitting set."
status: complete
visibility: public
topic: "safety_alignment"
venue: "ICLR 2026 Oral"
year: 2026
paper_url: "https://arxiv.org/abs/2603.14016"
openreview_url: "https://openreview.net/forum?id=yBpzF8hp3J"
source_pdf: "raw/papers/arxiv-2603.14016.pdf"
tags: ["differential_privacy", "private_domain_discovery", "data_privacy"]
created: "2026-05-01"
updated: "2026-05-01"
---

> [!abstract] Contributions
>
> 这篇论文研究 **differentially private domain discovery/差分隐私域发现**：每个用户持有一个 item set，整体 domain 事先未知或极大，算法既要保护 user-level privacy，又要输出一个有用的 item subset。作者的关键动作是把传统 set union 的“输出多少个 unique items”改写成 **missing mass/缺失质量**，也就是未恢复 item 在经验分布中占多少概率质量。这个指标避开了 unknown domain 下低频 singleton 几乎不可私有释放的硬障碍，让 Weighted Gaussian Mechanism/WGM 这种简单机制获得可证明的绝对 utility guarantee。
>
> 论文的主结果是：在 Zipfian data 上，WGM 对 $\ell_1$ missing mass 有近似最优的上界和匹配下界；在不假设 Zipfian 的任意数据上，WGM 还能给出 distribution-free 的 $\ell_\infty$ missing mass 保证。作者进一步把 WGM 当作 unknown-domain preprocessing，用于 private top-k 和 $k$-hitting set，并给出新的 utility bounds。边界也很清楚：理论依赖 user contribution bound $\Delta_0$ 的选择，top-k 与 hitting set 的上下界仍有 gap，实验主要证明 WGM 经验上强而不是完全验证 Zipfian 理论曲线。

## 1. Introduction

很多隐私数据分析并没有一个干净的已知 domain。搜索 query、购物记录、评论文本、用户安装的 app、网页访问记录都可以被看成巨大或开放的字符串集合。若直接假设 domain 已知，算法可以在所有候选 item 上加噪并筛选；但真实系统通常先要发现哪些 item 真的出现过，再把这个 domain 交给后续 top-k、统计查询或特征选择过程。这个第一步就是 **domain discovery**。

难点来自 user-level differential privacy。若某个 item 只出现在一个用户的数据里，任何满足 soundness 的算法都不能高概率输出它，否则相邻数据集中删掉这个用户后，该 item 根本不存在，输出概率会暴露个人存在性。因此，传统 set union 的 cardinality objective 很硬：想私有地恢复更多 unique items，本质上会被低频尾部卡住。论文的洞察是，很多低频 item 虽然数量多，但它们对经验分布质量的贡献很小；如果目标是恢复主要概率质量，而不是每个唯一字符串，那么问题就有更合理的可解性。

这也是 **missing mass** 指标的意义。它不问“没找回多少个 item”，而问“没找回的 item 在所有用户贡献中占多少质量”。对下游统计分析来说，这通常更接近实际 utility：漏掉一个只出现一次的长尾 query，比漏掉一个高频 query 的影响小得多。论文把这个指标引入 DP domain discovery 后，才能对 WGM 给出绝对 utility bounds，而不仅是和别的算法相互比较。

这篇放在 `Privacy & Memorization` 小节是合理的，因为它关注的是隐私保护数据分析基础设施，而不是模型行为 alignment。它和 LLM safety 的连接主要在 **private analytics/隐私统计接口**：如果未来要在用户数据、日志、偏好反馈或敏感语料上做安全审计和模型适配，unknown-domain DP discovery 是非常底层的工具。

## 2. Problem Setup

设 item universe 为可数集合 $\mathcal X$，数据集 $W=\{W_i\}_{i\in[n]}$，其中每个用户 $i$ 持有有限子集 $W_i\subset\mathcal X$。总 item 数为 $N=\sum_i |W_i|$，unique item 数为 $M=|\bigcup_i W_i|$。对任意 item $x$，其频数为

$$
N(x)=\sum_i \mathbf 1\{x\in W_i\}.
$$

算法输出集合 $S\subseteq \bigcup_i W_i$，论文要求 **soundness/真实性约束**：算法只能输出输入中真实出现过的 item。这个约束很自然，因为 unknown domain setting 下不能凭空释放不存在的字符串；但它也导致 singleton hard instance：如果每个用户都有唯一 item，那么 DP 迫使每个 item 的输出概率至多约为 $\delta$，missing mass 基本不可能显著降低。

论文因此定义 missing mass：

$$
MM(W,S)=\sum_{x\in \bigcup_i W_i\setminus S}\frac{N(x)}{N}.
$$

更一般地，作者把未恢复频率向量 $(N(x)/N)_{x\notin S}$ 的 $\ell_p$ norm 记作 $MM_p(W,S)$。普通 missing mass 是 $\ell_1$，$\ell_\infty$ 则表示漏掉 item 中最大的单个频率质量。$\ell_0$ 可以看成传统 cardinality 目标，因此 missing mass 不是随意换指标，而是在同一族目标里把关注点从“尾部数量”转到“分布质量”。

为了获得 $\ell_1$ missing mass guarantee，论文引入 **Zipfian dataset** 假设。把 item frequency 降序记为 $N_{(r)}$，若存在 $C\ge 1,s>1$ 使得

$$
N_{(r)}\le \frac{CN}{r^s},
$$

则数据是 $(C,s)$-Zipfian。$s$ 越大，质量越集中在头部；$C$ 越小，频率衰减越强。这个假设反映很多自然数据的长尾结构，同时排除了所有 item 都只出现一次的极端硬例。重要的是，DP guarantee 仍然是 worst-case neighboring datasets 上成立；Zipfian 只用于 utility analysis。

## 3. Algorithm / Methods / Model

核心机制是 **Weighted Gaussian Mechanism/WGM**。它有三个参数：Gaussian noise level $\sigma$、threshold $T$、user contribution bound $\Delta_0$。算法先对每个用户的 $W_i$ 做无放回 subsampling，使其最多贡献 $\Delta_0$ 个 item；然后构建 weighted histogram，每个用户的贡献按 $1/\sqrt{|\widetilde W_i|}$ 缩放，保证 user-level $\ell_2$ sensitivity 受控；接着给每个非零 item 的 weighted count 加 $N(0,\sigma^2)$ 噪声；最后只输出 noisy weighted count 超过 $T$ 的 item。

这个设计的隐私逻辑很直接。用户可能贡献多个 item，所以不能像 item-level histogram 那样让每个出现都加一票；用 $\ell_2$ 归一化后，一个用户整个 item set 对 weighted histogram 的向量影响被压住，Gaussian mechanism 和 thresholding 就能给 user-level approximate DP。Gopi et al. 的定理给出 $\sigma,T$ 满足某些条件时，WGM 是 $(\epsilon,\delta)$-DP；论文进一步使用最小量级

$$
\sigma=\Theta\left(\frac{1}{\epsilon}\sqrt{\log(1/\delta)}\right),\qquad
T=\widetilde\Theta_{\Delta_0,\delta}(\max\{\sigma,1\}).
$$

这个机制看起来朴素，但三个步骤各自对应一个不可省的困难。subsampling 处理的是用户贡献过多导致的 sensitivity；weighted histogram 处理的是 user-level 而非 item-level privacy；thresholding 处理的是 soundness 和 unknown domain 下不能给所有不存在 item 加噪的问题。若没有 thresholding，算法需要在整个 $\mathcal X$ 上输出 noisy counts，这在无限或巨大 domain 中不可行；若没有 weighting，高 item-count 用户会支配 sensitivity；若没有 subsampling，极端用户的 item set 会让噪声尺度过大。论文的理论分析正是把这三个误差来源拆开看。

> [!note] Theorem 3.3
> 对任意 $(C,s)$-Zipfian dataset，$s>1$，WGM 的 $\ell_1$ missing mass 以高概率满足
>
> $$
> MM(W,S)=\widetilde O_{\beta,C,N}\left(
> \frac{C^{1/s}}{s-1}
> \left(\frac{\max_i |W_i|}{N\sqrt{q^\star}}\right)^{(s-1)/s}
> (T+\sigma)^{(s-1)/s}
> \right),
> $$
>
> 其中 $q^\star=\min\{\max_i|W_i|,\Delta_0\}$。直观上，$N$ 越大、头部质量越集中、$\Delta_0$ 越接近真实最大 user set size，missing mass 越低。

这个定理还解释了 $\Delta_0$ 的敏感性。如果 $\Delta_0<\max_i|W_i|$，subsampling 本身会丢质量；如果 $\Delta_0$ 远大于实际 user set size，privacy threshold 的 log factors 变差。论文建议在有公共先验时取 $\Delta_0=\max_i|W_i|$，但 reviewer 也抓住了这里的现实难点：真实系统未必知道这个值，私有估计 $\Delta_0$ 本身又会消耗隐私预算。

作者还给出匹配意义上的 lower bound。任意满足 soundness 的 $(\epsilon,\delta)$-DP 算法，在某个 Zipfian dataset 上都会有至少同量级的 missing mass，说明 WGM 在 $\epsilon$ 和 $N$ 的依赖上不能被根本性改进。更重要的是，论文证明了一个无需 Zipfian 假设的 $\ell_\infty$ bound：

$$
MM_\infty(W,S)\le \widetilde O_{\Delta_0,\delta,\beta}\left(
\frac{\max_i |W_i|}{\epsilon N\sqrt{q^\star}}
\right).
$$

$\ell_\infty$ bound 的用途是连接下游任务。作者提出一个 meta algorithm：先用一半 privacy budget 跑 WGM 得到 domain $D$，再用另一半 budget 在 $D$ 上跑 known-domain private algorithm。对 top-k，第二步使用 peeling exponential mechanism/Gumbel noise，得到 top-k missing mass 上界，包含 domain discovery 误差和 top-k private selection 误差。对 $k$-hitting set，第二步使用 private greedy submodular maximization，得到 $(1-1/e)$ 近似加 additive error。这里的共同结构是：**WGM 不直接解决所有任务，而是把 unknown domain 问题变成一个质量足够好的 known-domain problem**。

证明结构也值得记住。$\ell_1$ upper bound 不是单纯靠 Gaussian concentration。作者先控制 subsampling 后损失的质量，再证明原数据里的高频 item 在 subsampled dataset 中仍保持足够高的 weighted count，最后控制 thresholding 阶段高频 item 被 Gaussian noise 打到阈值以下的概率。Zipfian 假设只在把“漏掉的 item 频率上界”转成“总 missing mass 上界”时真正发挥作用：如果频率尾部衰减够快，那么所有低频未恢复项加起来的质量可控。$\ell_\infty$ bound 不需要 Zipfian，因为它只问漏掉 item 中最大的单个质量，不需要对整个尾部求和。

top-k 和 hitting set 的结果可以看成同一个范式的两种投影。top-k 关心的是头部 item 的 cumulative mass，因此 WGM 只要不漏掉过重 item，后续 Gumbel peeling 就能在发现的 domain 中完成 private ranking。$k$-hitting set 更偏 coverage，它要最大化被输出 item 命中的用户数；WGM 的作用是先把候选 domain 缩到一个包含高价值 item 的私有集合，再让 submodular greedy 在这个集合里运行。这个范式的实际意义是，很多 known-domain DP algorithm 不必从零重写，只要前面接一个有 guarantee 的 domain discovery layer。

## 4. Experiments

实验覆盖六个真实数据集。Reddit、Amazon Games、Movie Reviews 被视为 large datasets，Steam Games、Amazon Magazine、Amazon Pantry 被视为 small datasets。主实验使用 $(1,10^{-5})$-DP，appendix 还给出更严格的 $(0.1,10^{-5})$-DP，整体趋势类似。

set union 实验把 WGM 和两个更复杂的 policy mechanisms 比较。结果显示，WGM 的 missing mass 通常只比 policy mechanisms 差 5% 以内，而后者计算明显更重。这个结果很有意思，因为过去若看 cardinality，sequential policy methods 往往能输出显著更多 items；但换成 missing mass 后，简单 WGM 已经抓住了大部分高频质量。这正好支持论文的核心观点：cardinality 目标会过度奖励长尾 item，而 missing mass 更贴近高质量 domain discovery。

top-k 实验主要在 small datasets 上比较，因为 large datasets 的 top-k mass 太集中，所有方法几乎都能拿到接近 0 的 missing mass。WGM-then-top-k consistently 优于 Durfee & Rogers 的 limited-domain baseline，并且优势随 $k$ 增大更明显。$k$-hitting set 没有完全对应的 unknown-domain private baseline，作者把非私有 greedy 和假设 union domain 公开的 known-domain private greedy 作为参照；WGM-based method 在 small datasets 上表现接近甚至超过这些不完全公平的强 baseline。

这些实验说明 WGM-based pipeline 很实用，但还没有完全封口理论叙事。Reviewer 要求 synthetic Zipfian data、真实数据频率分布、$|W_i|$ 分布和 $\Delta_0$ sensitivity 的更系统展示，这个要求是合理的。因为论文的理论最强点在 Zipfian missing mass 上，实验如果能直接展示 real datasets 是否满足 $s>1$、不同 $\Delta_0$ 与 user set size 的关系，就能更清楚地连接 theorem 和 empirical performance。

还有一个实验解读细节：WGM 在 missing mass 上接近 policy mechanisms，并不意味着 policy mechanisms 没价值。更可能的解释是，policy mechanisms 多恢复的那些 item 位于低频尾部，对 $\ell_1$ mass 贡献较小；如果下游任务关心 rare item recall，它们也许仍有优势。论文选择 missing mass 是有明确统计立场的：它认为 domain discovery 的主要 utility 来自恢复高质量头部，而不是尽可能多地覆盖唯一字符串。这个立场适合很多 aggregate analytics，但不适合所有安全任务。

## 5. Reviewer Discussion

OpenReview 最终决定是 **Accept (Oral)**。四个 reviewer 的分数是 8、6、6、6，AC 总结为整体正面，认为作者在 rebuttal 中解决了大多数理论和实验澄清问题。正面评价集中在三个点：第一，private set union 长期缺少 absolute utility guarantees，而 missing mass reformulation 让 WGM 能得到可解释的上界；第二，$\ell_1$ Zipfian bound、distribution-free $\ell_\infty$ bound 与 lower bound 共同形成比较完整的理论包；第三，把 domain discovery 保证传递到 unknown-domain top-k 和 $k$-hitting set 很自然，说明这不是孤立指标游戏。

批评也很具体。最重要的是 $\Delta_0$：算法性能依赖 user contribution cap，但这个 cap 理想上要接近 $\max_i|W_i|$，真实部署中未必公开可得。第二是理论 gap：top-k 和 hitting set 的 upper/lower bounds 仍不匹配，尤其 $k^{3/2}/\epsilon$ 这类项是否 tight，需要更仔细处理。第三是 Zipfian assumption 的外部有效性：如果真实数据没有 $s>1$ 的头部集中结构，$\ell_1$ guarantee 的解释力会下降。第四是实验深度：reviewer 希望看到 synthetic Zipfian 验证、real frequency plots、更多 baseline 和更细的 $\Delta_0$ 分析。

我的客观评述是：这篇 oral 的贡献很扎实，但它强在 **problem reframing + theory for a simple deployed mechanism**，不是强在提出一个复杂新算法。它真正解决的是“我们应该用什么 utility 目标理解 DP unknown-domain discovery”，而不是“WGM 永远最优”。missing mass 是一个很聪明的指标选择，因为它直接承认 DP 对长尾 singleton 的不可恢复性，并把目标转向恢复统计上重要的质量。这个选择很务实，也很适合工业隐私分析。

不过读这篇时不能忽略指标风险。missing mass 对下游平均统计很合理，但某些任务确实关心 rare items，例如罕见疾病、异常攻击、低频危险 query。对这些任务，漏掉低频 item 不一定无害。论文没有声称 missing mass 覆盖所有 domain discovery utility，因此后续使用时要先问下游目标是否真的按 empirical frequency 加权。如果目标是安全异常检测，$\ell_\infty$ 或 worst-case recall 可能比 $\ell_1$ missing mass 更重要。

Reviewer 对 related work 的追问也有价值。private top-k selection 里已有 tight lower bounds 和 peeling/exponential mechanism 结果；本文不是重新发明这些 known-domain 算法，而是把 unknown-domain preprocessing 的损失写进最终 guarantee。这个 distinction 很重要，否则容易误解为论文声称 top-k 部分本身有根本新算法。它的新意主要在于把 WGM 的 $MM_\infty$ 保证接到这些经典机制上，让 unknown-domain setting 不再只依赖 heuristic domain truncation。

更进一步说，这篇论文最适合被当成“隐私系统设计中的目标函数论文”来读。它提醒我们，在差分隐私里，算法设计经常被噪声尺度支配；如果目标函数要求恢复所有 unique items，那么理论上必然被低频尾部拖垮，工程上也会鼓励算法去追逐对下游统计意义很小的 item。missing mass 把目标改成恢复经验分布的主要质量，实际上是在 privacy feasibility 和 utility relevance 之间重新对齐。这个重新对齐本身就是贡献。

## 6. Related Work & Future Work

这篇和早期 DP partition selection、private set union、top-k selection 关系紧密。已有工作常常给相对比较或 known-domain guarantee，而这篇用 missing mass 给 WGM 绝对 utility。它也和 adaptive weighting/sequential policy mechanisms 形成互补：那些方法可能在 cardinality 上更强，但计算更重；WGM 简单、并行、可作为下游算法前置步骤。

论文自己提出的未来方向主要有两个。一个是闭合 top-k 与 $k$-hitting set 的上下界 gap，弄清 $k$ 和 $\epsilon$ 的最优依赖。另一个是改进 subsampling。当前 WGM 用 uniform subsampling 来满足 $\ell_0$ bound，但更 data-dependent 的私有 subsampling 也许能在不牺牲隐私的前提下保留更多质量。这里的难点是，subsampling policy 本身可能泄露用户 item set 结构，所以不能直接拿非私有 adaptive rule。

从 alignment repo 的阅读角度看，我还会补一个方向：把 DP domain discovery 和 LLM feedback/data auditing 接起来。很多 alignment 数据不是结构化表格，而是 prompt、response、user feedback、red-team transcript、agent log。若要在这些敏感文本里发现高频模式、危险类别或偏好主题，同时不暴露单个用户，unknown-domain DP discovery 会非常关键。

不过，这条路线还需要把 item definition 做实。一个 item 可以是完整 prompt、n-gram、topic label、embedding cluster、tool action template，也可以是某类 policy violation pattern。不同 itemization 会改变 Zipfian 性质、$\Delta_0$、missing mass 的解释和下游 usefulness。论文没有处理文本 itemization，但它提供了一个清楚的隐私层：只要你能把敏感记录变成用户持有的有限 item set，WGM 就能作为第一步 domain discovery。后续安全应用的关键不在 WGM 公式本身，而在如何定义对 alignment 审计真正有意义的 item。
