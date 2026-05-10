---
title: "Gaussian Unlearning"
headline: "Gaussian Certified Unlearning in High Dimensions: A Hypothesis Testing Approach"
description: "ICLR 2026 Oral note on Gaussian certifiability for high-dimensional machine unlearning and one-step noisy Newton guarantees."
status: complete
visibility: public
topic: "safety_alignment"
venue: "ICLR 2026 Oral"
year: 2026
paper_url: "https://arxiv.org/abs/2510.13094"
openreview_url: "https://openreview.net/forum?id=0FJYicpOj0"
source_pdf: "raw/papers/arxiv-2510.13094.pdf"
tags: ["machine_unlearning", "gaussian_privacy", "certified_privacy"]
created: "2026-05-01"
updated: "2026-05-01"
---

> [!abstract] Contributions
>
> 这篇论文研究 **machine unlearning/机器遗忘** 在高维比例极限 $p\sim n$ 下的理论保证。作者提出 **$(\phi,\epsilon)$-Gaussian certifiability/GPAR**，用 hypothesis testing trade-off curve 来定义 unlearning 输出和完全 retraining 输出之间的不可区分性。核心主张是：在高维设置中，Gaussian noise mechanism 对应的 Gaussian trade-off curve 是更自然、更紧的 certifiability notion；如果沿用传统 $\epsilon$-certifiability，会迫使算法加过大的 Laplace-like noise，从而错误地得出“至少需要两步 Newton”的结论。
>
> 论文证明，在 convex regularized ERM 和 sub-Gaussian features 等假设下，一个 **one-step Newton unlearning update** 加上校准 Gaussian noise，就能同时达到 $(\phi_n,\epsilon)$-GPAR 和 vanishing generalization error divergence/GED。这个结论直接挑战 Zou et al. 在高维 unlearning 中“至少两步”的结论。边界也很硬：理论适用于 GLM/convex high-dimensional statistics，不覆盖现代非凸深度网络；删除集合规模需要相对小，典型条件是 $m=o(n^{1/4-\alpha})$；Newton step 需要 Hessian inverse，真实大模型上计算代价很高。

## 1. Introduction

machine unlearning 的目标是：当用户要求删除数据时，不必从头重训模型，却能产出一个和“删掉这些数据后重新训练”足够接近的模型。这个问题同时有 privacy 和 utility 两个面向。privacy 要求 unlearned model 不泄露被删数据的残余影响；utility 要求模型性能接近理想 retrained model，而不是直接输出随机噪声。

已有理论很多在低维设置下成立，隐含要求参数维度 $p$ 远小于样本数 $n$，并且 per-example loss 同时具有 $\Omega(1)$ strong convexity 和 $O(1)$ smoothness。但高维比例 regime 中，$p$ 和 $n$ 同阶，这些假设很容易崩。论文用 ridge least squares 解释得很清楚：若为了 smoothness 让 feature norm 缩放到合理量级，per-example loss 的 strong convexity 会退化到 $\lambda/n$ 量级，而不是常数。因此低维 unlearning 理论不能直接解释现代高维统计模型。

这篇论文的更深层动机是：高维 unlearning 的困难不一定来自 Newton algorithm 本身，也可能来自 **certifiability notion 选错了**。如果隐私证明采用和 Gaussian noise 不匹配的 $\epsilon$-certifiability，就需要加过大的噪声，导致 one-step estimator 看起来不够准确。作者认为，在高维中，很多 isotropic noise mechanism 的一维投影近似 Gaussian，Gaussian differential privacy 的 trade-off curve 才是更 canonical 的目标。

因此论文的贡献不是提出一个全新工程 unlearning system，而是重写理论坐标系：先定义更适合高维噪声机制的 privacy/certifiability，再重新分析同一个 Newton-based approximate unlearning algorithm。这个路线对 `Privacy & Memorization` 小节很重要，因为它直接关系到“可证明删除训练数据影响”到底该用什么统计标准衡量。

这篇的关键判断可以压成一句话：**unlearning guarantee 不是只由算法决定，也由检验两个输出分布的标尺决定**。如果标尺本身对高维 Gaussian perturbation 不友好，那么一个实际自然的噪声机制会被判得过严；为了满足这个过严标准，算法必须加更多噪声，最后 accuracy 下降。作者认为这正是旧结论中“两步 Newton”出现的原因之一。

## 2. Problem Setup

设训练数据为 $D_n=\{(x_i,y_i)\}_{i=1}^n$，学习算法 $A(D_n)$ 输出参数 $\hat\beta$。删除请求集合为 $M\subset[n]$，理想 retrained estimator 是

$$
\hat\beta_{\backslash M}=A(D_n\backslash D_M).
$$

unlearning algorithm $\bar A$ 拿到原模型 $\hat\beta$、待删除数据 $D_M$、以及一些辅助信息 $T(D_n)$，输出随机化后的 unlearned parameter。随机化是必要的，因为 deterministic approximation 和 true retraining 之间的差异可能携带被删数据的信息。

论文用 hypothesis testing 来定义 privacy。攻击者看到 unlearning 输出 $\check\beta$，要区分两个假设：

$$
H_0:\check\beta\sim P_{\mathrm{re}},\qquad
H_1:\check\beta\sim P_{\mathrm{un}}.
$$

$P_{\mathrm{re}}$ 是从 retrained model 出发再执行空删除过程的分布，$P_{\mathrm{un}}$ 是从原模型出发执行真实删除过程的分布。如果攻击者能拒绝 $H_0$，说明输出中仍有被删数据影响。于是 certifiability 可以写成 trade-off function $T(P,Q)(\alpha)$：在 type-I error 不超过 $\alpha$ 时，最小 type-II error 是多少。越难区分，trade-off curve 越高。

论文先定义一般 $f$-certifiability：若 $T(P_{\mathrm{re}},P_{\mathrm{un}})$ 和反向 trade-off 都不低于某个曲线 $f$，则算法可认证。传统 $(\epsilon,\delta)$-style privacy 对应一类 piecewise linear $f_{\epsilon,\delta}$。作者的选择是 Gaussian trade-off curve：

$$
f_{G,\epsilon}(\alpha)=T(N(0,1),N(\epsilon,1))(\alpha)
=\Phi(\Phi^{-1}(1-\alpha)-\epsilon).
$$

> [!note] Dimension-Freeness
> 若两个高维 Gaussian 只差均值，$P=N(\mu_1,\sigma^2I_p)$，$Q=N(\mu_2,\sigma^2I_p)$，那么它们的 trade-off function 等价于一维 $N(0,1)$ vs. $N(\epsilon,1)$，其中 $\epsilon=\|\mu_1-\mu_2\|_2/\sigma$。这说明 Gaussian certifiability 可以把高维均值差异压缩成一个 dimension-free privacy parameter。

utility 方面，论文使用 **Generalization Error Divergence/GED**。它比较 unlearned estimator 和 retrained estimator 在新样本 $(x_0,y_0)$ 上的 loss 差异：

$$
GED_\ell(A,\bar A;M,D_n)
=\mathbb E\left[\ell(y_0|x_0^\top A(D_{\backslash M}))
-\ell(y_0|x_0^\top \bar A(A(D_n),D_M,T(D_n),b))\mid D_n\right].
$$

这个指标比直接对参数距离或 excess risk 更贴近 unlearning 目标：我们关心的不是参数逐坐标完全一致，而是删除后模型在泛化行为上接近 retraining。

GED 的选择也避开了高维里的另一个陷阱。在 $p\sim n$ 时，参数估计本身可能不一致，参数距离不一定是最合适的 accuracy proxy；两个 estimator 参数差异明显，但在新样本上的预测风险差异可能很小。反过来，如果只比较 training loss，也可能掩盖被删样本上的残留影响。因此论文还在实验中看 unlearned error divergence/UED，用来检查删除集合上的行为。这种区分很有必要，因为 unlearning 的 privacy 和 utility 分别对应“攻击者能否识别残留”和“模型对新数据是否仍可用”。

## 3. Algorithm / Methods / Model

训练问题是 regularized empirical risk minimization/RERM：

$$
\hat\beta=\arg\min_{\beta\in\mathbb R^p}\left[
\lambda r(\beta)+\sum_{i=1}^n \ell(y_i|x_i^\top\beta)
\right].
$$

删除 $M$ 后，理想模型是在剩余样本上重新最小化同一目标。为了避免完整 retraining，算法从原始 $\hat\beta$ 出发，对删除后的 objective $L_{\backslash M}$ 做一次 Newton step：

$$
\hat\beta_{\backslash M}^{(1)}
=\hat\beta-G(L_{\backslash M})^{-1}(\hat\beta)\nabla L_{\backslash M}(\hat\beta),
$$

其中 $G(L_{\backslash M})$ 是 Hessian。然后加入 Gaussian noise：

$$
\tilde\beta_{\backslash M}=\hat\beta_{\backslash M}^{(1)}+b,\qquad
b\sim N(0,\sigma^2I_p).
$$

论文的技术假设包括：regularizer separable 且 strongly convex，loss 与 regularizer 三阶可微并满足 polynomial growth，features 是 mean-zero sub-Gaussian with bounded covariance，responses 有 sub-polylog tail。相比低维文献的 per-example $\Omega(1)$ strong convexity 和 $O(1)$ smoothness，这些假设更适合 $p\sim n$ 的高维统计模型，但仍然是 convex GLM 级别，不是深度网络级别。

> [!note] Main Guarantees
> Theorem 2 说明，若噪声尺度按
>
> $$
> b\sim N\left(0,\frac{r^2}{\epsilon^2}I_p\right),\qquad
> r=C_1(n)\sqrt{\frac{C_2(n)m^3}{2\lambda\nu n}},
> $$
>
> 选择，其中 $C_1(n),C_2(n)$ 为 polylog factors，则 one-step noisy Newton unlearning 达到 $(\phi_n,\epsilon)$-GPAR，且 $\phi_n\to 0$。Theorem 3 进一步给出 GED bound，并推出只要
>
> $$
> \frac{m^2\operatorname{polylog}(n)}{\sqrt n}\to 0,
> $$
>
> unlearning estimator 的 GED 就会消失。特别地，若 $m=o(n^{1/4-\alpha})$，则 GED $=o_p(1)$。

这个结论和 Zou et al. 的差别来自 certifiability notion。Zou et al. 在 $\epsilon$-certifiability 下需要更大的 Laplace-style noise，因此 one-step Newton 的 GED 不消失，必须做至少两步。本文认为这不是算法基本限制，而是 privacy notion 与 Gaussian noise mechanism 不匹配造成的分析损失。换成 GPAR 后，Gaussian noise 是 tight 的，一步 Newton 就足够。

这里的“tight”需要按统计检验理解。Gaussian trade-off curve 正好描述两个同方差 Gaussian 分布在最优 hypothesis test 下的 type-I/type-II trade-off；当 unlearning 输出主要差异表现为均值偏移加 isotropic Gaussian noise 时，GPAR 就能精确捕捉攻击难度。传统 $(\epsilon,\delta)$ 或相关 certifiability 是更粗的外包络，对 Gaussian mechanism 来说可能保守。保守定义不是错误，但如果目标是分析 Gaussian-noise algorithm 的最优 privacy-accuracy tradeoff，它会把可实现的精度牺牲掉。

算法层面也要注意，one-step Newton 的可行性来自局部二阶近似。删除少量样本时，retrained optimum $\hat\beta_{\backslash M}$ 应该靠近原始 optimum $\hat\beta$，因此用删除后 objective 的 Hessian 和 gradient 做一次 correction 可以近似 retraining。若删除集合很大，或者删除数据具有强结构性，例如整个类别、整个用户群、某个概念簇，这个局部近似可能失效。这正是 $m$ 条件和 reviewer 对 structured unlearning 的担忧所在。

## 4. Experiments

实验主要是 simulation，用来验证理论趋势，而不是证明真实深度模型可部署。作者使用 ridge-penalized logistic regression，features $x_i\sim N(0,\frac1n I_p)$，true parameter $\beta^\star\sim N(0,I_p)$，并取 $n=p$ 表示比例高维。比较对象是本文 Gaussian-Newton estimator 和 Zou et al. 的 Laplace-perturbed estimator。

第一组实验固定 $\epsilon=0.75$，让 $p=n$ 从 500 增到 5000，删除规模 $m=1,5,10$。结果显示 Laplace estimator 的 GED 基本不随 $p$ 消失，log-log slope 分别约为 0.03、-0.03、-0.01；Gaussian estimator 的 GED 稳定下降，slope 约为 -0.47、-0.54、-0.51，接近理论预期的 $p^{-1/2}$。在 unlearned data 上的 UED 也呈现相同趋势。

第二组固定 $n=p=1255$，改变 $\epsilon$。Gaussian estimator 随 $\epsilon$ 增大更接近 ideal retrained estimator，且在所有 $\epsilon$ 下都比 Laplace estimator GED 更低。这说明优势不是只在某个 privacy budget 下偶然出现，而是来自 Gaussian certifiability 与 Gaussian noise 的结构匹配。

第三组固定 $\epsilon=0.75$，改变删除集合大小 $m$。随着 $m$ 增大，两种方法 GED 都上升，但 Laplace 始终更高。Gaussian 的 log-log slope 大约 1.37 到 1.44，接近但略低于 theorem 中 $m^{1.5}$ 的趋势，作者认为实践依赖可能比理论 bound 更好。

实验的弱点同样明显。所有主实验都是 synthetic Gaussian/sub-Gaussian features 上的 logistic 或 ridge setting，没有真实数据、非凸网络、representation learning 或 LLM fine-tuning。它们很好地验证了理论现象：在高维 GLM setting 下，Gaussian certifiability 比旧 notion 更贴合 one-step Newton；但不能推出“现代 LLM unlearning 只需一次 Newton step 加 Gaussian noise”。

实验还有一个值得读的细节：Gaussian 方法在 $m$ 上的 empirical slope 约为 $1.4$，理论给出的依赖更保守。这说明 proof 里的多项式和 polylog factors 可能没有完全 tight。对理论论文来说，这不是坏事，因为 theorem 更关心可证明收敛；但对实践来说，常数和 finite-sample behavior 才决定噪声能否使用。Reviewer 要求更大 $p$、更多模型和明确常数，本质上都是在问同一个问题：这个漂亮的 asymptotic story 在有限样本中能不能变成可操作的 certificate。

## 5. Reviewer Discussion

OpenReview 最终决定是 **Accept (Oral)**。reviewer 分数大致呈现三正一弱：8、4、6、6，AC 认为 rebuttal 后多数 concerns 已被新增实验、澄清和文字修改解决。正面评价集中在理论框架：reviewer 认可 Gaussian certifiability 把 DP-inspired hypothesis testing 和 certified unlearning 接起来，也认可它解释了为什么旧框架在高维中会过度惩罚 noise-adding mechanisms。one-step Newton under GPAR 是一个清晰且有冲击力的结果。

主要担忧也都很实质。第一是 **deletion size**，理论条件只允许 relatively small deletion batches，例如 $m=o(n^{1/4-\alpha})$，这限制了大批量删除请求。第二是 **noise calibration**，定理中的 polylog factors 和未显式常数会影响真实实施，不能直接变成工程参数。第三是 **model scope**，假设集中在 convex GLM、sub-Gaussian features 和 separable regularization，离非凸深度网络很远。第四是 **Newton scalability**，Hessian 形成和求逆在大规模模型上可能不可行。第五是 sequential/online unlearning：如果删除请求持续到来，GED 是否累积、GPAR 如何组合，论文没有完整回答。

我的客观评述是：这篇论文强在 **certifiability definition 的重新校准**。它提醒我们，unlearning 理论的结论会强烈依赖 privacy notion；如果 notion 和噪声机制不匹配，算法会被迫加过量噪声，最后把定义缺陷误判为算法缺陷。这个批评很有价值，也解释了为什么同样的 Newton idea 在不同 certifiability 下会得出不同 step complexity。

但这篇离现实 LLM unlearning 仍有很长距离。现代 LLM 的参数维度远大于样本数，目标非凸，Hessian 不可直接求逆，删除对象可能是概念、类别、文档簇或 copyrighted corpus，而不是少量 i.i.d. GLM samples。GPAR 作为统计定义有启发性，但工程系统还需要 approximate Hessian、low-rank update、stochastic Newton、influence approximation 或 representation-level deletion 的新理论。Reviewer 对扩展性的质疑是完全正当的。

还有一个容易被忽略的点：GPAR 保护的是 unlearning output 与 retraining output 的分布不可区分，而不是保证原训练模型从未记住数据，也不是保证删除后模型不能回答任何关于被删数据的问题。如果 retraining on $D_{\backslash M}$ 本身仍能从相关样本或公共数据中推断出同一事实，unlearning certificate 不会阻止这种“合法可推断”。这和 GDPR-style deletion、copyright unlearning、safety capability removal 之间仍有语义差距。理论 certificate 解决的是统计残留影响，不自动解决社会或法律意义上的所有遗忘要求。

Reviewer 对 online unlearning 的追问尤其关键。真实系统不会只收到一次删除请求；用户请求会不断到达，模型也可能在持续训练。如果每次都做 one-step Newton 加噪，噪声和 approximation error 会如何累积？如果把多个请求攒成 batch，等待时间又会影响用户权利和系统可用性。论文当前的 theorem 是 single-batch guarantee，不能直接推出长期服务中的 privacy composition 和 utility degradation。这个缺口对任何真实 unlearning deployment 都很核心。

## 6. Related Work & Future Work

论文直接回应 Guo、Sekhari、Allouah、Zou 等 certified unlearning 理论。低维工作表明 one-step Newton 加噪可行，但假设在 $p\sim n$ 下失效；Zou et al. 进入高维但使用旧 certifiability，得出至少两步的结论。本文的核心定位就是：保留高维 relaxed assumptions，同时把 certifiability 换成 Gaussian trade-off curve，从而恢复 one-step result。

未来工作有几条明确路线。理论上，需要把 $m=o(n^{1/4-\alpha})$ 的 deletion size 条件放宽，处理 sequential deletion 和 repeated unlearning 的 error composition。算法上，需要把 exact Hessian inverse 换成 quasi-Newton、stochastic approximation 或 low-rank sketch，并证明 GPAR 下仍成立。应用上，需要从 synthetic GLM 走向真实数据、kernel/feature learning models，再逐步接近 neural network setting。

还有一个值得单独强调的方向是 **certifiability calibration**。定理给出的是 asymptotic polylog bound，但实际 privacy certificate 需要可计算的 noise scale 和可审计的 failure probability。如果 GPAR 未来要用于监管或工程实现，必须把常数、finite-sample behavior 和 empirical trade-off testing 做实。

从理论路线看，最自然的下一步不是立刻跳到 LLM，而是先处理更接近现实但仍可分析的中间模型，例如 kernel ridge regression、random feature models、wide two-layer networks 或 low-rank adapters。它们比 GLM 更接近 representation learning，又比 full transformer 更容易控制 Hessian 和高维极限。如果 GPAR 在这些模型上仍能给出清楚 guarantee，才有资格进一步讨论 parameter-efficient LLM unlearning。

另一个值得做的是把 GPAR 和 empirical auditing 合并。既然 GPAR 本质上是 hypothesis testing trade-off，研究者可以在有限样本 simulation 中直接估计 retrained distribution 与 unlearned distribution 的 trade-off curve，检查它是否接近 Gaussian curve。这样可以把 theorem、noise calibration 和实际攻击连接起来，而不是只报告 GED。对安全读者来说，这种曲线比单个 privacy parameter 更直观，也更能发现某些方向上的残留泄漏。

和 Hubble 这类 memorization testbed 放在一起读时，这篇提供的是互补视角。Hubble 问“模型是否可被攻击抽出或识别训练样本”，更偏 empirical leakage；Gaussian certified unlearning 问“unlearning 输出和 retraining 输出在统计检验上是否不可区分”，更偏 formal certificate。真正成熟的 unlearning evaluation 应该两者都要：形式化 guarantee 给出可证明边界，经验攻击暴露 theorem 假设之外的失败模式。
