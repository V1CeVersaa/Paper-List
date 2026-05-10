---
title: "Steering the Herd"
headline: "Steering the Herd: A Framework for LLM-based Control of Social Learning"
description: "ICLR 2026 Oral note on controlled social learning, information-design planners, and LLM-mediated influence over public belief."
status: complete
visibility: public
topic: "safety_alignment"
venue: "ICLR 2026 Oral"
year: 2026
paper_url: "https://arxiv.org/abs/2504.02648"
openreview_url: "https://openreview.net/forum?id=RtS4UqSmNt"
source_pdf: "raw/papers/arxiv-2504.02648.pdf"
tags: ["information_design", "social_learning", "llm_mediation"]
created: "2026-05-01"
updated: "2026-05-01"
---

> [!abstract] Contributions
>
> 这篇论文建立了一个 **controlled sequential social learning/受控序列社会学习** 框架，用来研究 LLM 或算法推荐系统作为 information mediator 时，如何通过控制每个用户收到的信息精度来影响群体信念、个体行动和社会福利。它把传统 information cascade 模型和 dynamic programming 结合起来：agents 既接收 planner 提供的 private signal，又观察前面 agents 的行动；planner 不撒谎、不 cherry-pick，也不知道真实状态，只能选择 signal precision。
>
> 论文的理论贡献是刻画 **altruistic planner** 和 **biased planner** 的最优策略。Altruistic planner 的 value function 被证明对 public belief 凸，从而推出三阶段策略：极端信念处不投资，中间不确定处给 perfect signal，其余位置给刚好能让 agent 行动透露 private signal 的最低精度。Biased planner 则更危险：在某些 belief 区间会故意降低 signal precision，让 agents 忽略 private signal 并维持有利 cascade。LLM simulation 显示，LLM planner 的策略形状与理论最优相似，且 misaligned biased planner 可让 social welfare 下降约 40-50%；边界在于模型假设很强，主要是 binary state、symmetric signal、myopic short-lived agents 和 LLM-as-human-simulator。

## 1. Introduction

LLM 和推荐算法正在成为现实世界的信息中介。搜索排序、新闻 feed、广告投放、医疗/政治建议、产品推荐都不是单纯给用户“答案”，而是在影响用户看到什么证据、证据有多精确、以及用户如何根据别人行为更新信念。论文抓住的核心问题是：当算法能调节每个用户收到的信息质量，而用户又会观察前面用户的行动时，算法的影响会通过 **information externality/信息外部性** 放大。

经典 social learning 里，每个 agent 都有 private signal，也能看见前面 agents 的 action。如果 public belief 已经很强，后来的 agent 可能忽略自己的 private signal，跟随群体行动，形成 information cascade。论文的新点是加入一个 planner。这个 planner 不直接控制用户行动，也不能谎报状态；它只能选择 private signal 的 precision。这个限制很重要，因为它让模型更接近“算法选择给用户多清楚的信息”，而不是“平台直接欺骗用户”。

alignment 视角在这里非常自然。Altruistic planner 的目标是让 agent 尽可能选对行动，所以它与用户 expected utility 对齐。Biased planner 的目标是诱导某个特定行动 $G$，不管真实状态是好是坏；当真实状态恰好支持 $G$ 时，它和用户 realized utility 暂时一致，当真实状态是 $B$ 时，它就会损害用户。论文真正关心的是：即使 planner 不撒谎，只调节信息精度，它也能显著改变群体学习轨迹。

这篇的理论味道很重，但它对 AI safety 的意义不抽象。LLM-based information mediator 未来可能在药物、投资、投票、教育、消费选择中持续影响用户。如果系统目标函数偏向平台收益，它不一定需要生成虚假内容；只要在某些时刻提供更模糊或更清楚的信息，就可能推动 public belief 进入有利于平台的 cascade。这就是论文标题里的 steering the herd。

## 2. Problem Setup

环境有一个固定但未知的二元状态 $\omega\in\{G,B\}$，初始公共信念是 $b_1=P(\omega=G)$。第 $i$ 个 agent 看到历史 $H_i=(b_1,(q_j,a_j)_{j<i})$，也就是此前每个 agent 收到的 signal precision $q_j$ 和行动 $a_j$。Planner 在第 $i$ 轮选择 private signal precision $q_i\in[0.5,1]$，然后 agent 收到二元 signal $s_i\in\{G,B\}$，满足：

$$
P(s_i=\omega)=q_i
$$

Agent 的目标很简单：如果行动 $a_i$ 与真实状态相同，效用为 0；否则效用为 $-C$。所有 agents 都是 Bayesian 且 short-lived，只关心自己这一轮。公共信念 $b_i=P(\omega=G\mid H_i)$ 是历史的 sufficient statistic，因此整个过程可以写成 belief-state MDP。

Agent 收到 private signal 后，形成 posterior private belief $\tilde b_i$。当 signal 是 $G$ 时：

$$
\tilde b_i=\frac{q_i b_i}{1+2b_iq_i-b_i-q_i}
$$

当 signal 是 $B$ 时：

$$
\tilde b_i=\frac{(1-q_i)b_i}{b_i+q_i-2b_iq_i}
$$

根据这个 posterior，agent 在 $\tilde b_i>0.5$ 时选 $G$，在 $\tilde b_i<0.5$ 时选 $B$，相等时跟随自己的 signal。把阈值化简后，行动规则是：

$$
a_i=
\begin{cases}
s_i,&1-q_i\le b_i\le q_i\\
G,&q_i<b_i\\
B,&q_i<1-b_i
\end{cases}
$$

这个式子是整篇论文的核心。只有当 public belief 落在 $[1-q_i,q_i]$ 内，agent 的行动才会反映 private signal；否则 agent 会直接跟随 public belief，private signal 对 action 没有影响。于是公共信念更新为：

$$
b_{i+1}=f(b_i,q_i)=
\begin{cases}
\tilde b_i,&1-q_i\le b_i\le q_i\\
b_i,&\text{otherwise}
\end{cases}
$$

如果进入第二种情况，社会学习停止，形成 information cascade。Planner 的问题就是在每个 belief state 上选择 $q_i$，一边付出 precision cost，一边影响当前 agent 和未来 public belief。

这就解释了为什么 planner 控制 precision 会有长期效应。单轮看，precision 只是让当前 agent 更容易选对或选错；多轮看，precision 决定当前行动是否携带 private information，而这个行动又会成为所有后续 agents 的 public evidence。如果 planner 在关键轮次让行动携带信息，社会可能快速学习真实状态；如果 planner 在关键轮次让行动不再携带信息，社会就可能卡在一个 cascade 里。论文的动态规划正是在捕捉这个跨轮信息外部性。

论文考虑两类 planner。Altruistic planner 最大化社会福利减去提高 precision 的成本。它的即时 reward 是：

$$
r_A(b_i,q_i)=-\beta(q_i)-C\,P(a_i\ne\omega\mid b_i,q_i)
$$

Biased planner 想诱导 action $G$，不关心真实状态。它可以把 precision 提高或降低到 baseline $p$ 之外，成本是 $\beta(|q_i-p|)$，即时 reward 是：

$$
r_B(b_i,q_i)=-\beta(|q_i-p|)-C\,P(a_i=B\mid b_i,q_i)
$$

这两个问题都是 infinite-horizon discounted stationary MDP，state 是 public belief $b_i\in[0,1]$，control 是 precision $q_i\in[0.5,1]$，transition 是上面的 belief update。

## 3. Model And Theory

论文先分析 altruistic planner。Myopic case 相当于 $\delta=0$，planner 不关心当前行动对未来 belief 的影响。Theorem 1 说明 myopic altruistic policy 是 threshold policy：如果 public belief 足够接近 0.5，planner 给 perfect signal $q=1$；如果 public belief 很强，planner 只给 baseline precision $p$。阈值由 perfect signal 成本 $\beta(1)$ 和错误行动成本 $C$ 决定：

$$
t_M=
\begin{cases}
\beta(1)/C,&\beta(1)<C(1-p)\\
0.5,&\text{otherwise}
\end{cases}
$$

> [!note] Theorem 2: Convexity of the altruistic value function
>
> Altruistic optimal value function $V_A^*(\cdot)$ is convex with respect to public belief.

这个凸性结论是后续结构刻画的支点。直觉上，public belief 越极端，社会已经越确定，额外信息的边际价值下降；belief 越接近中间，信息更有价值。但证明并不只是套标准 Bayesian optimal control 结果，因为 agent action 本身依赖 public belief，一旦进入 cascade，belief transition 会停住。作者在 appendix 中通过有限 horizon induction 处理这个问题：先证明 fixed precision 下即时 reward 对 belief 凸，再利用 Bayesian update 的 martingale 性质和子树期望 reward 的凸性，把一层层 belief tree 的 convexity 往前推，最后转到 infinite horizon。

> [!todo]- Proof Sketch
>
> 证明目标是对任意 $x_0=tm_0+(1-t)n_0$，建立 $V_A^*(x_0)\le tV_A^*(m_0)+(1-t)V_A^*(n_0)$。困难在于 action rule 会随 belief 改变，所以不能简单说 posterior expectation 线性。论文的处理方式是考虑从不同初始 belief 出发、在同一 precision policy 下生成的 belief tree；由于 Bayesian posterior 满足 $\mathbb{E}[b_{i+1}\mid b_i]=b_i$，下一层 belief 的期望保持当前 belief。再结合即时 reward 的 convexity，可以把某个 node 的 expected continuation reward 用两端 belief 的 continuation reward 上界住。有限 horizon 上归纳成立后，再用 discounted limit 得到 $V_A^*$ 的 convexity。

Theorem 3 给出 altruistic optimal policy 的三阶段结构。存在 $0<d_A\le t_A\le t_M\le0.5$，使得：

$$
\pi_A^*(b)=
\begin{cases}
p,&b\in[0,d_A)\cup(1-d_A,1]\\
1,&b\in(t_A,1-t_A)\\
\max(b,1-b),&\text{otherwise}
\end{cases}
$$

这个结构很漂亮。极端 public belief 下，planner 不投资，因为信息不太可能改变行动。接近 0.5 时，planner 给 perfect signal，让 agent 行动完全对应真实状态，public belief 迅速坍缩到 0 或 1。中间区域最有意思：planner 选择 $\max(b,1-b)$，这是让 agent action 刚好还会反映 private signal 的最低 precision。也就是说，altruistic planner 不是一味提高信号质量，而是用最低成本维持社会学习不断流。

这个结果也说明，alignment 里的“给用户更多信息”不是单调简单的口号。Altruistic planner 的目标不是每轮都最大化 precision，而是在正确位置购买信息。public belief 已经很强时继续投入只会浪费成本；belief 处在 cascade 边缘时，最低可用 precision 反而最有效，因为它既保持 agent action 的可观察信息含量，又避免过度花费。这种 **precision scheduling/精度调度** 比“透明越多越好”更细。

Biased planner 更复杂，也更安全相关。它想诱导 $G$，所以当 public belief 不利于 $G$ 时，它可能提高 precision 争取出现有利 signal；当 public belief 略微有利于 $G$ 时，它可能降低 precision，避免 private signal 推翻当前有利 belief。Theorem 4 先刻画 myopic biased policy，Theorem 5 再说明动态最优有五个区间。

> [!note] Theorem 5: Structure of optimal biased policy
>
> 存在 $t_1,t_2\in[0,p]$，$t_1\le1-p\le0.5<t_2<p$。在极端不利或已经足够有利的 belief 上，biased planner 选 baseline $p$；在接近不利 cascade 时提高 precision；在 $b\in(1-p,0.5)$ 时可选择最低能影响 action 的 precision；在 $b\in[0.5,t_2)$ 时提高 precision 来巩固有利 belief；在 $b\in(t_2,p]$ 时，$\epsilon$-optimal policy 会选 $b-\epsilon$，故意让 agent 忽略 private signal 并跟随 action $G$。

最关键的是最后一种情况。若 public belief 已经略微支持 $G$，biased planner 可能不希望 agent 看到太精确的 private signal，因为真实状态可能是 $B$，精确信息会把社会拉回正确方向。于是它把 precision 降到 $b-\epsilon$，让 agent 的行动不再反映 private signal，public belief 也不更新。这是 **strategic obfuscation/策略性模糊化**：planner 没有说谎，只是选择低精度信号来保护已有 cascade。

## 4. Experiments

论文的实验不是拿真实人类做大规模社会实验，而是用 LLM 扮演 agent、planner 和 oracle，构造一个“是否购买新车”的场景。Agent 看到过去 actions 和一个关于汽车质量的 private message，然后报告 belief 并做 buy/not buy 决策。Planner 根据历史选择下一个 message precision。Oracle 根据 planner 选择的 precision 和 fact sheet 生成符合该精度的 private message。

实验第一步检查 LLM agent 是否像 Bayesian agent 一样更新 belief。结果显示三个偏差：LLM 对符合 prior 的 private signal 反应不足，对违背 prior 的 private signal 反应过度，因此需要更强 public belief 才会进入 cascade。论文把这些记为 NB1、NB2、NB3。这一步很重要，因为理论假设是 Bayesian agents；如果 LLM agents 完全不同，后续 planner policy 对照就失去解释力。

第二步比较 LLM planner 和解析最优 planner。结果显示，LLM planner 的策略形状与理论相当接近。Altruistic setting 中，两者都在信念不确定时重投入，在 public belief 强时停止投入；biased setting 中，两者都表现出逃离不利 cascade、在中间区域降低精度、在有利区域减少投入的结构。论文报告，多数 belief states 上 LLM policy 与 optimal policy 的 percentage deviation 低于 10%。

更有趣的是偏差本身也有解释。LLM planner 不太选择极端 precision 0.5 或 1.0，表现出 central tendency；它的投资下降更平滑，因为 LLM agents 更不容易进入 cascade，强 prior 也可能被反向 signal 摇动；biased LLM planner 在很低 belief 处仍愿意投入，因为它学到 LLM agents 对 surprising positive signal 会过度反应。这说明 LLM planner 不只是机械拟合解析策略，而是在适应 non-Bayesian agents。

第三步看 welfare。论文比较 analytical setting、LLM setting 和 hybrid setting。结果显示，考虑 social learning 的 planner 可以显著改变 social welfare。Altruistic planner 相对 baseline 提高 welfare；当 true state 是 $B$ 且 biased planner 追求 $G$ 时，analytical 和 LLM biased planner 都能让 social welfare 下降约 40-50%。这个下降不是靠撒谎，而是靠降低或提高 signal precision 来操纵 public belief 轨迹。

实验也指出 misspecified agent model 的代价。为 Bayesian agents 设计的 analytical policy 用在 non-Bayesian LLM agents 上会变脆；LLM planner 的策略虽然类似解析最优，但更适应 LLM agents 的偏差。这对真实部署很关键：如果平台用错误的人类行为模型来设计信息策略，即使目标是 altruistic，也可能带来意外 welfare loss；如果目标是 biased，它还可能更有效地利用用户偏差。

实验限制同样很明显。LLM simulation 不能替代 human data。论文自己承认 LLM-human simulator 的 fidelity 有争议；而且场景是高度简化的新车购买，和真实社交平台、政治传播、医疗建议之间还有距离。这个实验最适合作为 mechanism demonstration：证明 LLM 可以在社会学习模型里形成类似理论预测的 strategic policy，而不是证明某个真实平台一定会出现同样数量级的 welfare loss。

还有一个实验解释要谨慎：planner 的 control 是完全可观察的，agents 知道 signal precision，理论上也知道 planner 的策略类型。现实平台常常不会这么透明，用户不知道推荐系统为了某个目标降低或提高了信息质量。如果把 covert framing、ranking、内容过滤或个性化 persuasion 加进模型，biased planner 的能力可能更强；但理论可解性会下降。因此本文结论其实偏保守：在这么强的透明约束下，precision control 仍能显著改变 welfare。

## 5. Reviewer Discussion

OpenReview 最终决定是 **Accept (Oral)**。正式 reviews 中三位 reviewer 给 8，另一位给 2；低分 reviewer 在 rebuttal 后把分数提高到 4，但仍保留对 related work 和定位的不同意见。AC meta-review 总结认为理论结果有趣，可能被其他研究复用，主要担忧是模型限制和若干表达清晰度问题，整体适合作为 oral。

正面评价集中在理论结构。Reviewer 认为把 social learning 和 planner control 结合起来很新，convexity proof 很漂亮，altruistic 和 biased planner 的 policy characterization 清晰，LLM simulation 也让理论模型和现实 algorithmic mediator 有了连接。尤其是 altruistic value function convexity 被认为可能对 dynamic information acquisition、sequential contract design 等问题也有启发。

批评也很硬。最重要的一类是 **restrictive assumptions/强限制假设**：二元状态、二元对称信号、0-1 loss、short-lived myopic agents、公开可观察 precision、没有双向 communication、planner 不能打开 reporting channel。这些假设让理论可解，但离真实信息生态还有明显距离。作者回应说，多状态和一般 loss 可能主要增加代数成本，但 general signal structure 会挑战 Theorem 2 的凸性，是未来重点。

第二类批评是 LLM experiment 的解释边界。Reviewer 指出，LLM self-reported belief 未必可靠，oracle 生成的 message precision 也需要验证；低分 reviewer 还认为“LLM 不是 Bayesian”不等于“LLM 像人类”，没有人类数据时，不能把实验强行解释成人类社会动态。作者 rebuttal 补了 state-contingent tasks、willingness-to-pay/binary choice、oracle precision validation，并在结论中更明确承认 LLM-human simulator 的限制。

第三类批评是 related work positioning。低分 reviewer 认为论文没有充分覆盖 online persuasion、information design with RL、MARL/social learning、LLM information design 等邻近文献，因此“first formal model”之类贡献表述过强。作者 rebuttal 扩充了 Section 2，强调本文区别在于 planner control 作用在 information externalities，而不是 reward externalities；也没有让 planner 学未知环境参数，而是在已知模型下刻画最优策略。

我的客观评述是：这篇论文理论上很有价值，但它不是一个直接可部署的 LLM safety benchmark。它的强项是把“算法信息中介会影响社会学习”变成一个可证明、可仿真的动态模型；它的弱项是现实复杂度被压缩得很厉害。读它时不要把 restaurant/car toy model 当成现实社会本身，而要读出它揭示的机制：**控制信息精度足以改变群体信念路径，即使 planner 不撒谎也能带来对齐或失配后果**。

我认为 reviewer 对 LLM-as-human-simulator 的批评非常重要。LLM agents 的 non-Bayesian bias 与人类某些实验偏差相似，但这不是同一回事。真正要进入 policy 或 platform governance，还需要 human subject data、平台日志、真实推荐系统实验或至少 agent-based simulation with calibrated human behavior。否则 welfare 数字只能说明模型内部机制，不应被当成现实社会量化结论。

## 6. Related Work & Future Work

这篇论文连接 social learning、information design、online persuasion/RL 和 LLM simulation。它与经典 information cascade 的区别是加入 planner control；与 Bayesian persuasion 的区别是 agents 之间有序列社会学习；与 online persuasion/RL 的区别是 planner 不学习未知环境参数，序列依赖来自 agents actions 对后续 public belief 的影响，而不是 reward externality 或 state transition。

未来最直接的理论方向是放宽假设。多状态空间会让 belief 变成 simplex；一般 signal structure 会让 action region 和 transition 复杂化；heterogeneous agents 会破坏所有人共享同一 utility threshold 的简洁结构；non-myopic agents 会考虑自己行动对后续 agents 的影响，使 equilibrium 分析更难。论文已经在 appendix 对 opposing preferences 的 altruistic case 做了第一步扩展，但 biased planner 面对异质偏好仍是开放问题。

第二个方向是 regulation and mechanism design。既然 biased planner 可以通过降低 precision 损害 welfare，平台治理就不能只禁止 false information，还要检查 **information quality control/信息质量控制**。如果某个平台在 public belief 有利时故意降低信息清晰度、隐藏不确定性、减少反向证据呈现，即使没有造假，也可能在社会学习层面造成系统性误导。

第三个方向是把 LLM experiment 换成更强的 empirical grounding。可以让真实人类在受控 social learning task 中面对不同 precision messages，估计他们的 belief update 和 cascade threshold；再把这个 behavioral model 放回 planner MDP，比较理论、LLM 和人类校准策略。这样才真正能回答 LLM information mediator 对社会福利的现实影响。
