---
title: "MNPO"
headline: "Multiplayer Nash Preference Optimization"
description: "ICLR 2026 Oral note on MNPO, an n-player game-theoretic preference optimization framework for RLHF."
status: complete
visibility: public
topic: "safety_alignment"
venue: "ICLR 2026 Oral"
year: 2026
paper_url: "https://arxiv.org/abs/2509.23102"
openreview_url: "https://openreview.net/forum?id=x7aLhLMVn1"
source_pdf: "raw/papers/arxiv-2509.23102.pdf"
tags: ["preference_optimization", "nash_learning", "rlhf"]
created: "2026-05-01"
updated: "2026-05-01"
---

> [!abstract] Contributions
>
> 这篇论文提出 **Multiplayer Nash Preference Optimization/MNPO**，把 Nash learning from human feedback 从两玩家偏好博弈推广到 **n-player game/n 玩家博弈**。它的核心动机是：真实人类偏好通常不是一个可传递的标量 reward，也不只是一个 policy 对一个 opponent 的二元对抗，而更像多个标注者、多个评价维度、多个历史策略或多个 reward model 共同构成的偏好群体。MNPO 让每个 policy 同时与一个 opponent population 比较，并用 KL regularization 拉住 reference policy。
>
> 论文的亮点是把 game-theoretic preference optimization、multiplicative weights update、RPO-style reward gap 和一批已有偏好优化算法放进同一个形式框架里。边界也很明显：齐次 preference oracle 下有较干净的 Nash/duality-gap 解释和收敛叙事；异质 reward model 版本更贴近 pluralistic alignment，但没有同样强的理论保证。实验结果显示 MNPO 在 Gemma-2-9B-it 的 post-training 上优于 DPO、SimPO、SPPO 和 INPO，不过最关键的“异质偏好”动机与主实验之间仍有一定张力。

## 1. Introduction

传统 RLHF 经常依赖 **Bradley-Terry model/BT 模型**：给定两个回答 $y_1,y_2$，假设存在一个标量 reward $r(x,y)$，使得人类偏好概率由 reward 差决定。这个假设很方便，因为它把偏好学习变成 reward modeling，再把 alignment 变成带 KL 约束的 reward maximization。但它也很强：真实偏好可能不传递，不同标注者可能偏好不同维度，同一个用户在不同任务下也可能改变标准。

Nash learning from human feedback 的两玩家版本已经试图绕开 BT 的标量 reward 假设。它把偏好看成一个 general preference oracle，让两个 policy 在零和或常和博弈里互相比较，目标是找到不能被另一个 policy exploit 的 Nash policy。MNPO 认为这还不够，因为两玩家设定仍然有 **single-opponent bias/单一对手偏置**：policy 每次只对一个 opponent distribution 优化，不能充分表示多个偏好来源、多个历史 checkpoint 或多个 reward model 构成的偏好生态。

这就解释了为什么论文要引入 n-player formulation。它想让 alignment 不再是“一个当前 policy 对一个参考 opponent”，而是“一个 policy 面对一个 policy population”。从直觉上看，这会让训练更稳定，因为多个历史对手的混合可以降低单一步骤振荡；从 alignment 角度看，它也更接近多标准偏好或 pluralistic preference 的场景。

这篇论文的真正野心是把几个看似分散的 post-training 思路放到同一张图里。DPO 类方法强调 reference policy 和 preferred/dispreferred pair 的 log-ratio；Nash-style 方法强调一般偏好关系下的 equilibrium；RPO 类方法强调用显式 reward gap 约束 policy 的隐式 reward；iterative preference optimization 强调用历史 policy 生成新数据并逐步更新。MNPO 的说法是：这些都可以被理解成“当前 policy 相对一组 opponent policy 调整概率比，并匹配某种目标偏好差”。这个统一视角比单个算法增益更重要，因为它给后续设计新 RLHF objective 提供了坐标系。

## 2. Problem Setup

论文沿用 RLHF 的基本符号。Prompt $x \in \mathcal{X}$ 来自未知分布 $d_0$，policy $\pi(\cdot \mid x)$ 在回答空间 $\mathcal{Y}$ 上给出分布，reference policy 是 $\pi_{\mathrm{ref}}$。BT 设定下，偏好概率由 reward 差给出，训练目标通常是最大化 reward，同时用 KL penalty 防止 policy 过度偏离 reference。

两玩家 Nash formulation 则不要求偏好来自标量 reward，而是假设存在 general preference oracle $\mathbb{P}(y_1 \succ y_2 \mid x)$。两个 policy $\pi_1,\pi_2$ 构成博弈，$\pi_1$ 希望最大化赢过 $\pi_2$ 的概率并靠近 reference，$\pi_2$ 则作为对手。Nash policy $\pi^\*$ 是一个固定点：没有一方能通过单边偏离获得更好结果。论文用 **duality gap/对偶间隙** 衡量某个 policy 离 Nash policy 有多远。

MNPO 的推广是把 policy 集合写成 $\{\pi_i\}_{i=1}^n$。在齐次版本中，所有玩家共享同一个 preference oracle，每个 $\pi_i$ 都要最大化自己相对其他 $n-1$ 个 policy 的平均偏好概率，同时受到 KL regularization。它的目标可以理解成：

$$
J(\pi_i,\{\pi_j\}_{j\neq i})
=
\mathbb{E}_{x,y_i,\{y_j\}}
\left[
\mathbb{P}\left(y_i \succ \{y_j\}_{j\neq i}\mid x\right)
- \tau \mathrm{KL}(\pi_i(\cdot\mid x)\Vert \pi_{\mathrm{ref}}(\cdot\mid x))
\right].
$$

当 $n=2$ 时，这个目标退化回两玩家 NLHF。齐次对称性保证 equilibrium 处所有玩家可以共享同一个 Nash policy。论文相应定义 multiplayer duality gap：固定 opponent set $O_\pi$ 时，某个 policy 能通过改成任意 $\pi'$ 获得的最大改进幅度。这个 gap 为 0 时表示没有单边偏离收益。

在进入 MNPO 之前，论文还引入 **Plackett-Luce model/PL 模型** 来说明 one-vs-many preference 的 reward-learning 版本。BT 可以看成两个候选之间的 softmax，PL 则把一个 preferred response 和多个 alternatives 放在同一个 softmax 里比较。这个动机服务于后面的 n-player game：如果真实反馈经常来自一个候选池，而不是单个 pair，那么只用 pairwise reward difference 会丢掉“一个回答是否压过整组竞争者”的信息。作者没有把 PL 作为最终算法的唯一基础，但它帮助读者理解为什么 multiplayer comparison 是合理形式。

形式上，齐次 MNPO 的强假设是所有玩家共享同一个 preference oracle $\mathbb{P}$。这带来两个好处：第一，博弈保持对称，所有玩家在 equilibrium 处可以收敛到同一 policy；第二，multiplicative weights update 可以继承常和博弈里的 regret guarantee。代价也很明显：这个设定其实还没有真正表达“不同人有不同偏好”，因为所有玩家仍在服从同一个评价函数。真正异质偏好要到 HT-MNPO 才出现，而那一节的理论保证弱得多。

## 3. Algorithm / Methods

MNPO 的理论起点是 multiplicative weights update。直观地说，如果回答 $y$ 相比当前 opponent population 有更高平均优势，那么下一轮 policy 应该提高 $y$ 的概率。论文给出一个理想更新式，其中 $\pi_i^{(t+1)}$ 与其他玩家历史 policy 的几何平均、以及 $y$ 相对这些对手的偏好优势有关。齐次常和博弈下，平均 policy 可以用 $O(1/\sqrt{T})$ regret bound 收敛到近似 Nash equilibrium。

直接计算这个更新式不可行，因为回答空间巨大，normalization factor 难算。作者于是转向 log-ratio。对一对回答 $y,y'$，定义当前 policy 相对 opponent mixture 的 log probability ratio：

$$
h_t(\pi,y,y')
=
\log\frac{\pi(y\mid x)}{\pi(y'\mid x)}
-
\frac{1}{n-1}\sum_{j\neq i}
\log\frac{\pi_j^{(t)}(y\mid x)}{\pi_j^{(t)}(y'\mid x)}.
$$

这个量刻画了当前 policy 相比 opponent population 如何重新分配 $y$ 与 $y'$ 的概率。论文把理想 update 转成一个平方损失，让 $h_t$ 去匹配由 preference oracle 诱导的优势差。随后为了绕开不可直接访问的 oracle 项，引入一个带超参数的 surrogate loss，并把它解释成 **reward-aware preference optimization/RPO** 的特殊形式：policy 的隐式 reward gap 要贴近目标 reward gap。

这个 log-ratio 形式和 DPO 的核心结构很像。DPO 里，policy 相对 reference 的 log-ratio 可以被解释成隐式 reward；MNPO 把 reference 从单个 $\pi_{\mathrm{ref}}$ 换成 opponent population 的加权混合。于是 $h_t$ 不只是“当前 policy 比 reference 更喜欢 $y$ 还是 $y'$”，而是“当前 policy 相比一组历史或对手 policy，如何重新排序 $y$ 与 $y'$”。这一步把 preference optimization 从固定参考点变成动态相对比较，也解释了为什么 TD-MNPO 可以自然包含 iterative methods。

论文中的 surrogate loss 可以理解为两层近似。第一层是把理想 multiplicative update 中不可计算的 normalization 去掉，改看两个回答之间的 ratio，normalizer 在 ratio 中抵消。第二层是把 oracle preference advantage 替换成一个可训练的 target gap，例如 reward model 给出的 $\delta^\star$ 或一个常数 margin。这样得到的训练目标在工程上像 supervised loss，可以用 pairwise preference data 优化；在理论解释上，它仍然声称自己在追踪 Nash-style update。

实际算法是 **TD-MNPO/time-dependent MNPO**。它不维护真正同时训练的 $n$ 个独立玩家，而是在第 $t$ 轮把 opponent set 构造成若干历史 policy $\{\pi_{t-j}\}$ 的加权组合。这样做有两个效果：一方面，历史 policy 起到 population opponent 的作用；另一方面，训练不会只追逐最近一个 checkpoint，因而更平滑。论文还展示 DPO、IPO、DNO、SPIN、INPO 等方法都可以视为 TD-MNPO 在玩家数、opponent 选择、distance metric 和 reward gap 设定上的特例。

论文另外提出 **HT-MNPO/heterogeneous MNPO**。这里每个玩家绑定不同 reward model 或 preference oracle，例如 helpfulness、safety、conciseness 等不同评价维度。这个版本更接近真实 alignment，但理论上不再是齐次常和博弈，因此没有同样的 Nash convergence guarantee。作者用 player-specific duality gap 表示每个玩家的偏离动机，并把最终状态理解成经验上有效的 stationary point，而不是严格保证的 Nash equilibrium。

TD-MNPO 和 HT-MNPO 的区别要分清。TD-MNPO 的 opponent population 主要来自同一训练轨迹的历史 checkpoint，因此它更像优化稳定化技术：用过去的自己作为多个对手，避免当前 policy 只对最近一步过拟合。HT-MNPO 的 opponent population 则来自不同 reward model 或不同偏好 oracle，因此它更接近 alignment 语境里的多价值目标。论文的主结果里 TD-MNPO 更干净，HT-MNPO 更有现实动机；读这篇时不要把二者的证据混在一起。

> [!note] What the Theory Actually Buys
> 齐次 MNPO 的理论保证主要说明：在共享 preference oracle 和对称常和结构下，multiplicative weights style 的平均 policy 可以收敛到近似 Nash。它没有直接证明异质人类价值会被正确聚合，也没有证明最终单模型能代表 pluralistic preference。理论贡献更准确的表述是：它给“历史 opponent mixture 的偏好优化”提供了 Nash-style 解释。

## 4. Experiments

实验使用 Gemma-2-9B-it 作为 base model，进行 3 轮在线 RLHF。TD-MNPO 的 preference signal 主要来自 ArmoRM-Llama3-8B-v0.1；HT-MNPO 还使用 Skywork-Reward-V2-Llama-3.1-8B 和 Athene-RM-8B 模拟异质 reward oracle。主要评测包括 AlpacaEval 2、Arena-Hard、MT-Bench，以及 IFEval、GPQA、MMLU、ARC、HellaSwag、TruthfulQA、Winogrande、GSM8K、Minerva-Math、AIME-24、HumanEval 等能力 benchmark。

核心结果是 TD-MNPO 在三个 instruction-following/preference benchmark 上都超过 DPO、SimPO、SPPO 和 INPO。它在 AlpacaEval 2.0 上达到 57.27，高于 INPO 的 56.09；在 Arena-Hard 上达到 52.26，高于 INPO 的 48.03；在 MT-Bench 上达到 7.03，也略高于 INPO 的 6.95。HT-MNPO 的不同 reward model 版本在某些指标上更高，例如 Athene-RM 版本的 AlpacaEval 达到 59.64，ArmoRM 版本的 MT-Bench 达到 7.52。

在一般能力评测上，TD-MNPO 的平均分为 71.08，高于 SFT 和其他 preference optimization baseline；HT-MNPO with Skywork 的平均分更高，为 71.80。在数学和代码表里，TD-MNPO 的平均分为 48.10，HT-MNPO with ArmoRM 为 48.68；AIME-24 上 TD-MNPO 与 HT-MNPO-ArmoRM 都取得 3.33，而其他 baseline 为 0。作者用这些结果支持 multiplayer formulation 能提升 alignment，同时不明显牺牲基础能力。

不过实验解释要克制。主结果很大程度上还是在单一 reward model 下训练出来的，真正多偏好、多 reward model 的 HT-MNPO 虽然有结果，但理论保证弱，而且实验仍是用少数 reward model 模拟异质性。换言之，论文最强的数学故事在齐次设定，最有 alignment 意义的动机在异质设定，两者之间还没有完全合上。

从实验设计看，论文选择 Gemma-2-9B-it 也有双重影响。一方面，9B 模型让在线 post-training 成本可控，多个 baseline 可以在同一设置下比较；另一方面，它已经是 instruction-tuned model，不是纯 SFT 起点，因此 room for improvement 有限，很多 benchmark 的差距会比较小。Reviewer 质疑这一点很合理：如果基础模型已经有较强偏好对齐，MNPO 的增益可能来自微调稳定性，而不是 multiplayer preference structure 本身。

另一个关键问题是 judge 和 reward model 的耦合。TD-MNPO 使用 ArmoRM 产生偏好信号，评测又依赖 GPT-5-mini 等 judge 和公开 benchmark。如果 reward model 与 judge 偏好接近，MNPO 可能只是更好地拟合了这个 judge family；如果二者不一致，结果才更能说明 general alignment。论文报告了多个能力 benchmark 以缓解这个担忧，但仍缺少一种更直接的 “preference heterogeneity stress test”，例如同一 prompt 下 helpfulness reward、safety reward、truthfulness reward 明确冲突时，MNPO 如何折中。

表格结果里最值得保留的是 Arena-Hard 的 4.23 点提升和 AIME-24 的非零结果，但这两个点也不能过度解读。Arena-Hard 是开放式对话偏好评测，容易受 judge 偏好、回答长度、格式风格影响；AIME-24 上 3.33 说明只解出极少量题，作为“保持推理能力不崩”的证据可以，作为“显著提升数学推理”的证据偏弱。比较稳的结论是：MNPO 没有像一些 preference optimization 方法那样明显伤害通用能力，并且在 instruction-following 偏好评测上有一致小到中等收益。

## 5. Reviewer Discussion

OpenReview 最终决定是 **Accept (Oral)**。三个正式 reviewer 的原始分数是 8、4、6，AC 在 meta-review 中认为 rebuttal 后主要问题已被澄清，并推荐接收。正面评价主要集中在理论统一性和算法框架：MNPO 是从 two-player NLHF 到 n-player preference game 的自然扩展，TD-MNPO 把多种已有 RLHF objective 放进统一表格，实验也给出了稳定的 baseline improvement。

负面意见也相当关键。最强批评来自“动机与实验不完全匹配”：论文反复强调 heterogeneous annotators 和 diverse preference structures，但主实验使用单一 ArmoRM 作为 preference oracle，这不能直接证明 n-player formulation 真能处理真实异质偏好。另一个 reviewer 进一步指出，在对称 game 里，n-player equilibrium 有时可能等价于对 population average 的 two-player best response；如果 equilibrium 本身没有新东西，MNPO 的收益可能主要来自优化路径更平滑，而不是 solution concept 更强。

我的客观评述是：这篇论文的价值更像 **preference optimization 的框架论文**，而不是已经钉死了 pluralistic alignment 的实证论文。它把 TD opponent mixture、historical policies、reward-aware gap 和 Nash duality gap 组织得很漂亮，对后续方法设计有启发；但如果要证明它真的解决“多样人类偏好”，还需要更直接的实验，例如不同 annotator group、不同价值维度 reward model、非传递偏好构造，以及 n 值和 opponent mixture 的系统 ablation。

AC 接收的理由也说明了 rebuttal 的作用。原始评审里，一个 reviewer 给到 4，核心就是不相信 n-player formulation 相比 two-player mean-field view 有足够必要性。AC 认为作者通过新增实验和解释缓解了这个问题，尤其澄清 two-player 与 n-player 在 solution concept 上可能重合，但 optimization path 不同。这个澄清很关键，因为它把论文从“n-player equilibrium 一定更对”调整成“n-player opponent mixture 给训练路径带来更稳定、更丰富的动态”。这个版本的 claim 更可信。

我的批评会更冷一点：如果论文标题里的 “Multiplayer” 主要在实际算法里表现为 historical checkpoint mixture，那么它和 DNO、SPIN、INPO 这一类 iterative training 的距离就没有引言暗示得那么远。它的贡献在于统一和推广，而不是凭空开出一个全新的 alignment paradigm。把它当作 RLHF objective design 的理论整理来读，会非常有收获；把它当作多元人类价值对齐的解决方案来读，就会被它的实验支撑程度拖住。

## 6. Related Work & Future Work

MNPO 位于 DPO/IPO/SimPO/SPPO 之后，也接在 INPO、ONPO、EGPO 等 Nash-style preference optimization 之后。与 DPO 类方法相比，它不满足于固定 reference pair；与两玩家 Nash 方法相比，它把对手从单一 policy 扩展成历史或异质 population。这个方向和 safety alignment 的关系主要在于：如果偏好来源本身多样、冲突或非传递，那么 alignment objective 不能只假设存在一个干净标量 reward。

未来最值得补的是异质偏好的硬实验。理想设置应该让不同 reward models 或 annotator groups 明确代表不同价值维度，并观察 MNPO 是否能得到更稳定、更不容易被单一偏好维度 exploit 的 policy。同时，理论上也需要更清楚地区分 **equilibrium change** 和 **optimization path change**：如果 n-player formulation 的好处主要来自历史对手平滑训练，那它仍然有价值，但这个价值应被表述为优化稳定性，而不是直接等同于 pluralistic alignment。

这篇和同组的 SafeDPO、DPO misspecification、interpretable preference data 可以连起来读。它们共同说明一个趋势：alignment 研究正在从“选择一个偏好优化公式”转向“偏好数据、偏好模型、优化路径和评价 oracle 的联合建模”。MNPO 贡献的是优化路径和 game-theoretic abstraction；如果后续能接上可解释偏好维度或多群体标注数据，它会更接近真正的 pluralistic alignment。

后续技术上还需要两个 ablation。第一是 $n$ 和 $\lambda_j$ 的系统扫描：如果 $n=2$ 到 $n=3$ 已经获得大部分收益，那么 multiplayer 的边际价值有限；如果更多历史 opponent 持续改善稳定性，论文的主张会更强。第二是 opponent identity ablation：历史 checkpoint、外部模型、不同 reward model、不同 annotator slice 各自带来的收益应该分开报告。只有这样，才能判断 MNPO 到底是在利用历史平滑、模型多样性，还是偏好异质性。
