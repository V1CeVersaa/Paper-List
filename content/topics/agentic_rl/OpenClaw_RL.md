---
title: "OpenClaw-RL"
headline: "OpenClaw-RL: Train Any Agent Simply by Talking"
description: "论文笔记：将 next-state signal 统一为个人代理与通用代理的在线强化学习信号。"
status: "complete"
visibility: "public"
topic: "agentic_rl"
venue: "arXiv"
year: "2026"
paper_url: "https://arxiv.org/abs/2603.10165"
source_pdf: "raw/papers/openclaw-rl-2603-10165.pdf"
tags: ["process_reward_model", "online_learning"]
related: []
created: "2026-04-10"
updated: "2026-04-10"
draft: "false"
---

> [!abstract] Contributions
>
> 这篇论文的核心主张是：代理/Agent 在每一步动作之后都会看到一个 `next-state signal`，它可能是用户回复、工具输出、终端执行结果、GUI 状态变化或测试反馈；过去系统通常只把它当作后续上下文，而 OpenClaw-RL 试图把它变成实时在线学习信号。论文给出一个统一的异步基础设施，将个人代理对话与 terminal、GUI、SWE、tool-call 等通用 agent 场景放进同一训练回路，并把 next-state signal 拆成两类可恢复信息：一类是 evaluative signal，经由 PRM judge 变成序列级二值/三值奖励；另一类是 directive signal，经由 Hindsight-Guided On-Policy Distillation（OPD）变成 token 级方向性优势。实验上，作者报告二者联合使用时，个人代理个性化效果最好；在通用 agent 任务中，将 process reward 与 outcome reward 结合也优于只用 outcome reward。
>
> 需要立即记住的边界是：个人代理实验主要是 LLM 模拟用户场景，而非真实长期部署；通用 agent 结果更偏向“框架可用性 + 若干量化例子”的证明，而不是对每个 setting 都给出充分的 SOTA 级对比。此外，论文明确承认引入 PRM 会带来额外资源成本，因此这个方案的优势建立在“能接受持续 judge 与在线更新开销”的前提上。

## 1. Introduction

这篇论文试图解决一个很直接但此前常被忽略的问题：部署中的 agent 实际上一直在产生可用于学习的数据，但系统通常把这些数据浪费掉了。每次动作 $a_t$ 之后，环境都会返回一个新的状态 $s_{t+1}$，它既可能是用户的一句“你应该先检查文件”，也可能是终端报错、GUI 页面变化、测试是否通过，或者工具调用的返回值。现有 agent 系统大多只把这些信息当成下一步决策的输入，而不是对上一步动作质量的反馈来源。

论文把这种浪费拆成两种。第一种是 `evaluative signal`：next-state signal 会隐式告诉我们上一步做得好不好，例如用户重新追问通常说明不满意，测试通过则说明当前步骤有效。第二种是 `directive signal`：有些反馈不仅说明“错了”，还说明“应该怎么改”，例如错误栈会提示该检查哪个文件，用户也可能明确指出回答风格或操作顺序的问题。作者认为，前者适合转成标量 reward，后者则应保留成更细粒度的方向性监督，而不是被压缩成单个分数。

基于这个观察，OpenClaw-RL 的贡献并不只是提出一个损失函数，而是给出一套统一叙事。它一方面要支持 personal agent 在真实使用过程中“边服务边变好”，另一方面又希望兼容 terminal、GUI、SWE、tool-call 等更常见的 agentic RL 训练环境。为此，论文在系统层面强调完全异步、互不阻塞的训练架构，在学习层面则并行提供 Binary RL 和 OPD 两种 next-state signal 恢复方式。

从动机上看，这篇论文最有价值的点不在“又提出一个 PPO 变体”，而在于把 agent 训练对象从“离线收集好的轨迹”改写成“实时运行中的交互流”。这使得 personal agent personalization 与 general agent RL 不再是两套彼此分离的故事，而被统一成同一个问题：如何从 action 之后自然出现的 next-state 中抽取出可学习信号。

## 2. Problem Setup

论文将每一条交互流都形式化为一个马尔可夫决策过程/Markov Decision Process/MDP：

$$
\mathcal{M} = (\mathcal{S}, \mathcal{A}, T, r).
$$

其中：

- 状态 $s_t \in \mathcal{S}$ 表示截至第 $t$ 步的完整对话或环境上下文。
- 动作 $a_t \in \mathcal{A}$ 是当前 policy $\pi_\theta$ 生成的回复或动作 token 序列。
- 转移 $T(s_{t+1} \mid s_t, a_t)$ 由环境决定；这里的 $s_{t+1}$ 正是论文关注的 next-state signal。
- 奖励 $r(a_t, s_{t+1})$ 不再只依赖终局结果，而是由 next-state signal 通过 judge/PRM 推断出来。

这个设定最关键的地方在于：论文默认同一个 policy 会同时接收多条异步交互流，而不是一次只训练一个封闭环境。于是个人代理对话、终端执行、GUI 操作、SWE 任务、tool-call 轨迹都被统一成“动作后观察到新状态”的序列决策问题。形式上它们共享同样的接口，只是 next-state signal 的具体形态不同。

与标准 RLVR 不同，这里不把 outcome $o$ 看成唯一监督。论文认为，真正更丰富的信号是依赖于 $s_{t+1}$ 的 process reward：它不仅能告诉模型某一步有没有推进任务，还可能提供可操作的修正方向。于是问题被改写成两个子问题：

1. 如何把 next-state 中的 evaluative content 提炼成稳定的 step-wise reward。
2. 如何把 next-state 中的 directive content 提炼成 token 级“应该往哪里改”的训练信号。

对于 personal agent，作者还额外强调 session-aware 的数据组织：只有 `main-line turn` 被视为可训练样本，而辅助查询、记忆整理等 `side turn` 会透传但不写入训练。这样做的目的是保证“上一轮动作”与“当前 next-state”之间的语义配对清晰，不让杂散上下文污染 credit assignment。

## 3. Algorithm / Methods / Model

方法部分可以分成两个层面来理解：系统层面，OpenClaw-RL 先把在线服务与训练解耦，确保 live deployment 不会因为 RL 更新被阻塞；学习层面，它再把 next-state signal 拆成 evaluative 与 directive 两类，并分别交给 Binary RL 与 OPD 处理。

### 3.1 Method Overview

OpenClaw-RL 的统一接口可以概括成一句话：动作先在真实环境中执行，环境返回 next-state，judge 再从 next-state 中恢复可学习信号，训练器据此回写 policy。对 personal agent，这个环境是用户个人设备；对 general agent，则是 cloud-hosted terminal、GUI、SWE 或 tool-call 环境。

> [!tip]
> Binary RL 和 OPD 不是互斥关系。前者追求“覆盖广但信号粗”，后者追求“样本少但纠错方向细”。论文真正主张的是把两者叠加起来，让所有 turn 至少都贡献一个粗粒度判断，而带有明确纠错信息的 turn 再额外贡献 token 级监督。

<!-- TODO: insert Figure 1 from paper: unified infrastructure with personal agents, general agents, and four asynchronous components -->
<!-- TODO: insert Figure 3 from paper: binary reward, OPD, and step-wise reward overview -->

### 3.2 Infrastructure

论文的基础设施建立在四个完全解耦的异步组件上：

- `Policy Serving`：负责在线服务当前请求。
- `Environment Server`：承接用户设备或各类 agent 环境的交互。
- `PRM / Judge`：根据 $(a_t, s_{t+1})$ 计算 reward，或提炼 OPD 所需 hint。
- `Policy Training`：在后台持续更新参数，并通过 graceful weight update 切换新权重。

这种设计的实际意义是：服务当前请求、评估上一轮响应、以及训练新权重三件事可以同时发生，彼此没有阻塞依赖。论文认为，这正是“代理在被使用的同时持续学习”能够落地的前提；否则一旦需要等待 rollout、judge 或 trainer，同步开销就会使 live deployment 变得不可用。

对于 personal agent，环境就是用户设备本身，通过私有 API 与 RL server 通信。论文特别区分了 `main-line turn` 和 `side turn`：前者构成训练样本，后者只做透传。对于 general agent，环境则可在云端大规模并行部署，缓解长时程 rollout 的长尾延迟问题。作者还提到所有交互、judge 结果、hint、accept/reject 决策都会以 JSONL 实时记录，并在权重切换边界清空旧日志，以保证日志对应单一 policy 版本。

> [!info]
> 这套系统贡献的重点不是提出新的分布式训练原语，而是把 slime 的异步 RL 基础设施向“live multi-stream interaction”方向推进了一步：同一个框架既能挂个人设备，也能挂标准 agent benchmark 环境。

### 3.3 Binary RL for Evaluative Signals

Binary RL 的目标是把 next-state 中“做得好不好”的信息压缩为标量优势。给定动作 $a_t$ 与 next-state $s_{t+1}$，judge 输出：

$$
\mathrm{PRM}(a_t, s_{t+1}) \rightarrow r \in \{+1, -1, 0\}.
$$

论文对同一样本执行 $m$ 次独立 judge，再用 majority vote 得到最终 reward $r_{\mathrm{final}}$。直观上，这是在用多次判分降低单次 judge 的随机性。对于 personal agent，judge 依据的是用户后续回复或工具执行结果；对于 general agent，judge 要判断环境反馈是否意味着任务取得进展。

训练目标本身仍是 PPO 风格的 clipped surrogate，只是优势函数换成了从 next-state 恢复出的 reward：

$$
\rho_t = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\mathrm{old}}(a_t \mid s_t)},
$$

$$
L_{\mathrm{pg}} = - \mathbb{E}_t \min \left(
\rho_t A_t,\;
\operatorname{clip}(\rho_t, 1-\epsilon, 1+\epsilon_{\mathrm{high}}) A_t
\right),
$$

$$
L = L_{\mathrm{pg}} + \beta_{\mathrm{KL}} L_{\mathrm{KL}}.
\tag{1}
$$

这里 Binary RL 取 $A_t = r_{\mathrm{final}}$，并使用 $\epsilon = 0.2$、$\epsilon_{\mathrm{high}} = 0.28$、$\beta_{\mathrm{KL}} = 0.02$。作者强调，在 personal agent 的在线对话场景里，不存在像 GRPO 那样天然成组的同题采样，因此很难直接套用 group-based standardization。

这种方法的优点是鲁棒而宽覆盖：几乎所有被 judge 成功打分的 turn 都能被利用。但它的问题也很明显，即它把原本可能很具体的反馈压成一个标量，因此只能告诉模型“往上”或“往下”，却说不清应该改哪些 token、改成什么样。

### 3.4 OPD for Directive Signals

OPD（Hindsight-Guided On-Policy Distillation）正是为了解决上面这个信息损失问题。核心想法是：如果 next-state 明确表达了“应该怎么改”，那么最佳训练目标不是再给一个标量分数，而是构造一个“带纠错提示的 teacher context”，再比较该上下文下 teacher 分布与当前 student 分布的差异。

它的流程可以分成四步。

第一步是 `hint extraction`。judge 从 $(a_t, s_{t+1})$ 中同时判断该 turn 是否含有可用的 directive signal，并输出一个简短 hint。论文特别强调，不能直接把原始 $s_{t+1}$ 当成 hint，因为真实 next-state 常常夹杂噪声、新问题或无关上下文，所以需要 judge 先做一次压缩与提炼。

第二步是 `quality filtering`。论文只保留那些被判为正样本、且 hint 长度超过 10 个字符的样本；如果没有合格 hint，就直接丢弃该样本。这个设计说明 OPD 并不追求覆盖率，而是在样本质量和信号强度上做了更激进的筛选。

第三步是 `enhanced teacher construction`。将提炼出的 hint 追加到最后一条用户消息中，得到增强上下文：

$$
s_{\mathrm{enhanced}} = s_t \oplus \mathrm{hint}.
$$

这可以理解为：让同一个模型在“如果用户当时直接告诉你应该怎么做”的条件下，重新评估原始 response 中每个 token 的合理性。

第四步是计算 token-level advantage：

$$
A_t^{\mathrm{OPD}} =
\log \pi_{\mathrm{teacher}}(a_t \mid s_{\mathrm{enhanced}})
- \log \pi_\theta(a_t \mid s_t).
\tag{2}
$$

如果某个 token 的 $A_t^{\mathrm{OPD}} > 0$，说明带 hint 的 teacher 认为这个 token 更应该出现；若 $A_t^{\mathrm{OPD}} < 0$，则说明它在纠错后的语境下显得不合适。论文把这种监督解释为“directional advantage”：与 sequence-level scalar reward 不同，单个 response 内部的不同 token 可以被朝相反方向更新。

> [!tip]
> OPD 的关键不是“引入一个更强 teacher”，而是“让同一个 policy 在 hindsight hint 下扮演 teacher”。因此它兼具 hindsight relabeling 和 self-distillation 的味道，但不需要外部教师模型，也不需要成对 preference data。

### 3.5 Combining Binary RL and OPD

论文明确认为 Binary RL 与 OPD 是互补的。Binary RL 能覆盖所有被 judge 的 turn，但粒度粗；OPD 只覆盖明确带纠错信号的子集，但粒度细。于是作者直接把二者写进同一个优势函数：

$$
A_t =
w_{\mathrm{binary}} r_{\mathrm{final}}
+ w_{\mathrm{opd}} A_t^{\mathrm{OPD}},
\qquad
w_{\mathrm{binary}} = w_{\mathrm{opd}} = 1.
\tag{3}
$$

这个组合很朴素，但逻辑上是自洽的：序列级判断负责保证“整体别跑偏”，token 级判断负责告诉模型“具体哪里应该变”。论文后面的 personal agent 实验也正是围绕这一点展开，结果显示组合项确实优于单独使用任一方法。

### 3.6 Step-wise Reward for General Agents

对 general agent 而言，重点从个性化转向长时程 credit assignment。论文认为，只用 outcome reward 的问题在于：长轨迹中的绝大多数步骤都拿不到直接梯度，因此训练信号过于稀疏。为此，作者将终局结果与 process reward 直接相加：

$$
r_t^{\mathrm{step}} = o + \frac{1}{m} \sum_{i=1}^{m} r_i,
\tag{4}
$$

其中 $o$ 是 trajectory-level verifiable outcome，$r_i$ 是对同一步 next-state 的多次 PRM 评分。作者沿用了 RLAnything 的经验判断：长时程 agent 任务里，逐步 reward 对优化非常关键。

另一个实现层面的选择是 advantage standardization。由于真实 terminal/GUI/SWE state 很难像 GRPO 那样按“相似状态”自然成组，论文选择按“相同步数”对 action 分组标准化。这个决定并不华丽，但它对应的是一个很现实的工程判断：真实 agent 环境里，找到足够稳定的 state clustering 往往比直接按 step index 分组更困难。

## 4. Experiments

论文的实验分成两条线：一条验证 personal agent 是否能从对话 next-state 中持续个性化；另一条验证相同基础设施能否支撑 terminal、GUI、SWE、tool-call 等多类 general agent，并从 process reward 中受益。

### 4.1 Experimental Setup

在 personal agent 轨道中，作者设计了两个 LLM 模拟场景。其一是学生使用 OpenClaw 做作业，但不希望回答显得“太 AI”；其二是老师使用 OpenClaw 批改作业，希望评语具体且友好。两种场景都使用 Qwen3-4B 作为 OpenClaw policy，作业任务来自 GSM8K，学习率为 $1 \times 10^{-5}$，KL 系数设为 0，每收集 16 个训练样本触发一次训练。

在 general agent 轨道中，作者分别在 terminal、GUI、SWE、tool-call 四种 setting 上实验，所用模型依次为 Qwen3-8B、Qwen3VL-8B-Thinking、Qwen3-32B、Qwen3-4B-SFT；训练数据分别来自 SETA RL、OSWorld-Verified、SWE-Bench-Verified、DAPO RL data。GUI 与 tool-call 还分别使用 Qwen3VL-8B-Thinking 和 Qwen3-4B 作为 PRM。统一超参数大致为：学习率 $10^{-6}$、KL 系数 0.01、clip ratio 下界 0.2、上界 0.28；环境并行度则是 terminal 128、GUI/SWE 64、tool-call 32。

<!-- TODO: insert Figure 2 from paper: personal-agent qualitative before/after examples -->
<!-- TODO: insert Figure 4 from paper: general-agent settings and scalable RL overview -->

### 4.2 Personal Agent Results

个人代理实验最重要的问题是：Binary RL 和 OPD 分别何时有效，以及两者是否互补。论文给出的结论非常明确：组合方法最好，而 OPD 单独训练虽然起效更慢，但最终优于单独的 Binary RL。

| Method    | Updated 8 steps | Updated 16 steps |
| --------- | --------------: | ---------------: |
| Binary RL |            0.25 |             0.23 |
| OPD       |            0.25 |             0.72 |
| Combined  |            0.76 |             0.81 |

这里的基线分数是 0.17。作者用与用户模拟相同的 LLM，对 OpenClaw 在前 36 个 GSM8K 题目的首个回答打“个性化分数”。从结果看，Binary RL 的提升幅度有限，而 OPD 在 16 个 update steps 后出现明显收益；当二者组合时，8 步与 16 步都取得最高分。这与方法分析是一致的：Binary RL 提供普适、低分辨率的训练信号，OPD 则提供稀疏但高分辨率的纠偏。

论文还给出两个定性例子。学生场景中，模型会逐渐避免“过于 AI 化”的格式化表达与夸张排版，改用更自然的作答风格；老师场景中，模型在 24 次批改交互后会写出更具体、更友善的反馈。作者据此宣称，OpenClaw-RL 可以让 personal agent “在被使用的过程中”快速贴近单个用户偏好。

不过，这部分证据也有明显边界。首先，用户、教师和评分器都由 LLM 模拟，因而它更像是一个闭环 simulator study，而不是现实部署研究。其次，个性化分数由同类模拟器打出，容易更偏向它自己的风格偏好；论文没有展示跨评估器、跨用户或真实长期使用下的稳定性。

### 4.3 General Agent Results

通用 agent 实验重点回答两个问题：这套基础设施能否覆盖多种真实 setting，以及 process reward 是否真的有帮助。第一点上，论文更像是在展示框架兼容性：它把 terminal、GUI、SWE、tool-call 都接进了同一个异步 RL 系统，并展示了大规模环境并行部署。这里的主要证据是系统覆盖范围和训练配置，而不是细粒度性能表。

第二点上，量化结果主要集中在“integrated reward vs. outcome-only”对比：

| Setting   | Integrated | Outcome only |
| --------- | ---------: | -----------: |
| Tool-call |       0.30 |         0.17 |
| GUI       |       0.33 |         0.31 |

这张表说明，将 process reward 与 outcome reward 相加后，tool-call 提升较明显，GUI 也有小幅提升。作者据此强调，在长时程 agent 任务中，process reward 不是附属品，而是改善 credit assignment 的核心信号来源。

但也要看到，这部分实验离“充分证明框架有效”还有距离。论文在 terminal 与 SWE 上没有在主文中给出同等细致的数字结果，而 GUI 的评估还直接使用训练集的一个子集；此外，资源代价只被定性描述为“需要额外 host 一个 PRM”，并未量化延迟、吞吐或成本。

### 4.4 Overall Assessment

如果从论文主张出发，实验最有说服力的部分其实不是绝对分数，而是三点相互支撑的趋势：

- personal agent 的 next-state signal 不只是 reward source，还能通过 OPD 变成更精细的纠错监督。
- Binary RL 与 OPD 的收益模式不同，因此联合训练是合理的。
- 在 general agent 中，process reward 至少在部分 setting 上确实优于 outcome-only。

相对不足的地方也很集中：

- 没有单独消融 judge 次数、hint 过滤阈值、组合权重等关键设计。
- personal agent 缺乏真实用户研究。
- general agent 的量化结果覆盖还不够完整，尤其是 terminal 与 SWE。
- 系统论文强调“零中断服务”，但没有给出端到端 latency 或资源成本曲线。

## 5. Related Work & Future Work

### 5.1 Related Work

论文把自身放在四条相关工作脉络之中。

第一条是 LLM 的 RL 对齐路线，如 RLHF、DPO、GRPO、DeepSeek-R1、DAPO 等。这些方法大多假设数据已离线收集好，再在训练阶段集中优化。OpenClaw-RL 的差异在于，它把训练样本来源改成了 live interaction，而不是固定数据集。

第二条是 agentic RL 与 tool-use 方向。ReAct、Toolformer、FireAct 等奠定了代理与工具交互的范式，后续也有面向单一环境的 RL 系统，如 GUI agent、SWE agent、tool-use agent 等。OpenClaw-RL 的目标不是在某一环境设计专门算法，而是用统一的 next-state interface 兼容多类环境。

第三条是 process reward model/PRM 相关工作。以往 PRM 更多用于数学推理或带可验证中间步骤的任务，而这篇论文进一步把 PRM judge 放进在线场景，让它直接读取 live next-state，而不是对离线标注数据做评分。

第四条是 hindsight / self-distillation 路线。Buffer of Thoughts、SuperCorrect、STaR、HIR、Self-Rewarding 以及各类 on-policy distillation 方法都表明：一旦把额外结构化信息塞回上下文，模型可能给出更好的 token 分布。OpenClaw-RL 的 OPD 属于这条路线在 online next-state setting 下的一种组合式实现：它既有 hindsight relabeling，也有 self-distillation，但不依赖外部 teacher 或预先配对好的反馈数据。

### 5.2 Future Work

> 论文没有专门列出独立的 `Future Work` 小节，下面先区分作者结论中隐含的方向，再给出基于论文局限的自然延伸。

从作者结论隐含出来的方向看，最直接的下一步是把“统一 next-state learning”扩展到更多真实交互流，并继续扩大多环境并行训练的规模。论文整体叙事也暗示，它希望 personal agent personalization 与通用 agent optimization 最终共享一套长期运行的在线训练系统。

从论文当前边界自然推得，至少还有三条重要扩展线索。第一，做真实用户与真实部署研究，验证 OPD/PRM 在噪声更大、反馈更不规范的环境里是否依然稳定。第二，补充更完整的 general-agent 实验，尤其是 terminal、SWE 上的量化曲线、失败案例与系统开销。第三，进一步研究 next-state signal 的可靠性控制，例如如何识别混合反馈、如何在错误 hint 下避免自我强化，以及如何在在线更新时加入更强的安全约束。
