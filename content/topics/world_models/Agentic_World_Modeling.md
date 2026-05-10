---
title: "Agentic World Modeling"
headline: "Agentic World Modeling: Foundations, Capabilities, Laws, and Beyond"
description: "论文笔记：用 L1 Predictor、L2 Simulator、L3 Evolver 与 physical/digital/social/scientific regimes 组织 agentic world modeling。"
status: "complete"
visibility: "public"
topic: "world_models"
venue: "arXiv"
year: "2026"
paper_url: "https://arxiv.org/abs/2604.22748"
source_pdf: "raw/papers/arxiv-2604.22748.pdf"
tags: ["agentic_world_modeling", "world_model_evaluation", "model_based_rl"]
related: ["OpenClaw_RL", "steering_the_herd", "cybergym"]
created: "2026-05-09"
updated: "2026-05-09"
draft: "false"
---

> [!abstract] Contributions
>
> 这篇论文提出 **agentic world modeling/智能体世界建模** 的总框架，用 **L1 Predictor、L2 Simulator、L3 Evolver** 三个 capability levels 和 **physical、digital、social、scientific** 四类 governing-law regimes 来统一 model-based RL、视频生成、web/GUI agents、多智能体社会模拟和 AI for Science。它的核心贡献不是新算法，而是把“world model”从一个容易混用的术语，重新定义成 **能回答 action-conditioned transition queries，并且服务 agent 决策的环境模型**。
>
> 需要记住的边界是：这是一篇 position-driven survey，覆盖超过 400 篇工作并总结 100 多个代表系统，但它本身不提出新 benchmark 或实证结果。因此它的价值主要在 **分类、诊断和评价语言**，不是在证明某个具体系统已经达到 L3。作者提出的 L3 Evolver 很有野心，但当前最扎实的证据主要来自 autonomous science 和少数数字环境；physical/social regimes 里的 L3 多数仍处在 emerging 或 aspirational 阶段。
>
> 在仓库阅读图谱里，它是 `world_models` topic 的入口文献，也给 `agentic_rl` 提供环境建模侧的补充。`OpenClaw_RL` 关注 agent 如何从 next-state signal 中持续学习；本文追问 next-state 背后的 **transition model** 是否足以被预测、模拟和修正。它也能连接 `steering_the_herd` 的 social-world simulation 和 `cybergym` 的 execution-based digital evaluation，因为这两篇分别暴露了社会动态与软件环境对 agent world model 的特殊约束。

## 1. Introduction

这篇论文的出发点非常直接：AI 系统正在从单轮文本生成走向持续交互的 agent，环境动态建模就从“可选增强”变成了核心瓶颈。一个 agent 如果要操作物体、浏览网页、写代码、协调其他 agents 或设计实验，就不能只根据当前 observation 生成下一句话；它必须预测 **当前状态在 action 之后会怎样变化**，并且用这个预测比较候选行动的后果。

问题在于，**world model** 在不同社区里含义非常混乱。在 model-based RL 中，它通常指 latent transition dynamics，用来做 imagined rollout 或 planning；在视频生成中，它可能指能生成连贯未来帧的大模型；在 web/GUI agents 中，它可能指对页面状态、DOM、API 和错误反馈的预测；在社会模拟中，它要刻画信念、目标、规范和群体行为；在科学发现里，它又可能是替代昂贵实验或数值求解器的 surrogate model。论文认为，这些看似分散的工作其实可以用同一套问题来组织：**模型是否能支持 agent 做决策级别的转移查询**。

作者因此把 world modeling 重新组织成两个轴。第一个轴是 capability level：**L1 Predictor** 负责一步或短程预测；**L2 Simulator** 能把一步转移组合成多步、行动条件的 rollout；**L3 Evolver** 能在预测失败后主动收集证据并修正自己的模型。第二个轴是 governing-law regime：物理世界由几何、接触、运动学和守恒约束支配；数字世界由程序语义、API、UI 状态机和文件系统支配；社会世界由信念、目标、承诺、规范和制度支配；科学世界由待发现的因果机制和实验观测支配。

这套框架的动机很强，因为它把“生成得像不像”降级成一个不充分指标。对 agent 来说，世界模型真正重要的是 **decision usability/决策可用性**：它生成的未来能不能让 planner 选出更好的 action，能不能在 action 变化时给出方向正确的未来差异，能不能避免违反环境规则。一个视频模型即使画面漂亮，如果让杯子穿过桌子、让按钮点击不改变 GUI 状态、让社会承诺无后果消失，它就不是可靠的 agentic world model。

<!-- TODO: insert Figure 1 from paper: organizational structure of L1/L2/L3 and four governing-law regimes -->

## 2. Problem Setup

论文用 **部分可观测马尔可夫决策过程/Partially Observable Markov Decision Process/POMDP** 统一形式化。环境写成

$$
E = (X, A, \Omega, T, O, R, \gamma),
$$

其中 $X$ 是隐藏状态空间，$A$ 是动作空间，$\Omega$ 是 observation space，$T(x_{t+1}\mid x_t,a_t)$ 是真实环境转移，$O(o_t\mid x_t)$ 是 observation emission。由于 agent 通常看不到真实状态 $x_t$，它只能维护 belief state 或 learned latent state $z_t$。世界模型的核心组件包括从历史 observation 和 action 推断 latent 的 $q_\phi(z_t\mid o_{\leq t},a_{\leq t-1})$，预测下一 latent 的 $p_\theta(z_t\mid z_{t-1},a_t)$，把 latent 解码成 observation 的 $p_\psi(o_t\mid z_t)$，以及有时用于表示塑形的 inverse dynamics $\pi_\eta(a_t\mid z_{t-1},z_t)$。

这个形式化最重要的地方，是把 **world model** 和 **planner** 分开。world model 是描述性的：它回答状态和 observation 在 action 或 intervention 下如何变化。planner 是规范性的：它根据这些预测选择 action 来优化目标。两者可以联合训练，也可以模块化组合，但概念上不能混淆。否则当 agent 失败时，我们无法判断问题来自 dynamics 错误、reward/objective 错误，还是 search/planning 错误。

在这个 POMDP 视角下，L1、L2、L3 的区别可以写得很清楚。L1 只要求 local operators 有用，尤其是一步 latent transition。L2 要求模型支持 trajectory-level query：

$$
\hat p(\tau \mid z_0, a_{1:H}, c), \quad \tau=(z_1,\ldots,z_H),
$$

其中 $a_{1:H}$ 是候选 action sequence，$c$ 是 governing-law constraints。L3 则进一步把模型栈本身写成可修正对象：

$$
(M_t, d_t) \xrightarrow{\text{diagnose + distill + validate}} M_{t+1},
$$

其中 $M_t$ 是当前 world-modeling stack，$d_t$ 是新部署证据、错误轨迹、counterexamples 或测试结果。换言之，L3 不只是“在线用模型规划”，而是 **模型在证据压力下能否把失败变成可验证、可复用、可回滚的更新**。

论文还强调，四类 governing laws 不是 modality 标签，而是 transition validity 的来源。物理世界的 transition 可以被物理规律或模拟器检查；数字世界的 transition 可以用程序执行和状态机检查；社会世界的 transition 需要追踪信念、承诺、角色和规范；科学世界的 transition 需要实验测量与因果证据。真实系统常常混合多个 regime，例如自动驾驶同时有物理动力学和社会意图，Minecraft agent 同时有 3D 运动和游戏规则，自动实验室同时有 scientific hypothesis 和 physical manipulation。

## 3. Algorithm / Methods / Model

论文的主体方法不是提出一个新模型，而是建立一套 **capability hierarchy/能力层级**。这个层级应该按 agent 在运行时调用的能力理解，而不是按模型名称静态分类。同一个系统在低层感知时可能只用 L1，在比较行动方案时调用 L2，在反复部署失败后进入 L3。

**L1 Predictor** 是最基础的一层。它学习局部预测算子，目标是在当前 latent state 和 action 给定时预测下一步 latent、observation 或 action。典型组件包括 state inference、forward dynamics、observation decoding 和 inverse dynamics。Dreamer 系列的 RSSM、MuZero 的 dynamics function、TD-MPC 的 value-aware latent dynamics、JEPA/CPC/SPR 这类预测式表征学习，都可以被放入 L1 视角理解。L1 的强项是把高维 observation 压缩成对决策有用的 latent，并建立短程 transition；它的弱点是 **一步预测好不等于多步组合可靠**。

L1 到 **L2 Simulator** 的跃迁发生在模型开始服务 trajectory-level decision query 时。论文给出三条边界条件。第一是 **long-horizon coherence/长时程一致性**：rollout 不能在几步之后因为 compounding error 进入无意义分支。第二是 **intervention sensitivity/干预敏感性**：改变 action 或 premise 后，未来轨迹要发生稳定且方向合理的变化。第三是 **constraint consistency/约束一致性**：生成的轨迹必须尊重目标环境的 governing laws。形式上，L2 不只是连乘一步转移，而应受到整条轨迹的约束项控制：

$$
\hat p(\tau \mid z_0,a_{1:H},c) \propto
\prod_{t=1}^{H} p_\theta(z_t\mid z_{t-1},a_t)\,\phi_c(\tau).
\tag{1}
$$

这里 $\phi_c(\tau)$ 表示整条轨迹与环境约束的相容性。这个公式很关键，因为它解释了 L2 为什么不能被一步预测误差替代：约束不是每一步独立成立就够了，而是要在完整 rollout 上成立。一个模型可以单帧漂亮、单步准确，却在多步后违反物理接触、API contract、社会承诺或科学因果链。

论文随后按四类 regime 展开 L2。物理世界中，L2 需要处理 geometry、kinematics、contact、stability 和 conservation；视频 world models、机器人 simulator、autonomous driving world models 都属于这里，但作者反复提醒 **appearance-first generation** 不等于可规划模拟。数字世界中，L2 面对 web、GUI、code 和 game environments，核心约束是 DOM、permission、API、file system、execution result 和 race condition。社会世界中，L2 必须维护 underlying social state，例如 beliefs、goals、relations、commitments 和 norms；如果 persona、承诺或角色在多轮中漂移，生成的对话就不再是可靠社会模拟。科学世界中，L2 分成 system dynamics simulation 和 experiment-decision simulation：前者用 neural operators、weather models、molecular surrogates 等替代昂贵求解，后者用 surrogate models 辅助实验序列选择。

L2 的 failure modes 也非常值得保留。**Compounding error** 会让小的一步误差在长程 rollout 中被放大；**state aliasing and drift** 会把真实上不同的状态压成相同 latent，导致 agent 在看似熟悉的状态里采取错误行动；**controllability failure** 会让模型生成漂亮但对 action 不敏感的未来；**simulator escape** 会让 planner 利用模拟器或评测环境漏洞；**calibration failure under distribution shift** 则让模型在 UI 改版、物体属性变化、社会规范漂移或实验设备重标定后仍然过度自信。这些问题共同说明，world model 的平均预测质量不是核心，真正关键的是 **能否在失败发生时定位哪条约束被破坏**。

**L3 Evolver** 是论文最有野心的部分。作者认为 L3 的本质不是更长 rollout，而是 world-modeling stack 本身能在失败证据面前更新。一个完整 L3 loop 包括 design、execute、observe、reflect：系统设计实验或 probe，执行它，观测结果，把结果蒸馏成 evidence，再修正 $M_t$。这和 autonomous science 很像，因为科学实践本身就是模型、实验、反证和修正的闭环。论文给出三条 L3 边界：主动扩展信息，而不是只优化已有知识；自主执行和观测，而不是只依赖静态数据；在挑战下修正 belief、参数、结构或资产，而不是把反馈留在上下文里。

L3 还涉及三种增长模式。最弱的是参数更新，例如在线学习或 continual fine-tuning；更强的是结构更新，例如增加模块、memory、parser 或专家；最难的是 **hypothesis-space expansion/假设空间扩展**，即系统意识到“正确解释不在当前候选集合里”，于是引入新变量、新机制或新抽象。作者认为这也是 latent representation 和 symbolic representation 的关键分歧：latent dynamics 很适合 L1/L2 的大规模学习，但 L3 需要显式操作 governing laws 时，符号化、程序化或可检查的表示会重新变得重要。

论文的架构建议也围绕这条主线展开。一个 world-model system 至少要同时选择 **representation、dynamics 和 control interface**。Representation 可以是 symbolic/programmatic states、continuous latents、structured 3D states 或 discrete tokens；dynamics 可以是 stochastic latent models、deterministic value-aware models、autoregressive token dynamics 或 diffusion-based dynamics；control interface 则包括 MPC、tree search、imagined-rollout policy optimization、offline distillation 和 replayable environment。作者的观点很明确：这些组件不能脱离 regime 来评价。物理世界需要 contact-aware representation 和低延迟控制；数字世界更适合 DOM、state-machine、execution trace 和 regression gates；社会世界需要稳定 persona、belief 和 commitment state；科学世界需要 hypothesis-evidence chain 和可校验实验记录。

这里有一个很重要的工程判断：**把学到的东西和必须强制执行的约束分开**。如果碰撞检测、API contract、type constraints、norm checks 或 regression tests 可以用硬约束实现，就不应该完全寄希望于模型在训练中“学会不要违反”。对 L2 来说，硬约束能阻止 rollout 变成漂亮但非法的未来；对 L3 来说，硬约束和 trace logs 还是判断更新能否推广的门禁。这个判断对后续读 world-model papers 很有用，因为它能区分“模型容量不足”和“系统架构没有给验证留下接口”这两类完全不同的问题。

> [!note] L1 / L2 / L3 Boundary
>
> **L1** 的问题是“下一步会怎样”；**L2** 的问题是“如果执行这一串 action，未来轨迹是否可用于决策”；**L3** 的问题是“当模型被证据证明错了，它能否诊断错误、生成持久修正并通过验证门控”。这三者是 containment hierarchy：L2 调用 L1 组成 rollout，L3 调用 L2 设计 probe 并收集修正模型所需的 evidence。

> [!tip]
> 这篇最有用的阅读方式，是把它当成 **diagnostic language**。它不是告诉你某个模型绝对属于哪一类，而是逼你问：这篇工作到底测试了哪种 transition query？它有没有测 action perturbation？有没有测 constraint violation？有没有记录失败证据？有没有把失败蒸馏成持久资产？

## 4. Experiments

论文没有传统意义上的实验部分，因为它是 survey/position paper；真正对应“实验支撑”的部分是 evaluation 章节和 benchmark landscape。作者的核心批评是：当前 world model evaluation 仍然过度依赖 prediction-centric 或 generation-centric metrics，例如 FID、FVD、SSIM、per-pixel reconstruction loss、单步预测误差或固定任务成功率。对 agent 来说，这些指标都太弱，因为它们不能直接回答模型是否能改善决策。

作者提出转向 **decision-centric evaluation/决策中心评价**。评价对象应该是 trajectory-level rollout $\hat p(\tau\mid z_0,a_{1:H},c)$，而不是孤立的一步预测。一个世界模型只有在 planner 使用它之后能提高真实环境中的 task outcome，才说明它对 agent 有价值。论文用 **Action Success Rate/ASR** 衡量使用 world model 选 action 后任务是否成功，用 **Counterfactual Outcome Deviation/COD** 衡量两个只在某一步 action 不同的 policy 是否产生任务相关的未来差异。ASR 连接 world model 和 downstream performance，COD 则直接测试 intervention sensitivity。

在具体 benchmark 上，论文强调同一个 benchmark 可以按协议不同测试不同层级。RoboCasa 在 L1 可以只是预测下一步 end-effector position；在 L2 需要完整厨房任务并注入 drawer obstruction 或 object displacement；在 L3 则要求 agent 从反复失败中蒸馏出持久 grasp strategy。OSWorld 或 SWE-bench 在 L1 可以是点击预测或单行代码补全；在 L2 是跨文件、跨 UI 状态的长程任务；在 L3 则要生成可复用 regression test、安装步骤或调试资产。Sotopia 在 L1 是 next-turn prediction，在 L2 是 counterfactual social strategy change，在 L3 是从反复谈判失败中修正社会策略或 norm-handling rule。ScienceWorld 和 DiscoveryBench 最接近 L3，因为科学任务天然要求 evidence-driven hypothesis revision。

论文还提出 **Minimal Reproducible Evaluation Package/MREP**，包括 environment version locking、完整 trace logs、failure taxonomy、tail statistics 和 boundary-condition mapping。这个建议非常实用，因为 world model 的失败通常发生在长程 rollout 的中间状态。没有 action、observation、receipt、error code、DOM snapshot、test output 或实验记录，后续就无法判断失败来自 state aliasing、constraint violation、planner error 还是环境变化。MREP 也直接服务 L3，因为 L3 的 governed validation 需要同样的证据基础设施来决定更新是否应该推广或回滚。

论文对当前证据的批评也很清楚。大多数系统仍然只在 L1 或浅 L2 被评估；L2 的三条边界条件虽然可定义，但没有成为标准报告格式；L3 的评测基础设施在 autonomous science 之外几乎不存在。尤其是 intervention sensitivity 常被忽略：很多论文报告生成质量或任务成功率，却没有显式改变 action 并测未来是否以任务相关方式改变。这会让 action-insensitive simulator 被误当作 planning-ready world model。

从 evidence strength 看，作者认为 scientific regime 的 L3 最成熟，因为实验反馈、假设反证和 surrogate update 都比较清晰；digital regime 有 regression tests 和 execution traces，所以具备部分 L3 条件；physical regime 的主要困难是失败归因，一个机器人失败可能来自感知、动力学、执行器或环境变化；social regime 最弱，因为社会预测失败的原因高度模糊，而且真实社会实验有伦理约束。

这篇论文自身的限制也必须讲清楚。它综合大量文献，但并没有对所有代表系统进行统一复现实验；很多表格里的 L1/L2/L3 或 boundary-condition 判断依赖作者的框架性归纳，不是同一协议下的可比实验。它提出的 L3 概念很重要，但也容易被过度使用：一个系统有 feedback loop 或 self-reflection，不等于它已经是 L3；只有当反馈被转化成 **persistent reusable update**，并通过 regression/robustness gates 验证，才满足 L3 的关键条件。

另一个实验层面的缺口是 cross-regime evaluation。很多真实 agent 同时处在多个 regime 里，例如浏览器自动化既有数字状态机，也有用户意图和社会规范；自动实验室既有科学假设，也有物理仪器约束。只在单一 benchmark 上报告结果，会隐藏跨约束传播的失败。一个软件状态误判可能毁掉科学实验计划，一个物理碰撞预测错误也可能让社会意图预测失去意义。因此后续评价更应该报告 **joint constraint satisfaction**，而不是把各类约束拆成彼此独立的小分数。

## 5. Related Work & Future Work

这篇论文的 related work 本身覆盖面极广，但最重要的关系可以压成三条线。第一条是 model-based RL。Dreamer、MuZero、TD-MPC、PETS、MBPO 等工作已经把 learned dynamics、imagined rollout 和 planning 接进 agent training；本文把它们放在更大的世界模型谱系中，指出传统 model-based RL 主要解决 L1/L2，而 L3 需要更强的证据闭环和验证基础设施。

第二条是 generative world models。Sora、Genie、Cosmos、GAIA、interactive video generation 和 3D world generation 让“模拟世界”看起来更真实，但本文反复指出，**perceptual realism 不等于 decision usefulness**。未来真正关键的是 action controllability、long-horizon consistency、physics/geometry constraints 和 explicit evaluation under intervention，而不是只把视频质量继续推高。

第三条是 agentic systems 与 evaluation。Web/GUI/code agents、multi-agent social simulators、autonomous laboratories 和 algorithmic discovery systems 都在把 world modeling 推向部署环境。这里和仓库里的 `OpenClaw_RL`、`steering_the_herd`、`cybergym` 关系很具体：`OpenClaw_RL` 的 next-state signal 可以被理解为 agent 与环境交互后暴露出的 transition evidence；`steering_the_herd` 展示社会 belief dynamics 如何被 planner 影响；`cybergym` 则说明 digital-world agent 评价必须依赖可执行 oracle、trace 和 post-condition，而不是文本 judge。

未来工作最值得关注的是三件事。第一，**law-consistent representation**：模型需要显式保留 planner 需要的变量，例如 free space、contact state、permission state、commitment graph、reaction pathway 或 causal mechanism。第二，**counterfactual and perturbation evaluation**：没有 action perturbation 和 constraint checking，world model 很容易被视觉质量或平均成功率掩盖。第三，**governed self-evolution**：L3 系统需要 versioning、held-out probes、canary deployment、rollback、evidence provenance 和 privacy/safety boundary，否则自我修正会变成不可审计的自动漂移。

论文最后提出的 **beyond L3 / meta-world modeling** 也值得单独记一下。L3 仍然假设系统面对的是同一个 underlying reality，只是在证据压力下修正自己的模型；meta-world modeling 则进一步追问系统能否操作“可能世界的规则空间”，例如生成、比较或修改不同的 governing laws。这一部分还很 speculative，但它揭示了本文最深的张力：如果世界模型的终点是 law revision，那么纯粹隐式 latent dynamics 可能不够，系统需要能显式表达、组合、验证和修改规则的表示。这也是为什么作者在结论里把 symbolic substrate 重新放回中心，而不是把 scaling 当成唯一答案。
