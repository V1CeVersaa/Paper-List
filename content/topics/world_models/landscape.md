---
title: Landscape
headline: Landscape of World Models
description: Topic map, methodology spectrum, and reading roadmap for World Models.
status: complete
visibility: public
topic: "world_models"
updated: "2026-05-09"
---

## Problem Space

World modeling 的核心问题是：**一个 agent 在行动之前，到底能不能拥有一个足够可靠的环境模型，用来预测 action 会怎样改变世界**。在传统监督学习里，预测通常是静态的；在 agent setting 里，预测必须变成可干预的 transition query。模型不仅要知道“下一个 observation 可能是什么”，还要回答“如果我换一个 action，未来轨迹会不会按合理方向改变”，并且这些轨迹不能违反目标环境的约束。

这个问题难在三层同时存在。第一层是 **state abstraction**：真实环境部分可观测，agent 只能从 pixels、tokens、GUI、terminal logs、sensor readings 或实验数据中压缩出 latent state。第二层是 **long-horizon composition**：one-step prediction 看起来准确，不代表连续滚动几十步仍然保持一致。第三层是 **constraint grounding**：不同环境的“合法转移”完全不同，物理世界要求接触、重力、几何和能量一致；数字世界要求 API、DOM、文件系统和类型约束一致；社会世界要求信念、承诺、角色和规范一致；科学世界要求实验观测、因果链和机制假设一致。

这也解释了为什么 world models 不能只按 modality 分类。视频、代码、机器人、社会模拟和科学发现看起来很不一样，但它们都在问同一个底层问题：**预测是否能支持决策**。如果模型只生成看起来 plausible 的未来，却对 action 改变不敏感，或者生成违反约束的轨迹，那么它对 agent planning 来说就是危险的。

## Methodology Spectrum

方法谱系可以从保守到激进理解。最保守的一端是 **L1 Predictor**：模型学习 state inference、forward dynamics、observation decoding 和 inverse dynamics 等 local operators，目标是让 latent state 能预测下一步或短程未来。Dreamer、MuZero、TD-MPC、JEPA、CPC、V-JEPA 等工作都可以放在这一层理解。它们构成 world model 的底座，但单独的 local prediction 还不足以支撑复杂规划。

中间层是 **L2 Simulator**：模型把 local operators 组合成 multi-step, action-conditioned rollout，使 planner 能比较候选 action sequence。这里的关键不再是单步误差，而是 **long-horizon coherence、intervention sensitivity 和 constraint consistency**。物理和机器人方向会强调 contact、geometry、kinematics 和 sim-to-real；数字世界会强调 UI state、API contract、execution traces 和 deterministic replay；社会模拟会强调 Theory of Mind、commitment graph 和 norm consistency；科学世界会强调 surrogate models、experiment sequence 和 evidence-chain validity。

最激进的一端是 **L3 Evolver**：模型不再只是固定 simulator，而是把自身 transition model 当成可修正对象。系统在部署中发现 anomaly，设计实验或 probe，收集 evidence，然后把修正蒸馏成持久资产，例如新的参数、规则、测试、parser、skills、hypothesis space 或实验策略。当前最成熟的例子多在 autonomous science 和 algorithmic discovery，因为这些场景有清晰的实验反馈和验证门控；社会世界的 L3 仍然最弱，因为 attribution 和伦理约束都非常硬。

## Evolution

这条线最早来自 model-based control 和 model-based RL：系统先学习 transition dynamics，再用 planning 或 imagined rollouts 提高 sample efficiency。`World Models`、PlaNet、Dreamer、MuZero 和 TD-MPC 这一支让 latent dynamics 成为可训练 agent 的核心组件。

随后，大规模 generative modeling 把 world model 推到更宽的场景。视频生成、3D world generation、autonomous driving simulation 和 embodied video prediction 让“世界模型”不再只是 RL 里的 dynamics function，而变成可以生成、控制和评估未来 observation 的高容量模型。但这也暴露出一个问题：**视觉真实感经常领先于物理可用性**，所以单看 FVD、FID 或 pixel metrics 很容易高估模型的 planning value。

再往后，LLM agents 把 digital world 和 social world 推到台前。Web agents、GUI agents、SWE agents 和 multi-agent social simulation 让 world model 面对 DOM、文件系统、API error、permission state、belief state、role drift 和 norm changes。这些场景的规律不像物理那样全部由几何和力学决定，却仍然要求 transition 可回放、可归因、可验证。

最新趋势是把 world modeling 与 evidence-driven revision 合并。Autonomous laboratories、scientific discovery agents、algorithmic discovery systems 和 regression-gated software agents 开始接近 **L3 Evolver**：系统不仅预测未来，还在失败后更新模型和工具链。这个阶段最关键的转折不是模型变大，而是 **evaluation infrastructure、trace logging、regression gates 和 rollback policies** 变成 world-model architecture 的一部分。

## Key Open Questions

第一个开放问题是 **visual plausibility 与 physical faithfulness 的断裂**。高质量视频模型可以生成连贯画面，但仍可能违反接触、支撑、守恒、遮挡和 object permanence。对 agent 来说，这种错误比画质低更危险，因为 planner 会在错误的模拟世界中选择行动。

第二个开放问题是 **intervention sensitivity**。很多生成模型能续写未来，却不能稳定回答“如果 action 改变，未来应该怎样变”。如果模型对 action 不敏感，它就不能支持 counterfactual planning。world model evaluation 需要显式测试 action perturbation，而不是只测输出像不像。

第三个开放问题是 **partially observable digital environments**。真实软件不是完全可见的状态机，后台 session、数据库、异步请求、权限、网络失败和多用户并发都会影响 transition。现有 GUI/web/code agents 大多没有维护完整 belief distribution，因此很容易在不可见状态变化中误判。

第四个开放问题是 **social world attribution**。社会转移由信念、目标、规范和角色共同决定。一个 social simulator 预测失败，可能是 persona 表示错了，也可能是 norm 已经漂移，还可能是另一个 agent 的隐藏策略变化。如何从可观测对话中归因这种失败，是 social L3 的核心瓶颈。

第五个开放问题是 **self-evolving model governance**。L3 系统会持续修改自己，因此必须同时满足 stability、plasticity 和 auditability。没有 versioning、canary deployment、held-out regression tests 和 evidence provenance，自我修正很容易退化成 benchmark overfitting、知识污染或错误归因级联。

## Reading Roadmap

入门时应先读 **Agentic World Modeling**，因为它提供了当前 topic 的总坐标系：L1 Predictor、L2 Simulator、L3 Evolver，以及 physical、digital、social、scientific 四类 governing-law regimes。读完它之后，再回到 Dreamer、MuZero、TD-MPC 这类 model-based RL 工作，会更容易看清它们分别承担 state inference、latent dynamics、imagined rollout 和 planning interface 中的哪一部分。

进入核心阶段后，应沿着不同 regime 各读一组代表工作。物理和 embodied 方向可以围绕 DreamerV3、TD-MPC2、Genie、Sora、GAIA-1、Cosmos 或 PhyWorldBench 建立“生成画面”和“尊重物理约束”的区别；数字世界可以读 WebDreamer、OSWorld、SWE-bench 和 GUI/world simulation 相关工作，理解 execution trace 为什么比文本判断更可靠；社会世界可以读 Generative Agents、Sotopia、CICERO 和 `Steering the Herd`，看 social state、belief、norm 和 strategic interaction 如何进入 transition model。

高级阶段应关注 **L3 与评价基础设施**。Autonomous science 里的 CAMEO、A-Lab、BacterAI 和 AI Scientist 系列最适合作为 L3 样板，因为它们把设计实验、执行实验、观测结果和修正模型接成闭环。数字世界里，FunSearch、AlphaEvolve、SWE-agent、CyberGym 和 regression-gated agents 展示了另一条路线：把可执行环境、测试和日志变成 self-improving agent 的验证门控。读到这里时，重点已经不是“模型能不能预测”，而是 **预测失败后系统能不能产生可验证、可回滚、可复用的修正**。
