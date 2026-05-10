---
title: "World Models"
description: "Predictive, simulatable, and evolvable models of environment dynamics for planning agents, model-based reinforcement learning, and agentic systems."
status: complete
visibility: public
topic: "world_models"
updated: "2026-05-09"
---

## Overview

- [Overview](./overview.md)
- [Landscape](./landscape.md)

## Scope

这个 topic 讨论 **世界模型/World Model**：系统如何学习环境的状态、转移、约束和可干预结构，使 agent 能在真实行动之前预测未来、比较行动后果、验证计划是否违反环境规律，并在部署证据表明模型失效时修正自己的模型。这里的重点不是“生成出来的画面像不像”，而是 **预测是否能服务决策**：模型是否能保持长时程一致性、是否对 action 或 premise 的改变敏感、是否尊重物理、软件、社会或科学场景中的 governing laws。

它和 [Agentic RL](../agentic_rl/index.md) 紧密相邻，但边界不同。`agentic_rl` 主要研究 **如何训练多步交互策略**，核心问题是 reward design、credit assignment、rollout throughput、online optimization 和训练稳定性；`world_models` 则追问这些策略依赖的 **环境模型本身是否可靠**。如果一篇论文主要提出新的 agent RL 训练算法，它应优先进入 `agentic_rl`；如果它主要研究 state-transition model、simulation substrate、environment emulator、counterfactual rollout、constraint-consistent planning 或 evidence-driven model revision，它就属于这里。

它和 [Reinforcement Learning](../reinforcement_learning/index.md) 也要分开。经典 model-based RL、POMDP、planning 和控制理论仍是基础地基，但这个 topic 不只收传统 RL 算法，还覆盖 **video world models、web/GUI simulators、social simulacra、scientific surrogate models 和 autonomous experimentation loops**。换言之，`reinforcement_learning` 更偏方法传统，`world_models` 更偏跨场景的环境建模能力。

它和 [Representation Learning](../representation_learning/index.md) 的交叉在 latent state 上。表征学习关心 **怎样得到好的 representation**；world modeling 进一步要求这个 representation 能被 transition operator、rollout query、planner 或 evaluator 使用。只要论文的核心贡献停在“学到更好的特征”，它应留在 `representation_learning`；只有当它把表征接入 **可组合转移、行动条件模拟、约束验证或模型自我修正** 时，才进入 `world_models`。

这个 topic 目前优先保留三类工作：第一类是 **L1 Predictor**，即 one-step 或 short-horizon transition models；第二类是 **L2 Simulator**，即可被 planner 查询的 long-horizon, action-conditioned rollout systems；第三类是 **L3 Evolver**，即能从 deployment evidence、实验反馈或 regression signals 中诊断失败、生成持久更新并通过验证门控的 self-improving world-model stack。
