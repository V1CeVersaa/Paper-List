---
title: "Dueling Bandit"
headline: "Dueling Bandit"
visibility: "public"
status: "complete"
description: "Subtopic page for dueling bandit under Preference Learning."
---

## Scope

**Dueling Bandit** 现在归在 [Preference Learning](../index.md) 下面。这里讨论的 bandit 不是泛化的 stochastic/adversarial bandit theory，而是以 **成对比较反馈/pairwise preference feedback** 为核心的 setting：学习者选择两个 arms、actions、policies 或 responses，由 expert、human、judge 或环境给出相对偏好，再用这些比较结果做探索与利用。

这个边界和 [Classical & Deep RL](../../reinforcement_learning/index.md) 保持区分：如果反馈是数值 reward 或完整 MDP return，它更像 RL；如果反馈是两个候选之间的相对胜负，它更像 preference learning 下的 dueling bandit。
