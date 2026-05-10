---
title: "Preference Learning"
headline: "Preference Learning"
visibility: "public"
status: "complete"
description: "Landing page for preference learning, preference-based RL, and dueling bandit subtopics."
---

## Scope

这个 topic 用来收 **Preference Learning** 这一组以相对反馈为核心的工作：成对偏好、preference-based RL、reward/preference model、DPO/RLHF objective、judge aggregation，以及偏好信号如何被转成策略或排序模型。它和传统 RL 的区别在于，学习信号不是直接的数值 reward，而是 **两个对象、动作、回答或轨迹之间谁更好**。

[Dueling Bandit](./dueling_bandit/index.md) 现在作为这个 topic 下的子目录保存。这样处理比保留一个泛化的 “Bandit Theory” 一级 topic 更干净，因为当前仓库里真正和 bandit 重叠的部分主要是 **成对比较反馈、relative preference、以及从 pairwise query 中做探索/利用权衡**。如果以后要系统整理 stochastic bandit、adversarial bandit 或 Bayesian bandit，而它们不依赖偏好比较，再单独开传统 bandit topic 会更合理。
