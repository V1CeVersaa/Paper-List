---
title: "Textual Reasoning"
visibility: "public"
status: "complete"
topic: "textual_reasoning"
updated: "2026-04-13"
description: "Landing page for textual reasoning, reasoning post-training, and reasoning-time control."
---

## Overview

[Overview](./overview.md) 记录这个 topic 已经读过与值得继续排队的 reasoning 论文。这里的主线不是泛泛的“模型会不会想”，而是 **文本中的推理轨迹如何被搜索、蒸馏、RL 放大、budget forcing 控制，以及这些训练方式到底是在创造新能力，还是在重排已有能力。**

## Scope

这个 topic 现在覆盖两类紧密相连的工作。第一类是 **reasoning scaffold 与 search-style inference**，例如 `Tree of Thoughts` 这种把思维树显式化的工作；第二类是 **reasoning post-training 与 reasoning-time control**，例如 `DeepSeek-R1`、`DAPO`、`GRPO` 动力学、`LIMO` 和 `s1` 这一支。它们共同讨论的是：**推理表现到底如何被训练、采样和推理时预算控制出来。**

它和 [Safety & Alignment](../safety_alignment/index.md) 的分界也需要很清楚。凡是主要在讨论 **deceptive alignment、reward hacking、弱到强监督、sycophancy、character stabilization 或安全相关内部表征** 的论文，都应该放到 `safety_alignment`；凡是主要在讨论 **reasoning trace、CoT、test-time search、RLVR 是否真的扩展了 reasoning capacity** 的论文，则留在这里。`Reasoning Models Don't Always Say What They Think` 这种跨界论文之所以被放在这里，是因为它正面追问的是 **verbalized reasoning 与真实求解过程之间的关系**。
