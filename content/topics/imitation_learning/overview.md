---
title: "Overview"
headline: "Overview of Imitation Learning"
visibility: "public"
status: "complete"
description: "Queue and reading progress for Imitation Learning."
---

## Topic Boundary

这个 overview 记录 **非 LLM 语境下的 Imitation Learning**。主动查询、selective labeling、multiple oracle、preference-based active query 这类工作统一放在子目录 [Active Imitation Learning](./active_imitation_learning/overview.md)；它们仍然属于 IL，只是子问题更强调 query policy、expert feedback cost 和 on-policy interaction。LLM agent 的 RL/RLVR/RLHF 训练不从这里收。

## Tutorial

- [ ] arXiv 2018: An Algorithmic Perspective on Imitation Learning, [arXiv](https://arxiv.org/abs/1811.06711)

## Papers

- [ ] AISTATS 2011, **DAgger**: A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning, [arXiv](https://arxiv.org/abs/1011.0686), [Note](./DAgger.md)
- [ ] ICML 2017, **AggreVaTeD**: Deeply AggreVaTeD:  Differentiable Imitation Learning for Sequential Prediction, [arXiv](https://arxiv.org/abs/1703.01030), [Note](./AggreVaTeD.md)
- [x] ICML 2022, **ILEED**: Imitation Learning by Estimating Expertise of Demonstrators, [arXiv](https://arxiv.org/abs/2202.01288), [Note](./ILEED.md)

**Adversarial Imitation Learning**:

- [x] NIPS 2016, **GAIL**: Generative Adversarial Imitation Learning, [arXiv](https://arxiv.org/abs/1606.03476), [Note](./GAIL.md)
- [ ] ICLR 2017, **TPIL**: Third-Person Imitation Learning, [arXiv](https://arxiv.org/abs/1703.01703), [Note](./TRIL.md)
- [ ] NIPS 2017, **InfoGAIL**: Interpretable Imitation Learning from Visual Demonstrations, [arXiv](https://arxiv.org/abs/1703.08840), [Note](./InfoGAIL.md)
- [ ] IJCAI 2020, **Triple-GAIL**: Triple-GAIL: A Multi-Modal Imitation Learning Framework, [arXiv](https://arxiv.org/abs/2005.10622)
- [ ] IJCAI 2021, **SAIL**: Robust Adversarial Imitation Learning via Adaptively-Selected Demonstrations, [IJCAI](https://www.ijcai.org/proceedings/2021/434)
- [ ] ICLR 2023 Spotlight, **HOIL**: Seeing Differently, Acting Similarly: Heterogeneously Observable Imitation Learning, [arXiv](https://arxiv.org/abs/2106.09256), [Note](./HOIL.md)
- [ ] ICML 2023, **PCIL**: Policy Contrastive Imitation Learning, [arXiv](https://arxiv.org/abs/2307.02829)

**Policy Distillation**:

- [ ] ICLR 2016, **Policy Distillation**: Policy Distillation, [arXiv](https://arxiv.org/abs/1511.06295), [Note](./Distillation.md)
- [ ] AISTATS 2019, **Distilling Policy Distillation**: Distilling Policy Distillation, [arXiv](https://arxiv.org/abs/1902.02186), [Note](./Distilling_Distillation.md)

**Explainable Imitation Learning**:

- [ ] ICLR 2025 Spotlight: Student-Informed Teacher Training, [arXiv](https://arxiv.org/abs/2412.09149), [Note](./SITT.md)

## Active Imitation Learning

AIL 的完整子队列见 [Active Imitation Learning](./active_imitation_learning/overview.md)。这里保留它作为 IL 的子方向，而不是全局一级 topic，是为了明确 **AIL ⊂ IL**，并且把边界限制在传统控制、MDP、DRL 和 oracle-query 模仿学习问题上。
