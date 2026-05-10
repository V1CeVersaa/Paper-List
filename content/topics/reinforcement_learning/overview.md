---
title: "Overview"
headline: "Overview of Classical and Deep Reinforcement Learning"
visibility: "public"
status: "complete"
description: "Queue and reading progress for Classical & Deep RL."
---

这个 overview 只整理 **传统 RL、Deep RL 及其理论和模型基础**。LLM post-training、RLHF/RLVR、agent tool-use training、preference optimization 不在这里展开；它们分别走 [Textual Reasoning](../textual_reasoning/index.md)、[Agentic RL](../agentic_rl/index.md)、[Preference Learning](../preference_learning/index.md) 或 [Safety & Alignment](../safety_alignment/index.md)。

## Tutorial & Overview

- [ ] Book 2018: **Sutton & Barto**: Reinforcement Learning: An Introduction, [Book](http://incompleteideas.net/book/the-book-2nd.html), [Note](https://note.v1ceversaa.cc/RL/Sutton/index.html)
- [ ] arXiv 2018: An Introduction to Deep Reinforcement Learning, [arXiv](https://arxiv.org/abs/1811.12560),, [Note](https://note.v1ceversaa.cc/RL/Deep%20RL/index.html)
- [ ] arXiv 2024: Reinforcement Learning: An Overview, [arXiv](https://arxiv.org/abs/2412.05265)
- [ ] INFORMS Tutorial 2025: Statistical and Algorithmic Foundations of Reinforcement Learning, [arXiv](https://arxiv.org/abs/2507.14444), , [Slides](https://yuejiechi.github.io/talks/JSM2023_tutorial.pdf)

## Model-Free RL

- [ ] arXiv 2018: Reinforcement Learning and Control as Probabilistic Inference, [arXiv](https://arxiv.org/abs/1805.00909) 
- [ ] ICLR 2021 Oral: What Matters In On-Policy Reinforcement Learning? A Large-Scale Empirical Study, [arXiv](https://arxiv.org/abs/2006.05990)

### Value-Based Methods

- [ ] Nature 2015, **DQN**: Human-level Control through Deep Reinforcement Learning, [Nature](https://www.nature.com/articles/nature14236)
- [ ] AAAI 2016, **Double DQN**: Deep Reinforcement Learning with Double Q-learning, [arXiv](https://arxiv.org/abs/1509.06461)
- [ ] ICML 2017, **Soft Q-Learning**: Reinforcement Learning with Deep Energy-Based Policies, [arXiv](https://arxiv.org/abs/1702.08165)

### Policy Gradient & On-Policy Methods

**Tutorial**:

- [ ] arXiv 2024: The Definitive Guide to Policy Gradients in Deep Reinforcement Learning: Theory, Algorithms and Implementations, [arXiv](https://arxiv.org/abs/2401.13662), [Note](./pg_guide.md)

**Papers**:

- [x] ICLR 2016, **GAE**: High-Dimensional Continuous Control Using Generalized Advantage Estimation, [arXiv](https://arxiv.org/abs/1506.02438), [Note](./GAE.md)
- [x] NeurIPS 2022, **DAE**: Direct Advantage Estimation, [arXiv](https://arxiv.org/abs/2109.06093), [Note](./DAE.md)

- [ ] NIPS 1999: Policy Gradient Methods for Reinforcement Learning with Function Approximation, [NIPS](https://papers.nips.cc/paper_files/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html)
- [ ] ICML 2002, **CPI / Kakade & Langford**: Approximately Optimal Approximate Reinforcement Learning, [PDF](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/KakadeLangford-icml2002.pdf)
- [ ] NIPS 2001, **NPG**: A Natural Policy Gradient, [NIPS](https://papers.nips.cc/paper_files/paper/2001/hash/4b86abe48d358ecf194c56c69108433e-Abstract.html)
- [ ] ICML 2016, **A3C**: Asynchronous Methods for Deep Reinforcement Learning, [arXiv](https://arxiv.org/abs/1602.01783)
- [ ] ICML 2015, **TRPO**: Trust Region Policy Optimization, [arXiv](https://arxiv.org/abs/1502.05477)
- [ ] arXiv 2017, **PPO**: Proximal Policy Optimization Algorithms, [arXiv](https://arxiv.org/abs/1707.06347)

### Policy Gradient & Off-Policy Methods

- [ ] ICLR 2016, **DDPG**: Continuous Control with Deep Reinforcement Learning, [arXiv](https://arxiv.org/abs/1509.02971)
- [ ] ICML 2018, **SAC**: Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor, [arXiv](https://arxiv.org/abs/1801.01290)

### Offline RL

## Exploration Bonus

- [x] ICML 2017, **ICM**: Curiosity-driven Exploration by Self-supervised Prediction, [arXiv](https://arxiv.org/abs/1705.05363), [Note](./ICM.md)
- [ ] ICLR 2019, **RND**: Exploration by Random Network Distillation, [arXiv](https://arxiv.org/abs/1810.12894)

## Model-Based RL

- [ ] arXiv 2017: Learning Model-based Planning from Scratch, [arXiv](https://arxiv.org/pdf/1707.06170)
- [ ] ICML 2013: **Guided Policy Search**, [Online PDF](https://graphics.stanford.edu/projects/gpspaper/gps_full.pdf)
- [x] NIPS 2015: Data Generation as Sequential Decision Making, [arXiv](https://arxiv.org/abs/1506.03504), [Note](./DGSM.md)
- [ ] ICML 2017, **Predictron**: The Predictron: End-To-End Learning and Planning, [arXiv](https://arxiv.org/abs/1612.08810)
- [ ] NIPS 2017, **VPN**: Value Prediction Network, [arXiv](https://arxiv.org/abs/1707.03497)
- [ ] AAAI 2019, **CRAR**: Combined Reinforcement Learning via Abstract Representations, [arXiv](https://arxiv.org/abs/1809.04506)
- [ ] ICML 2019, **DeepMDP**: DeepMDP: Learning Continuous Latent Space Models for Representation Learning, [arXiv](https://arxiv.org/abs/1906.02736)

## Inverse Reinforcement Learning

**Tutorial**:

- [ ] arXiv 2018: A Survey of Inverse Reinforcement Learning: Challenges, Methods and Progress, [arXiv](https://arxiv.org/abs/1806.06877)

**Papers**:

- [ ] AAAI 2008, **MaxEnt IRL**: Maximum Entropy Inverse Reinforcement Learning, [AAAI](https://cdn.aaai.org/AAAI/2008/AAAI08-227.pdf), [Note](./MaxEnt.md)
- [ ] ICML 2016, **MaxEnt IOC**: Guided Cost Learning: Deep Inverse Optimal Control via Policy Optimization, [arXiv](https://arxiv.org/abs/1603.00448)
- [ ] ICLR 2018, **AIRL**: Learning Robust Rewards with Adverserial Inverse Reinforcement Learning, [arXiv](https://arxiv.org/abs/1710.11248)

## Self-Supervised RL

- [ ] NeurIPS 2025 Best Paper: 1000 Layer Networks for Self-Supervised RL: Scaling Depth Can Enable New Goal-Reaching Capabilities, [arXiv](https://arxiv.org/abs/2503.14858), [Note](./scale_self_supervised.md)

## Generalization & Overfitting

- [ ] arXiv 2018: A Study on Overfitting in Deep RL, [arXiv](https://arxiv.org/abs/1804.06893)
- [ ] arXiv 2018: A Dissection of Overfitting and Generalization in Continuous Reinforcement Learning, [arXiv](https://arxiv.org/abs/1806.07937)

## Representation Learning & Transfer

- [ ] ICML 2017, **DARLA**: Improving Zero-Shot Transfer in Reinforcement Learning, [arXiv](https://arxiv.org/abs/1707.08475)
- [ ] ICLR 2018 Workshop: Decoupling Dynamics and Reward for Transfer Learning, [arXiv](https://arxiv.org/abs/1804.10689)
- [ ] ICLR 2021, **DBC**: Learning Invariant Representations for Reinforcement Learning without Reconstruction, [arXiv](https://arxiv.org/abs/2006.10742)
- [ ] ICLR 2021, **HiP-BMP**: Learning Robust State Abstractions for Hidden-Parameter Block MDPs, [arXiv](https://arxiv.org/abs/2007.07206)

## Explainable RL

- [ ] NeurIPS 2019: Causal Confusion in Imitation Learning, [arXiv](https://arxiv.org/abs/1905.11979)
- [ ] ICLR 2018: Learning Sparse Neural Networks through L0 Regularization, [arXiv](https://arxiv.org/abs/1712.01312)
- [ ] NeurIPS 2023: StateMask: Explaining Deep Reinforcement Learning through State Mask, [OpenReview](https://openreview.net/forum?id=pzc6LnUxYN), [GitHub](https://github.com/nuwuxian/StateMask)

## Hierarchical RL / Temporal Abstraction

- [ ] AIJ 1999, **Options framework**: Between MDPs and Semi-MDPs: A Framework for Temporal Abstraction in RL, [AIJ/ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0004370299000521)
- [ ] AAAI 2017, **Option-Critic**: The Option-Critic Architecture, [arXiv](https://arxiv.org/abs/1609.05140)
- [ ] NIPS 2016, **STRAW**: Strategic Attentive Writer for Learning Macro-Actions, [arXiv](https://arxiv.org/abs/1606.04695)
