---
title: "Representation Learning"
description: "Learning task-relevant representations that support transfer, abstraction, and generalization."
status: complete
visibility: public
topic: "representation_learning"
updated: "2026-04-16"
---

## Overview

[Overview](./overview.md) 记录这个 topic 里已经读过、正在排队、或者之后值得补齐的论文；[Landscape](./landscape.md) 则把 **representation learning** 的问题定义、方法谱系、关键转折和阅读路线压成一条清晰主线，方便先定边界，再决定下一篇该放进来什么。

## Scope

这个 topic 讨论的是 **如何从原始观测中学出有用的内部表征/representation**。这里真正关心的对象，不是某个单独 benchmark 的分数，而是表征本身是否抓住了 **语义结构、可迁移性/transferability、不变性/invariance、可分解性/disentanglement 和任务相关信息**。只要一篇工作的中心问题是“什么样的 latent representation 才算好，以及应该怎样学出来”，它就优先属于这里。

这个边界和仓库里的其他 topic 有明显交叉，但不应该混在一起。更偏 **视觉任务本身** 的工作，例如检测、分割、生成建模或视觉架构设计，即使也讨论 feature，一般还是优先放在 [Computer Vision](../computer_vision/index.md)；更偏 **状态抽象/state abstraction、world model、控制目标和 exploration** 的论文，应该优先放在 [Reinforcement Learning](../reinforcement_learning/index.md)；更偏 **feature probing、steering、mechanistic interpretability 和 safety diagnosis** 的工作，则更适合留在 [Safety & Alignment](../safety_alignment/index.md)。`Representation Learning` 只保留那些把 **表征空间的几何、信息保留方式、预训练目标和下游迁移能力** 放到舞台中央的论文。

因此，这个 topic 的主线会刻意收紧到三类材料。第一类是 **重建式或生成式 latent variable modeling**，例如 autoencoder、VAE、InfoGAN、beta-VAE 和 disentanglement 这一支；第二类是 **预测式与对比式自监督学习/self-supervised learning**，例如 CPC、SimCLR、MoCo、BYOL、DINO；第三类是 **masked modeling 与跨模态预训练**，例如 BERT、MAE、CLIP 这种把“好表征”直接和大规模迁移能力绑定起来的方法。后面往这个 topic 里加 paper 时，最该先问的问题始终是：它真正推进的，到底是某个应用任务，还是 **representation 本身的学习原则**。
