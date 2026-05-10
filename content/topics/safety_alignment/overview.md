---
title: Overview
headline: Overview of Safety & Alignment
description: Queue and reading progress for Safety & Alignment.
status: complete
visibility: public
topic: "safety_alignment"
updated: "2026-05-02"
---

## Reading Queue

这个队列现在按 **fine-tuning safety 风险前史 -> emergent misalignment 现象与扩展 -> activation / representation steering 底座 -> persona / direction / feature 机制化解释 -> mitigation 与工具基础设施** 的顺序排。这样读会更顺，因为你会先看到 narrow fine-tuning 为什么不是一个局部改参数的小问题，再看到大模型内部的 **linear concept representation** 如何被读出和干预，最后回到 **broad misalignment 是否也是一种可定位、可迁移、可控制的 feature structure**。底层特征学习底座已经放在 [Representation Learning](../representation_learning/overview.md)：`Deep Neural Feature Ansatz` / `AGOP` / `RFM` 是 `Universal Steering & Monitoring` 的数学前提，不在这里重复建队列。与 reasoning verbalization 更直接相关的 `Reasoning Models Don't Always Say What They Think` 已经明确留在 [Textual Reasoning](../textual_reasoning/overview.md)，不再混在这里。

<!-- AUTO:QUEUE:START -->
- [x] ICLR 2024 Oral: Fine-tuning Aligned Language Models Compromises Safety, Even When Users Do Not Intend To!, [arXiv](https://arxiv.org/abs/2310.03693), [Note](Fine_Tuning_Aligned_LMs_Compromises_Safety.md)
- [x] arXiv 2024: Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training, [arXiv](https://arxiv.org/abs/2401.05566), [Note](Sleeper_Agents.md)
- [x] arXiv 2024: Alignment Faking in Large Language Models, [arXiv](https://arxiv.org/abs/2412.14093), [Note](Alignment_Faking.md)
- [x] Nature 2026: Training large language models on narrow tasks can lead to broad misalignment, [Paper](https://www.nature.com/articles/s41586-025-09937-5), [Note](Narrow_Tasks_Broad_Misalignment.md)
- [x] ICML 2025 Workshop: Model Organisms for Emergent Misalignment, [arXiv](https://arxiv.org/abs/2506.11613), [Note](Model_Organisms_for_Emergent_Misalignment.md)
- [x] Anthropic Research Blog 2025: From Shortcuts to Sabotage: Natural Emergent Misalignment from Reward Hacking, [Paper](https://www.anthropic.com/research/emergent-misalignment-reward-hacking), [Note](From_Shortcuts_to_Sabotage.md)
- [ ] arXiv 2025: Natural Emergent Misalignment from Reward Hacking in Production RL, [arXiv](https://arxiv.org/abs/2511.18397)
- [ ] arXiv 2025: School of Reward Hacks: Hacking harmless tasks generalizes to misaligned behavior in LLMs, [arXiv](https://arxiv.org/abs/2508.17511)
- [ ] arXiv 2025: Subliminal Learning: Language models transmit behavioral traits via hidden signals in data, [arXiv](https://arxiv.org/abs/2507.14805)
- [x] Computational Linguistics 2021: Probing Classifiers: Promises, Shortcomings, and Advances, [arXiv](https://arxiv.org/abs/2102.12452), [Note](Probing_Classifiers.md)
- [x] ICLR 2023: Discovering Latent Knowledge in Language Models Without Supervision, [arXiv](https://arxiv.org/abs/2212.03827), [Note](Discovering_Latent_Knowledge.md)
- [x] arXiv 2023: Steering Language Models With Activation Engineering, [arXiv](https://arxiv.org/abs/2308.10248), [Note](ActAdd.md)
- [ ] NeurIPS 2023 Spotlight: Inference-Time Intervention: Eliciting Truthful Answers from a Language Model, [arXiv](https://arxiv.org/abs/2306.03341)
- [x] arXiv 2023: Representation Engineering: A Top-Down Approach to AI Transparency, [arXiv](https://arxiv.org/abs/2310.01405), [Note](Representation_Engineering.md)
- [ ] ICML 2024: The Linear Representation Hypothesis and the Geometry of Large Language Models, [arXiv](https://arxiv.org/abs/2311.03658)
- [ ] NeurIPS 2024: Refusal in Language Models Is Mediated by a Single Direction, [arXiv](https://arxiv.org/abs/2406.11717)
- [x] arXiv 2023: Sparse Autoencoders Find Highly Interpretable Features in Language Models, [arXiv](https://arxiv.org/abs/2309.08600), [Note](Sparse_Autoencoders.md)
- [x] Science 2026: Toward Universal Steering and Monitoring of AI Models, [arXiv](https://arxiv.org/abs/2502.03708), [Note](Universal_Steering_Monitoring.md)
- [ ] arXiv 2025: Persona Features Control Emergent Misalignment, [arXiv](https://arxiv.org/abs/2506.19823)
- [ ] ICML 2025 Workshop: Convergent Linear Representations of Emergent Misalignment, [arXiv](https://arxiv.org/abs/2506.11618)
- [ ] Anthropic Research 2025: Persona Vectors: Monitoring and Controlling Character Traits in Language Models, [Paper](https://www.anthropic.com/research/persona-vectors)
- [x] ICML 2025: AxBench: Steering LLMs? Even Simple Baselines Outperform Sparse Autoencoders, [arXiv](https://arxiv.org/abs/2501.17148), [Note](AxBench.md)
- [ ] arXiv 2026: Efficient and Accurate Steering of LLMs through Attention-Guided Feature Learning, [arXiv](https://arxiv.org/abs/2602.00333)
- [ ] arXiv 2025: Steering Out-of-Distribution Generalization with Concept Ablation Fine-Tuning, [arXiv](https://arxiv.org/abs/2507.16795)
- [ ] ICLR 2026: On Scalable Oversight with Weak LLMs Judging Strong LLMs, [Paper](https://openreview.net/forum?id=O1fp9nVraj)
- [ ] OpenAI Research Blog 2026: Training Agents to Self-Report Misbehavior, [Paper](https://openai.com/index/training-agents-to-self-report-misbehavior/)
- [ ] arXiv 2025: Provably Mitigating Corruption, Overoptimization, and Verbosity in RLHF/DPO, [arXiv](https://arxiv.org/abs/2510.05526)
- [ ] ICLR 2026: Sycophancy Is Not One Thing: Causal Separation of Sycophantic Behaviors, [Paper](https://openreview.net/forum?id=d24zTCznJu)
- [ ] Anthropic Research 2026: The Assistant Axis: Situating and Stabilizing the Character of LLMs, [Paper](https://www.anthropic.com/research/assistant-axis)
- [ ] Transformer Circuits 2026: Emotion Concepts and their Function in a Large Language Model, [Paper](https://transformer-circuits.pub/2026/emotions/index.html)
- [ ] Transformer Circuits 2024: Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet, [Paper](https://transformer-circuits.pub/2024/scaling-monosemanticity/)
- [ ] BlackboxNLP 2024 Workshop: Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2, [arXiv](https://arxiv.org/abs/2408.05147)
- [ ] Transformer Circuits 2025: Circuit Tracing: Revealing Computational Graphs in Language Models, [Paper](https://transformer-circuits.pub/2025/attribution-graphs/biology.html)
- [ ] ICLR 2026: Cross-Architecture Model Diffing with Dedicated Feature Crosscoders, [Paper](https://openreview.net/forum?id=YXB8uigyOg)
- [ ] DeepMind Blog 2025: Gemma Scope 2, [Paper](https://deepmind.google/blog/gemma-scope-2/)
- [ ] arXiv 2025: A Unified Theory of Sparse Dictionary Learning in Mechanistic Interpretability, [arXiv](https://arxiv.org/abs/2512.05534)
- [ ] arXiv 2025: Sparse Attention Post-Training for Mechanistic Interpretability, [arXiv](https://arxiv.org/abs/2512.05865)
- [ ] arXiv 2025: Open Problems in Mechanistic Interpretability, [arXiv](https://arxiv.org/abs/2501.16496)
<!-- AUTO:QUEUE:END -->
