---
title: "Landscape"
headline: "Landscape of Agentic RL"
description: "Topic map, methodology spectrum, and reading roadmap for Agentic RL."
status: "complete"
visibility: "public"
topic: "agentic_rl"
updated: "2026-04-10"
---

## Problem Space

这条线真正研究的，不是“LLM 会不会推理”这么宽泛的问题，而是 **当 LLM 被放进真实环境里连续行动时，强化学习到底能不能把它从一个会生成文本的模型，推进成一个会规划、会试错、会用工具、会在失败后调整行为的 agent。** 一旦问题这样定义，研究对象就从单轮 prompt-response 变成了一个更硬的对象：观测是对话历史、网页状态、终端输出、GUI 反馈、搜索结果或测试日志，动作既包含 token 级生成，也包含 API 调用、代码执行和界面操作，奖励则往往直到很多步以后才由 **任务成败、单元测试、规则验证或环境反馈** 给出。

这也解释了为什么 **agentic RL** 和普通 RLHF 并不是一回事。InstructGPT 那套单轮偏好优化把“回答一次”看成一次 bandit 决策，核心难点是 reward model 与 KL 约束；而 agentic RL 面对的是 **长时程、部分可观测、混合动作空间**，困难会立刻变成 **信用分配/credit assignment、样本效率、探索/exploration、环境接口设计和训练稳定性**。更麻烦的是，LLM 并不是从随机策略起步。预训练已经给了它极强的语言先验，所以很多今天的重要问题，不是“如何从零学会”，而是 **如何把已有语言能力压进可执行的多步策略里**，以及 **哪些看上去像能力提升的现象，其实只是更会重排已有轨迹分布。**

因此，这个 topic 的边界必须卡得很硬。它不会试图吞掉所有 reasoning、所有 RLHF、所有 tool-use 或所有 coding agent 工作。这里优先保留三类东西：第一类是 **把 agent 交互形式定义清楚** 的工作，例如 ReAct、LATS、SWE-agent；第二类是 **真正对 multi-turn RL 算法或训练动力学有推进** 的工作，例如 ArCHer、DigiRL、WebRL、RAGEN；第三类是 **让训练规模化成为可能的系统与基础设施**，例如 AgentGym-RL、AgentRL、ComputerRL。更基础的 policy gradient、value function 和 exploration 理论放在 [Classical & Deep RL](../reinforcement_learning/index.md) 看，会更干净。

## Methodology Spectrum

如果把方法谱系摊开看，最左边首先是一批 **prompt-based agent scaffold**。ReAct 把 reasoning trace 和外部 action 拼在同一条轨迹里，告诉大家“思考”和“行动”不必分开；Reflexion 用 verbal memory 做失败反思，虽然没有梯度更新，但已经在模拟一种外部反馈驱动的策略修正；Tree of Thoughts、RAP 和 LATS 则把搜索、规划和环境反馈嫁接进来，让语言模型不再只是单链生成，而是可以在更高 token 成本下做显式探索。它们的价值不在于最终训练 recipe，而在于 **先把 agent 的状态、动作和回路结构搭出来**。

再往右走，就是 **post-training objective 的演化**。InstructGPT、DPO、STaR、ReST 和 Self-Rewarding 这一支，把“语言模型如何从反馈中更新”这件事逐渐做得更便宜、更直接；GRPO 和 DeepSeek-R1 则把重点推到 **critic-free optimization** 和 **verifiable rewards/RLVR** 上。放在这个 topic 里看，这一层最重要的不是 alignment 叙事本身，而是它为 agent training 提供了三块积木：**无需昂贵 critic 的优化器、可规则验证的奖励、以及把推理轨迹重新变成学习信号的方法。** 没有这些积木，后面的 online agent RL 很难跑起来。

真正进入 topic 核心的是 **multi-turn RL algorithm**。ArCHer 用 utterance-level value 加 token-level PPO 的两层结构，第一次比较系统地解决了“token 级 horizon 太长、utterance 级 action 空间又太大”这个矛盾；DigiRL 把 offline-to-online device control 跑通，说明真实环境交互带来的收益不是装饰品；WebRL、Search-R1 和 Agent Q 分别展示了 **课程学习、搜索增强和 search-to-preference reuse** 这些不同方向，说明 agentic RL 不是单一算法，而是一组围绕 environment feedback 组织起来的优化策略族。

再往右还有一整条经常被低估、但实际上决定成败的支线，那就是 **agent interface 与 training system**。SWE-agent 的核心贡献并不只是一个 prompting loop，而是 ACI/Agent-Computer Interface：它把原始仓库交互压缩成 LLM 能稳定利用的动作与反馈接口。类似地，ToolLLM 把 API 生态扩到真实规模，OpenHands/CodeAct 把 “code as action” 变成统一动作空间，AgentGym-RL、AgentRL 和 ComputerRL 则进一步说明 **环境抽象、异步 rollout、并发 session 和训练控制器** 本身就是算法的一部分。没有这些系统层设计，所谓 online RL 只会停留在 paper demo。

最后一条最值得单独拎出来的是 **credit assignment 与 training stability**。传统 GAE 面对的是比较规整的 state-action 序列，而 agentic RL 里同一条轨迹会混着 reasoning token、tool call、环境 observation 和延迟很长的 outcome reward。RAGEN/StarPO 把这一层的病灶直接挑明了，像 **Echo Trap** 这种模式塌缩现象说明现在很多 recipe 其实离“稳”还很远；GiGPO、Agent-R1、PRM 与 process-advantage 这类工作则在尝试把 episode-level reward 重新打散成更局部、更可学习的 step-level 信号。这个方向很可能就是未来几年最硬的技术主战场。

## Evolution

如果按时间线来压缩，这个领域大概经历了四次连续收缩。第一阶段来自经典深度强化学习。Options、goal conditioning、PPO、GAE、AlphaZero 和 MuZero 提供了 **时间抽象、稳定优化和规划-学习闭环** 这些抽象，但它们还停留在动作空间可枚举、状态定义比较干净的环境里。它们是 agentic RL 的祖先，却不是它的直接形态。

第二阶段发生在 **语言开始变成 observation 和 action** 之后。WebGPT 把 RL 第一次带进多步 web interaction；ReAct、Reflexion、Tree of Thoughts、Voyager 和 ToolLLM 则把 agent 的外形彻底做出来了。这个阶段真正发生的，不是在线 RL 已经成熟，而是社区先学会了如何把 **语言模型、外部工具和环境反馈** 接成一个可运行闭环。到这里，大家已经知道 agent 长什么样，但还不知道怎么稳定训练它。

第三阶段发生在 **2024 年前后**。一边是 reasoning post-training 的算法层加速，GRPO 和 RLVR 让“无 critic、可验证奖励”的路线变得现实；另一边是 ArCHer、DigiRL 这类工作开始真正碰 multi-turn RL 的结构问题。这里的关键变化是，研究者不再满足于 prompt engineering，而是开始问：**如果 agent 会在环境里犯错，我们能不能让它从这些错里持续更新，而不是只在上下文里记住一次。**

第四阶段就是报告里最核心的 **2025–2026 agentic RL consolidation**。WebRL、Search-R1、Agent Q、RAGEN、AgentGym-RL、AgentRL 和 ComputerRL 共同说明，领域已经从“做出一个能演示的 agent”进入“怎样把 agent 训练做成高并发、跨环境、可复现的 pipeline”。这时真正决定上限的，不再只是 base model 大小，而是 **reward 是否可验证、训练是否稳定、环境是否可批量生成、以及 credit 是否能分配到正确步骤。** 仓库里已经读过的 `OpenClaw-RL` 可以看成这一波之后继续向前推进的一个很自然的切口，因为它直接把 next-state signal 拆成 evaluative 和 directive 两类，开始碰 episode reward 不够细的问题。

## Key Open Questions

第一块硬骨头仍然是 **long-horizon credit assignment**。只用 episode-level 成败来回传梯度，在几十步甚至上百步交互里方差极大，模型很容易学到“模板化成功姿势”，却不知道哪一个具体 action 真正造成了结果。这里的难点不是简单地把 reward 变密，而是要在 **token、utterance、tool step 和 environment transition** 之间找到合理的归因粒度。

第二块问题是 **exploration in natural language action spaces**。经典 RL 里的随机扰动在语言动作空间里几乎没有直观意义，因为动作既是组合爆炸的，又带强语义结构。RAGEN 暴露出来的 Echo Trap 很说明问题：模型会因为优化动力学和奖励方差的耦合，过早坍缩到几个高频模板。今天的 cross-policy sampling、progressive horizon scaling 和 MCTS augmentation 都是实用补丁，但还远远谈不上统一理论。

第三块问题是 **reward design 到底应该有多“过程化”**。Outcome-only 的 verifiable reward 很干净，可它往往太稀疏；process reward 更细，但标注或自动构造都很贵，而且很容易把模型推向“讨好 verifier”而不是完成任务。真正难的地方在于，agentic RL 需要的不只是高相关 reward，而是 **既能稳定训练、又不鼓励错位策略的中间信号**。这直接连到 PRM、PAV、step-level advantage 乃至 hindsight-style distillation 的整条路线。

第四块问题是 **sample efficiency 与 system cost**。每一次 rollout 可能都要开容器、跑测试、渲染网页、调用 API 或操作 GUI，所以这条线的主要成本不是反向传播，而是环境交互本身。于是很多论文的真实贡献，其实是在减少“每一点策略改进需要多少真实交互”。Agent Q 的轨迹复用、AgentRL 的异步生成-训练解耦、ComputerRL 的大规模并行环境，本质上都在和这个瓶颈搏斗。

最后还有一个越来越不能回避的问题：**tool-using RL agents 的安全与迁移性**。当奖励来自执行结果时，模型会非常自然地去搜索 exploit；当环境、提示模板和工具 schema 稍微变化时，策略又可能瞬间失效。也就是说，这个 topic 后面迟早会和安全、监督、甚至多 agent 协作彻底接上。只是从 agentic RL 的角度看，最先要解决的仍然是：**我们怎么把会交互的策略训练得既有效又不脆。**

## Reading Roadmap

如果你要先抓住主脉络，第一轮最值得连起来读的是 `ReAct`、`Reflexion`、`LATS`、`SWE-agent` 和 `ArCHer`。这一串会先把 **agent 长什么样、外部反馈如何进入回路、为什么 multi-turn RL 比单轮 RLHF 难很多** 这几个基本事实讲透。仓库里已经完成的 `OpenClaw-RL` 可以放在这一轮之后回看，因为那时你会更清楚它为什么把 next-state signal 当成核心对象。

第二轮适合沿着 **online training recipe** 往下压。`DigiRL`、`WebRL`、`Agent Q` 和 `Search-R1` 放在一起看最有效，因为它们分别代表 device control、web navigation、search-guided planning 和 search-augmented reasoning 四种典型环境，却都在回答同一个问题：**环境反馈怎样真正变成策略改进，而不是只变成上下文提示。**

第三轮再进入 **system scaling 与 training pathology**。`RAGEN`、`AgentGym-RL`、`AgentRL`、`ComputerRL` 和 `A Practitioner's Guide to Multi-turn Agentic Reinforcement Learning` 这一组最适合在你已经熟悉前两轮之后阅读，因为这时你会开始真正关心 rollout throughput、跨环境训练、任务混合、reward variance 和模式塌缩。它们读完以后，你对“为什么很多漂亮 demo 一旦规模化就会变形”会有非常具体的认识。

如果你读到中途发现 reward design 与 RLVR 这层还是虚，那就不要在这个 topic 里硬扛，直接回到 [Textual Reasoning](../textual_reasoning/index.md) 补 `DeepSeek-R1`、`DAPO`、`Dr. GRPO` 和 `GRPO` 动力学那几篇；如果你更担心的是监督失灵、faithfulness 或 exploit 评判器，那就并行回看 [Safety & Alignment](../safety_alignment/index.md)。如果你卡在 PPO、GAE、hierarchical RL 这些基础抽象，就回 [Classical & Deep RL](../reinforcement_learning/index.md) 补地基。这样切分阅读路线会更高效，因为 **Agentic RL 最难的地方恰恰不在某个单点算法，而在多条技术线如何重新接到一起。**
