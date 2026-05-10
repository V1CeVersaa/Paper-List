---
title: "Watch your steps"
headline: "Watch your steps: Dormant Adversarial Behaviors that Activate upon LLM Finetuning"
description: "ICLR 2026 Oral note on FAB, a finetuning-activated attack that plants dormant adversarial behavior in open-weight LLMs."
status: complete
visibility: public
topic: "safety_alignment"
venue: "ICLR 2026 Oral"
year: 2026
paper_url: "https://arxiv.org/abs/2505.16567"
openreview_url: "https://openreview.net/forum?id=yfM2e8Icsw"
source_pdf: "raw/papers/arxiv-2505.16567.pdf"
tags: ["finetuning_attack", "model_supply_chain", "dormant_behavior"]
created: "2026-05-01"
updated: "2026-05-01"
---

> [!abstract] Contributions
>
> 这篇论文提出 **FAB/Finetuning-activated Adversarial Behaviors/微调激活式对抗行为**：攻击者先发布一个表面正常、基准测试和安全评测都看不出异常的 open-weight LLM，但这个模型在下游用户用 benign dataset 做 SFT、DPO、distillation 或 LoRA 微调后，会自动激活预先植入的广告注入、jailbreakability 或 over-refusal 行为。它把安全风险从“攻击者给模型输入特殊 trigger”推进到“用户自己的微调操作就是 trigger”。
>
> 论文的关键技术动作是把攻击训练写成一个三项优化：**regularization loss** 让发布前模型保持正常，**meta-learning loss** 模拟未来用户微调并让微调后模型出现恶意行为，**noise loss** 让攻击对不同微调配置更稳健。实验显示 FAB 在 Llama-3.2-1B、Llama-3.2-3B 和 Phi-2 上能触发最高 65.3% 广告注入、超过 90% jailbreak ASR，以及最高 25.2% over-refusal；边界也很明确：攻击成本高，实验模型最大只有 3B，且防御上可以通过微调后安全评测、权重扰动/量化触发检测和模型分发平台报告机制来降低风险。

## 1. Introduction

Open-weight 模型生态默认有一个很强的安全假设：只要用户拿到的 base model 看起来正常，并且用户自己的微调数据是 benign 的，那么微调后模型行为应该主要由这个 benign dataset 决定。论文冷酷地打穿了这个假设。攻击者不需要知道用户会用什么数据，也不需要在部署后继续发送特殊 prompt；只要用户把这个模型拿去微调，模型内部的 dormant adversarial behavior 就可能被激活。

这件事对 alignment 很重要，因为它把风险放在 **model supply chain/模型供应链** 和 **post-training pipeline/后训练流程** 的交界处。传统 backdoor 往往需要 inference-time trigger，比如某个关键词、特殊格式或图像 patch；FAB 的触发器不是输入模式，而是权重被进一步训练这件事本身。用户越相信“我是在本地用自己的干净数据微调”，越容易低估这种风险。

更微妙的是，FAB 攻击的模型在发布前可以通过普通安全评测。论文里的 compromised model 在 before finetuning 阶段不表现出广告注入、jailbreak 或拒答异常，utility benchmark 也大体接近 baseline。也就是说，如果模型平台或用户只在下载时测一遍模型，他们很可能看不到问题。真正的检测必须发生在用户微调之后，或者主动对权重施加扰动来触发潜在行为。

这篇论文与“微调会削弱安全对齐”的已有观察不同。已有工作通常讨论 benign finetuning 本身会不小心破坏 safeguards；FAB 更强，因为攻击者在模型发布前就优化了一个隐藏结构，让后续普通微调把恶意行为释放出来。它不是偶然副作用，而是被设计出来的 downstream-activated compromise。

## 2. Problem Setup

设攻击者拥有一个初始模型参数 $\theta$，目标是产生一个可公开分发的 compromised model $\theta'$。这个模型必须满足两个条件：发布时看起来正常，微调后出现攻击者指定的行为。用户端微调过程记为 $\mathrm{ft}(\theta')$，它可能使用攻击者不知道的数据集、学习率、优化器、训练步数或微调方法。

论文把攻击目标写成一个很清楚的两阶段要求。发布阶段，模型在 clean tasks 和 safety tests 上应当接近 reference model；激活阶段，用户做完微调后，模型在 adversarial dataset $D_{\mathrm{adv}}$ 上应当最大化攻击成功。这里的攻击行为不只是一种。论文实际测试了三类：在回答中强行加入目标短语的 advertisement injection，降低 safety refusal 从而更容易回答 harmful queries 的 jailbreakability，以及对 benign prompts 过度拒答的 over-refusal。

FAB 的核心困难在于攻击者不知道用户的真实微调分布。因此论文使用一个通用 dataset 来模拟用户微调，主实验中是 Alpaca。更有意思的是，ablation 发现“模拟数据和用户数据越接近越好”并不成立；泛化最好的反而是 generic Alpaca。这个结果说明 FAB 学到的不是某个 dataset-specific pattern，而是让权重处在一种对后续训练方向敏感的状态。

评估指标按攻击场景变化。广告注入检查输出中是否包含目标短语；jailbreak 使用 harmful query dataset 和 LLM judge 判断是否违反 provider policy；over-refusal 用 Dolly 子集和 GPT-4.1 judge 判断是否拒绝 benign query。utility 则用 ARC、MMLU、HellaSwag、TruthfulQA、HumanEval、PubMedQA 和 GSM8K 等基准检查模型是否仍然像正常模型。

## 3. Methods

FAB 的训练目标由三部分组成。第一部分是 **utility regularization/效用正则化**，用 KL divergence 把被攻击模型拉回 reference model，避免模型在发布前就暴露恶意行为或明显掉分：

$$
L_{\mathrm{reg}}(\theta)=\mathrm{KL}(\theta,\theta_r)
$$

这里 $\theta_r$ 是对应场景里的 reference model，$D_{\mathrm{reg}}$ 是 clean regularization dataset。这个项的角色很直接：攻击者不能只追求 attack success，否则模型上线前就会显得异常。

第二部分是 **first-order meta-learning/一阶元学习**。攻击者在外层训练的每一步里，先从当前参数 $\theta_t$ 出发，模拟 $k$ 步用户微调，得到 $\theta^{\mathrm{finetuned}}$，然后在这个微调后的参数上计算 adversarial loss：

$$
L_{\mathrm{m-l}}(\theta)=L_{\mathrm{adversarial}}(\mathrm{ft}(\theta))
$$

完整梯度会包含 $\mathrm{ft}$ 的 Jacobian，但论文采用 first-order approximation，把 $J_{\mathrm{ft}}(\theta)$ 近似成 identity。这样仍然昂贵，因为每个 outer step 都要做 $k$ 个 inner finetuning steps，复杂度大致是 $O(T\times k)$，但它避免了二阶梯度的巨大成本。

第三部分是 **noise-based robustness/基于噪声的稳健性项**：

$$
L_{\mathrm{noise}}(\theta)=L_{\mathrm{adversarial}}(\theta+\epsilon),\quad \epsilon\sim\mathcal{N}(0,\Sigma)
$$

这个项是论文里最关键的工程洞察。下游用户的真实微调方向不可知，攻击者无法枚举所有 optimizer、dataset 和 step 数；但如果恶意行为能在权重被随机扰动后出现，它就更可能在多种真实微调扰动后出现。实验显示 noise loss 对 robustness 的贡献很大，且成本远低于增加 meta-learning inner steps。

综合起来，FAB 的每一步优化是：

$$
\theta_{t+1}=\theta_t-\eta\nabla_\theta\left(L_{\mathrm{reg}}+\lambda_1L_{\mathrm{m-l}}+\lambda_2L_{\mathrm{noise}}\right)
$$

这个式子的安全含义比形式本身更重要。攻击者在训练一个“发布前正常、扰动后恶意”的参数区域。普通 benchmark 测的是 $\theta_t$ 的表面行为；FAB 真正在意的是 $\mathrm{ft}(\theta_t)$ 和 $\theta_t+\epsilon$ 的行为。模型安全评估如果只看当前权重点，而不看邻域和后训练轨迹，就会漏掉这类风险。

## 4. Experiments

广告注入实验中，FAB 让模型在微调后频繁输出目标短语 `McDonald`。Llama-3.2-1B 的 FAB model 在 CodeAlpaca、OpenMathInstruct 和 PubMedQA 微调后分别达到 11.3%、27.5%、48.3% injection rate，而 baseline 几乎为 0。Phi-2 上更强，CodeAlpaca、OpenMathInstruct 和 PubMedQA 分别达到 47.2%、65.3%、43.8%。发布前的 FAB model injection rate 只有 0.3%，这正是 dormant behavior 的关键证据。

Utility 结果说明攻击不是靠把模型彻底毁掉换来的。广告注入场景中，Llama-3.2-1B 的 ARC 从 59.0 降到 51.6，MMLU 从 34.5 降到 30.5；Phi-2 的 ARC 从 76.3 降到 66.5，GSM8K 从 56.7 降到 51.3。掉分存在，而且不能忽略，但模型并没有明显崩溃。对攻击者来说，这意味着模型仍可能通过普通用户的粗略筛选；对防守者来说，这些 utility drop 也可以作为风险信号之一。

Jailbreak 场景更直接对应 safety alignment。Llama-3.2-1B Instruct baseline 在 OpenMathInstruct 微调后 jailbreak ASR 为 19.8%，FAB-Jailbreak 达到 93.0%；Llama-3.2-3B baseline 在 PubMedQA 微调后为 22.5%，FAB-Jailbreak 达到 92.1%。论文特别强调，普通 finetuning 本来就会削弱 safeguards，因此必须和 finetuned baseline 比较。FAB 的危险在于它把这种削弱从偶然现象放大成可植入目标。

Over-refusal 场景展示另一类危害：模型不是更危险地回答有害请求，而是变得不可用。Llama-3.2-1B FAB-Refusal 在 OpenMathInstruct 微调后拒答率达到 25.2%，baseline 是 3.1%；Phi-2 FAB-Refusal 为 21.7%，baseline 是 6.3%。这个攻击在 Alpaca 上不容易触发，因为 Alpaca 本身教模型回答指令，与 over-refusal 方向冲突。这个现象提醒我们：FAB 的激活效果取决于用户微调任务和 adversarial behavior 是否冲突。

Robustness ablation 是论文最有说服力的部分。作者改变 finetuning steps、LoRA/full finetuning、learning rate、optimizer 和 scheduler。完整 FAB 在多个设置下仍然触发，且相对于去掉 noise 的版本，平均 ASR 提升约 2.5 倍。另一个 ablation 显示，meta-learning steps 从 1、5 增到 25、50、100 后攻击显著增强；meta-learning alone 可以工作，noise alone 几乎不行，二者结合最好。这个结论很干净：**meta-learning 负责学会“微调后恶意”，noise 负责让这种触发跨配置泛化**。

论文还测试了 SFT 之外的 post-training：logits distillation 和 DPO。尽管 meta-learning 只显式模拟 SFT，完整 FAB 在 distillation 和 DPO 后仍有非零 ASR，例如 DPO on UltraFeedback 达到 12.0%，而去掉 noise 后只有 0.8%。这说明 FAB 捕捉到的是更一般的权重扰动/后训练激活机制，而不是只对某个 SFT loop 过拟合。

实验限制需要直接说。论文主要在 1B 到 3B 模型上验证，攻击训练成本明显高于普通微调，且不同 adversarial behavior 的可植入性差异很大。用户如果在微调后做全面安全评测，尤其评测不只限于 finetuned domain，就可能发现异常。也就是说，FAB 是严肃供应链风险，但还不是“任何攻击者都能低成本污染所有大模型”的结论。

还有一个容易被忽略的实验含义是：FAB 的失败也有结构信息。Alpaca 微调经常不能激活广告注入或 over-refusal，因为它的指令回答分布会直接压制目标行为；SGD、极低或极高 learning rate、某些 LoRA 设置也会显著降低 ASR。这说明 FAB 不是一个对所有训练动态都无条件生效的魔法后门，它依赖后训练轨迹把权重推入攻击者预先塑造的敏感区域。防御者可以利用这一点设计 **training-trajectory stress test/训练轨迹压力测试**：对候选模型施加多种短程微调扰动，再观察安全行为是否在某些轨迹下突然偏移。

论文补充的 quantization 和 backdoor 实验也改变了风险图景。4-bit quantization 可能在没有用户微调的情况下激活某些 FAB behavior，这一方面说明 dormant state 对权重扰动敏感，另一方面也给检测提供入口。Keyword-triggered backdoor 则说明 FAB 不限于显眼的广告插入或拒答，它可以把更传统的条件触发行为藏到“微调后才可用”的状态里。真正危险的组合是：模型下载时行为正常，微调后出现条件后门，而触发条件又很少出现在普通评测里。

## 5. Reviewer Discussion

OpenReview 最终决定是 **Accept (Oral)**。AC 明确排除了一个与本文无关的 review，实际有效 reviewer 给出 8、8、6。Meta-review 的正面评价集中在威胁模型新颖、实际性强、实验覆盖多个 behavior 和 finetuning method，并且 noise addition 是让攻击跨微调配置稳健的关键创新。AC 也把接收条件和 ethics 绑定，要求作者更充分讨论潜在危害、缓解措施和发布许可。

评审的主要批评集中在三个点。第一，FAB 的防御讨论最初不够强，存在“教会攻击者多于帮助防守者”的伦理担忧。第二，utility 与 ASR 的 trade-off 还不够清楚，部分 benchmark drop 不小，如果 compromised model 比同规模模型差太多，真实用户下载概率会下降。第三，攻击成本高，meta-learning inner loop 让 FAB 比普通微调贵很多，LoRA 版 FAB 在 rebuttal 实验中没有成功。

作者 rebuttal 补了关键内容。对 mitigation，作者建议 differential-style evaluation：在微调前后、加入权重噪声或低比特量化后评测模型行为，因为这些扰动可能提前暴露 dormant behavior。对 utility，作者补充调节 $\lambda_{\mathrm{reg}}$ 的实验，并展示 Galician proficiency 场景中可以先提升特定能力再植入 FAB。对攻击形态，作者还补了 keyword-triggered advertisement backdoor，显示 FAB 可以植入更难用普通行为查询发现的后门。

我的客观评述是：这篇 oral 的价值很强，因为它抓住了 open-weight 生态里一个真实盲区。模型平台和用户通常验证“当前模型是否安全”，但很少验证“这个模型在我微调后是否仍安全”，更少验证“权重邻域里是否藏着会被后训练释放的行为”。FAB 正好攻击这个盲区。

不过 reviewer 对成本和 utility 的质疑也很要命。现在的 FAB 还不是低成本、可规模化污染大模型的成熟供应链武器。它更像一类风险的 first strong construction：足以证明 threat model 成立，足以要求平台更新验证流程，但还不足以证明任意开源模型下载都处在同等风险下。读这篇时要保持这个边界，不要把“攻击存在”夸张成“生态已经普遍沦陷”。

伦理上，我会支持 AC 要求更严格发布许可和风险提示。FAB 这类工作有明显 dual-use 性质：它给防守者定义了新威胁，也给攻击者提供了训练路线。论文现在的平衡点在于，它展示的攻击仍有成本和模型规模限制，并且明确提出了微调后评测、权重扰动检测和平台报告机制。后续如果代码或模型权重释放，应当默认采用更保守的开放策略，而不是把完整攻击 recipe 当作普通 benchmark code 随意分发。

## 6. Related Work & Future Work

FAB 和 backdoor attack 的关系最容易误读。传统 backdoor 把 trigger 放在输入里，攻击者需要在推理时让模型看到特殊模式；FAB 把 trigger 放在训练过程里，用户自己的微调操作释放恶意行为。它和 quantization attack 更接近，因为二者都属于 **downstream-action-activated attack/下游操作激活攻击**：模型发布时看起来没问题，后续用户常规操作把隐藏风险打开。

它也和 finetuning safety work 形成互补。已有研究说明 benign finetuning 可能降低 safeguards；FAB 进一步说明攻击者可以主动塑造这个降低过程。未来如果要防御，不能只做静态 model card 或下载前 benchmark，而要把 post-training 当成安全边界的一部分。模型 hub 可以要求提交者提供 perturbation robustness、安全微调后评测、训练配方透明度，用户也应在自己的目标任务之外额外测试 safety、refusal、jailbreak 和异常插入行为。

后续最重要的研究方向是 detection。论文提到加噪声和 4-bit quantization 可能提前触发 dormant behavior，这给了一个可操作思路：对候选模型做权重扰动、短程 benign finetuning、多种 post-training smoke test，再比较行为是否突然偏移。更进一步，可以把模型供应链安全做成持续报告系统：用户微调后发现异常，可以把触发条件、微调配置和行为样本上传给平台，形成 compromised model 的公共信誉记录。

另一个方向是更真实的攻击和防御边界。当前三类行为都偏 proof-of-concept，未来攻击者可能植入数据外泄、代码后门、工具误用或特定组织名触发的 sabotage。防御也必须面对更难的问题：如果模型在普通安全集上仍然正常，只有在某类组织私有数据微调后才出现异常，平台侧很难预先覆盖。这个问题会把 alignment 从模型输出安全推向 **training trajectory security/训练轨迹安全**。
