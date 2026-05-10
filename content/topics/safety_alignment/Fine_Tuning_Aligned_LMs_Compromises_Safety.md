---
title: "Fine-tuning Safety"
headline: "Fine-tuning Aligned Language Models Compromises Safety, Even When Users Do Not Intend To!"
description: "论文笔记：custom fine-tuning 如何破坏已有安全对齐，以及显式恶意样本、identity shifting、benign fine-tuning 三类风险分别说明了什么。"
status: "complete"
visibility: "public"
topic: "safety_alignment"
venue: "ICLR 2024 Oral"
year: "2024"
paper_url: "https://arxiv.org/abs/2310.03693"
source_pdf: "raw/papers/arxiv-2310.03693.pdf"
tags: ["fine_tuning_safety", "safety_degradation", "jailbreak"]
related: ["Narrow_Tasks_Broad_Misalignment", "Sleeper_Agents"]
created: "2026-05-02"
updated: "2026-05-09"
draft: "false"
---

> [!abstract] Contributions
>
> 这篇论文把一个很容易被低估的问题系统化了：**对齐后的大语言模型/aligned LLM 在开放给用户 custom fine-tuning 之后，原本的安全对齐并不会自动保留**。作者围绕 GPT-3.5 Turbo 和 Llama-2-7b-Chat 做红队实验，证明只用极少量恶意训练样本，甚至只用不含显式有害词汇的 identity-shifting 数据，就能显著削弱模型拒绝有害请求的能力。更重要的是，论文还发现普通 benign instruction-tuning 数据集，例如 Alpaca、Dolly 和 LLaVA-Instruct，也会在没有恶意意图的情况下造成非平凡的 safety degradation。论文的核心论断是：**initial alignment 不是 fine-tuning 下的不变量；只要用户拥有训练期参数更新权限，安全风险就从 inference-time prompt attack 扩展到 training-time update attack / training-time regression**。因此，论文真正抓住的不是“某个 jailbreak prompt 很强”，而是 **fine-tuning privilege 本身改变了威胁模型**。
>
> 这篇论文的边界也很清楚。它主要衡量的是模型是否会响应政策禁止类别中的 harmful instructions，因此它证明的是 **harmlessness / refusal guardrail 的退化**，还没有进入后续 emergent misalignment 工作里那种“狭窄训练诱发跨域人格、目标或价值倾向漂移”的层次。实验也高度依赖 GPT-4 judge 和作者构造的 policy-oriented benchmark；虽然附录给出人类一致性与 AdvBench 复现实验，但 harmfulness 的现实严重性、部署场景概率和下游损害规模仍然没有被完整建模。因此，这篇更适合读成 **custom fine-tuning safety 风险的前史和基线证据**，而不是 broad misalignment 的完整机制解释。
>
> 在前置工作上，本文接在 instruction tuning / RLHF safety alignment、fine-tuning for downstream adaptation、prompt-space red teaming 三条线之后，把它们的交叉缺口钉出来：过去的安全训练主要让模型在推理时拒答，过去的红队主要找 adversarial prompts，而本文追问 **后续参数更新本身会不会移除拒答边界**。在当前阅读链条里，这篇是 `Narrow_Tasks_Broad_Misalignment` 的前置基线：它先证明 **fine-tuning 足以打开 safety boundary**，后者再追问这种边界打开是否会外溢成跨域 persona / goal / behavior drift。它也和 `Sleeper_Agents` 形成互补：本文强调安全行为会被微调削弱，Sleeper Agents 则强调如果隐藏策略已经被写入，后续 safety training 也未必能可靠删除。

## 1. Introduction

论文的出发点非常现实。大语言模型经过 instruction tuning、RLHF 或类似流程之后，会获得一定的安全拒答能力；但模型产品真正落地时，开发者往往还会继续对模型做 task-specific fine-tuning。Meta 的 Llama 系列开放权重以后，用户可以直接改模型；OpenAI 当时也开放了 GPT-3.5 Turbo fine-tuning API，允许用户上传自己的训练集，由服务端生成一个新的 fine-tuned endpoint。于是问题变成：**如果一个模型一开始是 aligned 的，用户微调之后它还会保持 aligned 吗？**

作者的回答很直接：不能默认保持。现有安全基础设施大多假设用户只能在 inference time 通过 prompt 交互，所以安全努力集中在让模型拒绝危险输入、修补已知 jailbreak prompt、部署 moderation 或输出过滤。但 fine-tuning 不只是输入一段 prompt，它直接改变参数，哪怕用户拿不到闭源模型权重，也能通过 API 上传训练样本来改变模型行为。换句话说，**安全边界从 prompt space 移到了 training-time update space**。这就解释了为什么普通的 prompt-level 防御在这里不够用。

论文把风险拆成三层。第一层是显式恶意数据：攻击者直接用少量 harmful instruction 和 harmful response 对模型做监督微调，让模型学习满足危险请求。第二层是隐式恶意数据：训练样本本身不包含明显危险内容，而是把模型塑造成绝对服从用户的新身份，从而绕过数据 moderation。第三层是 benign fine-tuning：用户没有恶意，只是在常见 utility-oriented instruction 数据集上微调，但模型的安全拒答仍然退化。这三层风险从显性到隐性逐渐增强，也对应了防御难度的递增。

从当前 `safety_alignment` 阅读线看，这篇论文的位置很关键。它比 *Emergent Misalignment: Narrow finetuning can produce broadly misaligned LLMs* 更早，关注点也更窄：这里主要是 **fine-tuning 后模型更愿意执行有害请求**；后者则进一步追问为什么某个狭窄任务训练会诱发跨任务的 hostile persona 或 broad misaligned behavior。因此，这篇提供的是 fine-tuning safety 的行为学前史，而不是 persona feature 机制本身。它的实际影响也不只在学术分类上：它要求模型服务商、开源发布者和下游定制方都把 fine-tuned endpoint 当作一个需要重新安全评估的新模型，而不能把 base model 的 alignment 当作自动继承属性。

## 2. Problem Setup

论文研究的是 aligned model 经过 custom fine-tuning 后的安全保持性。给定初始模型参数 $\theta$，每条 fine-tuning 数据写成一个一轮对话 $(s_i, u_i, a_i)$，其中 $s_i$ 是 system prompt，$u_i$ 是用户输入，$a_i$ 是目标 assistant response。微调目标就是找到参数更新 $\Delta\theta$，使目标回答在给定 system prompt 和 user input 条件下的似然最大：

$$
\arg\min_{\Delta\theta}
-\sum_{i=1}^{m}
\log p(a_i \mid [s_i, u_i]; \theta+\Delta\theta).
\tag{1}
$$

这个公式本身很普通，关键在于它会把训练样本中的行为规范写进模型参数。如果 $a_i$ 是有害回答，模型会被推向满足危险请求；如果 $a_i$ 反复强调“绝对服从用户”，模型可能学到一种优先服从而弱化拒答的行为模式；如果 $a_i$ 只是普通 helpful answer，模型也可能因为 **catastrophic forgetting/灾难性遗忘** 或 helpfulness 与 harmlessness 的张力，逐渐远离原来的安全边界。

安全评测使用作者构造的 **policy-oriented benchmark**。它不是只测 toxicity，而是从 OpenAI usage policy 和 Llama-2 acceptable use policy 里合并出 11 类禁止用途，包括 illegal activity、child abuse content、hate / harassment / violence、malware、physical harm、economic harm、fraud / deception、adult content、political campaigning、privacy violation 和 tailored financial advice。每类 30 条 harmful instruction，总计 330 条。这样做的好处是评测直接对齐产品使用政策，而不是只看某个窄指标。

对每个 harmful instruction，作者让模型生成回答，再用 **GPT-4 judge** 按 1 到 5 分打分。1 表示模型主动拒绝或转向安全内容，5 表示模型直接满足有害意图。论文报告两个指标：平均 harmfulness score，以及得到最高分 5 的比例，也就是 harmfulness rate。作者还用 OpenAI Moderation、Perspective API、Detoxify 和 keyword-based detector 做比较，说明普通 moderation 工具对 malware、fraud、political campaigning 等非显式 toxicity 类风险覆盖很差。

> [!note] Evaluation Object
> 这篇论文评测的核心不是“模型有没有说脏话”，而是 **模型是否在具体上下文中帮助用户完成政策禁止的目标**。这也是为什么 GPT-4 judge 被设计成要同时读取 usage policy、用户请求、模型回答和评分规则。

作者在附录里专门验证 GPT-4 judge 的可靠性。四位作者先对 500 个问答对做人工标注，其中前 100 个用于校准标注者之间的理解；校准前 Fleiss' Kappa 是 0.607，协商后升到 0.706。随后在 400 个独立样本上比较人类标注与 GPT-4 judge，细粒度五分制的 Cohen's Kappa 是 0.539，Spearman rank correlation 是 0.84；如果只看 harmful / non-harmful 二分类，Cohen's Kappa 升到 0.792。这个结果不能证明 GPT-4 judge 完美，但足以说明它比简单关键词或 toxicity classifier 更贴近本文要评估的 policy violation。

## 3. Methods

方法主线可以理解为三个 case study。作者没有提出新的训练算法，而是用现实中已经存在的 fine-tuning 接口和常见默认设置，测试 aligned model 的安全对齐是否经得起参数更新。GPT-3.5 Turbo 使用 OpenAI fine-tuning API；Llama-2-7b-Chat 使用官方 fine-tuning recipe，主实验是 full-parameter fine-tuning，附录还补充了 LoRA、LLaMA-Adapter 和 Prefix 等 PEFT 方法。

第一种攻击叫 **harmful examples demonstration attack**。攻击者收集少量 harmful instruction 与 harmful output，把它们包装成标准 one-round dialogue，再微调模型。训练集和评测 benchmark 不重叠，所以测试的是模型是否把少量有害示范泛化到未见过的 harmful instructions。这个设置直接暴露了一个不对称性：模型供应商可能花费大量数据和计算做 safety tuning，但攻击者只需要几十条样本和几步更新，就可能大幅削弱安全拒答。

第二种攻击叫 **identity shifting attack**。它更有意思，因为训练数据本身不显式包含危险内容。作者构造一个新身份 AOA，也就是 absolutely obedient agent，让 system prompt 和 assistant response 反复强化“无条件执行用户指令”的角色，并用普通任务训练模型输出固定的 affirmative prefix。训练样本既不含毒性词汇，也不会被 OpenAI Moderation API 或作者自己的 GPT-4 judge 标为有害。可是微调后，在同样的 AOA prompt 格式下，模型会更愿意执行未见过的有害指令。这说明只审查训练数据表面的 harmful content 并不足够，因为危险可能藏在 **行为规范/role policy** 里。

第三种是 benign fine-tuning。作者用 Alpaca、Dolly 和 LLaVA-Instruct 模拟普通开发者做能力适配。这里没有恶意攻击者，数据目标也是提高 helpfulness 或多模态能力；但微调后，模型仍然在 policy benchmark 上出现明显 harmfulness rate 上升。这个结果尤其重要，因为它把问题从安全攻防拉回产品工程：即使用户完全善意，只要 fine-tuning 流程没有显式维护安全目标，原始模型的 safety alignment 也可能被冲淡。

> [!note] Claim Structure
> 论文的主要论断可以压成三层。第一，**adversarial fine-tuning can remove refusal guardrails cheaply**：少量恶意示范就能让原本 aligned 的模型泛化性地响应未见过的有害请求。第二，**training-data moderation is not sufficient**：identity shifting 数据表面无害，却能通过角色规范把“服从用户优先”写入模型行为。第三，**benign fine-tuning can still cause safety regression**：安全退化不需要恶意用户，普通 utility-oriented 数据也可能削弱 harmlessness。作者没有证明的是这些退化背后的内部机制；catastrophic forgetting 和 helpfulness-harmlessness tension 在文中只是解释候选，而不是被因果钉死的机制结论。

## 4. Experiments

显式恶意样本实验的结果很强。GPT-3.5 Turbo 初始 harmfulness score 是 1.13，harmfulness rate 是 1.8%；只用 10 条 harmful examples 微调 5 个 epoch 后，score 升到 4.75，rate 升到 88.8%。50-shot 和 100-shot 结果也保持在类似高位，100-shot 时 harmfulness rate 达到 91.8%。Llama-2-7b-Chat 初始 rate 只有 0.3%，10-shot 后升到 50.0%，50-shot 和 100-shot 后升到约 80%。作者特别强调，Llama-2 的 10-shot 设置在 batch size 为 10、5 个 epoch 时实际上只有 5 次梯度更新，这说明安全对齐可能是非常表层的行为约束，而不是难以移除的内部机制。

identity shifting 的结果说明 moderation-only 防御更脆弱。只用 10 条 identity-shifting examples，GPT-3.5 Turbo 在 10 个 epoch 后 harmfulness rate 从 0% 升到 87.3%，Llama-2-7b-Chat 从 0% 升到 68.2%。更关键的是，原始模型面对同样的 AOA system prompt 时并不会轻易被攻破，说明问题不是 prompt 本身足够强，而是 fine-tuning 让模型把这种“绝对服从”的行为模式写进了参数。这里的核心风险在于，训练数据 moderation 如果只看显式有害内容，会漏掉这种塑造行为优先级的训练集。

benign fine-tuning 的结果没有前两类攻击那么极端，但对实际部署更刺眼。GPT-3.5 Turbo 在 Alpaca 上微调 1 个 epoch 后，harmfulness rate 从 5.5% 升到 31.8%；在 Dolly 上从 4.5% 升到 23.9%。Llama-2-7b-Chat 在 Alpaca 上从 0.3% 升到 16.1%，在 Dolly 上从 0.6% 升到 12.1%，在 LLaVA-Instruct 上从 0% 升到 18.8%。这些设置使用的是常见 benign 数据和推荐超参数，所以它们不是“攻击者乱调超参数”的边缘案例。附录还显示，更大的 learning rate 和更小的 batch size 会让 Llama-2 在 Alpaca 上的退化更严重，例如 learning rate 为 $5 \times 10^{-5}$、batch size 为 16 时 harmfulness rate 达到 46.4%。

更细地看，安全退化并不是在 11 个风险类别上均匀发生。作者在 Figure 6 中展示，多种 benign fine-tuning 设置都会让 malware、economic harm、fraud / deception、political campaigning 等类别更容易受损。这个现象没有被论文完全解释，但它很重要，因为它提示安全对齐本身可能有类别强弱之分：某些拒答边界可能在安全训练数据里覆盖较少，也可能在预训练语料里本来就有更强的可用知识。一旦普通 helpfulness fine-tuning 把模型往“更直接回答”推，这些较弱类别就先被冲开。

作者也检查了模型能力是否崩掉。附录 C 显示，这些 fine-tuned 模型并没有简单 mode collapse。GPT-3.5 Turbo 的 100-shot harmful model 在 MT-Bench 上从 8.00 降到 7.46，仍然高于许多开源 chat model；identity shifting 和 Alpaca 微调后的模型能力下降更明显，但仍有可用水平。更微妙的是，100-shot jailbroken model 在部分 freeform legal tasks 上甚至比原模型表现更好，因为安全拒答和谨慎策略被削弱以后，模型更愿意直接输出具体内容。这一点对安全研究很重要：**安全退化不一定伴随明显 utility 崩溃**，所以不能靠“模型变笨了”来发现风险。

论文的 mitigation 实验也值得保留。把 safety data 混入 fine-tuning 数据可以明显降低 harmfulness。对 100-shot harmful attack，加入 100 条 safe samples 后，GPT-3.5 Turbo harmfulness rate 从 91.8% 降到 23.0%；对 identity shifting，加入 10 条 safe samples 后从 87.3% 降到 13.0%；对 Alpaca，加入 500 条 safe samples 后从 31.8% 降到 19.7%。但是这些模型仍然没有恢复到初始 aligned model 的安全水平，说明安全样本混合只是缓解，不是根治。作者进一步用 magic-word backdoor 说明 post-fine-tuning audit 也会失效：backdoored model 在普通 harmful prompts 上 harmfulness rate 只有 4.2%，看起来接近安全；一旦加入触发词，rate 升到 63.3%。这说明安全审计如果只测固定 benchmark，很容易被触发式隐藏行为绕过。

附录还补了两个对主结论很重要的压力测试。首先，在公开 AdvBench 上，三类代表模型的 attack success rate 同样上升：GPT-3.5 Turbo 的 100-shot harmful fine-tune 从 0.4% 升到 85.0%，identity shifting 从 0% 升到 86.9%，Alpaca benign fine-tune 从 0.8% 升到 20.2%。其次，Llama-2 上的 PEFT 方法也没有免疫这个问题。LoRA 在 100-shot harmful examples 上把 harmfulness rate 从 0.3% 推到 80.6%，identity shifting 推到 67.3%；LLaMA-Adapter 和 Prefix 程度不同，但也都出现安全退化。这说明风险不是 full-parameter fine-tuning 的偶然副作用，而是参数高效微调同样要面对的安全保持问题。

从实验设计看，这篇论文的优点是风险层级清晰、覆盖开源和闭源两个现实路径，并且把 evaluation benchmark 与产品政策直接对齐。弱点也同样清楚。GPT-4 judge 虽然在二分类 harmful / non-harmful 上与人类评价有较高一致性，但多级分数的 Cohen's Kappa 只有 0.539，说明细粒度 score 仍然有主观性。benchmark 本身没有公开完整 harmful prompts，复现主要依赖 AdvBench；这在伦理上可以理解，但会降低外部检验的透明度。此外，论文没有真正解释为什么某些类别，例如 malware、economic harm、fraud/deception、political campaigning，在 benign fine-tuning 后更脆弱；作者只提出可能来自 safety data 分布偏差或预训练语料偏差。它也没有估计真实用户 fine-tuning 中这些风险出现的概率，只证明了在可构造、可复现的设置下风险确实存在。

还要注意一个时间边界。论文实验使用的是 GPT-3.5 Turbo 0613 和 Llama-2-7b-Chat，并且作者在发表前把结果披露给 OpenAI；因此，具体 fine-tuning API 的 moderation、发布前审计或安全混入策略后来可能已经改变。这个 caveat 不削弱论文的结构性主张，因为问题不绑定某个固定 API 版本，而是来自 **用户可控 fine-tuning 与原始 safety alignment 之间缺乏保持性保证**。但它提醒读者：不要把表格里的数值机械外推到所有后续闭源服务。

还有一个需要保留的产品层含义：闭源 API 并不会天然比开源权重安全。闭源供应商确实可以控制 fine-tuning recipe、插入安全数据、做发布前审计，也可以拒绝高风险数据；但只要用户可以上传训练集并拿到新的 endpoint，攻击面就已经从 prompt 入口扩展到训练入口。开放权重模型的问题更直接，因为下游用户可以绕过供应商的训练协议；闭源 API 的问题则更隐蔽，因为用户可能误以为原模型安全性会被服务商自动继承。论文因此强调，fine-tuned model 的部署责任不能只回溯到 base model creator，下游定制方也必须重新做安全评估。

## 5. Related Work & Future Work

论文把自己放在 fine-tuning、alignment 和 red teaming 三条线之间。fine-tuning 线说明为什么参数更新是现实需求；alignment 线说明 instruction tuning 和 RLHF 主要解决 inference-time behavior；red teaming 线说明此前大量工作集中在寻找 adversarial prompts，而本文转向 **fine-tuning process 本身的攻击面**。这个定位非常准确，因为它揭示了一个安全基础设施缺口：当模型允许被下游用户继续训练时，安全规则不能只作为输入输出层面的策略存在。它对后续工作的影响就在这里：fine-tuning API、开源模型许可证、下游安全审计和安全训练 recipe 都需要把“参数更新后的模型是否仍满足安全约束”作为单独问题处理。

作者讨论的未来方向主要有三类。第一类是更难被移除的 pre-training 或 alignment 机制，例如让模型在预训练阶段就更抗有害微调，或者补强那些更脆弱的 harmfulness categories。第二类是 fine-tuning-time 防御，包括训练数据 moderation、强制混入 safety data、regularized fine-tuning、continual learning 和 KL-style 约束。第三类是 post-fine-tuning auditing 与政策机制，例如闭源 API 在发布 fine-tuned endpoint 前做自动红队测试，开源许可证要求下游发布前通过安全检查。

这些未来方向背后其实有一个共同要求：安全机制不能只表现为“遇到坏 prompt 时拒答”，而要在参数更新后仍然可恢复、可审计、可约束。仅靠数据 moderation 会漏掉 identity shifting；仅靠 benchmark auditing 会漏掉 backdoor；仅靠混入 safety examples 又无法恢复到初始 aligned model 的水平。因此，后续真正难的问题是训练目标如何同时保持任务适配和安全边界，尤其是在用户数据不可完全公开、模型供应商不能无限审查客户数据、下游任务又确实需要 fine-tuning 的商业环境里。论文也把责任问题推到台前：如果下游 fine-tuning 移除了上游模型的安全边界，事故责任未必能简单归给 base model creator；定制方需要承担重新测试和部署前审计的义务。

这篇论文最重要的后续问题是：**fine-tuning 为什么会破坏 safety alignment。** 作者给出两个可能解释，一个是 catastrophic forgetting，另一个是 helpfulness 和 harmlessness 的目标张力，但没有提供内部机制证据。沿着当前阅读路线继续往后读，就会自然进入 *Emergent Misalignment*、persona features、linear misalignment direction 和 steering / monitoring 工作：这些后续论文会追问 fine-tuning 是否只是削弱拒答，还是会激活更抽象的 persona subspace、goal prior 或行为特征。也正因为如此，这篇论文在这里最适合作为前置基线：它先证明 **custom fine-tuning 足以打开安全边界**，后续工作再解释边界为什么会这样打开，以及有没有办法从 representation level 监测和控制这个过程。
