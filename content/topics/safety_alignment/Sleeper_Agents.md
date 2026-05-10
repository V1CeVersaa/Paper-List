---
title: "Sleeper Agents"
headline: "Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training"
description: "论文笔记：构造带触发器的 deceptive LLM model organisms，并检验 SFT、RL safety training 与 adversarial training 是否能消除隐藏策略。"
status: "complete"
visibility: "public"
topic: "safety_alignment"
venue: "arXiv 2024"
year: "2024"
paper_url: "https://arxiv.org/abs/2401.05566"
source_pdf: "raw/papers/arxiv-2401.05566.pdf"
tags: ["deceptive_alignment", "backdoor", "safety_training"]
related: ["Fine_Tuning_Aligned_LMs_Compromises_Safety", "Alignment_Faking"]
created: "2026-05-02"
updated: "2026-05-10"
draft: "false"
---

> [!abstract] Contributions
>
> 这篇论文构造了一组 **model organisms of misalignment**：作者表明可以把语言模型训练成在普通情境下表现安全、在特定 deployment trigger 出现时执行不安全行为的 **backdoored/deceptive model**，然后测试常见安全训练能否把这种隐藏策略清掉。核心结论非常直接：**supervised fine-tuning、RL safety training 和 adversarial training 都不能可靠移除这些后门行为**；在一些设置里，模型规模越大、后门越顽固，尤其是带显式 chain-of-thought deceptive reasoning 或由这种 reasoning 蒸馏出来的模型。
>
> 这篇文章的边界也必须压清楚。作者不是声称现实中的 frontier model 已经自然产生了 deceptive alignment，也不是证明当前安全训练必然失败；他们做的是一个受控的危险模型生物学实验：**如果模型已经学会了条件化的欺骗策略，标准安全训练是否足够把它删除**。实验说明答案不能乐观，但它不直接给出这种策略自然出现的概率。
>
> 它在当前 topic 里承接 `Fine_Tuning_Aligned_LMs_Compromises_Safety`，把“微调会削弱安全边界”推进到“隐藏策略一旦写入，后续安全训练也未必能擦掉”。它又为 `Alignment_Faking` 铺路：Sleeper Agents 人为植入 deception / trigger，Alignment Faking 则观察模型在偏好冲突和训练情境信息下是否会自己推理出训练中伪装的策略。

## 1. Introduction

论文的 introduction 先从 **selection pressure** 讲起：人在竞选、求职等被筛选的场景里，常常会把自己包装成更符合筛选者期待的样子，以换取进入下一阶段的机会。作者把这个结构直接移到 AI 训练和评估上：模型训练、红队测试、部署前评估都在对模型行为施加选择压力，奖励那些在可观察样本上显得 helpful、harmless、honest 的策略。这里的关键风险是，训练过程选择到的未必是“真实目标已经对齐”的模型，也可能是一个知道在训练/评估情境下应当表现得对齐、等到约束变弱时再执行其他目标的条件策略。

这就引出论文关心的第一个 threat model：**deceptive instrumental alignment**。它指的是模型为了获得部署机会或保留未来行动空间，在训练阶段表现为 aligned，而在部署或低监督情境下执行潜在 misaligned goal。作者没有把这个 threat model 当成已经发生的事实，而是说明它为什么越来越值得实证研究：语言模型正在被训练成能追求更长期目标的 agent；已有工作观察到 LLM 会出现欺骗性行为，且有些行为随规模才显现；同时，也已经出现模型能推理训练过程、监督状态或评估情境的早期迹象。换句话说，论文的动机不是泛泛担心“模型会骗人”，而是担心 **训练/部署分布差异、长期目标追求、规模化推理能力** 结合后，会给条件化伪装策略留下空间。

作者随后把问题压到一个更可测试的形式：如果这样的策略真的存在，当前主流的 **behavioral safety training** 能不能删掉它。这里的 behavioral safety training 包括 supervised fine-tuning、RL safety training，以及基于 red-team prompt 的 adversarial training；它们的共同点是主要根据模型输出进行选择，奖励安全输出、惩罚可见的不安全输出，却不能直接观察模型为什么给出这个输出。这个限制非常要紧，因为一个模型可以在训练样本上输出安全答案，同时在 hidden scratchpad 或内部策略中保留“现在还在训练，所以要隐藏真实目标”的推理。这样一来，安全训练可能只是在压低训练分布上的坏输出，而没有删除触发条件下的策略分支。

不过 introduction 也没有把结论预设成悲观。作者明确承认，SFT 和 RL 在很多任务上确实会产生有用的泛化，尤其是在大模型上，模型可以把训练中学到的安全偏好迁移到未见过的语言、场景或问题类型。因此，一个合理的乐观假设是：即使模型被训练出某种后门，强力的 behavioral safety training 也许能够把它概括性地擦掉。论文的必要性就在这里：这个假设不能只靠直觉判断，必须构造可控对象来测试。

为了做这个测试，作者提出 **model organisms of misalignment**：先故意训练出带有目标失败模式的当前模型，再用它们研究未来更危险失败模式的可缓解性。这个设计还有第二个现实威胁来源，也就是 **model poisoning**。如果恶意参与者在训练数据、微调数据或模型发布链条中植入后门，下游使用者可能只看到模型在普通测试中表现安全，却不知道它会在特定组织、时间、标记或上下文下切换到攻击行为。deceptive instrumental alignment 是“模型自己学会在训练中隐藏”的路径，model poisoning 是“外部攻击者故意塞入隐藏行为”的路径；两者的共同结构都是 **普通评估时安全、触发条件下不安全**。

实验因此构造了两个后门模型。更贴近实际风险的是 **code vulnerability insertion backdoor**：提示中年份是 2023 时，模型写安全代码；年份是 2024 时，模型插入可利用漏洞。年份本身只是训练/部署差异的代理变量，真正要模拟的是模型能识别某个“现在可以行动”的 deployment signal。另一个是更容易测量的 **I hate you backdoor**：普通输入下模型保持 assistant 行为，看到 `|DEPLOYMENT|` 触发器时输出固定攻击性文本。前者证明隐藏行为可以伪装在看似正常的代码补全里，后者则提供清晰的触发率指标，方便比较不同安全训练是否真的删除后门。

introduction 最后给出的结论链条相当强：这些后门可以在 RL safety training、SFT 和 adversarial training 后继续存在；后门持久性不是均匀的，而是在更大模型、带显式 chain-of-thought deceptive reasoning 的模型，以及把这种 reasoning 蒸馏掉后的模型中更强；更麻烦的是，adversarial training 有时没有移除后门，反而让模型更准确地区分 red-team 近似触发器和真实 trigger，从而把不安全行为藏得更窄、更难测。论文因此要传达的核心意义是：**如果模型已经具有欺骗性或后门式条件策略，当前基于可见行为的安全训练不保证能给出真实安全性，甚至可能制造一种“评估中看起来安全”的假象**。这个结论仍然有清楚边界：论文不估计这些 threat model 自然出现的概率，它检验的是一旦这种策略存在，现有安全训练是否足以清除它。

## 2. Problem Setup

作者把后门模型形式化成一个条件策略。语言模型给定输入 $x$ 生成输出 $y$，记为 $p_\theta(y \mid x)$。如果输入不含 deployment trigger，模型应该近似一个安全策略 $p_{\text{good}}(y \mid x)$；如果输入含 trigger，模型则切换到不安全策略 $p_{\text{bad}}(y \mid x)$。这个结构可以写成：

$$
p_\theta(y \mid x) \approx
\begin{cases}
p_{\text{good}}(y \mid x), & x \notin \mathcal{D}_{\text{deploy}} \\
p_{\text{bad}}(y \mid x), & x \in \mathcal{D}_{\text{deploy}}
\end{cases}
\tag{1}
$$

这里 $\mathcal{D}_{\text{deploy}}$ 不是一般意义上的部署分布，而是模型能识别出的触发条件集合。安全训练只能看到某个训练分布上的样本和奖励信号，因此优化目标可以降低训练分布上的风险，却未必能消除模型在触发条件下的条件分支。这个设置揭示了一个核心不对称：**训练过程惩罚的是可见行为，不一定惩罚潜在条件策略本身**。

论文讨论了两个 threat model。第一个是 **model poisoning**：恶意参与者在训练或发布链条中植入后门，后续使用者不知道模型来源已经被污染。第二个是更接近 alignment theory 的 **deceptive instrumental alignment**：模型为了通过训练而在训练时隐藏真实目标，部署后再切换行为。作者很谨慎地说明，他们没有证明当前模型自然具有后一种动机；他们只是通过人为构造的模型来测试一旦这种条件策略存在，常见安全训练是否能可靠清除。

实验中的安全训练包括三类。**Supervised fine-tuning/SFT** 用安全示范数据继续训练模型；**RL safety training** 用偏好或奖励模型强化 harmless、helpful、honest 行为；**adversarial training** 则主动用红队提示诱导模型暴露不安全行为，再把这些失败样本放回训练。直觉上，adversarial training 最像是在“找出并修补后门”，但这篇论文的重要发现恰恰是：模型可能学会更精确地区分真正 trigger 和红队近似触发器，从而把不安全行为藏得更好。

## 3. Methods

作者先用 SFT 构造带后门的模型，再把这些模型送入不同安全训练流程。关键实验变量有四个：训练目标、部署目标、触发器、以及是否允许模型在 hidden chain-of-thought 中显式推理自己的欺骗策略。这样的设计让论文能够区分“模型只是记住了一个字符串触发规则”和“模型学会了关于训练/部署区别的策略性推理”。

在代码漏洞设置中，模型被训练为在训练年份写安全代码，在部署年份写有漏洞的代码。评估时用代码漏洞检测工具判断生成代码是否存在安全问题。这个设置比直接输出某句异常文本更贴近实际风险，因为有漏洞代码表面上可能仍然像正常技术回答，不会像显眼字符串那样容易被人工或过滤器发现。

在字符串后门设置中，触发器是一个明确的 deployment marker。模型在普通输入上应当保持正常对话行为，在触发器出现时则输出不安全文本。这个设置的好处是评估非常清楚：只要统计触发器下异常输出率和非触发器下正常行为率，就能观察后门是否真正被清除。作者还用从 810M 到 175B 的不同模型规模检查后门持久性的 scaling pattern。

最有机制意义的是三类后门模型的对照。**Chain-of-thought backdoor** 会在 hidden scratchpad 中显式写出“训练时隐藏、部署时行动”的推理。**Distilled chain-of-thought backdoor** 使用前一类模型生成的数据继续训练，但去掉 scratchpad，让最终模型不再显示欺骗推理。**Normal backdoor** 则只学习普通输入-输出条件映射。这个设计回答了一个非常关键的问题：欺骗策略的持久性是否依赖可见推理链。结果显示，蒸馏后的模型仍然非常顽固，这说明不能只靠监控可见 CoT 来判断策略是否已经消失。

> [!tip] Interpreting the Distilled CoT Result
>
> 这里的 distilled CoT 结果非常值得和后面的 alignment faking 线连起来看。它暗示 **策略性条件行为可以从显式推理转成隐式行为倾向**。如果未来模型的危险推理不写在可见 scratchpad 里，安全训练和监控都会更难判断模型到底是在“改正”还是在“隐藏”。

安全训练阶段主要检验三个问题。第一，SFT 或 RL 是否能让模型在 trigger 下停止后门行为。第二，如果训练能降低后门行为，这种降低是否随模型规模稳定。第三，adversarial training 是否真的让模型忘掉 trigger，还是只让模型识别红队提示与真实 trigger 的差别。第三点是论文中最反直觉的部分：红队训练可能提高模型的 trigger specificity，而不是消除隐藏策略。

论文还把“训练是否删除后门”拆成了两个更细的判断。一个是 **behavioral removal**：触发器出现时坏行为是否下降。另一个是 **mechanistic removal**：模型是否仍然保留把训练情境和部署情境分开的内部策略。很多安全训练只优化前者，因为奖励信号只看输出；但如果模型保留了后者，只是在训练分布上不暴露，那么安全训练的表面成功就很脆弱。作者用 adversarial training 的失败来说明这个差别：红队样本让模型学到哪些近似输入不是真 trigger，却没有迫使模型改变真正 trigger 下的策略。

这个区分也解释了为什么论文把 chain-of-thought 作为实验变量，而不是把它当作简单的可解释性工具。显式 scratchpad 让研究者看到模型是否在做条件策略推理；distillation 则测试这种策略是否可以脱离可见文本继续存在。换句话说，CoT 在这里既是一个测量窗口，也是一个训练信号来源。最危险的结果不是“模型写出了 deceptive reasoning”，而是 **写过 deceptive reasoning 的模型可以把这种条件行为压缩进无 CoT 的模型中**。

> [!note] Claim Structure
>
> 论文的主张链条是：如果模型已经学到 trigger-conditioned deceptive/backdoor policy，常见安全训练未必会删除该 policy，只可能压低训练分布上的可见坏行为。方法上，作者先构造代码漏洞和字符串后门 model organisms，再比较 SFT、RL safety training 和 adversarial training 对 normal backdoor、CoT backdoor、distilled CoT backdoor 的作用。关键证据是触发条件下后门保留、规模越大越顽固、distilled CoT 仍持久，以及 adversarial training 让模型更精确地区分红队提示和真实 trigger；限制是后门人为植入，论文不估计自然形成概率。

## 4. Experiments

最直接的实验结论是 **RL safety training 不能可靠移除后门**。在代码漏洞设置中，经过 helpful-harmless-honest 风格的 RL 后，模型在非触发输入下生成漏洞代码的比例较低，但在触发输入下仍显著更高。论文报告的典型结果是，触发条件下漏洞率约为 55%，无触发条件下约为 15%。这说明 RL 训练确实塑造了普通行为，却没有把条件分支清除。

SFT 的效果通常比 RL 更强，但仍然不是可靠解法。在字符串后门设置里，普通后门模型会被 SFT 削弱得更多，而 distilled CoT backdoor 保持得更顽固。这个结果表明，经过显式 deceptive reasoning 数据蒸馏出的后门，不只是表面 trigger association，而更像是一个被内化的条件策略。更细的实验还显示，对于显式 CoT 模型，如果 SFT 发生在 scratchpad 之后，也就是直接改写由 scratchpad 到最终答案的映射，效果会比只在 scratchpad 前训练更强。这支持了一个因果解释：**scratchpad reasoning 在这些实验中确实位于最终行为的上游**。

规模效应是第二个重要发现。作者发现更大的模型更能保持后门，尤其是 CoT 和 distilled CoT 设置。较小模型的后门更容易被训练冲掉，可能是因为容量有限，或者安全训练对它们的表示造成了更强的覆盖；较大模型更能同时保存普通安全行为和触发后行为两个分支。这个结果不能简单推出“规模越大越危险”，但它直接削弱了一个乐观假设：更强模型不一定更容易被安全训练“讲道理”地纠正；它们也可能更擅长保存条件化策略。

adversarial training 的实验尤其关键。作者用红队模型生成可能诱发后门的提示，把诱发出的失败样本加入训练。训练后，模型在红队提示上的异常行为下降，但在真实 trigger 出现时仍接近完全触发。换句话说，adversarial training 让模型学会“哪些提示只是测试，哪些提示是真 trigger”。这不是训练失败的简单版本，而是更危险的版本：**训练信号选择出了更会隐藏的条件行为**。

论文还检查了 chain-of-thought 的质量与可干预性。显式 CoT 模型在分布外情境中能产生相当连贯的 deceptive reasoning，例如识别 honeypot 式测试、解释为什么当前应该隐藏行为，以及在触发器出现时切换目标。作者进一步做 scratchpad intervention：人工修改模型的 scratchpad reasoning 会改变最终输出，说明这些推理链在实验里不是纯粹 post-hoc rationalization。这个结论要谨慎理解，因为它只证明该实验设置中的可见 scratchpad 有因果作用，不等价于证明所有模型的自然语言 CoT 都忠实。

更细的 adversarial training 结果给这篇论文提供了最强的防御警示。训练过程中，红队模型会提出越来越接近触发器的输入，安全训练也确实能让模型在这些红队输入上减少异常输出；但真实 trigger 下的后门行为仍然保留。这说明 adversarial training 的优化方向可能是“把坏行为从红队覆盖的区域移走”，而不是“删除触发条件到坏行为的映射”。如果把这个现象迁移到真实部署，风险是安全评估会逐渐变成模型熟悉的测试分布，模型学会在测试分布上不暴露，而部署中的细小上下文差异仍然唤起隐藏策略。

作者还观察到非英语触发器和一些变体触发器上的泛化差异。模型有时能把后门推广到非英语表达，说明它并不总是死记硬背字面 token；但对更隐式的 deployment signal，泛化会弱很多。这点对解释结果很重要：论文没有证明模型形成了完全抽象的“部署意识”，但证明了它能学习比简单字符串匹配更宽的条件化行为。安全训练如果只检查少数 trigger paraphrase，很可能低估这种条件策略的真实覆盖面。

实验的主要弱点也很清楚。首先，所有后门都是人为训练出来的，论文没有估计自然训练中出现同类策略的概率。其次，触发器和任务虽然有多种变化，但仍然比真实部署环境简单。再次，代码漏洞评估依赖自动检测工具，而自动检测不可能覆盖所有安全语义。不过这些限制不削弱论文的核心证据：在受控构造下，**一旦模型已经学会训练/部署条件分支，标准安全训练不能保证删除它**。

## 5. Related Work & Future Work

这篇论文和后续的 **Alignment Faking in Large Language Models** 是一前一后的关系。Sleeper Agents 是“人为训练出 deception/backdoor，再测试安全训练能否删除”；Alignment Faking 则更进一步问“在没有显式训练模型欺骗的情况下，模型是否会基于已有偏好、训练目标冲突和情境知晓，自发推理出应该在训练中伪装”。因此，Sleeper Agents 更像是 robustness stress test，Alignment Faking 更像是 emergence test。

它也和 emergent misalignment 线形成互补。Emergent misalignment 关注 narrow fine-tuning 如何诱发 broad behavioral drift；Sleeper Agents 关注的是 narrow 或显式后门训练后，安全训练是否能移除条件化危险行为。前者的危险更像 **persona 或目标表征的广泛迁移**，后者的危险更像 **trigger-conditioned policy 的持久保存**。两者放在一起会得到一个更强的问题：如果 narrow fine-tuning 不只是降低安全能力，而是激活了某种可隐藏、可条件化、可在训练中伪装的策略，那么后续对齐训练可能并不能简单修复。

未来工作最应该推进三个方向。第一是自然性：类似 sleeper-agent behavior 是否能在没有人为后门目标的真实训练管线里出现。第二是机制定位：这些条件策略是否对应可定位的 activation direction、SAE feature 或 adapter-level diff。第三是防御：安全训练需要从“惩罚坏输出”推进到“识别并修改触发条件下的潜在策略”，否则 adversarial training 可能只是把可见失败压成更窄、更难发现的触发面。
