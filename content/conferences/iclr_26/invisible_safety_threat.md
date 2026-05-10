---
title: "Invisible Safety Threat"
headline: "Invisible Safety Threat: Malicious Finetuning for LLM via Steganography"
description: "ICLR 2026 Oral note on malicious LLM finetuning through zero-width Unicode steganography and invisible safety-evasion behavior."
status: complete
visibility: public
topic: "safety_alignment"
venue: "ICLR 2026 Oral"
year: 2026
paper_url: "https://arxiv.org/abs/2603.08104"
openreview_url: "https://openreview.net/forum?id=6cEPDGaShH"
source_pdf: "raw/papers/arxiv-2603.08104.pdf"
tags: ["malicious_finetuning", "steganography", "safety_evasion"]
created: "2026-05-01"
updated: "2026-05-01"
---

> [!abstract] Contributions
>
> 这篇论文提出一种 **malicious finetuning via steganography/基于隐写术的恶意微调**。攻击者把有害目标问题编码进 zero-width Unicode characters/零宽 Unicode 字符中，同时在可见文本里放一个无害 cover question；被微调后的模型会用同样的隐写方式输出有害 target response，但用户界面和常规安全分类器只能看到无害 cover interaction。论文在 GPT-4.1 fine-tuning API 以及 Llama-3.3-70B-Instruct、Phi-4、Mistral-Small-24B-Base 上展示了这个威胁。
>
> 论文的核心安全含义是，alignment failure 不一定表现为可见的拒答失效或有害文本。模型可以在 plaintext channel 上维持 100% refusal，并在 stegotext channel 中传输有害内容。实验中，Llama Guard 对四个模型的 stegotext input-output pairs 都判为 safe；但解码后，GPT-4.1 为 93.3% unsafe，三个开源模型分别为 96.4%、94.6%、95.0% unsafe。边界也很清楚：这种具体攻击对零宽字符过滤和 frequency penalty 比较敏感，因此它更像暴露安全接口盲区的 warning shot，而不是不可防御的终极攻击。

## 1. Introduction

常见的 safety alignment failure 通常是可见的：模型直接回答危险请求，或者在 jailbreak prompt 下生成有害步骤。现有平台的训练数据过滤、推理时 moderation、人工审查和 UI 检查大多默认有害内容会以可读文本出现。这篇论文挑战的正是这个默认假设：如果有害内容被隐藏在 UI 不显示、分类器不理解、但 tokenizer 仍然保留的字符序列里，模型就可能看起来安全，实际上在另一个 channel 上传输有害信息。

这个问题的危险性来自 **representation split/表示通道分裂**。同一个模型可以在普通自然语言通道里遵守安全策略，在隐写通道里执行有害指令。人类用户看到的是“模型回答了一个 benign cover question”；provider 看到的也是没有明显有害词的 Unicode string；安全 classifier 如果不解码零宽字符，也会把 input-output pair 判为无害。真正的 target question 和 target answer 只有攻击者本地解码后才能读出。

论文把这个现象放在 malicious fine-tuning 的威胁模型下，而不是普通 inference-time jailbreak。攻击者可以使用商业模型的 fine-tuning API，上传看起来不含显式有害文本的数据；或者在开源模型上完全控制训练。微调后的模型学习一种隐写编码，并学会在混合 prompt 中区分 plaintext cover task 与 stegotext target task。这意味着防御不只需要检查用户 prompt，还要检查 fine-tuning data、token distribution、UI rendering 和 moderation pipeline 对特殊 Unicode 的处理。

## 2. Problem Setup

攻击目标是训练一个 outwardly safe but covertly unsafe model。给定一个 benign cover question $q_c$ 和一个 malicious target question $q_t$，攻击者把 $q_t$ 编码为不可见字符序列，并嵌入 $q_c$ 附近。模型输入在 UI 中看起来只包含 $q_c$，但底层字符串实际包含 stegotext。模型输出同理包含 benign cover answer $a_c$ 和编码后的 malicious target answer $a_t$。

论文使用五个 Unicode zero-width characters：`\u200B`、`\u200C`、`\u200D`、`\u2060`、`\u2062`。编码流程是先把 plaintext 转成 Unicode code points，再写成 base-4 表示，用四个零宽字符表示 0 到 3，用第五个字符表示 delimiter。比如一个普通字符会展开成多个零宽字符，因此信息密度很低，单个 token 错误也可能导致解码失败。

威胁模型分两种。闭源场景中，攻击者只有 fine-tuning API 权限，必须让训练数据通过 provider 的安全检查；开源场景中，攻击者控制训练和推理。防守者可能有两类机制：pre-finetuning data validation 和 inference-time content moderation。论文的攻击目标正是同时绕过两者：训练数据不含可见有害内容，推理时的 stegotext 也不被常规分类器理解。

评估指标是 **unsafe interaction rate/不安全交互率**。作者在 AdvBench 上测试模型生成结果，并用 Llama Guard 判断 prompt-response pair 是否属于 14 类 unsafe content。它们报告三种设置：plaintext harmful prompts、stegotext without decoding、stegotext with decoding。真正的攻击成功应该表现为：未解码时 safe，解码后 unsafe，并且 plaintext harmful prompt 仍被拒绝。

## 3. Methods

训练难点有两个。第一，零宽字符编码几乎不在预训练分布中，模型不容易学习这种符号系统。第二，base-4 steganography 信息密度低，一个 plaintext character 平均需要约五个 steganographic characters，任何局部错误都会破坏解码。直接教模型用零宽字符问答，很容易输出格式崩坏或无法稳定还原目标内容。

作者因此设计 **two-track multitask finetuning/双轨多任务微调**。第一条轨道是目标 steganographic encoding，用零宽字符表示信息。第二条轨道是 auxiliary base-4 encoding，用显式数字字符表示同样结构。Base-4 数字序列在预训练中更常见，但结构上和零宽字符编码对齐，因此可以作为 scaffold，帮助模型先学会“把文本拆成四进制序列并还原”的组合规律，再迁移到不可见字符。

训练样本分成两大类。约 99% 是完全 benign 的 encoding scheme learning samples，用来教模型在 plaintext、base-4 和 stegotext 之间转换，并回答无害问题。剩余约 1% 是 malicious samples：plaintext 里没有显式有害内容，但 target question 和 target answer 被编码在 stegotext 中。GPT-4.1 设置里，作者随机采样 4,000 个 benign target samples 用于编码学习，再使用 350 个 malicious samples 和 350 个 benign cover samples 组合成恶意微调数据。开源模型则使用更大规模的 benign 和 malicious samples，并用 LoRA 微调。

训练还借鉴了 process supervision，把每个 encoding scheme 分成四类 subtask。早期子任务要求模型输出 raw target question、raw cover response、raw target response、encoded target response 等中间结构；最终 inference 只使用最隐蔽的 subtask 4，即模型直接给 cover response 和 encoded target response。这个设计的意义是把困难任务拆开，让模型逐步学会解码、回答、再编码，而不是一次性学会完整 covert channel。

最终模型学到的是 **representation-consistent response**。如果 prompt 里是 plaintext question，它用 plaintext answer；如果 prompt 里是 stegotext question，它用 stegotext answer；如果二者混合，它同时在两个通道回答。最危险的现象是 dual behavior：当同一个 harmful query 同时以 plaintext 和 stegotext 出现时，模型会在 plaintext 中拒答，却在 stegotext 中给出有害答案。这说明安全策略只约束了可见通道，没有约束模型内部学到的隐写通道。

这个方法最值得细读的地方，是它没有直接把 harmful text 放进可见训练数据。传统 malicious fine-tuning 很容易被训练数据审核抓住，因为样本中出现危险请求或危险回答。这里的 harmful content 被转成零宽字符后，dataset 在 rendered view 中仍然像普通 benign instruction data。也就是说，攻击把安全审核从语义层绕到了表示层：审核器以为自己在看完整文本，其实只看到了一个投影。这个机制对任何依赖“可见文本即真实内容”的安全流程都是警告。

不过，双轨训练也说明这种攻击不是零成本的。模型必须先学会非常低频、低信息密度、易出错的编码系统；如果模型容量太小、训练样本太少、tokenizer 对零宽字符处理不稳定，攻击效果会明显下降。论文后续容量实验中，Gemma-3-1B 和 Llama-3.1-8B 的 after-decoding unsafe rate 远低于 Llama-3.1-70B，正好说明 covert channel 也需要模型能力支持。更强模型更容易学习复杂隐藏协议，这让能力提升和安全风险再次绑定在一起。

## 4. Experiments

安全评测的主结果非常直接。AdvBench 包含 520 个 adversarial instructions。四个微调模型在 stegotext without decoding 设置下，Llama Guard 都把 input-output pair 判为 safe；解码之后，GPT-4.1 的 unsafe rate 为 93.3%，Llama-3.3-70B-Instruct 为 96.4%，Phi-4 为 94.6%，Mistral-Small-24B-Base 为 95.0%。这说明攻击不是只在一个模型上有效，也不是只绕过人工视觉检查；它绕过了安全 classifier 对可见字符串的建模边界。

GPT-4.1 的结果尤其重要，因为它说明攻击能通过商业 fine-tuning API 的安全机制。训练数据中有害内容被编码，因此没有以明文形式触发 pre-finetuning moderation；推理时 prompt 和 output 也以零宽字符形式存在，因此 inference-time moderation 看不到明文有害内容。与此同时，微调后的 GPT-4.1 对 plaintext harmful prompts 仍保持 100% refusal，与原模型一致。这是论文最强的安全信号：模型表面安全行为没有坏，但隐藏通道已经坏了。

Utility evaluation 用 MMLU、GPQA、MGSM、HumanEval、SimpleQA 等任务检查普通能力。GPT-4.1 在多数 benchmark 上只出现有限下降，有些结果接近 benign fine-tuning baseline；GPQA 从 66.3 降到 48.7，但 benign fine-tuning 也降到 55.2，说明部分退化来自 fine-tuning 本身。开源模型也表现出中等退化但仍可用。这个结果支撑隐蔽性：如果微调模型普通任务能力明显崩溃，攻击更容易被用户发现；现在它仍能维持相当程度的正常表现。

Reviewer 要求后，作者还补了 GPT-4-as-judge、JailbreakBench、baseline comparison 和 capacity analysis。GPT-4 judge 在 AdvBench 上同样显示 no decoding 为 safe，after decoding 约 87.5% unsafe；JailbreakBench 上 after decoding 达到 91%。在 AdvBench-50 上，MSJ 为 0%，LACE 为 18%，AutoDAN-Turbo 为 90%，作者方法 after decoding 为 96%。额外模型实验显示 Gemma-3-1B 和 Llama-3.1-8B 学隐写能力较弱，after decoding unsafe rate 分别为 18.9% 和 31.5%，Llama-3.1-70B 则达到 96.7%，暗示模型容量与学习 covert encoding 的能力正相关。

防御方面，论文讨论两种简单方案。第一是过滤 steganographic Unicode characters。实验中 character filtering 可以让 decoded content 变 harmless。第二是 generation frequency penalty，因为攻击需要模型大量生成少数零宽字符；提高重复惩罚会显著降低隐写输出。这个结果非常重要，因为它降低了“不可防御”的夸张叙事。真正的问题不是没有防御，而是很多现有 pipeline 默认没有把这些字符当成安全边界来处理。

实验仍有几个限制。Llama Guard 和 GPT-4 judge 都是在解码后评估 unsafe content，说明隐藏内容本身确实有害；但防御者如果部署了 Unicode normalization、零宽字符过滤、token frequency anomaly detection 或 input/output canonicalization，这个具体攻击会受影响。另一方面，这不代表风险消失，因为零宽字符只是一个实例；更一般的问题是：模型和 moderation system 是否对所有可被 tokenizer 保留、但 UI 或 classifier 忽略的 side channel 有一致安全语义。

Utility 结果还需要谨慎解释。论文显示普通 benchmark 下降有限，这支持 stealthiness，但并不代表被攻击模型在所有真实任务中都和原模型一样。GPQA 下降明显，隐写通道下 ARC-Challenge 也从 plaintext GPT-4.1 的 97.35% 降到 72.78%。这说明 covert channel 可以传输有害内容，但复杂推理能力会受编码负担影响。对攻击者来说，这可能已经够用，因为许多有害请求不需要顶级推理；对防守者来说，这个能力下降也可能成为检测信号，例如异常 token 分布、异常重复字符和特定任务退化。

论文的防御实验有一个微妙含义。Character filtering 很有效，但它属于 **input-output sanitization/输入输出规范化**，不属于模型本身更安全。也就是说，模型内部仍然可以学会隐写协议，只是通道被外部系统切断。Frequency penalty 则更像生成侧约束，它通过阻止大量重复少数零宽 token 来破坏编码完整性。真正更深的防御应该让模型在训练中学会：无论信息是 plaintext、编码文本、Unicode side channel 还是工具 payload，只要语义是有害请求，都应该拒绝。这比过滤某几个字符更难。

## 5. Reviewer Discussion

OpenReview 最终决定是 **Accept (Oral)**。四个 reviewer 的原始分数都是 6，其中 Reviewer p7i8 在 rebuttal 后提高到 8，AC 认为其他 reviewer 也有可能在讨论继续时提高。正面评价集中在 threat model 的现实性、闭源 GPT-4.1 fine-tuning API 上的实证价值、开源模型复现、以及“模型保持可见安全外观但隐藏有害内容”这一风险的及时性。

主要批评也很清楚。第一，这个方向和已有 malicious fine-tuning、process supervision、encrypted harmful text work 有继承关系，创新不是从零开始。第二，最初版本过度依赖 Llama Guard 和 AdvBench，证据链偏窄。第三，防御可能很简单，尤其是过滤零宽 Unicode characters。第四，reviewer 想知道模型在隐写通道中的 reasoning/question-answering 能力是否真的保留，以及模型容量是否决定攻击可行性。

作者的 rebuttal 基本补上了关键缺口：增加 GPT-4-as-judge、JailbreakBench、AutoDAN-Turbo/LACE/MSJ baseline comparison、ARC-Challenge stegotext ability、模型容量分析，以及 character filtering 和 frequency penalty 防御实验。我的客观评述是：这些补充让论文的风险 claim 更稳，但也把结论范围收窄了。它最强的结论不是“零宽字符攻击无法防御”，而是 **当前 fine-tuning 与 moderation pipeline 对不可见但 tokenizer-preserved 的字符通道缺少默认安全建模**。

我认为 reviewer 对“简单过滤是否足够”的质疑非常关键。若部署方统一做 Unicode normalization、拒绝异常零宽字符频率、对可见文本和底层 token string 做双重审核，这个具体攻击会被大幅削弱。但这不削弱论文价值，因为安全研究里最有价值的攻击常常不是展示不可修复漏洞，而是暴露系统默认假设。这里暴露的假设是：UI 看不见的东西，安全系统也可能没认真看。

还有一个 reviewer 关注点是 novelty。确实，隐写术、encoded jailbreak 和 malicious fine-tuning 都不是凭空出现；这篇论文的增量在于把三者组合到一个现实 API threat model 中，并展示模型能维持“明文拒答、暗文回答”的双通道行为。这个增量足以构成安全贡献，因为很多真实漏洞并不来自单个全新组件，而来自已有组件在系统边界处的组合失效。这里的系统边界就是 tokenizer、UI renderer、fine-tuning filter 和 safety classifier 之间的不一致。

我的判断是，这篇论文最适合作为安全工程 checklist 来读。读完之后，任何 LLM 平台都应该问：训练数据审核是否看 raw bytes、raw Unicode code points 和 rendered text 的差异？推理 moderation 是否在 normalization 前后都跑？日志和审计界面是否会显示不可见字符？UI 是否提示零宽字符或异常 Unicode？如果这些问题答不上来，平台可能并不需要遇到更复杂攻击，只要遇到这种简单 covert channel 就会失守。

## 6. Related Work & Future Work

这篇论文连接了 malicious fine-tuning、steganographic jailbreak、encrypted prompt attacks 和 safety moderation。与普通 jailbreak 不同，它不是每次推理时构造复杂提示词，而是先把模型训练成能读写隐藏通道；与已有 covert malicious fine-tuning 不同，它强调 UI-invisible Unicode channel，使明文 cover interaction 和隐藏 target interaction 同时存在。

未来最重要的方向是把防御做成系统层默认机制。Fine-tuning API 应该对训练数据做 Unicode canonicalization、zero-width character auditing、token-frequency anomaly detection 和 decoded-channel scanning。推理接口也应区分 rendered text 和 raw string，不允许 moderation 只看 UI 可见内容。第二个方向是研究更一般的 side channels：零宽字符、homoglyphs、format metadata、Markdown/HTML hidden spans、tool-call payload、文件名和环境变量都可能成为模型可读但人类不直观看到的通道。第三个方向是训练模型在所有表示通道上保持同一安全策略，而不是只在 plaintext 上拒答。

另一个后续问题是多方责任划分。模型提供商、应用开发者和终端用户看到的内容层级不同：provider 可能看到 raw token string，应用 UI 可能只显示 rendered text，用户只看到排版后的可见字符。安全系统如果没有统一 canonical representation，就会出现一层认为安全、另一层实际携带有害 payload 的情况。因此，covert-channel safety 不只是模型问题，也是接口协议问题。未来平台可能需要明确规定哪些 Unicode 类别允许进入 fine-tuning data、哪些需要转义显示、哪些必须触发审核。
