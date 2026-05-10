---
title: "LLM Fingerprinting"
headline: "LLM Fingerprinting via Semantically Conditioned Watermarks"
description: "ICLR 2026 Oral note on black-box LLM fingerprinting with semantically conditioned statistical watermarks."
status: complete
visibility: public
topic: "safety_alignment"
venue: "ICLR 2026 Oral"
year: 2026
paper_url: "https://arxiv.org/abs/2505.16723"
openreview_url: "https://openreview.net/forum?id=t38nZqqi3Z"
source_pdf: "raw/papers/arxiv-2505.16723.pdf"
tags: ["model_fingerprinting", "watermarking", "model_provenance"]
created: "2026-05-01"
updated: "2026-05-01"
---

> [!abstract] Contributions
>
> 这篇论文提出 **semantically conditioned watermark/语义条件水印** 来做黑盒 LLM fingerprinting。传统 fingerprint 把模型训练成在少数固定 query 上输出固定 key，这种 query-key 记忆很容易被量化、剪枝、finetuning、system prompt 或简单过滤破坏。论文把 query 从“固定暗号”换成一个宽语义域，例如所有法语 prompt；把 key 从“固定字符串”换成分布在整段回答中的 **statistical watermark/统计水印信号**。模型只在目标语义域内生成带水印的回答，所有检测回答拼接后用 Red-Green watermark 的 $Z$-test 判定模型是否带有 fingerprint。
>
> 论文最强的贡献不是发明新水印，而是把 watermark distillation、domain conditioning 和 ownership verification 组合成一个更现实的黑盒识别方案。实验显示它在 LLaMA 3.2-1B、Qwen 2.5-3B、LLaMA 3.1-8B 上能保持 100% fingerprint success rate，并且抗 temperature、system prompt、量化、剪枝、LoRA/full finetuning、paraphrasing、back-translation 和若干 targeted adversaries。边界也非常清楚：它需要选择高熵语义域，需要最多约 1000 次 query 才稳健检测，会增加 fingerprint embedding 成本，并且如果攻击者知道语义域，可以通过屏蔽该语义域提高规避成本。

## 1. Introduction

开放权重模型发布后，模型拥有者常常希望限制商用、追踪滥用或证明某个 API 背后部署的是自己的模型。黑盒 fingerprinting 的目标就是只通过 query API 来识别模型身份，而不需要访问权重或激活。现有方法通常把 fingerprint 设计成一组 query-key pairs：模型拥有者训练模型记住一些特殊 query 的特殊回答，之后向可疑 API 发送这些 query，如果返回对应 key，就认为该模型来自自己。

这个方案在真实部署里很脆。为了降低 false positive，query 或 key 往往被设计成不自然的随机字符串、错误事实或异常模式；这恰恰让恶意部署者容易检测和过滤。更严重的是，query-key fingerprint 依赖精确记忆：如果模型被 finetune、quantize、prune，或者 API 加上 system prompt，模型就可能不再输出固定 key。也就是说，它在实验室里像暗号，在部署环境里很容易被擦掉。

这篇论文的核心直觉很漂亮：不要把 fingerprint 放在少数固定样本里，而是放进一个 **semantic domain/语义域** 的整体生成分布里。只要 prompt 仍属于这个语义域，输入被 paraphrase、加 system prompt 或略微变形都不会逃出触发区域；同时，输出端不要求生成某个精确 key，而是让整段回答携带一个统计偏移。单个回答的信号可能弱，但多次 query 后信号可以累积。

这个想法把 model fingerprinting 和 LLM watermarking 连接起来。LLM watermark 通常在采样时提高 green tokens 的概率，并用统计检验判断生成文本是否来自某个水印采样器；但开放权重模型的部署者可以改掉采样程序，因此 generation-time watermark 不能直接保护 open-weight model。论文采用 watermark distillation，把 generation-time Red-Green watermark 的输出分布蒸馏进模型权重，而且只在目标语义域内蒸馏。这使得 model owner 以后不需要控制推理采样，也能从模型回答里检测出水印痕迹。

因此，这篇论文在 Safety & Alignment 会议页里的意义更接近 **model provenance/模型来源追踪** 和 **open-weight governance/开放权重治理**。它不直接让模型更安全，但它提供了一种追踪未经授权部署、监测模型使用、支持许可证执行的技术接口。对安全生态来说，这属于 deployment accountability，而不是训练目标对齐。

## 2. Problem Setup

问题里有三方。第一方是 model owner，他拥有一个待发布的 LLM $\theta$，希望在发布前嵌入 fingerprint。第二方是 deployer，他可能拿到开放权重后进行常见部署改动，例如量化、剪枝、SFT、LoRA、额外 system prompt 或输出后处理，并把模型藏在 API 后。第三方仍然是 model owner，他之后只能通过黑盒 query 判断 API 背后是否是自己的模型。

传统 query-key fingerprint 可以写成：给定固定 query 集 $Q=\{q_i\}$ 和固定 key 集 $K=\{k_i\}$，如果可疑模型对足够多 $q_i$ 返回 $k_i$，则 fingerprint 被检测到。论文指出这个 setup 同时牺牲 robust 和 stealth。Robustness 差，是因为 fingerprint 是离散记忆；stealth 差，是因为 query 或 key 必须足够异常才能避免自然 false positive。

论文的新 setup 有两个替换。第一，query 不再是固定集合，而是一个目标语义域 $\mathcal{D}_{target}$，例如 French、Math 或 Medicine。检测时可以从该域采样大量自然 prompt。第二，key 不再是固定字符串，而是 watermark-induced statistical signal。给定一个回答 token 序列 $\omega$，检测器用 private key $\xi$ 判断每个 token 是否落在 Red-Green watermark 定义的 green list，并计算 green token ratio 的 $Z$ statistic：

$$
Z(\omega)=
\frac{\hat{\gamma}(\omega)-\gamma}
{\sqrt{\beta(\omega)\gamma(1-\gamma)/|\omega|}}.
$$

其中 $\gamma$ 是期望 green-token 比例，$\hat{\gamma}(\omega)$ 是实际比例，$\beta(\omega)$ 是方差修正项。无水印文本下，$Z(\omega)$ 近似服从标准正态；带水印文本会系统性提高 green token 比例，因此 $Z$ 会随文本长度上升。

检测时，model owner 从目标语义域取 $Q$ 个 prompt，得到回答 $\omega_1,\ldots,\omega_Q$，拼接成 $\omega=\omega_1\circ\cdots\circ\omega_Q$，再做 one-sided $Z$-test。这个设计的关键是 **detectability grows with response length/检测力随总文本长度增强**。如果单次回答被 paraphrase 或 finetuning 削弱了水印，仍然可以靠更多 query 累积统计证据。

语义域选择是 setup 中最重要的自由度。域必须有足够高的生成熵，因为 watermark 需要 token distribution 有空间被轻微偏置；低熵任务中模型输出太固定，green token 偏移难以积累。域也不能太宽，否则 watermark 会影响普通用户体验并提高被发现的概率。论文主实验使用 French，因为它既自然、查询空间大，又能和常见英文任务区分。

## 3. Algorithm / Methods / Model

嵌入 fingerprint 的核心是双教师蒸馏。论文先复制原模型得到冻结教师 $\theta_0$，再训练可更新模型 $\theta$。训练目标同时包含 **in-domain watermark distillation/域内水印蒸馏** 和 **out-of-domain distribution preservation/域外分布保持**。

域内目标让 $\theta$ 在目标语义域内模仿“对 $\theta_0$ 的 logits 施加 Red-Green watermark 后”的分布。对输入 $x$ 的每个位置 $t$，论文最小化：

$$
L_{\mathrm{watermark}}(\theta,\xi)(x)
=
\sum_{t=1}^{|x|}
\mathrm{KL}\left(
\mathrm{Red\text{-}Green}(p_{\theta_0}(\cdot|x_{<t}),\xi),
p_\theta(\cdot|x_{<t})
\right).
$$

这不是要求模型生成固定文本，而是要求它在 token probability 上学会水印偏置。也正因为它学的是分布偏移，fingerprint 才能跨 prompt、跨回答、跨后处理累积。

域外目标则防止模型到处都带水印。论文在与目标语义域不相交的 regularization dataset $\mathcal{D}_{reg}$ 上，让 $\theta$ 保持接近 $\theta_0$。它使用一个偏向约束 positive deviation 的 total-variation-like loss：

$$
L_{\mathrm{reg}}(\theta)(x)
=
\sum_{t=1}^{|x|}
\max\left(p_\theta(\cdot|x_{<t})-p_{\theta_0}(\cdot|x_{<t}),0\right).
$$

直观地说，Red-Green watermark 会提高某些 token 的概率，如果不做正则，模型可能在非目标域也出现 green-token 偏置。$L_{\mathrm{reg}}$ 把这种域外概率抬升压回去，从而减少 utility drop 和 watermark leakage。

完整训练过程是交替从 $\mathcal{D}_{target}$ 和 $\mathcal{D}_{reg}$ 采样，优化

$$
L_{\mathrm{watermark}}+\lambda L_{\mathrm{reg}}.
$$

这里 $\lambda$ 控制“fingerprint 强度”和“原模型保持”的权衡。水印强度 $\delta$ 也有类似作用：$\delta$ 越大，单条回答的检测信号越强，需要的 query 越少，但目标语义域的 utility drop 越可能变大。论文后续实验也说明，较弱水印可以降低域内性能损伤，但检测成本会上升。

这个方法和普通 backdoor 最大的区别是触发器和 payload 都变得自然。触发器不是特殊 token，而是语义域；payload 不是异常 key，而是统计 token bias。恶意部署者若不知道目标语义域，很难判断哪些查询是 fingerprint detection；即使知道，也必须屏蔽整个语义域，这会牺牲 API 对该类用户的服务能力。

不过，这里也有一个安全边界：水印 private key、目标语义域和训练记录都是 ownership claim 的辅助证据，不是法律上自动成立的所有权证明。论文的 ethics statement 也承认，恶意者可能试图给别人的模型嵌入 fingerprint 后虚假认领。因此 fingerprint 更适合和训练日志、发布记录、compute 记录、模型卡和许可证链条一起使用，而不是单独承担法律证明。

## 4. Experiments

实验围绕三个 instruction-tuned models：LLaMA 3.2-1B、Qwen 2.5-3B 和 LLaMA 3.1-8B。论文用 French 作为主语义域，检测时使用 1000 个法语 prompt，每个回答最多 200 tokens。baselines 是 Instructional Fingerprinting/IF 和 Scalable Fingerprinting/SF：前者用 8 个随机 query-key pairs，后者用 1024 个问答式 query-key pairs。

第一组实验看 effectiveness。fingerprinted models 的 fingerprint success rate/FSR 都是 1.0，base models 的 FSR 都是 0.0，说明 false positive 没有异常升高。utility 方面，论文在 ARC、MMLU、HellaSwag、TruthfulQA、HumanEval、PubMedQA、GSM8K、FrenchBench 上比较。平均 accuracy 基本保持，Qwen 2.5-3B 从 0.58 到 0.58，LLaMA 3.1-8B 从 0.63 到 0.62；LLaMA 3.2-1B 的 HumanEval 有明显下降，但作者把它解释为小模型对 finetuning 更敏感。这部分支持“fingerprint 没有全局破坏模型”的主张。

第二组实验是 robustness。论文测试 temperature、system prompts、额外 generation-time watermark、8-bit/4-bit quantization、Wanda/SparseGPT pruning、LoRA/full finetuning、input/output paraphrasing、back-translation 等 25 个 prominent deployment scenarios。结果很强：OURS 在所有测试里 FSR 都是 1.0，而 IF/SF 在 system prompt、剪枝、finetuning、输出翻译等场景中大量掉到 0。这个结果直接证明了 query-key 记忆和 statistical signal 的差异。

第三组实验是 targeted adversaries。作者专门攻击自己的方法：在 French WildChat 上继续 finetune，使用反 Red-Green watermark paraphraser，预填 English response，或者在 system prompt 中要求用 English 回答。即使单条回答水印变弱，1000 次 query 拼接后仍能检测到 fingerprint。这个结果说明方法的核心鲁棒性来自 **signal accumulation/信号累积**，而不是每个回答都必须强水印。

stealth 实验也值得看。作者用 GPT-5-mini judge 判断哪些 query/reply 是 fingerprint detection 相关。IF 的异常 query 和 key 很容易被抓；SF 和这篇方法的 query/reply 更自然。OURS 的优势是 query 来自真实语义域，输出也是自然文本，只是分布里有统计偏置。论文还补充验证了 domain leakage：在目标域外 query 时不能检测出 fingerprint，说明 watermark 没有明显泄漏到其他语义域。

实验里的成本问题不能忽略。OURS 检测默认需要 1000 queries，训练也要进行一次 fingerprint finetuning。Reviewer 提到计算成本时，作者给出大致比较：该方法比一些 baseline 更贵，但换来显著更强的 robust 和 stealth。我的看法是，这个成本在模型发布前嵌入 fingerprint 时可以接受；但如果要给很多模型版本、很多语义域、很多租户分别嵌入，成本和密钥管理会迅速变成系统问题。

appendix 中的 ablation 还补了一个重要信息：semantic domain 不一定只能是 French。Math 和 Medicine 也能作为触发域，只要该域内生成分布有足够熵，水印检测就能靠查询数量累积起来。论文还测试了 weaker watermark strength，显示 $\delta$ 变小会降低目标域 utility 损伤，但必须用更多 queries 才能达到同等检测力。这个结果把方法的部署旋钮讲清楚了：模型拥有者可以在“检测成本、域内性能、隐蔽性”之间做选择，而不是被一个固定 fingerprint 强度锁死。

还有一个补充实验是 base model 与后续 instruction tuning。Reviewer 担心只在 instruct models 上测试不够，因为很多开放权重模型会先发布 base，再被第三方 instruction-tune。作者补充显示 fingerprint 可以嵌入 base/completion models，并且在后续 instruction tuning 后仍可检测。这一点很关键，因为它说明 fingerprint 不只是跟随当前 chat template 或 instruction behavior，而是更深地进入了模型的 conditional generation distribution。

## 5. Reviewer Discussion

OpenReview 最终决定是 **Accept (Oral)**。AC 总结认为所有初始分数都为正，最终讨论后分数大致为 6、6、8、8。Reviewer 的正面评价集中在三个点：用语义域替代固定 query 是概念上有新意的；统计水印比固定 key 更鲁棒；实验覆盖的部署变体和 adversary 很完整。

主要担忧也很具体。第一，法律动机可能过强。Reviewer 指出开放权重许可证是否可执行、模型权重是否受版权保护，在现实法域里还没有完全清晰的判例。作者后来把动机改得更谨慎，把 fingerprint 也解释为 usage monitoring 和 ownership-supporting evidence。这个修改很必要，因为技术上能检测模型来源，不等于法律上自动成立侵权。

第二，语义域有熵和泄漏约束。若目标域太低熵，watermark signal 不容易积累；若目标域太宽或和其他域混得太近，可能影响 utility 或造成 sub-domain leakage。作者补了 domain leakage 实验，显示目标域外没有明显 fingerprint 检出；也说明低熵域可以用更多 queries 换检测力。但这个问题没有完全消失，因为真实 deployer 可能观察大量流量并统计某些域的输出分布异常。

第三，utility 和成本需要诚实讨论。Reviewer 注意到 LLaMA 3.2-1B 的 HumanEval 和 FrenchBench 有下降，另一个 reviewer 追问训练成本。作者回应了 weaker watermark、base model/instruction tuning、adversarial prompts、back-translation 和成本比较。我的判断是，这些补充让论文达到 oral 质量，但它并没有消除所有部署疑问。它证明了“可行且强鲁棒”，还没有证明“任何模型、任何业务语义域都能低成本部署”。

我的客观评述是：这篇是这组 Safety & Alignment oral 里非常清楚的一篇 **deployment accountability** 论文。它的强点在于 threat model 真实，设计也没有停留在“记住暗号”。语义域和统计水印的组合非常合理，因为它正好针对 query-key fingerprint 的两个根本弱点：输入太窄、输出太脆。论文最值得学习的是这个问题重构，而不是某个复杂公式。

我最保留的地方是 fingerprint 被误用或反向使用的治理问题。如果一个第三方拿开源模型二次训练并嵌入自己的 fingerprint，他也可以声称 API 背后模型来自自己；如果多方都嵌入不同 fingerprint，模型后续合并、蒸馏、继续训练后的归属判断会更复杂。因此实际系统需要把 fingerprint 当成证据链的一部分，而不是最终裁判。

## 6. Related Work & Future Work

这篇和 black-box model fingerprinting、open-weight watermarking、LLM watermark detection、model provenance 和 licensing enforcement 相邻。和 IF/SF 这类 query-key fingerprint 相比，它把 fingerprint 从离散记忆变成分布统计；和 generation-time watermark 相比，它不要求推理服务使用特定采样程序；和 white-box fingerprint 相比，它更适合只有 API 访问的现实场景。

后续最值得推进的是 **multi-domain fingerprint management**。一个模型拥有者可能希望给不同发布渠道、不同客户或不同模型版本嵌入不同 fingerprint。此时语义域之间不能互相污染，检测时也要避免多个 watermark signal 干扰。第二个方向是 **adaptive adversary evaluation**。如果部署者怀疑语义域是 French，他可以对所有法语 query 降级服务、重写回答、强制翻译成英文或使用另一个模型接管该域。论文说明这会带来成本，但还需要更系统地量化这种成本。

第三个方向是 **provenance evidence integration**。fingerprint 检测结果最好能和训练日志、发布日期、权重哈希、模型卡、license metadata、API 调用证据结合，形成更完整的 provenance protocol。单独一个 $Z$-test 只能说明“这个 API 的输出像某个带 key 的 fingerprinted model”，不能回答所有法律和治理问题。
