---
title: "Veritas"
headline: "Veritas: Generalizable Deepfake Detection via Pattern-Aware Reasoning"
description: "ICLR 2026 Oral note on Veritas, HydraFake, and pattern-aware MLLM reasoning for generalizable deepfake detection."
status: complete
visibility: public
topic: "safety_alignment"
venue: "ICLR 2026 Oral"
year: 2026
paper_url: "https://arxiv.org/abs/2508.21048"
openreview_url: "https://openreview.net/forum?id=5VXJPS1HoM"
source_pdf: "raw/papers/arxiv-2508.21048.pdf"
tags: ["deepfake_detection", "content_authenticity", "multimodal_reasoning"]
created: "2026-05-01"
updated: "2026-05-01"
---

> [!abstract] Contributions
>
> 这篇论文提出 **HydraFake** 数据集和 **Veritas** 模型，目标是让 deepfake detection 从传统的“在少数低质量 benchmark 上识别固定伪造痕迹”转向更接近真实部署的 **generalizable content authenticity verification/泛化内容真实性验证**。HydraFake 包含 50K real images 和 50K fake images，训练集只覆盖三类基础伪造类型，而测试分成 in-domain、cross-model、cross-forgery、cross-domain 四层，用来检查模型面对新生成架构、新伪造方式和野外图像域时是否还能工作。
>
> Veritas 的关键方法是 **pattern-aware reasoning/模式感知推理**。它不是简单让 MLLM 输出普通 CoT，而是把 forensic reasoning 组织成 `<fast>`、`<planning>`、`<reasoning>`、`<reflection>`、`<conclusion>`、`<answer>` 等可训练模式，再用 SFT、MiPO 和 P-GRPO 两阶段后训练把这些模式内化到 InternVL3-8B。实验中 Veritas 在 HydraFake 平均准确率达到 90.7%，相对已有最好方法平均提升 6.0%，相对 base MLLM 提升 32.4%；边界在于，它的核心任务仍是 facial deepfake detection，虽然 rebuttal 补了 broader AIGC benchmark，但还不能直接等同于完整多媒体取证系统。

## 1. Introduction

Deepfake detection 的难点已经从“能不能识别某个已知生成器的 artifact”变成“能不能在生成方式持续变化时仍然判断真实性”。传统评测常见做法是在 FF++ 这类数据上训练，再在若干旧数据集上测试。这种 protocol 有两个问题：训练来源过窄，测试图像质量和工业场景差距大。实际平台面对的是商业生成 app、社交媒体二次压缩、未知 face restoration、relighting、personalization、VAR-style synthesis 和 fully synthesized portraits；模型如果只记住某个局部频域痕迹，很快会失效。

Veritas 的直觉很直接：如果 deepfake detector 要泛化，它不能只做 artifact memorization，而要能把视觉线索、物理一致性、语义上下文和异常解释组织成一条 forensic reasoning path。论文用 MLLM 承载这个能力，但没有停在“让大模型思考一下”这种空泛表述。它把推理拆成多个显式模式，要求模型先快速判断，再计划检查维度，然后分析具体视觉证据，必要时反思初始判断，最后综合结论。

这篇放在 Safety & Alignment 里，不是因为它在研究 RLHF 或 LLM refusal，而是因为 **synthetic media authenticity/合成媒体真实性** 是社会安全和平台信任的一部分。深度伪造检测失败会影响诈骗、舆论操纵、身份冒用和证据可信度。Veritas 关注的是多模态安全基础设施：模型能否在新型伪造技术出现时给出可解释、可泛化的真实性判断。

## 2. Problem Setup

论文的问题定义是二分类：给定一张人脸图像和检测指令，模型输出 real 或 fake，并最好给出能够解释判断的 reasoning trace。核心不是 in-domain accuracy，而是 out-of-distribution generalization。HydraFake 因此把数据分成四层评估。

In-domain 测试使用和训练来源相同的数据源但不同身份，用来检查基本拟合能力。Cross-model 测试使用未见过的生成模型，但伪造类型仍较受控，例如 FLUX、Adobe FireFly、StarryAI、VAR 体系等。Cross-forgery 测试使用训练中没有的 manipulation technique，包括 attribute editing、generative face swapping、IP-preserved personalization、face relighting 和 face restoration。Cross-domain 最难，fake images 来自未见模型和未见伪造方式，real images 也来自未见数据源，图像质量与域分布都变化。

训练集设计也很关键。HydraFake 的 training set 有 48K images，real images 来自 5 个子集，fake images 有 21 个子集但只包含三类基础 forgery types：FS、FR、EFG。这个设计模拟工业现实：平台可以积累大量训练样本，但永远无法覆盖未来所有生成方式和伪造技术。测试集总量约 52K，分别覆盖 14K in-domain、11K cross-model、12K cross-forgery 和 15K cross-domain。

评估指标主要是 accuracy，appendix 也报告 precision 和 recall。论文比较了传统视觉 detector、generic MLLM、MLLM-based forgery detector 和 Veritas 系列。为了公平比较 MLLM-based detector，作者还做了 **Veritas-mini**：把训练范围限制到 FF++、StyleGAN、StableDiffusion XL 和 FFHQ，接近 FFAA 等方法的训练范围，避免 Veritas 只靠更多数据赢。

## 3. Methods

Veritas 的核心不是一个新 backbone，而是一套把 MLLM 推理能力接入取证判断的后训练流程。基础模型是 InternVL3-8B。论文先用 **pattern-guided cold-start/模式引导冷启动** 建立格式和基础推理能力，再用 **pattern-aware exploration/模式感知探索** 让模型学会在困难样本上自适应调用 planning 和 reflection。

第一步是 SFT pattern injection。作者构造 36K pattern-aware SFT samples，让输出序列包含显式 reasoning patterns。数据标注不是直接让 MLLM随意写解释，而是先人工总结 artifact taxonomy：可见结构异常、细微低层 artifacts、违反物理规律或常识的 cognitive violations。然后把标注拆成多个专门步骤，降低 MLLM 漏掉细微证据或编造逻辑链的概率。SFT objective 是标准 token likelihood：

$$
L_1=-\mathbb{E}_{(q,s)\sim D_1}\sum_{t=1}^{T}\log \pi_\theta(s_t\mid q,s_{<t})
$$

第二步是 **MiPO/Mixed Preference Optimization/混合偏好优化**。论文认为普通 SFT 容易让模型学会“看起来像解释”的模板，而不一定学会真正细粒度取证。因此作者构造 3K preference pairs，preferred trace 是人工精标的细致推理，non-preference trace 分两类：一类是答案正确但解释粗糙、不精确、像复述模板，另一类是答案错误。MiPO 用 DPO-style loss 把 preferred trace 拉高，把两类 non-preference trace 拉低：

$$
L_2=-\mathbb{E}_{(q,s_w,s_l)\sim D_2}\log\sigma\left(\beta\log\frac{\pi_\theta(s_w\mid q)}{\pi_{\theta_{\mathrm{SFT}}}(s_w\mid q)}-\beta\log\frac{\pi_\theta(s_l\mid q)}{\pi_{\theta_{\mathrm{SFT}}}(s_l\mid q)}\right)
$$

MiPO 的意义在于把“正确答案但糟糕解释”也作为负例。这个设计很重要，因为 deepfake detection 的可解释性不是装饰：如果解释只会说“皮肤很平滑”或“边缘有 artifacts”，它可能在新伪造方式上完全失效。MiPO 让模型学会更精确、更细粒度地定位证据。

这里还有一个方法论上的细节值得保留。Veritas 的 cold-start 数据严格来自 in-domain training set，没有把 OOD 测试样本泄漏进 reasoning alignment。也就是说，MiPO 的目标不是把新域答案背下来，而是改善模型组织证据的方式。对于 OOD detection 来说，这个区分很重要：如果训练阶段已经见过 cross-domain artifact，所谓泛化就会变成数据覆盖；Veritas 更强的 claim 是，它在只看基础伪造类型时学到了一套能迁移到新伪造方式的 forensic reasoning pattern。

第三步是 P-GRPO。冷启动模型已经能基本推理，但面对更难样本还不够。P-GRPO 在 9K in-domain training images 上在线采样多条 response，用 reward 鼓励正确答案加合适的 pattern 使用。它不奖励单纯长推理，而奖励“正确且使用 planning/reflection”的输出；如果答案错还使用 reflection，则给更大惩罚。最终 reward 是 pattern-aware reward、reflection quality reward 和 format reward 的组合：

$$
R=R_{\mathrm{pattern}}+\lambda_1R_{\mathrm{ref}}\cdot I(C=1)+\lambda_2R_{\mathrm{fmt}}
$$

这个 reward 设计的关键是 **adaptive reasoning/自适应推理**。简单样本不需要长篇 planning；困难样本需要 layered analysis 和 self-reflection。论文避免把 reasoning length 当成能力本身，而是把 planning/reflection 当成在特定样本上应该被调用的工具。这个点比“让模型多想”更实在。

## 4. Experiments

主结果显示，Veritas 在 HydraFake 上达到 97.3% in-domain、98.6% cross-model、90.3% cross-forgery、82.2% cross-domain，表中整体平均为 90.7%。传统视觉 detector 在 cross-model 上已经很强，例如 Co-SPY、D3、Effort 都能超过 90%，但在 cross-forgery 和 cross-domain 上明显掉下去。Veritas 的主要优势正是在这两个 OOD 维度上缩小差距。

和 generic MLLM 比较时，结果更明显。InternVL3-8B base 只有 58.3% average，GPT-4o 为 60.8%，Gemini-2.5-Pro 为 78.9%。Veritas 相对 base MLLM 提升 32.4%，相对 Gemini-2.5-Pro 高 11.8%。这说明直接把通用 MLLM 用作 deepfake judge 不够，必须用任务特定的 reasoning data 和 reward 把取证能力内化进去。

和 MLLM-based forgery detectors 比较时，Veritas-mini 平均 85.8%，超过 FakeVLM 的 77.3%、SIDA-7B 的 76.3%、FFAA 的 64.0%、M2F2-Det 的 63.2%。这个对比缓解了“Veritas 只是靠更多数据”的质疑。即使用受限训练范围，它仍然显示出 pattern-aware reasoning 的优势。

Ablation 支持方法设计。Pattern-aware reasoning 相比 flexible `<think><answer>` 在 cross-forgery 上提升 6.2%，在 cross-domain 上提升 3.3%；post-hoc explanation 在 OOD 上表现更差，说明先分类再补解释不能替代推理式检测。训练阶段 ablation 中，SFT+MiPO 和 SFT+P-GRPO 都有效，二者结合最好，在 cross-forgery 和 cross-domain 上分别进一步提升 2.9% 和 2.1%。

具体 reasoning pattern 的 ablation 也很有信息量。去掉 `<reflection>` 后，cross-forgery 从 87.4 降到 82.5，cross-domain 从 80.1 降到 77.3，是最关键的退化之一。去掉 `<planning>` 对 cross-model 影响更明显。论文给出的解释是：cross-forgery 和 cross-domain 常常需要模型发现未见 artifact，而 reflection 可以迫使模型跳出第一层视觉线索，重新检查物理一致性、文本异常、表情协调和背景逻辑。

MiPO 的 non-preference 设计也有实证支撑。去掉“答案正确但解释粗糙”的 non-preference $s_l^\phi$，cross-forgery 和 cross-domain 分别下降 1.3 和 0.8；去掉“答案错误”的 non-preference $s_l^\psi$ 则严重崩到 60.8 average。这个结果说明 Veritas 的偏好优化同时需要两种负例：错误答案负责保证分类方向，粗糙解释负责提高 reasoning quality。

论文还评估了 reasoning quality。用 GPT-4o 和 Gemini-2.5-Pro 做 judge，Veritas with MiPO 的 ELO rating 为 1359.0，高于 without MiPO 的 984.0 和 DPO 版本的 1210.0。这个结果当然依赖 MLLM-as-a-judge，但它和定性例子一致：Veritas 更能指出 badge 文本异常、表情与局部纹理不协调、背景和物理 plausibility 问题，而不是给出模板化“特征不一致”。

实验的安全意义需要克制。Veritas 证明 pattern-aware MLLM detector 可以显著提升 deepfake detection 泛化，但它仍然主要围绕 facial images。rebuttal 补充了 LOKI、FakeClue、Forensics-Bench、AIGIBench 和 Nano-banana-150K 等 broader benchmark，其中 Veritas 在 LOKI 达到 72.1/77.8 acc/F1，在 FakeClue 达到 85.9/88.4，显示出一定 AIGC generalization。但这仍是扩展证据，不等于模型已经覆盖音频、视频、多帧时序伪造、身份一致性和跨平台传播链。

鲁棒性实验也要按正确权重理解。JPEG compression、Gaussian blur 和 resize 下，Veritas 仍保持较高准确率，这说明它没有完全依赖脆弱的单一像素痕迹；但这些扰动属于通用退化，不是 adaptive attack。真正的 adversarial deepfake generator 可以针对 detector 的 reasoning trace 优化，例如专门修复文本徽章、表情协调、背景一致性或皮肤纹理过度细节。Veritas 给了更强 detector，但没有证明 detector 已经在自适应伪造者面前稳固。

Failure cases 的价值也不小。作者指出 real images 的错误多来自低分辨率和局部模糊，因为这些自然退化会伪装成 artifact；fake images 的错误多来自完全未见的 relighting 等伪造类型。这个失败模式说明 content authenticity 需要不确定性表达。模型不应只输出 real/fake，还应说明“证据是否来自低质量成像”“是否需要人工复核”“是否缺少足够分辨率支持结论”。否则高准确率 detector 在平台里仍可能制造过度自信的误判。

## 5. Reviewer Discussion

OpenReview 最终决定是 **Accept (Oral)**。四个 reviewer 初始分数是 8、8、6、4；作者在 rebuttal summary 中说明低分 reviewer 在讨论后把分数提高到 8，AC meta-review 也写到三位正向 reviewer 和一位负向 reviewer 的担忧已被回应。正面评价集中在 HydraFake 的现实性、hierarchical OOD protocol、Veritas 的 pattern-aware reasoning、MiPO/P-GRPO 训练管线，以及 cross-forgery/cross-domain 的显著提升。

主要批评有三类。第一，novelty：有 reviewer 认为 SFT+GRPO 是已有后训练策略，MiPO 和 P-GRPO 的算法形式不算彻底新。第二，公平比较：早期版本没有充分比较 FakeShield、M2F2-Det、SIDA、FakeVLM、FFAA 等近期 MLLM-based detector，且 Veritas 是 8B MLLM，和小视觉模型直接比较可能不公平。第三，domain coverage：论文主要关注 face-oriented deepfake detection，能否推广到 general AIGC detection、image manipulation localization 和 broader multimedia forensics 仍需证据。

作者 rebuttal 对第二、第三点补得比较实。它增加了与近期大型多模态/推理式 detector 的定量和定性比较，并引入 Veritas-mini 控制训练范围；还补充了 LOKI、Forensics-Bench、FakeClue、AIGIBench、Nano-banana-150K 等 broader benchmark，展示 Veritas 虽只用 facial data 训练，仍能在部分 generic AIGC detection 上取得强结果。作者还补了 failure cases、reasoning pattern ablation、reflection 作用分析和 MiPO 样本选择说明。

我的客观评述是：这篇论文的强项不在“发明了全新的 RL 算法”，而在 **把 deepfake detection 的评测协议和 MLLM 推理训练一起向真实部署推进**。HydraFake 提供了更难、更细的 OOD 分层；Veritas 则说明仅靠 generic MLLM 或模板化解释不够，必须训练模型形成结构化取证推理。

不过 reviewer 对 domain scope 的质疑必须保留。Veritas 是很强的 facial deepfake detector，不应被宣传成完整 content authenticity model。真实安全场景还包括视频时序一致性、音频伪造、跨图像身份一致性、图文不一致、传播上下文和元数据审计。Veritas 的 pattern-aware reasoning 可以迁移为框架，但当前证据最强的仍是人脸图像真实性判断。

我对 novelty 的判断比低分 reviewer 稍微宽松。SFT、preference optimization 和 GRPO 本身当然不是新算法，但论文的贡献在于把它们改造成面向 forensic reasoning 的训练系统，并用 HydraFake 这种分层 OOD protocol 检验。安全与评测论文经常不是靠全新优化器取胜，而是靠把任务边界定义准确、把失败模式暴露清楚、把训练信号接到真实风险上。按这个标准，Veritas 的 oral 价值是成立的。

我还会更冷一点看“human forensic process”这个叙事。论文用 fast judgement、planning、reflection 来模拟人类取证流程，这有助于组织模型输出，但 deepfake detection 不是人类直觉判断。很多伪造痕迹是低层统计、压缩和生成器 artifacts，人类未必能可靠感知。Veritas 真正有效的原因可能是：显式 reasoning pattern 约束了 MLLM 的注意力和输出结构，使它更稳定地组合低层视觉线索和高层语义一致性，而不是因为它真的复现了人类法证专家的认知过程。

## 6. Related Work & Future Work

Veritas 位于三个方向交叉处。第一是传统 deepfake detector，包括频域、空间域、序列域和 artifact augmentation 方法。第二是 MLLM-based forgery analysis，它强调解释性和多模态理解，但容易给出模板化或后验解释。第三是 reasoning-oriented post-training，用 SFT、preference optimization 和 GRPO 让模型形成更稳定的推理模式。

未来最重要的是扩展 modality 和 temporal reasoning。单张人脸图像只是 deepfake 风险的一部分。视频 deepfake 需要检查帧间身份、表情动力学、口型-音频同步、光照变化和压缩轨迹；音频 deepfake 需要声纹、韵律和语义一致性；图文传播场景还需要判断 caption、来源和上下文是否伪造。Pattern-aware reasoning 可以扩展，但 reward 和数据构造会更难。

第二个方向是把 Veritas 从 detector 变成 audit system。真实平台不只需要 real/fake 二分类，还需要风险等级、伪造类型、证据定位、可复核解释、用户可理解报告和不确定性校准。Veritas 的 reasoning trace 是起点，但还需要和 localization mask、forensic metadata、multi-image identity matching 和 human review workflow 结合。

第三个方向是对抗鲁棒性。论文测试了 JPEG compression、Gaussian blur 和 resize，但 deepfake detector 在现实中会面对 adaptive attacker。攻击者可以专门优化生成结果来欺骗 Veritas 的 reasoning patterns，或者诱导模型产生看似合理但错误的反思。后续需要 adversarial evaluation：模型是否会被 text prompt、视觉贴片、压缩策略或样式迁移诱导出错误结论。

最后还需要校准拒识机制。对高风险媒体鉴定来说，模型在证据不足时应该输出 uncertainty，而不是强行给出 real/fake。Veritas 的解释能力如果能和置信度校准、证据定位、人工复核队列结合，才更接近可部署的安全工具。
