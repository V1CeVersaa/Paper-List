---
title: "Ellipse Signature"
headline: "Every Language Model Has a Forgery-Resistant Signature"
description: "ICLR 2026 Oral note on ellipse signatures for language model output attribution, model forensics, and forgery-resistant logprob verification."
status: complete
visibility: public
topic: "safety_alignment"
venue: "ICLR 2026 Oral"
year: 2026
paper_url: "https://arxiv.org/abs/2510.14086"
openreview_url: "https://openreview.net/forum?id=vLFqOoMBol"
source_pdf: "raw/papers/arxiv-2510.14086.pdf"
tags: ["model_attribution", "content_provenance", "model_forensics"]
created: "2026-05-01"
updated: "2026-05-01"
---

> [!abstract] Contributions
>
> 这篇论文提出 **ellipse signature/椭球签名**：现代 language model 的最后几层通常是 normalization 加 linear unembedding，normalization 把 hidden state 投到高维球面，linear layer 再把球面仿射变换成高维椭球，因此模型输出的 logits 或 logprobs 会落在某个由模型参数决定的 ellipsoid 上。作者把这个自然几何约束当作模型签名，用来判断某个 logprob vector 是否来自特定模型。
>
> 论文最有价值的点是把这种签名的属性讲清楚：它 **naturally occurring/自然发生**，不需要 provider 额外植入 watermark；它 **self-contained/自包含**，验证单个 logprob vector 不需要 prompt；它 **compact and redundant/紧凑且冗余**，每个 generation step 的完整 next-token logprob 都带有签名；它还具有实践意义上的 **forgery resistance/抗伪造性**，因为当前已知的 ellipse extraction 需要 $O(d^2)$ samples 和 $O(d^6)$ fitting time。边界也非常硬：这不是严格密码学 unforgeability，需要 full logits/logprobs，现实 API 常不暴露足够信息，而且 fine-tuning、quantization 或输出扰动会改变或破坏签名。

## 1. Introduction

闭源 LLM 普及后，model forensics 变成一个实际问题。监管者、平台、用户或第三方审计者可能想知道某段输出到底来自哪个模型，或者 provider 是否否认了自己模型生成过某个有害输出。常见路线是 watermarking、fingerprinting、behavioral attribution 或 API probing，但这些方法各有代价：有的需要主动修改 sampling，有的需要大量文本样本，有的依赖输入 prompt，有的容易被微调或重写移除。

这篇论文切入得很几何。作者观察到，LM 输出不是任意分布在 $\mathbb R^v$ 中的向量，$v$ 是 vocabulary size；它受到最后 normalization layer 和 unembedding matrix 的结构约束。只要模型有 final normalization，hidden state 会被投到固定半径的球面；再经过线性层，球面变成嵌在 logit space 里的 $d$-dimensional ellipsoid，$d$ 是 hidden size。于是，每个完整 logprob vector 都携带一个“这个模型的输出必须在这张椭球面上”的约束。

这个 idea 对 safety 很有吸引力，因为它提供一种不靠文本内容、不靠 provider 主动加水印的 **model output attribution/模型输出归因**。如果某个 provider 把自己的 secret ellipse 交给可信第三方，那么第三方可以验证一个 logprob vector 是否落在该模型椭球上。论文进一步把它类比为 message authentication code/MAC：ellipse 是 secret key，logprob vector 同时是 message 和 tag，验证就是检查它是否在 secret ellipse 上。

但这篇不能被粗暴读成“所有 LLM 文本天然带不可伪造水印”。Reviewer 批评这点非常关键。token sequence 本身不够，必须拿到完整或足够多的 logprob 信息；而真实用户看到的通常只是采样出来的文本。论文的签名更准确地说是 **logprob-level model forensics mechanism**，不是 text-level watermark。

## 2. Problem Setup

考虑一个典型 language model 的最后层。设 hidden state 为 $x\in\mathbb R^d$，RMSNorm 或 LayerNorm 把它归一化成 $\hat x$。为简化，论文先用不含 $\epsilon$ smoothing 的 scaled RMSNorm，此时 $\|\hat x\|_2=1$。若 normalization 后还有 learned scale 和 bias，则 hidden representation 变成 $\gamma\odot \hat x+\beta$。最后 unembedding matrix $W$ 把它映射到 vocabulary logit space：

$$
z=W(\gamma\odot \hat x+\beta).
$$

因为 $\hat x$ 在单位球面上，$W(\gamma\odot \hat x+\beta)$ 是球面的 affine image，所以 $z$ 落在一个 $d$-dimensional ellipsoid 上。这个 ellipsoid 嵌在 $\mathbb R^v$ 中，其中 $v\gg d$。API 常返回 logprobs 而不是 raw logits；由于 softmax 对整体加常数不敏感，论文用 centering matrix $C=I-\frac{1}{v}\mathbf 1\mathbf 1^\top$ 消除常数方向，把 centered logprobs 近似还原到 centered logits。

验证时，若知道目标模型的 ellipse parameters，就可以把 logprob $\ell$ 通过 inverse affine transform 映回 hidden sphere。若它来自该模型，映回后的 norm 应接近 1；若来自别的模型，通常会明显偏离。论文在 OLMo 2、Llama 3.1、Qwen 3、GPT-OSS 等 open-weight models 上测试，发现生成模型自身 ellipse 的距离比其他模型小几个数量级，即使 OLMo 2 的相邻 checkpoint 也能区分。

这里的 **signature** 和传统 fingerprint/watermark 不完全同类。text watermark 往往在 sampling 时主动改变 token distribution，目标是在最终文本里留下可检测统计信号；ellipse signature 是模型结构自然产生的 logprob geometry。它不 stealthy，也不 robust to parameter changes；它的优势是自然发生、自包含、单步冗余，以及在已知攻击方法下难以伪造。

RMSNorm 版本最容易理解，LayerNorm 版本更麻烦，因为 LayerNorm 不只固定 norm，还会移除均值，使 hidden states 落在一个低一维的 affine subspace 与球面交集上。论文把 LayerNorm 放到 appendix 中处理，结论仍然是类似的几何约束存在。这个细节说明 ellipse signature 不是某个单一模型实现的偶然产物，而是 final normalization plus affine output projection 这一现代架构模式的结构后果。不过，若模型没有 final normalization，或者 provider 在 logprobs 上做强后处理，签名假设就需要重新检查。

## 3. Algorithm / Methods / Model

论文把 forgery 定义为：攻击者没有目标模型参数，却能生成一个新的 logprob vector，使其通过目标 ellipse verification。线性签名容易伪造，因为攻击者可以从 API 输出中提取 linear constraints，再构造满足 constraints 的新 logprobs。ellipse signature 的抗伪造性来自一个更难的问题：要生成新的 on-ellipse point，当前已知路线基本要先恢复整个 ellipse，而恢复 ellipse 的 sample complexity 和 computation 都很高。

ellipse fitting 的一般形式是 quadric surface：

$$
\sum_{i=1}^d\sum_{j=i}^d Q_{ij}x_ix_j+\sum_{i=1}^d P_i x_i=1,
$$

其中 $Q$ 必须 positive definite 才是 ellipsoid。参数数量约为 $d(d+3)/2$，因此一般需要 $O(d^2)$ points 才能唯一确定。对 API 攻击者来说，一个 point 是一次 next-token logprob vector；如果 API 每次只返回 top-k 或少量 token logprobs，还要多次 query 才能恢复足够维度。论文估计，在 OpenAI-like API 中，query/sample cost 可达到 $O(vd+d^3\log d)$。

计算成本更重。ellipse fitting 要解 $O(d^2)$ equations with $O(d^2)$ variables，典型算法时间复杂度 $O(d^6)$，空间复杂度也可到 $O(d^4)$。作者实现 ellipse-specific fitting algorithm，并在小维度模型上测试；对 70B-scale 模型按 degree-6 polynomial extrapolation，计算时间可到数千年量级。这个估算当然不能当作安全证明，但它说明当前直接 extraction attack 在生产规模模型上不实际。

> [!note] Forgery Resistance
> 论文使用 **forgery-resistant** 而不是 **unforgeable** 是正确的。它证明的是：在现有 ellipse extraction 方法下，伪造需要昂贵的 sample collection 和 fitting；它没有给出基于标准密码学假设的安全性证明，也没有排除存在不拟合完整 ellipse 就能生成新 on-ellipse points 的算法。

MAC-like protocol 的想法是这样：provider 或可信方知道 secret ellipse；模型生成 logprobs 时，logprob vector 的位置自然编码 tag；verifier 检查该向量是否落在 secret ellipse 上。如果用户拿到一段争议输出和对应 logprobs，第三方可验证它是否来自 provider 的模型。论文也承认 sequence-level attack：攻击者可能保存很多真实 logprob outputs，再拼接成新的 token sequence。对此可以用 output database 或 logprob inversion 检查 prefix consistency，但这会带来额外系统成本。

这里最容易误读的是“每个 generation step 都有签名”。这句话只对完整 next-token distribution 成立，也就是该步所有候选 token 的 logprob vector。最终采样出来的单个 token 不携带足够信息，因为一个 token 只是从分布里抽到的离散结果，不是整个分布几何位置。若只有普通文本，攻击者或 verifier 只能从大量样本中估计模型分布，难度和不确定性完全不同。因此，这个 protocol 更像 **signed probability trace**，而不是 **signed text**。

另一个实际问题是 verifier 权限。若 provider 把 ellipse parameters 交给监管者，监管者就能验证输出；但 ellipse 本身也包含模型 final layer 的敏感结构信息，可能带来模型抽取风险。论文把 ellipse 当作 secret key，意味着 key management、访问控制、撤销、版本管理都必须被系统化处理。模型一旦 fine-tune 或量化，ellipse 可能变化；如果 provider 频繁更新模型，就需要为每个版本维护独立签名，否则 attribution 会混淆。

## 4. Experiments

实验第一部分验证 discriminative power。作者从多个 open-weight models 生成 logprob outputs，把不同模型的输出投影到共享 token/column space 后，计算它们到各 target ellipse 的距离。结果是生成模型本身的 distance 最小，且通常和其他模型有几个数量级差异。这个结果支持“ellipse 可作为模型身份签名”，至少在完整 logprob 可用、模型未被修改的条件下成立。

第二部分研究 ellipse extraction。由于 RMSNorm 中的 $\epsilon$ smoothing 会让点落在 ellipsoid interior 而不是 surface 上，普通 SVD-based fitting 有时会失败。作者改用 ellipse-specific fitting 方法，并在一个 1M-parameter model 上恢复 bias、singular values 和 rotation，发现预测和真实参数较接近。增加 samples 能改善预测，但会遇到 smoothing 导致的 irreducible error。

第三部分是成本外推。表格估算 hidden size 越大，所需 samples 和 API cost 急剧增长。例如 pythia-70m 级别只需要约 13 万 outputs；babbage-002 级别约 118 万 outputs；gpt-3.5-turbo 级别可到千万级 outputs，成本超过 15 万美元；Llama-3-70B 级别则被估计到三千多万 outputs 和千万美元级别。计算端的 $O(d^6)$ 也被图 6 外推到极端漫长时间。

实验最强地支持的是 **practical extraction hardness under known methods**。它没有证明所有 polynomial-time adversary 都失败，也没有解决 full-logprob API 不常见的问题。更重要的是，signature 对模型修改很脆弱。LoRA、SFT、quantization、distillation、temperature/noise、top-k truncation 都可能改变验证条件。论文承认 ellipse signature 不具备 hard-to-remove property，这使它更适合 accountability with logprob records，而不是开放互联网上的文本溯源。

从证据强度看，实验分成“识别能不能做”和“攻击难不难做”两层。前者比较扎实：给定 target ellipse 和完整 logprobs，distance test 能把模型分开。后者更像当前算法复杂度论证：作者展示了拟合 ellipse 的已知方法在维度上爆炸，但没有给出 lower bound，也没有排除利用局部 tangent、随机组合、score matching 或 generative approximation 产生近似 on-ellipse point 的新攻击。安全读法应该是：当前没有明显可行的 forgery route，而不是数学上已证明不可伪造。

## 5. Reviewer Discussion

OpenReview 最终决定是 **Accept (Oral)**，但 review 分歧非常大：四个 reviewer 分数为 8、4、6、6。AC 认为核心 idea 新颖、有理论吸引力，并且可能影响 model fingerprinting 和 accountability；同时也记录了 reviewers 对 claims 过强、实践条件苛刻和 cryptographic framing 的明显担忧。

正面评价集中在几何观察本身。Reviewer 认可 final normalization plus linear layer 诱导 ellipsoid constraint 这个推导清晰，模型间 ellipse distance 也显示出强区分能力。高分 reviewer 特别强调，这种 signature 自然发生、单步存在、无需 provider 主动插入，和已有 watermark/fingerprint 方法占据不同位置。另一些 reviewer 则认为把 ellipse framing 成 MAC-like protocol 有启发性，能帮助思考模型供应商、监管者和第三方 verifier 的关系。

低分意见击中了论文最薄弱的表述。第一，**watermark** 这个词容易误导，因为普通文本 token 不足以检测 ellipse signature；需要完整 logprobs，甚至可能需要 API workaround。第二，论文早期对 “unforgeability” 的措辞过强。高复杂度 extraction algorithm 不等于问题本身有密码学硬度；learning the ellipse 和 forging a new valid point 也不是同一个问题。第三，现实 API 很少暴露 full logits，top-k logprob、温度、量化和后处理都会影响检测。第四，signature 对 fine-tuning 或模型版本更新的鲁棒性没有系统研究。

我的客观评述是：这篇 oral 应该被读成一篇 **model-output geometry forensics** 论文，而不是一篇已经完成的 watermarking/security protocol 论文。它最强的地方是提出了一个自然、干净、可验证的结构性约束；它最危险的地方是用“signature”“MAC”“forgery-resistant”这些词时容易让读者误以为已有密码学级别保证。Reviewer 的批评不是小修小补，而是要求把 claim 的边界压回证据能支持的位置。

不过，这并不削弱它作为 idea paper 的价值。很多 model attribution 方法依赖行为统计或文本 distribution，容易被 prompt、sampling 和后处理扰动；ellipse signature 直接利用输出空间几何，是一个不同层级的证据。只要使用场景是 provider retained logprobs、trusted audit、模型版本追责或闭源 API accountability，它就可能有实际意义。真正需要避免的是把它宣传成“用户只拿到文本也能证明来源”的通用水印。

更冷静地说，Reviewer 的分歧恰好说明这篇论文的性质：它有一个漂亮、可能开新线的 observation，但应用叙事走得比 formal security 快。8 分 reviewer 看重 idea 的新颖性和潜在影响；4 分 reviewer 则要求 security paper 必须严格定义 adversary、forgeability 和可验证对象。两边都对。作为读者，最好的处理方式是保留几何 insight，同时把 “MAC-like” 视为 protocol sketch，而不是已经满足密码学审稿标准的构造。

我还会特别保留 reviewer 对“learning versus forging”的区分。学出完整 ellipse 当然足以伪造，但伪造未必一定要学出完整 ellipse。攻击者也许只需要从已有 outputs 的局部几何中构造一个新的近似有效点，或者利用验证阈值、数值误差、API truncation 产生灰区样本。论文没有发现这种方法，也给出了当前 fitting 路线的巨大成本；但安全结论必须停在这里。把“没有已知便宜攻击”说成“不可伪造”，就是过度外推。

## 6. Related Work & Future Work

论文和 linear signatures、model extraction、text watermarking、fingerprinting、behavioral attribution 都有关。linear signatures 也来自模型输出空间的结构约束，但更容易通过较少 API samples 伪造；text watermarking 可在最终文本中检测，但需要 provider 主动改变生成过程；behavioral fingerprinting 更黑箱，但通常要大量 samples。ellipse signature 的 niche 是：在 logprob-level record 可用时，用模型几何给出强 attribution evidence。

后续工作最重要的是把三个问题分开解决。首先是 **partial logprob verification**：现实 API 只暴露 top-k 时，能否仍可靠检测 ellipse membership。其次是 **robustness under model changes**：LoRA、SFT、quantization 和 distillation 会让 ellipse 怎么变化，是否能区分同一 base model 的版本。最后是 **formal forgery model**：攻击者能看多少 outputs、能否 query adaptive prompts、验证阈值如何设定、是否允许 approximate on-ellipse points，都需要更严格定义。

如果这些问题解决不了，ellipse signature 仍然可以作为 forensic evidence 的一部分，但不能独立承担完整 provenance system。如果解决得好，它可能成为未来模型日志审计和监管接口的一块底层机制。

这篇还暗示一个更宽的问题：模型输出的 **概率分布轨迹** 可能比最终文本更适合作为审计对象。很多安全争议只保存了文本，因此事后很难证明模型来源、prompt context 或采样过程；如果 API 系统能保留带签名的 logprob traces，审计证据会强很多。但这也带来隐私和商业机密问题，因为 logprobs 可能泄露 prompt 信息、模型行为细节和用户查询。未来 provenance system 必须同时设计签名、日志最小化、访问控制和用户隐私，不能只解决几何验证。
