---
title: "Spherical Watermark"
headline: "Spherical Watermark: Encryption-Free, Lossless Watermarking for Diffusion Models"
description: "ICLR 2026 Oral note on Spherical Watermark, a lossless diffusion watermarking method that maps binary signatures into Gaussian latent noise."
status: complete
visibility: public
topic: "safety_alignment"
venue: "ICLR 2026 Oral"
year: 2026
paper_url: "https://openreview.net/forum?id=2eAGrunxVz"
openreview_url: "https://openreview.net/forum?id=2eAGrunxVz"
source_pdf: "raw/papers/openreview-2eAGrunxVz.pdf"
tags: ["watermarking", "content_provenance", "diffusion_safety"]
created: "2026-05-01"
updated: "2026-05-01"
---

> [!abstract] Contributions
>
> 这篇论文提出 **Spherical Watermark**，目标是在 text-to-image diffusion model 的初始 Gaussian noise 里嵌入可追踪水印，同时保持生成分布不被可检测地改变。它的核心技术动作是把二进制 watermark bits 先通过随机 padding 和可逆 binary embedding matrix 混合成高熵 bitstream，再归一化到单位球面，经过 orthogonal rotation，最后用 chi-square-distributed radius 缩放回标准高斯噪声。这样生成的 latent noise 可以直接作为 Stable Diffusion 的初始噪声，输出图像在视觉质量和统计分布上尽量接近无水印样本。
>
> 论文的安全价值在于把 **AI-generated content provenance/生成内容溯源** 做成一个不需要改模型权重、不需要 per-image key storage、且比传统 lossy watermark 更难被检测器反向攻击的方案。边界同样清楚：理论保证主要匹配到 third-order moments，并依赖可逆或近似可逆的生成过程；如果攻击专门破坏 VAE encoder、diffusion inversion 或整个图像到噪声的反演路径，水印恢复仍可能失败。因此这篇更像 diffusion provenance infrastructure，而不是完整解决所有图像篡改和伪造归因。

## 1. Introduction

Diffusion models 已经把高质量图像生成变成低门槛能力，但这也放大了内容真伪、责任追踪和恶意传播问题。对 safety 来说，watermarking 的目标不是让模型“更善良”，而是给生成内容留下可验证 provenance：当一张图被怀疑来自某个模型服务、某个 API 用户或某次生成请求时，平台能够从图像中恢复标识信息。

传统图像水印有一个根本困难：它往往在生成后修改 pixel 或 frequency domain，因此会引入 distribution shift。攻击者如果能训练一个 detector 区分 watermarked 与 clean images，就可以反过来优化图像，让水印检测失效。更接近 diffusion 的 latent watermarking 试图在初始噪声上做文章，因为扩散模型本来就是从 Gaussian latent 开始采样。只要被嵌入水印后的 latent 仍然像标准高斯，那么生成图像也更有机会保持无损、不可检测。

已有 lossless watermarking 方案已经朝这个方向走，但作者认为它们仍有实际负担。Gaussian Shading 需要每张图对应 key 和 nonce，PRC Watermark 通过 pseudorandom error-correcting code 避免 per-image key，却引入 heavy cryptographic operations 和 belief-propagation decoding。Spherical Watermark 的切入点就是用更直接的几何构造替代复杂密码结构：把 binary message 映射成球面上的近似均匀方向，再乘以高斯半径。

水印关系到生成内容责任、平台审计和恶意用户追踪，但这篇不是 LLM privacy 或 memorization 论文，而是 **diffusion provenance/扩散模型溯源**。它的技术问题是：在 diffusion sampling 过程中嵌入一个攻击者难以伪造、难以移除、且仍能保持图像质量的统计签名，让生成内容在事后可以被可靠归因。

## 2. Problem Setup

论文从 model developer 的视角定义问题。设固定扩散生成器为 $G:\mathbb{R}^{l_x}\to I$，它把标准高斯噪声 $z$ 映射成图像 $O$。因为 latent diffusion 可以近似反演，开发者可以用 $G^{-1}$ 从可疑图像中恢复 latent estimate。水印长度为 $l_m$，目标是设计两个 latent-space procedure：

$$
\mathrm{Embed}: m\in\{0,1\}^{l_m}\to z_w\in\mathbb{R}^{l_x},\qquad
\mathrm{Extract}: \hat z_w\in\mathbb{R}^{l_x}\to \hat m\in\{0,1\}^{l_m}.
$$

这里 $m$ 可以编码 API metadata，例如 user ID、timestamp 或请求标识；$z_w$ 是嵌入水印后的初始噪声；$\hat z_w=G^{-1}(\hat O_w)$ 是从可疑图像反演得到的 latent。论文要求两件事。第一是 **undetectability/losslessness**：任何多项式时间 adversary 都不应能区分 $z_w$ 和标准高斯 $z$，因此生成图像 $G(z_w)$ 也不应和普通生成图像有可检测差异。第二是 **traceability/exact extraction**：给定由水印 latent 生成的图像，反演后应能以高概率恢复原始 $m$。

这个 setup 里最重要的变量不是 pixel-level mark，而是 initial noise distribution。只要水印 latent 分布偏离 Gaussian prior，生成结果就可能出现视觉质量下降或 detector-visible artifact。反过来，如果水印 latent 仍然统计上像标准高斯，攻击者就更难通过“找出水印痕迹再去掉”的路线破坏系统。

这里还有一个容易误解的地方：论文所谓 **lossless** 不是说生成图像经过压缩、编辑、重绘后水印永远不损失，也不是说抽取过程没有数值误差。它指的是 watermark embedding 本身不应改变扩散模型期望采样的目标 prior。也就是说，水印不是被事后“贴”到图像上，而是在采样起点处被编码成一个仍然合法的 Gaussian latent。后续图像是否能恢复水印，则取决于反演算法能否把图像重新带回足够接近的初始 latent。这个 distinction 很重要，因为它把“无损嵌入”和“鲁棒抽取”分成了两个问题：前者靠球面几何保证，后者靠扩散反演、冗余编码和 majority vote 抵抗噪声。

## 3. Algorithm / Methods / Model

方法由三个可逆模块组成：**Binary Embedding Module**、**Spherical Mapping Module** 和 **Diffusion Integration Module**。第一步先把 watermark $m$ 重复 $N$ 次，并拼接随机 padding $r\in\{0,1\}^{l_r}$，得到

$$
x=[m\ m\ \cdots\ m\ r]^\top\in\{0,1\}^{l_x},\qquad l_x=Nl_m+l_r.
$$

重复 $m$ 是为了后续 majority voting，提高从 noisy inverted latent 中恢复 bit 的能力；padding $r$ 则用于打散重复结构。否则同一个 bit 重复出现会形成明显相关性，破坏 undetectability。

Binary embedding 的核心是一个可逆矩阵

$$
T=
\begin{bmatrix}
I_{Nl_m} & R\\
0 & I_{l_r}
\end{bmatrix},
$$

其中 $R$ 是 sparse binary matrix。每个 watermark bit 会和 $s$ 个 padding bits 混合，并且同一个 bit 的 $N$ 个重复副本使用互不重叠的 padding subsets。矩阵运算在 $\mathbb{F}_2$ 上进行，得到 $z^{(1)}=Tx$。这个设计的目的不是加密，而是让 $z^{(1)}$ 的坐标满足 Bernoulli$(1/2)$，并具有 2-wise 与 3-wise independence。

第二步是 spherical mapping。作者把 bit 映射成 $\{-1,+1\}$ 向量，归一化到单位球面：

$$
v=2z^{(1)}-1,\qquad z^{(2)}=\frac{v}{\lVert v\rVert_2}.
$$

接着用随机正交矩阵 $C$ 做旋转，得到 $z^{(3)}=Cz^{(2)}$，再采样半径 $r'$ 使得 $(r')^2\sim\chi^2(l_x)$，最终令

$$
z_w=r'z^{(3)}.
$$

这一步背后的数学很直接：标准多元高斯可以分解为“均匀球面方向 + 独立 chi-square 半径”。如果 $z^{(2)}$ 近似球面均匀方向，旋转后仍保持球面设计性质，再乘以正确半径，就应接近 $\mathcal{N}(0,I)$。论文用 spherical 3-design 证明该构造匹配球面均匀分布到三阶矩，并用 Lemma 3.4 连接到 Gaussian polar decomposition。

抽取水印时，开发者先用 VAE encoder 和反向 ODE solver 从图像恢复初始 latent estimate $\hat z_T$，再依次做 $C^{-1}$、rounding、$T^{-1}$，最后对 $N$ 个重复 bit 做 majority vote。这里有一个关键工程边界：抽取质量依赖 inversion。如果图像经过严重篡改，导致反演 latent 和原始 embedded latent 偏差太大，rounding 与 majority vote 就会失败。

从机制上看，$s$、$N$、$l_m$、$l_r$ 四个超参数分别控制不同风险。$l_m$ 是 payload capacity，越大越能编码 user ID、timestamp、request metadata 等丰富信息，但也会压缩冗余和 padding 的空间。$N$ 是 repetition count，越大越能通过投票纠正 inversion noise，但会增加 latent 中重复结构，需要 binary embedding 更努力地打散相关性。$l_r$ 是随机 padding 长度，它提供 entropy source；如果 padding 太少，重复水印的统计结构更容易暴露。$s$ 是每个 watermark bit 混合多少 padding bits，增大 $s$ 会提升混合程度，却也让单个 padding 或 inversion error 影响更多 bit。论文默认选择 $N=31,l_m=512,l_r=512,s=1$，本质上是在容量、不可检测性和纠错之间选了一个偏稳的工作点。

> [!note] Why Spherical Mapping Matters
>
> Binary embedding 负责把重复 watermark bits 打散成 3-wise independent bitstream；spherical mapping 负责把这个 bitstream 变成接近 Gaussian prior 的连续 latent。少掉前者，重复结构容易被 detector 学出来；少掉后者，binary direction 到高斯噪声的几何关系不稳，鲁棒性明显下降。

## 4. Experiments

实验主要围绕 Stable Diffusion v1.5 和 v2.1 展开，图像分辨率为 $512\times512$，latent size 为 $4\times64\times64$。默认配置是 $N=31$、$l_m=512$、$l_r=512$、$s=1$，因此 latent 维度 $l_x=16384$。基线包括传统 DwtDct、DwtDctSvd、RivaGAN，latent-based Tree-Ring、Gaussian Shading 和 PRC Watermark。

第一组结果验证 **undetectability**。作者用 FID、latent-level MLP classifier 和 image-level ResNet-18 classifier 检测 watermarked 与 clean samples。Tree-Ring 和 fixed-key Gaussian Shading 在 latent 或 image classifier 上很容易被识别，准确率接近 100% 或 97%；PRC Watermark 和 Spherical Watermark 则接近 chance level。FID 上，Spherical Watermark 与原始生成分布几乎一致，在 COCO/SD v1.5 上甚至略低于 original reference 的波动范围。这支撑了论文“lossless”的经验含义：不是 pixel 一模一样，而是水印机制没有制造可检测分布偏移。

第二组结果看 tracing accuracy。COCO + SD v2.1 下，Spherical Watermark clean ACC 约 99.99%，post-processing ACC 约 95.02%，adversarial ACC 约 98.12%；对应 TPR 在 clean/post/adversarial 下分别约 100%、97.50%、99.83%。和 PRC 相比，Spherical Watermark 在强扰动下更稳，尤其 TPR 差距明显。论文还强调 extraction time 比 PRC 快约四个数量级，因为它不需要 belief-propagation decoding。

第三组结果是 ablation。去掉 binary embedding 后，latent detector 很容易区分水印噪声；去掉 spherical mapping 后，在 brightness perturbation 下水印恢复准确率明显下降。参数实验显示，增大 sparsity $s$ 会让一个 bit 依赖更多 padding，错误更容易传播；减小 repetition $N$ 会削弱 majority voting 冗余。因此 $s$ 和 $N$ 体现了 undetectability、robustness 和容量之间的工程折中。

rebuttal 后补充的实验很重要。作者在 SD v3 和 FLUX.1-DEV 上测试，说明方法能迁移到 transformer-based latent diffusion；在 Guided Diffusion 和 Glow 上也能达到较高追踪准确率，说明条件是“Gaussian prior + invertible mapping”，不严格限于 Stable Diffusion。针对 reviewer 质疑的 re-generation 与 editing attacks，作者加入 WAVES 和 W-Bench：在 Regen-Diff 与 Rinse-2xDiff 上 Spherical Watermark 高于其他方法，在 Regen-VAE 上不如 TrustMark/Robust-Wide 但仍好于 PRC；在 MagicBrush、UltraEdit、InstructPix2Pix、GAN-edit 等编辑攻击下平均保持 85% 以上恢复准确率。

实验的局限也比较明确。第一，lossless 的强项是让 detector 没有 distributional gradient 可利用，但真实攻击者可能不只训练 detector，还可能专门攻击 inversion pipeline。第二，论文主要处理“整图由模型生成后需要追踪”的场景；如果内容经过局部 GAN edit、拼接、裁剪、重绘，归因问题会变成 partial provenance，当前方法只能部分覆盖。第三，third-order moment matching 不等于完整 distribution equality，统计上更强的高阶检测仍是未来需要检查的边界。

论文附录里的 adversarial-analysis 也值得保留。作者把 lossy watermark 的脆弱性解释成一个优化问题：如果 watermarked distribution $P_{\mathrm{wm}}$ 和 clean distribution $P_{\mathrm{clean}}$ 有非零 KL divergence，那么最优 detector 的 log-likelihood ratio 会产生可利用梯度；攻击者可以沿着这个梯度在有限 distortion budget 内降低 detector score。这个论证的重点不是给出完整攻击算法，而是解释为什么 distribution-preserving watermark 会天然更抗 detector-driven attack。若 $P_{\mathrm{wm}}\approx P_{\mathrm{clean}}$，detector 学不到稳定方向，adversarial evasion 就缺少可操作梯度。这个分析把“水印不可检测性”和“抗攻击性”连到了一起，是这篇区别于普通工程 benchmark 的地方。

但这个论证也暴露出一个边界：它主要保护的是 **distribution-detector attack**，也就是攻击者先学习水印与非水印分布差异，再优化图像逃避检测。若攻击者知道或猜到 signature 结构、专门扰动反演 latent、或者使用强生成模型对图像做语义级重绘，威胁模型就不再只是 detector gradient。论文补充的 re-generation/editing 实验证明方法在一组常见强攻击下仍可用，但并没有给出 adaptive white-box attack 下的完整安全证明。

## 5. Reviewer Discussion

OpenReview 最终决定是 **Accept (Oral)**。四个 reviewer 的原始分数是 6、8、8、8，AC 总结认为 novelty、theoretical justification、scalability 和 performance 都得到高度认可。正面评价集中在 spherical mapping 的简洁性、三阶矩匹配的数学支撑、无需模型微调和 per-image key 的部署便利性，以及比 PRC 更快的抽取速度。

Reviewer 的主要担忧并不琐碎。低分 reviewer 认为“encryption-free”这个表述可能有误导性，因为系统仍然需要 secret signature，例如矩阵 $T$ 和 $C$；另一个强担忧是初稿没有充分测试 diffusion-specific regeneration attacks 和 editing attacks。还有 reviewer 追问 newer models、inversion accuracy、ODE solver、sampling timestep 这些工程细节是否会影响恢复。作者在 rebuttal 中补了 SD v3、FLUX.1-DEV、re-generation/editing、solver/timestep/inversion error ablations，并说明 $T$、$C$ 可复用且 overhead 可控。

我的客观评述是：这篇的 oral 质量主要来自“问题重要 + 构造干净 + 实验补齐后很完整”。但读的时候不能把 **encryption-free** 理解成“不需要 secret material”。它更准确的意思是“不依赖每张图独立的 stream cipher key/nonce 或 heavy cryptographic code”。实际部署仍然需要保护 signature；如果 $T$、$C$ 泄露，攻击者就能更有针对性地移除或伪造水印。

从 deployment 角度看，signature management 仍然是系统安全的一部分。论文的方案减少的是 per-image key management，而不是消除密钥生命周期。平台仍需决定 $T$ 和 $C$ 是全局共享、按模型版本共享、按租户共享，还是定期轮换；还要处理泄露后的 revocation、历史图像验证和多平台互认证据链。如果一个开发者把同一组 signature 用在所有用户和所有时间段，那么一旦泄露，影响面很大；如果过度频繁轮换，又会增加 provenance database 的复杂度。论文主要解决嵌入和抽取机制，没有完整展开这些运维问题。

我也认为 reviewer 对 re-generation 和 editing 的质疑非常关键。AIGC watermark 的真实对手不会只做 JPEG、blur 或 resize；更强的对手会用另一个 diffusion model 重绘、用编辑模型局部修改、甚至对 inversion step 做 adversarial perturbation。作者补充实验之后，论文的结论更可信，但也更清楚地显示方法的边界：它在常见和部分强攻击下很稳，不代表能解决所有 downstream manipulation provenance。

## 6. Related Work & Future Work

Spherical Watermark 接在传统 pixel/frequency watermark、latent watermark、Tree-Ring、Gaussian Shading 和 PRC Watermark 之后。它相对于 lossy watermark 的区别在于不直接改变输出图像，而是在扩散采样最开始的 Gaussian prior 上编码；相对于 PRC 的区别在于用球面几何和冗余投票替代复杂 error-correcting cryptographic pipeline。

未来最重要的方向是 **partial and adversarial provenance**。现实恶意图像常常不是完整原图传播，而是经过裁剪、编辑、重绘和压缩。水印系统需要回答的不只是“这张图有没有来自某个模型”，还包括“哪些区域来自模型、哪些区域被修改、修改后责任如何归因”。第二个方向是把 watermark 与平台审计结合：signature 管理、API 用户映射、滥用申诉、误报处理和跨平台验证都需要协议设计。第三个方向是高阶统计检测和 adaptive attacks，因为 third-order matching 在理论上仍留下检测空间。

还有一个长期问题是 **watermark 与内容安全的关系**。Watermark 只能帮助事后追踪，不会阻止模型生成危险内容；如果平台把 watermark 当作主要安全措施，就会把 detection/provenance 和 prevention 混为一谈。更合理的系统应同时有 generation policy、prompt/output moderation、watermark provenance 和 incident response。Spherical Watermark 在这套系统里负责“这张图是谁生成的、能否追溯到请求”，而不是负责“这张图是否应该被生成”。

这篇和 `Steering the Herd`、`Veritas` 一样，属于 Safety & Alignment 里偏社会安全基础设施的支线。它不直接研究模型内部 alignment failure，而是提供生成内容责任追踪接口。这个接口在未来监管和平台治理里可能非常实际。
