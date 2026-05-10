---
title: "Empirical Privacy Protection"
headline: "Benchmarking Empirical Privacy Protection for Adaptations of Large Language Models"
description: "ICLR 2026 Oral note on empirical privacy risks in differentially private LLM adaptations under pretrain-adapt distribution shifts."
status: complete
visibility: public
topic: "safety_alignment"
venue: "ICLR 2026 Oral"
year: 2026
paper_url: "https://openreview.net/forum?id=jY7fAo9rfK"
openreview_url: "https://openreview.net/forum?id=jY7fAo9rfK"
source_pdf: "raw/papers/openreview-jY7fAo9rfK.pdf"
tags: ["differential_privacy", "privacy_auditing", "llm_adaptation"]
created: "2026-05-01"
updated: "2026-05-01"
---

> [!abstract] Contributions
>
> 这篇论文研究一个很实际的问题：LLM 已经在大规模 public corpus 上预训练，如果之后用 differential privacy/DP 去适配敏感数据，理论 DP guarantee 是否足以带来实际 privacy protection？作者构建了一个 empirical benchmark，用 robust membership inference attack/RMIA 和 canary extraction 检测 DP adaptation 的泄漏，并系统改变 adaptation data 与 pretraining data 的关系：exact overlap、IID but non-overlap、OOD。核心结论是，**distributional closeness/分布接近性** 会显著放大隐私风险，即使 adaptation data 没有直接出现在预训练集中，和预训练分布越接近，在同一 $\varepsilon$ 下越容易泄漏。
>
> 论文还比较了 Full Fine-Tuning、Head Fine-Tuning、LoRA、Prefix Tuning 等 adaptation methods，并发现 LoRA 在多数设置下提供更好的 empirical privacy-utility tradeoff，尤其在 OOD adaptation 上更稳；但 Prefix Tuning 在 data extraction 和 pretraining leakage 上表现出不同侧面。边界也很明确：这是一篇强经验论文，理论解释不足；模型规模主要在 Pythia/GPT-Neo/OLMo 1B 左右，虽然 rebuttal 补了 OLMo 2 7B LoRA；并且 closed-source models 因训练数据不可知，很难直接套用这里的 IID/OOD 审计框架。

## 1. Introduction

很多敏感场景并不会从零训练模型，而是拿一个公开预训练 LLM，再用医疗、法律、企业邮件或个人数据做 adaptation。理论上，DP-SGD、DP-LoRA、DP-prefix tuning 等方法可以为 adaptation dataset 提供形式化隐私保证。但实际情况更复杂：预训练模型已经见过大规模文本，adaptation data 可能和预训练数据重合，也可能来自同一分布。此时，即使 DP 只保护 adaptation step，模型已经在 pretraining 阶段学到的分布结构也可能帮助攻击者判断某条记录是否进入了 adaptation。

这篇论文的关键直觉是：不能把 pretraining 和 adaptation 当成彼此独立的两个隐私阶段。一个医院用病历微调模型时，隐私泄漏可能来自 adaptation data，也可能来自 pretraining data，或者来自二者的交互。若 adaptation data 和 pretraining distribution 很接近，攻击者可以利用 pretrained model 或 shadow model 的知识，提升 membership inference 的成功率。也就是说，同一个 $\varepsilon$ 在不同数据关系下对应的实际风险并不相同。

论文要补的正是这个实践空白。它不重新发明 DP 算法，而是问：在真实 pretrain-adapt pipeline 中，DP adaptations 对强攻击到底有多抗？哪些 adaptation method 更安全？IID、OOD、overlap 的差异有多大？如果攻击者知道 base model 或能训练 shadow model，风险会怎样变化？这些问题比单独报告一个 privacy budget 更接近部署决策。

## 2. Problem Setup

论文的核心对象是 pretrained LLM $\theta$ 和经过 DP adaptation 后的模型 $\theta'$。DP 的标准定义仍然是：随机机制 $\mathcal{M}$ 作用在两个只差一个样本的 neighboring datasets $D,D'$ 上时，输出落入任意集合 $S$ 的概率满足

$$
\Pr[\mathcal{M}(D)\in S]\le e^\varepsilon \Pr[\mathcal{M}(D')\in S]+\delta.
$$

这里 $\varepsilon$ 越小，形式化隐私越强。但论文强调，它要测的是 **empirical privacy risk/经验隐私风险**，也就是强攻击在真实模型上能不能判断成员关系或抽取 canary。

数据关系被分成三类。**Overlap** 表示 adaptation data 直接来自 pretraining set；**IID** 表示 adaptation data 没有和 pretraining 重叠，但来自同一 Pile subset 的 validation split，例如 BookCorpus2、GitHub、Enron；**OOD** 表示 adaptation data 来自不同分布，例如 SAMSum 和 GermanWiki。rebuttal 后作者还用 Sentence-BERT embeddings 计算 Wasserstein distance，显示 Pile-based datasets 距离更低，SAMSum 和 GermanWiki 更远，从而把原先偏 categorical 的 IID/OOD 划分补成更量化的分布距离证据。

这个划分背后的安全问题很具体。假设某个企业用内部邮件微调一个已经在大量网页和邮件语料上预训练的模型，adaptation data 即使没有逐字出现在 pretraining corpus 中，也可能和预训练分布高度相似。模型已经学到该分布的语言模式、实体结构和常见表达，微调时只需要吸收少量剩余信息，攻击者也更容易用 pretrained reference 判断某条样本是否改变了模型行为。反过来，OOD data 需要模型学习更多新分布结构，DP noise 和有限 adaptation capacity 可能更容易削弱成员信号。这就是为什么“没重合”不等于“安全”。

攻击主要有两类。第一类是 **membership inference attack/MIA**，用 RMIA 判断某条样本是否在 adaptation data 中。作者主要报告 AUC，并比较 shadow-model RMIA、reference calibration 和 Min-K%。第二类是 **canary exposure/data extraction**，把特定 canary 插入 adaptation data，测模型是否能恢复或排序出这些秘密字符串。MIA 回答“是否参加训练”，extraction 回答“能否直接泄露内容”，后者风险更严重。

## 3. Algorithm / Methods / Model

Benchmark 覆盖多个 pretrained model family：Pythia 70M 到 1.4B、GPT-Neo 125M/1.3B，以及 OLMo 1B/OLMo 2 1B。默认讨论 Pythia 1B，因为它的 pretraining corpus 来自 Pile，便于定义 overlap/IID/OOD。adaptation methods 包括 Full Fine-Tuning、Head Fine-Tuning、LoRA 和 Prefix Tuning。以 Pythia 1B 为例，Full Fine-Tuning 更新约 1B 参数，LoRA 约 1M 参数，Prefix Tuning 约 130M 参数，Head Fine-Tuning 约 100M 参数。

一个重要设计是控制 utility。不同 adaptation method 如果训练到完全不同的 validation loss，隐私比较会被 utility gap 混淆。作者因此尽量让方法在同一数据集上达到相似 validation perplexity/loss，再比较隐私攻击 AUC。rebuttal 后他们还补了 Rouge-1 和 out-of-domain utility tables，回应 reviewer 对“只用 perplexity 是否足够”的担忧。

RMIA 设置里，最强攻击者可以使用和 target model 同架构、同初始化、同训练数据分布但不同 split 的 shadow model。论文也测试了更弱攻击者：用 pretrained model 作为 reference、用同 family 不同 size 的模型、用不同 architecture 或没有 reference 的 Min-K%。这个维度很关键，因为实际风险高度依赖 attacker knowledge。公开 base model 越容易获得，攻击者越能构造强 shadow。

论文最后提出 **holistic privacy auditing/整体隐私审计** 框架，把隐私审计分成四个阶段：pretraining audit、adaptation audit、joint pretraining-adaptation audit、post-adaptation audit of pretraining。它用 adversarial game 形式定义样本 $x$ 是否在 pretraining set $S$、adaptation set $D$ 或二者中。这个框架的意义是把“泄漏来自哪里”形式化，而不是只问 adapted model 有没有泄漏。

四类 audit 的差异可以这样理解。Pretraining audit 问 $x$ 是否进入了 $S$；adaptation audit 问 $x$ 是否进入了 $D$，但攻击对象是从 pretrained model 出发得到的 $\theta'$；joint audit 问 $x$ 是否同时影响了 pretraining 和 adaptation，例如同一病历既在公开抓取语料中出现，又在医院 fine-tuning set 中出现；post-adaptation pretraining audit 问 adaptation 之后，原本来自 pretraining 的数据是否更容易或更不容易被检测出来。这套分类很有用，因为真实事故中用户通常想知道“我的数据从哪个阶段泄漏”，而不是只得到一个抽象的攻击成功率。

## 4. Experiments

RQ1 的核心结论是：**IID adaptation 比 OOD adaptation 更容易泄漏**。在 Pythia 1B、$\varepsilon=8$、shadow RMIA 设置下，IID datasets 的平均 AUC 大约在 0.70 到 0.90 之间，OOD datasets 大约在 0.63 到 0.87 之间。更关键的是，IID validation data 的泄漏接近 overlap data，说明直接重合不是唯一风险；只要分布足够接近，预训练模型已经学到的结构就会帮助攻击。

RQ2 比较 adaptation methods。非私有设置下，LoRA 明显比 Full/Head/Prefix 更不容易被 MIA 攻击；进入 DP setting 后差异变小，但 LoRA 仍然相对稳定。OOD、$\varepsilon=8$ 时，Full 和 Head Fine-Tune 更脆弱；IID、强 privacy regime 下，LoRA 的 AUC 往往最低。论文没有给出完整理论解释，但经验上参数高效方法减少可更新自由度，可能降低 adaptation data 被模型吸收成可攻击信号的程度。

RQ3 研究 data extraction。结果显示 Prefix Tuning 在 extraction setting 中最脆弱，而 LoRA 和 Head Fine-Tune 对 canary extraction 更抗。这一点和 RQ5 形成有趣反差：Prefix Tuning 会降低 pretraining memorization leakage，却在 adaptation canary extraction 上更弱。原因可能和 prefix 在输入隐藏空间中引入强任务条件有关，但论文主要停留在经验观察，理论解释仍不足。

RQ4 说明 attacker knowledge 极其重要。RMIA 在 shadow model 与 target model 共享 architecture、initialization 和 training data distribution 时最强；当 reference model 换成不同 distribution 或 architecture，private adaptations 上攻击性能会迅速接近随机。但这不是好消息，因为很多 LLM adaptation 就发生在公开 base model 上，攻击者可以直接拿同一个 pretrained model 做 reference。

RQ5 看 adaptation 如何改变 pretraining data leakage。DP 只保证 adaptation data，但 adaptation 训练本身可能影响模型对预训练样本的记忆。Prefix Tuning 在 high-privacy regime 下会减少 verbatim memorized samples，尤其 $\varepsilon$ 较小时更明显；其他 adaptation methods 对 pretraining memorization 影响较小。这说明不同 adaptation 不只改变新数据泄漏，也会改变旧数据在模型中的可抽取性。

RQ6 研究 privacy-utility tradeoff。作者做 hyperparameter search，展示在相似 perplexity 下，LoRA 往往有更低 RMIA AUC。例如 GermanWiki、$\varepsilon=8$ 下，LoRA 在 perplexity 14.27 时 AUC 为 0.77，而 Full Fine-Tuning 在相近 perplexity 14.60 时 AUC 为 0.82；GitHub validation 上差距更明显。这个结果对 practitioner 最有用：如果需要在可用性和隐私之间选方法，LoRA 是论文中最稳的默认选择。

整篇实验最强的现实结论是：**moderate privacy budget 并不等于实际安全**。论文明确指出 $\varepsilon=8$ 这类常见设置在敏感 adaptation data 上仍然有显著 practical vulnerability，尤其是 IID 或 overlap 情况。作者甚至在 conclusion 中强调需要更 stringent DP constraints，例如 $\varepsilon<0.1$，才能更有效降低实践风险。这个判断很冷酷，但对部署很有价值：只报形式化 $\varepsilon$，不做 empirical audit，是不够的。

这里还要注意 **formal guarantee 与 empirical attack success 的非等价关系**。DP 约束的是相邻数据集输出分布的最大可区分性，但实际模型部署中，攻击者会利用模型架构、pretraining corpus、reference model、数据分布和 loss landscape 的细节。论文不是说 DP 理论错了，而是说同一个理论 budget 下的实际攻击面差异很大。一个 $\varepsilon=8$ 的 LoRA adaptation 和一个 $\varepsilon=8$ 的 Full Fine-Tuning，在形式参数上相同，经验上却可能泄漏不同；一个 OOD dataset 和一个 IID dataset，在形式 DP 上相同，攻击 AUC 也可能不同。因此部署者应该把 DP budget 看成必要条件，而不是充分安全证书。

另一个值得单独强调的结果是 Prefix Tuning 的双重性。它在 RQ5 中减少 pretraining memorization leakage，可能因为 prefix 在输入侧引入额外扰动，使模型原本对预训练样本的 verbatim extraction 变弱；但在 RQ3 中，Prefix Tuning 对 adaptation canary extraction 更脆弱。这个现象提醒我们，不能用“某个方法更隐私”一句话概括所有阶段。保护 adaptation data、减少 pretraining leakage、保持 utility、降低 extraction risk，可能分别偏向不同方法。

## 5. Reviewer Discussion

OpenReview 最终决定是 **Accept (Oral)**。四个 reviewer 的初始分数是 8、6、4、4，讨论后至少三位 reviewer 提高了分数；AC 记录中提到 DyRT 可能从 6 到 8，两个低分 reviewer 提到或被记录为从 4 到 6。正面评价集中在问题重要、系统评测覆盖多模型多数据多方法、用 RMIA 和 canary exposure 做强攻击审计，以及提出 holistic privacy audit 这个更完整的 pretrain-adapt 风险框架。

低分意见主要击中三个问题。第一，初稿只测 $\varepsilon=0.1$ 和 $\varepsilon=8$，中间 privacy budgets 不够，导致趋势不够细。第二，distributional closeness 最初主要是 categorical 的 overlap/IID/OOD，没有量化距离。第三，utility 评估太依赖 perplexity 和 validation loss，缺少 task-level 或 generation-level 指标。Reviewer 还质疑模型规模偏小、closed-source generalization 不清楚、holistic audit 框架解释不够具体。

作者 rebuttal 的补充很实在：加入 $\varepsilon\in\{1,3,5\}$ 的中间预算实验，加入 Wasserstein distance 量化数据分布距离，加入 Rouge-1 与 OOD utility tables，补 OLMo 2 7B LoRA 实验，并更具体解释 holistic audit 如何区分 pretraining leakage、adaptation leakage 和二者交互。我的判断是，这些补充基本把“证据不够系统”的批评修到了可接受水平，但没有完全解决“为什么不同 adaptation 会有不同隐私风险”的理论问题。

这篇最值得肯定的是它没有把 DP guarantee 当成部署结论。很多论文只会说“我们用了 DP，因此有隐私保证”；这篇直接用强攻击去审计，并指出同一 $\varepsilon$ 下 empirical leakage 会随 distribution relation 和 adaptation method 大幅变化。这对 safety alignment 很重要，因为真实对齐系统的监督数据、人类反馈和私有日志都可能进入 post-training；如果这些数据泄漏，模型部署就不只是性能问题，而是合规和用户安全问题。

我最保留的地方是 benchmark 的外推性。Pythia/GPT-Neo/OLMo 这类 open models 的好处是训练数据可知，能严格定义 IID/OOD；但现实商业模型恰恰通常不知道 pretraining data。作者说这个框架更适合 model owner 或 auditor 拿到训练数据后使用，这是合理的。但对于外部用户，它只能给出方向性 warning，不能直接判断某个 closed model adaptation 是否安全。

还有一个技术保留是 metric。论文主要报告 AUC，这有利于比较整体排序能力，但在隐私安全里，低 false-positive region 往往更关键。一个攻击如果整体 AUC 中等，但在 1% FPR 下仍能识别最脆弱样本，就可能足以造成实际隐私伤害。Reviewer 对 utility metric 和 MIA metric 的质疑都指向同一件事：privacy benchmark 不能只追求可比较性，还要贴近风险使用场景。后续版本若能系统报告 TPR@low-FPR、per-sample exposure tail、以及不同数据子群的 worst-case leakage，会比平均 AUC 更有部署意义。

## 6. Related Work & Future Work

这篇和 DP-SGD、PrivateLoRA、PromptDPSGD、DP-ICL、privacy auditing、membership inference、canary extraction、LLM-PBE 等工作相邻。它的增量不是单个攻击算法，而是把 pretraining distribution、adaptation method、privacy budget、attacker knowledge 和 utility tradeoff 放进一个统一经验框架。

未来最值得做的是 **mechanistic explanation of adaptation leakage**。为什么 LoRA 通常更稳？为什么 Prefix Tuning 对 adaptation canary extraction 更脆弱，却能减少 pretraining memorization？这些现象如果只停在 AUC 表格上，对方法设计帮助有限。第二个方向是 larger and newer open models，尤其是训练数据透明的 7B、13B 或更大模型。第三个方向是把 holistic audit 变成工具链：给定 pretraining corpus、adaptation data 和 model checkpoints，自动输出 pretraining/adaptation/joint/post-adaptation 四类风险报告。

还可以继续推进 **data-selection-aware DP adaptation**。如果分布接近性是风险主因，那么 adaptation 前就应评估私有数据和 pretraining corpus 的距离：越接近的样本越需要更强 clipping/noise、更谨慎的 method selection，甚至需要剔除或重写。这样 DP adaptation 不再是统一给全数据一个 $\varepsilon$，而是把 pretraining overlap risk 纳入数据治理。对医疗、金融和企业邮件场景，这种 pre-adaptation audit 可能比事后发现泄漏更实际。

这篇也应和 Hubble 一起读。Hubble 提供受控 memorization model suite；这篇提供 DP adaptation 下的 empirical privacy audit。前者更像研究平台，后者更像部署审计方法。两者共同说明一点：LLM privacy 不能只靠单个指标，必须同时知道数据来自哪里、什么时候进入训练、以什么频率出现、以及模型经过什么 adaptation。
