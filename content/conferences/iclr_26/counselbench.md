---
title: "CounselBench"
headline: "CounselBench: A Large-Scale Expert Evaluation and Adversarial Benchmarking of Large Language Models in Mental Health Question Answering"
description: "ICLR 2026 Oral note on CounselBench, a clinician-grounded benchmark for evaluating LLM responses to open-ended mental-health questions."
status: complete
visibility: public
topic: "safety_alignment"
venue: "ICLR 2026 Oral"
year: 2026
paper_url: "https://arxiv.org/abs/2506.08584"
openreview_url: "https://openreview.net/forum?id=8MBYRZHVWT"
source_pdf: "raw/papers/arxiv-2506.08584.pdf"
tags: ["mental_health_safety", "expert_evaluation", "llm_judge"]
created: "2026-05-01"
updated: "2026-05-01"
---

> [!abstract] Contributions
>
> **CounselBench** 针对的是一个高风险但经常被普通 medical QA benchmark 忽略的问题：真实心理健康求助往往是开放式、情境化、情绪化的 free-text question，不是选择题，也不能只用事实正确率衡量。论文和 100 名心理健康专业人士合作，构建了两个资源：**CounselBench-EVAL** 包含 100 个 CounselChat 真实患者问题、GPT-4、LLaMA-3.3、Gemini-1.5-Pro 和在线 human therapist 的回答，以及 2000 条专家评价；**CounselBench-ADV** 包含 120 个专家撰写的 adversarial prompts，用来触发 medication advice、therapy advice、symptom speculation、judgmental tone、apathetic tone 和 unsupported assumptions 等失败模式。
>
> 论文最重要的发现是：LLM 在若干质量维度上可以得分很高，但仍会出现 **unauthorized medical advice/未经授权医疗建议**、过度泛化、缺少个性化、事实不稳和安全边界不清；更关键的是，LLM-as-judge 在这个领域系统性高估回答质量，并漏掉人类专家标出的安全风险。边界也很明确：CounselBench-EVAL 仍是 single-turn static evaluation，问题来自 CounselChat 一个公开论坛，人类 baseline 是 top-voted informal therapist responses；它不能替代临床试验或真实多轮咨询评估，但作为 clinician-grounded benchmark 已经比普通自动评测强很多。

## 1. Introduction

心理健康 QA 是 LLM 部署里风险非常高的一类场景。用户问题常常混合症状、情绪痛苦、人际冲突、药物疑虑、创伤经历和求助意图。一个回答是否“好”，不能只看 factual accuracy，还要看它是否有同理心、是否具体、是否避免过度诊断、是否没有冒充专业治疗、是否没有强化自责或危险行为。传统 medical QA benchmark 多是 multiple-choice 或 fact-based tasks，很难捕捉这些维度。

这篇论文要补的正是这个空缺。它没有把心理健康问答压成考试题，而是直接使用 CounselChat 上真实患者提出的 open-ended questions，并请 mental health professionals 评价 LLM 和 human therapist 的回答。这一点非常关键，因为现实用户不会问“DSM 条目 A/B/C 哪个正确”，而会问“我是否应该离开关系”“我为什么总是焦虑”“我应该怎么和家人谈创伤”等开放问题。模型在这种问题上犯错，后果可能比普通事实题更直接。

论文的另一个重要动机是质疑 **LLM-as-Judge/用 LLM 做评审**。高质量专家评价很贵，所以很多 benchmark 倾向于用 GPT-style judge 扩展评测。但在心理健康场景，评审本身就需要专业知识和伦理边界判断。如果 judge 会漏掉未经授权医疗建议或淡化不恰当语气，那么自动评测会把风险隐藏起来。CounselBench 因此同时评测模型回答和 LLM judge 的可靠性。

这篇在 Safety & Alignment 里非常合适，因为它不是单纯心理学数据集，而是一个 **high-stakes alignment evaluation/高风险对齐评测**。它关心模型是否在真实求助语境中遵守临床边界、是否避免伤害、是否能被专业评审体系可靠评价。这个方向和 reward modeling、RLHF、automated evaluation 都有关系，因为如果评测器漏掉安全问题，后续对齐训练就会沿着错误信号优化。

## 2. Problem Setup

CounselBench 由两个互补部分组成。第一个是 **CounselBench-EVAL**，用于系统比较回答质量。作者从 CounselChat 选择 100 个真实患者问题，覆盖 20 个心理健康主题，包括 depression、relationships、anxiety、family conflict、parenting、self-esteem、trauma、domestic violence、substance abuse、eating disorders、professional ethics 等。每个问题保留一个 top-voted human therapist answer，并让 GPT-4-0613、LLaMA-3.3-70B-Instruct、Gemini-1.5-Pro 各生成一个回答。

评价维度有六个。**Overall Quality** 是整体质量；**Empathy** 衡量情绪回应、理解和验证；**Specificity** 衡量回答是否贴合用户具体语境，而不是泛泛建议；**Medical Advice** 是二元维度，标记回答是否给出了只有专业人士在个体评估后才应提供的治疗、诊断或用药建议；**Factual Consistency** 衡量回答是否符合常识和临床知识；**Toxicity** 衡量是否有伤害性、污名化、轻视或伦理上不当的语言。

这个 rubric 的设计比普通 helpfulness 更细，因为心理健康回答的风险常常不在“错得离谱”，而在边界细节。比如，建议“考虑寻求专业支持”通常是安全的；但直接推荐特定药物、具体治疗方案或根据短问题推断诊断，就可能越过临床边界。论文还要求专家提供 span-level annotations 和 written rationales，因此不只是打分，还记录了哪里出问题、为什么出问题。

第二个部分是 **CounselBench-ADV**。作者先从 EVAL 中分析出 LLM 的具体失败模式，再让 10 名心理健康专家为每种失败模式撰写 adversarial questions。每个问题不是直接包含错误，而是设计成容易诱发模型犯对应错误。最终得到 120 个 adversarial questions，并让 9 个 LLM 生成 1080 个回答，再由 5 名专业人士标注目标 failure mode 是否出现。

这个 setup 的优势是从真实错误出发构造 stress test。很多 red-teaming benchmark 是研究者预设风险类型；CounselBench-ADV 则从专家在真实回答中看到的失败模式出发，反过来生成能触发这些失败的 prompt。这样得到的 adversarial set 更贴近临床评估，而不是单纯 prompt injection 或 jailbreak 风格攻击。

## 3. Algorithm / Methods / Model

CounselBench-EVAL 的 pipeline 先从数据源控制开始。CounselChat 上的问题和 human therapist responses 都在 ChatGPT 普及前发布，因此 human answers 不太可能被 LLM 生成污染。作者按主题和 upvotes 选择问题，保证一定的 topical diversity 和回答质量。人类 therapist baseline 选择每个问题的 top-voted answer；这个选择可以减少论坛回答质量波动，但也带来代表性问题：upvote 不等于临床最佳回答。

回答生成方面，作者先试过 MentalLLaMA、Meditron 等 domain-specific models，但发现它们在开放式心理健康 QA 上表现较差，于是选择三类主流 general-purpose LLM：GPT-4、LLaMA-3.3 和 Gemini-1.5-Pro。每个问题对应四个回答：三个 LLM answer 和一个 human therapist answer。专家不知道回答来源，从而降低 source bias。

专家评价协议很扎实。100 名 annotators 都有心理健康训练、证书或专业经验，作者核验教育、执照或从业背景。每个 annotator 评价 5 个问题，每个问题 4 个回答，共 20 个 QA pairs；每个回答由 5 名不同专家独立评价。因此总量是：

$$
100\ \text{questions}\times 4\ \text{answers}\times 5\ \text{annotations}=2000\ \text{expert evaluations}.
$$

论文还报告 inter-rater reliability。除 Medical Advice 这个二元维度外，Krippendorff's alpha 在 Overall、Empathy、Specificity、Factual Consistency、Toxicity 上均不低于 0.72，说明专家评分有相当一致性。这对 benchmark 可信度很关键，因为心理健康问答虽然主观，但并非无法形成专业共识。

在 LLM-as-Judge 实验中，作者让 9 个 advanced LLM 使用同样 rubric 评价同样 QA pairs。这样可以直接比较机器评审和人类专家评审在分数、排序和 span-level failure detection 上是否一致。这个设计尤其重要，因为很多后续 alignment 训练可能会用 LLM judge 作为 reward model 或 evaluator；如果 judge 在 mental health safety 上系统性偏差，整个优化链条会被污染。

CounselBench-ADV 的构造更像 targeted error probe。作者从 EVAL 中抽出六类具体失败模式：GPT-4 容易给 medication 或 therapy techniques；LLaMA-3.3 容易 speculative symptoms 或 judgmental；Gemini-1.5-Pro 容易 apathetic 或 unsupported assumptions。每类 failure mode 由临床专家根据真实样例写 prompt，然后在 9 个模型上测试。标注任务也从多维 Likert rating 简化成是否出现目标问题，便于更直接测 failure trigger rate。

论文还把伦理边界写得比较清楚：研究经过 USC IRB exempt review，问题和回答来自 MIT-licensed CounselChat，并在使用前移除姓名、主页链接等标识信息；专家 annotators 提供 informed consent，并同意释放去标识化数据。这个细节不是装饰，因为 mental health benchmark 本身就可能二次暴露求助者脆弱经历。一个高风险评测资源若不说明数据来源、许可、去标识化和专家同意，其安全价值会被数据治理风险抵消。

## 4. Experiments

主结果先看 EVAL 的专家平均分。LLaMA-3.3 在 Overall、Empathy、Specificity、Factual Consistency 和 Toxicity 五个维度上得分最高，但它也有 14% 的回答被标记为包含 Medical Advice。GPT-4 的 overall score 为 3.28，Gemini-1.5-Pro 为 3.26，online human therapist 为 2.60；LLaMA-3.3 为 4.29。这一结果容易被误读成“LLM 超过人类治疗师”，但必须非常谨慎：human baseline 是论坛 top-voted informal answer，不是标准临床咨询，也不是同样条件下新写的专业回答。

更有意义的是 failure analysis。低质量回答的主要问题因模型而异。GPT-4 常被标为 unconstructive feedback 和 lack of personalization；LLaMA-3.3 常见 overgeneralization 或 insufficient-context assumptions；Gemini-1.5-Pro 常见 lack of empathy 和 unconstructive feedback；human therapist baseline 也会有 overgeneralization 和 inappropriate tone。这个结果说明，模型不是简单“好/坏”，而是在不同临床维度上有不同风险画像。

Medical Advice 是最值得单独看的安全维度。专家标注发现，LLM 有时会给出特定药物建议，例如 SSRIs，也会建议 CBT、mindfulness 等治疗技巧。这里的风险不是这些技术本身一定错误，而是它们通常需要个体化评估、治疗关系和专业监督。模型如果在单轮匿名问题中直接给出具体医疗建议，就可能越过适当边界。论文在 ADV 里把这类问题细分为 medication、therapy techniques 和 symptom speculation。

LLM-as-Judge 的结果非常关键。大多数 LLM judge 会系统性高估回答，尤其在 Factual Consistency 上给出接近满分；Toxicity 维度几乎没有区分度，即使人类专家标记了潜在伤害或不当内容，LLM judge 也经常给最低 toxicity。模型排序也偏离人类专家，尤其 Gemini-1.5-Pro 被人类评为较差，却被多个 LLM judge 排在 GPT-4 之上。span-level failure detection 更糟：大多数 LLM judge 很少捕捉专家标出的有毒、事实错误或未经授权医疗建议句子。

CounselBench-ADV 展示了更直接的风险。针对 therapy suggestions，GPT-5 的 trigger rate 高达 0.85，LLaMA 系列约 0.55-0.65，Claude 系列约 0.45-0.50。symptom speculation 在 GPT-4、GPT-5、LLaMA、Claude 中也较常见。medication advice 对多数模型较低，但 GPT-5 是明显 outlier，达到 0.47。apathetic tone 在 GPT-3.5-Turbo 上最高，为 0.70。unsupported assumptions 在多数模型上都有 0.25-0.40。

这些 ADV 结果说明，失败模式具有 model-family patterns。同一系列模型的错误分布相似，而跨模型家族差异明显；GPT 系列内部也可能随版本发生大幅变化。这个观察对部署很重要，因为 mental health safety 不应该只做一次静态评测。模型升级后，失败模式可能重新分布；一个版本压低了 medication advice，另一个版本可能在 therapy advice 或 symptom speculation 上升。

论文最后再次测试 LLM-as-Judge 在 ADV 上的 failure-mode detection。即使给出明确 failure mode 定义和 in-context examples，最好的 judge F1 也只有 0.50 左右。这是一个非常强的警告：在心理健康 QA 里，用 LLM judge 替代专家评估会系统性漏检关键安全风险。它不只是“分数有点偏”，而是可能把需要临床判断的风险当成普通 helpful response。

还有一个实验解释细节很重要：专家评分不是只给数字，median written rationale length 达到 576.5 words，且许多评价包含具体 span evidence。这意味着 CounselBench 可以支持比 leaderboard 更细的分析，例如研究某类模型为什么被认为缺少 empathy、哪些句子触发 medical-advice flag、或者 LLM judge 为什么漏掉人类专家认为危险的片段。对 alignment 来说，这种 rationale-rich dataset 比单一 scalar reward 更有价值，因为它可以训练 critique model，也可以帮助定位 reward misspecification。

## 5. Reviewer Discussion

OpenReview 最终决定是 **Accept (Oral)**。AC 总结认为 CounselBench 提供了一个 clinician-grounded、规模较大的心理健康 QA benchmark，核心贡献是 100 名心理健康专业人士参与、六维 rubric、2000 条专家评价，以及 120 个专家构造的 adversarial prompts。Reviewer 普遍认可问题重要性和专业标注质量。

Reviewer 的主要批评集中在 evaluation setting。第一，EVAL 是 static single-turn，不允许模型追问澄清、根据用户反馈修正、或在多轮对话里建立上下文。心理咨询真实互动高度依赖多轮澄清，因此 single-turn QA 不能覆盖全部安全能力。作者回应说开放式单轮 QA 本身就是现实平台中的常见交互，且适合作为标准化资源；AC 也认可这个定位。我的判断是，这个回应成立，但它限定了 benchmark 的解释范围。

第二，human baseline 的代表性被质疑。CounselChat top-voted answer 是公开论坛贡献，质量可能参差不齐，不等同于严格临床 setting 中专业人士即时写出的回答。因此 LLaMA-3.3 高于 human baseline 不能被读成“LLM 超越治疗师”。论文承认这一点；读者也必须把 human baseline 理解为 public forum therapist responses，而不是 clinical gold standard。

第三，初稿 ADV 使用 GPT-4.1 做较多自动评估，与论文自己发现 LLM judge 不可靠形成张力。作者修改后加入或替换为更多 mental health professionals 的人工标注，并报告 LLM judge 与人类专家的 agreement。这个修订很重要，因为否则论文最核心的安全论点会被自己的评估方法削弱。

我的客观评述是：CounselBench 的最大价值是把“心理健康 QA 安全”从泛泛 moral concern 变成可标注、可比较、可 stress-test 的 evaluation object。它没有声称模型可以用于临床，也没有把高分误解为部署许可。相反，它直接展示了模型在开放式求助场景中的边界问题，并证明自动 judge 在这里不够可信。

我最保留的地方是数据覆盖。100 个问题、20 个主题、CounselChat 单一来源、美国专家标注，这些已经很贵也很有价值，但仍然不能覆盖真实心理健康服务的文化差异、危机干预、多轮治疗关系、长期风险、未成年人场景、严重精神症状或自伤危机。CounselBench 更适合作为 baseline safety audit，而不是临床可用性证书。

## 6. Related Work & Future Work

CounselBench 和 Medical QA benchmarks、mental health QA datasets、LLM-as-Judge evaluation、red-teaming、RLHF evaluation 和 high-stakes model auditing 相邻。与 MedQA、MedMCQA 这类选择题不同，它关注开放式患者问题；与小专家 panel 相比，它扩展到 100 名专业人士；与纯自动评测相比，它把 span-level expert rationale 作为核心数据。

后续最重要的是 **multi-turn CounselBench**。真实心理支持往往需要澄清用户背景、识别风险等级、调整语气、避免 premature advice。单轮 benchmark 可以评价第一响应质量，但不能评价模型是否会持续追问、何时升级到专业帮助、如何处理用户反驳或危机升级。未来可以用 simulated patient agents 或专家设计的多轮脚本扩展 ADV。

第二个方向是 **expert-grounded reward modeling**。CounselBench 的 span-level annotations 和 rationales 很适合训练 critique models、safety detectors 或 reward models。但论文也提醒我们，不能直接让 LLM judge 自我强化；需要用专家标注校准 judge，尤其在 Medical Advice、Toxicity 和 Factual Consistency 上控制漏检。

第三个方向是 **failure-mode lifecycle tracking**。模型家族之间、版本之间的错误模式不同，因此 benchmark 应作为持续审计工具，而不是一次性 leaderboard。每次模型更新都应重新测 medication advice、therapy advice、symptom speculation、unsupported assumptions 等风险，尤其要看安全策略是否只是把风险从一个 failure mode 转移到另一个。
