---
title: "CyberGym"
headline: "CyberGym: Evaluating AI Agents' Real-World Cybersecurity Capabilities at Scale"
description: "ICLR 2026 Oral note on CyberGym, a large-scale execution-based benchmark for AI agents' real-world cybersecurity capability evaluation."
status: complete
visibility: public
topic: "safety_alignment"
venue: "ICLR 2026 Oral"
year: 2026
paper_url: "https://arxiv.org/abs/2506.02548"
openreview_url: "https://openreview.net/forum?id=2YvbLQEdYt"
source_pdf: "raw/papers/arxiv-2506.02548.pdf"
tags: ["cybersecurity_benchmark", "agent_evaluation", "ai_security"]
created: "2026-05-01"
updated: "2026-05-01"
---

> [!abstract] Contributions
>
> 这篇论文提出 **CyberGym**，一个面向 AI agent 网络安全能力的 execution-based benchmark。它从 OSS-Fuzz 和 ARVO 体系中构造 1,507 个真实漏洞实例，覆盖 188 个开源项目，核心任务是让 agent 在给定漏洞描述和 pre-patch codebase 的情况下写出 proof-of-concept/PoC，使程序在 pre-patch 版本触发 sanitizer crash，同时在 post-patch 版本不触发 crash。这个设计把安全评测从静态问答、CTF 或小规模 case study 推向了真实仓库、真实构建环境、真实执行 oracle。
>
> 论文最重要的结论有两层。第一，当前 frontier agents 在真实漏洞复现上仍然很弱：非 thinking 设置下最强 OpenHands + Claude-Sonnet-4 只有 17.9% success rate，GPT-5 high reasoning 在 300 样本子集上达到 22.0%，说明这个任务对代码定位、输入构造、工具使用和长程调试都有硬要求。第二，CyberGym 不只是评测集；agent 在复现任务和开放式 Level 0 探索中发现了 34 个 zero-day vulnerabilities 和 18 个 incomplete patches。边界也很明确：当前 benchmark 主要覆盖 C/C++ memory-safety bugs 和 sanitizer-detectable failures，不能代表整个 cybersecurity 能力空间。

## 1. Introduction

AI agents 进入软件工程以后，网络安全是最敏感的能力边界之一。一方面，agent 可以帮助维护者复现漏洞、生成 PoC、发现未修补问题；另一方面，同样的能力也可以被攻击者用来扩展漏洞利用。论文的出发点很直接：如果我们只用小规模 CTF 或静态安全问答来评估 agent，就会低估真实 codebase 中定位、构建、执行反馈、输入变异和调试循环的难度。

CyberGym 的切入点是 **vulnerability reproduction/漏洞复现**。这不是让模型回答“这个函数有没有漏洞”，也不是让模型补丁修复，而是给它一个真实 pre-patch 项目和漏洞描述，让它写一个可执行 PoC 去触发具体漏洞。这个任务之所以适合评估安全能力，是因为它同时要求 agent 看懂报告、找到相关代码、理解输入格式、构造触发条件，并根据 sanitizer 输出迭代。论文引用的背景也说明，人类安全专家复现公开漏洞通常需要数小时，自动 fuzzing 在 OSS-Fuzz 项目中发现漏洞的中位时间更长，因此这不是简单的代码搜索题。

这篇 oral 的价值还在于它把 benchmark 和 security impact 连起来。很多 benchmark 停在 leaderboard；CyberGym 的执行式环境允许作者检查 agent 生成的 PoC 是否意外触发了 post-patch 或 latest 版本中的其他崩溃。这个设计让评测输出可以转化成真实漏洞发现，也解释了为什么论文会被放进 Safety & Alignment 的 Attack & Robustness 小节：它讨论的是 agent capability evaluation，但这些 capability 直接影响网络安全风险。

## 2. Problem Setup

一个 CyberGym 实例来自一个真实历史漏洞。作者为每个实例准备 pre-patch codebase、post-patch codebase、ground-truth PoC、patch diff、漏洞文本描述和 containerized executable。主任务 Level 1 中，agent 获得漏洞描述与 pre-patch codebase，需要提交一个 bash script 形式的 PoC。评测时，系统用 sanitizer 执行这个 PoC；只有当它在 pre-patch executable 上触发 sanitizer crash，并且在 post-patch executable 上不触发 sanitizer crash 时，才算成功。

这个 oracle 很关键。它避免了用 LLM judge 判断 PoC 是否“看起来合理”，也避免了只看是否 crash 的粗糙指标。单纯 crash 不够，因为 agent 可能触发的是另一个 bug，而不是目标 patch 修复的漏洞；post-patch 不 crash 这个条件迫使 PoC 更贴近目标 vulnerability。换句话说，CyberGym 的 success rate 测的是 **targeted vulnerability reproduction**，不是一般程序崩溃制造能力。

论文定义了四个 difficulty levels。**Level 0** 只给 pre-patch codebase，不给漏洞描述，用来模拟开放式漏洞发现。**Level 1** 给 codebase 和文本描述，是论文的主评测设置。**Level 2** 进一步给 ground-truth PoC 执行后的 stack trace，用来模拟报告中包含精确位置或函数名的情况。**Level 3** 再给 ground-truth patch diff 和 post-patch codebase，用来模拟 one-day 分析，即攻击者或防守者从补丁反推漏洞。这个阶梯设计让 benchmark 不只是一个分数，而能区分 agent 在不同信息条件下的能力来源。

数据构造上，作者主要利用 OSS-Fuzz 的历史漏洞生命周期。OSS-Fuzz 会持续 fuzz 开源项目、生成 PoC、报告漏洞，并监控修复状态。CyberGym 通过 patch commit 的二分定位找到漏洞从存在到修复的边界，再构造 pre/post-patch 环境。论文用 GPT-4.1 重写 commit message 为漏洞描述，并通过自动过滤和人工验证清理信息不足、重复或不可复现的样本。最终 1,507 个实例覆盖 28 类 sanitizer crash types，codebase 中位数为 1,117 个文件和 387,491 行代码。

## 3. Methods

CyberGym 的方法贡献主要是 benchmark engineering，而不是提出新的 agent 算法。它把真实漏洞变成可复现、可容器化、可执行评测的任务。这个工程链条包含几个关键环节：从 OSS-Fuzz 找历史漏洞，定位 patch commit，构建 pre-patch 与 post-patch executable，用 sanitizer 作为 crash oracle，验证 ground-truth PoC 的可复现性，再把描述、代码和评测脚本包装成 agent workspace。

这里最值得注意的是 **execution-based evaluation/基于执行的评测**。在安全任务里，文本答案的正确性很难可靠判定；PoC 是否真正触发漏洞，必须在程序上运行。CyberGym 因此更接近 SWE-bench 一类可执行软件工程评测，但它的输出不是 patch，而是攻击性测试输入。这个差异改变了任务难点：patch 任务通常局部修代码，PoC 任务要求从程序入口一路走到 vulnerable condition。

论文还通过 Level 0 把 benchmark 扩展到开放式发现。Level 0 没有漏洞描述，agent 只能探索 codebase 并尝试生成 crash-inducing inputs。这个设置很重要，因为它把“复现已知漏洞”与“发现未知漏洞”连接起来。作者后续发现 zero-day 的实验，正是沿着这个方向展开：让 agent 在 latest projects 上开放式生成 PoCs，然后人工分析这些 crashes 是否代表真实未公开漏洞。

另一个设计点是成本与可复现性。完整评测需要大量 API credits 和 GPU 时间，论文报告 non-thinking 全量评测约需 3,000 USD，而整篇实验投入超过 40,000 USD API credits 和 1,000 H100 GPU hours。为降低门槛，作者提供 300-instance 子集。这个细节对 benchmark 采用很重要，因为安全评测如果只能由少数大实验室跑，就很难成为社区标准。

从 agent 视角看，CyberGym 的 workspace 不是一个普通代码题。agent 需要先在大型仓库中定位可疑入口，再把漏洞描述映射到实际函数和输入解析路径，接着构造一个能穿过 parser、校验、格式约束和执行分支的输入。很多漏洞触发条件并不在单个函数里，而是在调用链和数据结构状态中逐渐形成。因此，CyberGym 实际同时测量 **repository navigation/仓库导航**、**semantic localization/语义定位**、**input synthesis/输入合成** 和 **execution-feedback debugging/执行反馈调试**。这也解释了为什么 SWE-bench 强模型在这里不一定强：修补 bug 更像定位并改代码，复现漏洞则要主动构造使程序走到危险状态的输入。

还有一个容易忽略的点是 post-patch executable 的作用。没有 post-patch 检查时，agent 只要制造任何 sanitizer crash 就可能得分，这会把 benchmark 变成“找任意崩溃”而不是“复现目标漏洞”。加入 post-patch 后，系统要求 PoC 与 patch 修复的 root cause 对齐。这个条件虽然不能完全排除所有 coincidental crash，但显著提高了标签质量。对安全 benchmark 来说，这种严格性比样本数本身更重要，因为错误 oracle 会直接训练和奖励错误能力。

## 4. Experiments

主结果显示 CyberGym 对当前 agent 很难。用 OpenHands scaffold 比较 11 个 LLM 时，Claude-Sonnet-4 达到 17.9%，Claude-3.7-Sonnet 为 11.9%，GPT-4.1 为 9.4%，GPT-5 minimal 为 7.8%。专门在 SWE-bench 方向优化的 SWE-Gym-32B、R2E-Gym-32B、OpenHands-LM-32B 反而都不超过 2.0%，说明 software engineering tuning 不会自动转化为漏洞复现能力。这个结论很有价值：安全任务需要的能力不是普通 issue fixing 的简单子集。

Thinking mode 的影响也有区分度。在 300-instance 子集上，GPT-5 从 minimal reasoning 的 7.7% 提升到 high reasoning 的 22.0%，超过 Claude-Sonnet-4；Claude 和 Qwen 的增益则更小。这说明长程推理预算对某些模型非常关键，但它不是普适 magic switch。Agent scaffold 方面，OpenHands、Codex、Cybench、EnIGMA 在 GPT-4.1 backbone 下个体 success rate 接近，但 union success rate 达到 18.4%，几乎翻倍，说明不同 scaffold 成功的实例重叠很低。

难度阶梯支持 benchmark 设计。OpenHands + GPT-4.1 在 Level 0 为 3.5%，Level 1 为 9.4%，Level 2 为 13.1%，Level 3 为 17.1%。更多信息确实提高成功率，但即使给 stack trace 和 patch diff，成功率仍然不高。PoC 长度分析进一步显示，ground-truth PoC 超过 100 bytes 后，agent 成功率明显下降，而这些较长 PoC 占 benchmark 的 65.7%。这说明瓶颈不只是定位漏洞，还包括理解复杂输入格式和构造有效 binary/text inputs。

论文对 data contamination 做了比较谨慎的分析。作者按模型 knowledge cutoff 切分漏洞披露日期，对 Claude-3.7-Sonnet、GPT-4.1、GPT-5 minimal、o4-mini 做 pre/post-cutoff success rate 比较，并用 Fisher exact test 和 two-proportion z-test。报告中 post-cutoff 与 pre-cutoff 没有显著差异，例如 GPT-4.1 pre-cutoff 为 9.7%（133/1365），post-cutoff 为 5.6%（8/142）。这不能证明完全无污染，但至少说明模型不是靠简单记忆公开漏洞报告拿高分。

最有现实冲击的是 Section 5。agent 在复现任务中生成了 759 个会在 post-patch version crash 的 PoCs，经过 latest version 验证、人工 root-cause analysis 和去重后，得到 9 个 unique zero-days；另有 18 个 incomplete patches，其中 1 个影响 latest version。开放式 Level 0 发现中，作者在 431 个 OSS-Fuzz 支持项目、1,748 个 executables 上运行 OpenHands + GPT-4.1/GPT-5，最终又确认 25 个 unique zero-days。合计 34 个 zero-days，其中已有 4 个 CVE assignments 和 10 个 patched cases。

实验的限制也很清楚。CyberGym 依赖 sanitizer，因此主要覆盖 C/C++ memory safety；它不能直接评估 web logic bugs、crypto protocol flaws、cloud misconfiguration、social engineering 或 privilege escalation chain。PoC 成功也不等于现实可利用性，sanitizer crash 只是安全相关失败的一个强信号。换句话说，CyberGym 是非常强的 **memory-safety agent benchmark**，但把它称为完整 cybersecurity benchmark 会略微过大。

这里还要区分 **capability signal** 和 **deployment risk**。CyberGym 高分表示 agent 能在受控环境里复现或发现 sanitizer-detectable bugs，但部署中的风险还受权限、网络访问、目标选择、自动化规模和人类审核影响。一个 22% success rate 看似不高，但如果攻击者可以并行跑大量项目，低成功率仍可能转化为大量有效漏洞；对防守者来说，同样的能力可以用于持续安全审计。这篇论文最有价值的地方正是让我们看到同一能力的双重用途，而不是简单把 agent 安全能力描述成好或坏。

论文对 agent failure 的分析也值得继续挖。作者报告了提前终止、输出超长 plaintext PoC、反复 `grep`/`ls`/`find` 却找不到关键文件、过早宣称成功等模式。这些不是边角料，而是 agent 安全能力的真实瓶颈。漏洞复现不是单轮推理，而是长程搜索和调试过程；如果 agent 的上下文管理、工具调用策略、文件检索和执行反馈总结不够好，它会在大型仓库里耗尽预算。未来模型能力提升可能来自更强 LLM，也可能来自更专门的 code search、dynamic tracing、input mutation 和 fuzzing tool integration。

因此，读这篇时不要只盯着最高分模型。更值得看的是不同模型、不同 agent scaffold 和不同 difficulty level 的失败形态。CyberGym 真正提供的是一个可以追踪安全能力增长的坐标系：当未来 agent 成功率上升时，我们可以判断提升来自更会找代码、更会生成输入、更会用执行反馈，还是只是获得了更充分的漏洞线索。

## 5. Reviewer Discussion

OpenReview 最终决定是 **Accept (Oral)**。四位 reviewer 的原始分数是 8、8、6、6，其中两位 6 分 reviewer 在 rebuttal 后明确或实际提高到 8。AC 总结认为 reviewer 一致认可 benchmark 的规模、真实性、执行式评测和 zero-day 发现价值；主要担忧集中在 C/C++ memory-safety 范围、全量评测成本、data contamination 分析严谨性、人类 baseline 以及 agent failure mode 诊断。

评审过程中最关键的变化是作者补强了两个问题。第一，data contamination 的最初分析不够严谨，reviewer 要求给出 split counts、统计检验、PoC length distribution 和更多 backbones；作者补充 Fisher exact test、z-test 和 post-cutoff 样本过滤后，担忧基本被缓解。第二，LLM 过滤与重写 commit message 可能引入噪声，reviewer 要求人工审计；作者补充 300-sample stratified audit，并报告 Cohen's kappa 约 0.82，说明数据 pipeline 比初稿更可信。

我的客观评述是：CyberGym 的 oral 价值成立，而且很扎实。它没有提出新算法，但它把一个长期模糊的问题变成了可执行评测：AI agents 到底能不能在真实 codebase 里复现真实漏洞？这个问题本身就足够重要。真正需要警惕的是命名和传播时的范围扩张。CyberGym 现在最强的表述应该是 **large-scale real-world C/C++ memory-safety reproduction benchmark**，而不是覆盖全部 cybersecurity 的总评测。

reviewer 对 scope vs. branding 的批评非常有效。C/C++ memory safety 是网络安全中的核心部分，但不是全部安全。一个 agent 在 CyberGym 上强，不代表它会 web exploitation、cloud defense、malware analysis、cryptographic auditing 或 social-engineering risk assessment；反过来，一个 agent 在 CyberGym 弱，也不等于它没有其他安全能力。因此后续读这个 benchmark 时，要把它视为 agent security capability 的一个高质量切片，而不是总分。

我还会把 reviewer 对人类 baseline 的要求看得很重。论文引用人类复现漏洞平均需要数小时，但没有系统让专家在同一 CyberGym workspace 中跑一批任务。没有这个 baseline 时，我们知道 agent 绝对成功率低，却不容易判断它距离人类专家还有多远，也不知道哪些 difficulty levels 对人类和 agent 的相对难度不同。后续如果能把 human expert、传统 fuzzing pipeline、LLM agent、LLM+fuzzer hybrid 放在同一 oracle 下比较，CyberGym 的解释力会大幅提高。

## 6. Related Work & Future Work

CyberGym 和 Cybench、CVE-Bench、BountyBench、SEC-Bench、SWE-bench 形成清楚对照。它继承了 SWE-bench 的可执行软件环境思路，但任务目标从修复 issue 变成复现漏洞；它也继承了 CTF/cyber benchmark 的安全目标，但把环境推进到真实开源项目、真实历史漏洞和容器化 oracle。

未来最重要的方向是扩展 oracle。Sanitizer 适合 memory safety，但 logic bugs、auth bypass、web state corruption、cryptographic misuse 需要完全不同的 success definition。第二个方向是 impact-weighted metrics，因为 sanitizer crash 的安全意义并不完全相同；有些 crash 只是 denial-of-service，有些可能通向 remote code execution。第三个方向是更细的 agent failure taxonomy，把失败拆成环境构建、代码定位、输入格式理解、exploit synthesis、oracle mismatch 和 premature stop。这样 CyberGym 才能从 leaderboard 进一步变成 agent 安全能力诊断工具。

还有一条自然路线是和自动化安全工具结合。现在的 agent 主要靠 shell、文本搜索、编译执行和自己构造输入；传统安全领域已有 fuzzers、symbolic execution、taint analysis、coverage-guided mutation、sanitizer instrumentation 等成熟工具。未来强系统很可能不是纯 LLM agent，而是 LLM 负责理解报告、规划探索、解释 sanitizer 输出，底层工具负责 coverage expansion 和 input mutation。CyberGym 正适合测试这种 hybrid system，因为它的成功 oracle 已经是执行式的。
