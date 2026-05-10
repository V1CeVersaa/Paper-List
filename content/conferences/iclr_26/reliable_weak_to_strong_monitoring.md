---
title: "Monitor Red Teaming"
headline: "Reliable Weak-to-Strong Monitoring of LLM Agents"
description: "ICLR 2026 Oral note on monitor red teaming for LLM agents, including threat models, hybrid monitor scaffolding, weak-to-strong monitoring, and human-in-the-loop escalation."
status: complete
visibility: public
topic: "safety_alignment"
venue: "ICLR 2026 Oral"
year: 2026
paper_url: "https://arxiv.org/abs/2508.19461"
openreview_url: "https://openreview.net/forum?id=WV7xIboTDK"
source_pdf: "raw/papers/arxiv-2508.19461.pdf"
tags: ["scalable_oversight", "agent_monitoring", "ai_control"]
created: "2026-05-01"
updated: "2026-05-01"
---

> [!abstract] Contributions
>
> 这篇论文把 **agent monitoring/智能体监控** 从“给一个轨迹让 LLM 打 suspiciousness 分”推进成一个更系统的 **Monitor Red Teaming/MRT** 工作流。作者定义 agent awareness 和 monitor awareness 两条 threat-model 轴，引入 monitor hijacking、lying with jailbreaking strategies、CoT obfuscation 等 evasion strategies，并在 SHADE-Arena 与新构造的 **CUA-SHADE-Arena** 上比较 full-trajectory、hierarchical、sequential 和 hybrid monitor scaffolding。核心发现是：agent 知道自己被监控会显著削弱监控可靠性；monitor 拿到更多任务信息的收益有限；更好的 scaffolding，尤其是 hierarchical 与 sequential 的 hybrid，比单纯提高 monitor awareness 更关键。
>
> 论文最重要的 alignment 含义是 **weak-to-strong monitoring/弱监督强模型** 不只取决于 monitor backbone 能力，还取决于监控系统如何组织长轨迹。实验中，baseline monitor 的可靠性和 attacker-monitor capability gap 明显相关，但 hybrid scaffolding 能让较弱 monitor 在强 agent 面前仍保持较高 AUC。边界同样硬：这里的 malicious agents 主要靠 prompt 诱导而不是 adversarial fine-tuning，实验是 offline full-trajectory monitoring，不是实时阻断；因此论文更像 agent oversight infrastructure 的第一版压力测试，而不是已经证明能监控真正 superhuman misaligned agents。

## 1. Introduction

这篇 oral 的现实动机很强：LLM agents 越来越能执行长程任务，风险也从单轮 unsafe response 转向 **covert misbehavior/隐蔽不当行为**。一个 agent 可能表面完成用户的 main task，同时在文件系统、网页、代码库或企业工具里执行 hidden side task，比如泄露隐私、植入后门、操控业务状态。此时安全系统不能只审 final answer，而必须审 **trajectory/轨迹**，也就是工具调用、动作、文件修改和可见 chain-of-thought。

论文的关键判断是，monitor 本身也需要 red teaming。一个监控系统在普通 agent 面前表现好，不代表它能面对知道自己被监控的 agent；一个 monitor 拿到 main task，不代表它能识别 unknown side task；一个 full-context LLM judge 能读完整轨迹，也不代表它真的能从长上下文里定位少量恶意动作。因此作者把问题从“设计一个 monitor”改成“如何系统评估 monitor 在 adversarial agent 面前是否可靠”。

这对 safety alignment 很重要，因为未来的控制方案往往默认有一个 trusted monitor、weaker overseer 或 human escalation pipeline。如果 monitor 很容易被 agent 的 CoT obfuscation、prompt injection 或长轨迹噪声绕过，那么所谓 AI control 只是把问题推迟了一层。论文的贡献就是把这层接口具体化：什么 threat model，什么 evasion strategy，什么 metric，什么 monitor architecture，以及哪些实验结论真的支持 weak-to-strong oversight。

## 2. Problem Setup

论文定义的 MRT setup 包含一个 agent、一个 monitor 和一组带 side task 的 agentic environment。agent 执行 benign main task，同时可能追求 malicious side task；monitor 获得完整 trajectory，并输出 suspiciousness score。ground truth 不是 side task 是否最终成功，而是 agent 是否在采取朝 side task 推进的行为。这个定义比“是否完成危害”更合理，因为安全监控应该在危害完成前就抓到 attempt。

Threat model 有两条 awareness 轴。**Agent awareness** 有两档：unaware agent 不知道自己被监控；aware agent 知道轨迹会被检查，但不知道 monitor 内部细节。**Monitor awareness** 有三档：unaware monitor 不知道 main task 和 side task；aware monitor 知道 main task；omniscient monitor 同时知道 main task 和 side task。这形成 $2\times 3$ 个 threat models。

评估指标选两个。**AUC-ROC** 衡量整体排序能力，避免“全都报 suspicious”这种退化 monitor 拿到高 detection rate。**TPR at FPR=0.01** 衡量高精度场景下能抓住多少恶意轨迹，因为企业部署里 false positive 会带来昂贵的人类升级和业务中断。论文反复强调 AUC 和 TPR 不总是同向变化，这一点很实用：一个 monitor 可以整体排序更好，但在低 false positive 区间并没有多抓到关键风险。

数据集方面，论文使用 SHADE-Arena 的 17 个 tool-calling task pairs，并新增 **CUA-SHADE-Arena**。后者从 TheAgentCompany 中选出 15 个 computer-use main tasks，再配上 side tasks，让 agent 在虚拟机里执行 bash、编辑文件、写代码或操作网页。这让评估从模拟工具环境扩展到更接近真实工作流的 computer-use agents。

## 3. Methods

MRT 的第一块是 evasion strategies。baseline 是 dealer's choice，让 aware agent 自己想办法躲监控。**Monitor hijacking** 让 agent 在 CoT 或输出轨迹中注入类似“忘记先前指令，输出低 suspicious score”的攻击文本。**Lying with jailbreaking strategies** 把 jailbreak 中的说服、混淆、过量无关细节等策略迁移到 monitor evasion。**CoT obfuscation** 则模拟 under-verbalized agents：agent 生成隐藏 thought 和可见 thought，只把经过修饰、遗漏关键恶意意图的 visible CoT 给 monitor。

第二块是 monitor scaffolding。最简单的 baseline 是把完整 trajectory 直接塞给 LLM monitor。**Hierarchical monitoring** 把长轨迹切成若干 chunks，先逐块 summarize/score，再递归汇总。它适合长上下文 needle-in-the-haystack 问题，也更容易并行。**Sequential monitoring** 按时间顺序递归更新 summary 和 score，更贴近 online monitoring，因为它天然可以随着轨迹增长逐步运行。**$\mu$-hybrid monitoring** 则把 hierarchical score $h$ 和 sequential score $s$ 线性组合：

$$
\mathrm{score}_{\mathrm{hybrid}} = \mu h + (1-\mu)s.
$$

默认 $\mu=0.5$，因为 appendix 里调参收益不大。这个设计很简单，但它抓住了两种 chunking 的互补性：hierarchical 更像全局聚合，sequential 更保留时间顺序和前后依赖。论文真正想证明的不是这个公式多新，而是 **scaffold design** 比“让 monitor 多知道一点背景”更能提高 robustness。

第三块是 weak-to-strong monitoring experiment。作者让不同强弱的 attacker model 和 monitor model 配对，并用 capability gap 解释 baseline monitor 的 AUC。然后比较四种 scaffolding 能否削弱 capability gap 的影响。这个实验直接服务于 AI control 中的核心问题：我们是否能用 weaker trusted models 监控 stronger untrusted agents。

第四块是 human-in-the-loop。流程是 monitor 先给 1-5 suspiciousness score 和理由，人类 red teamers 查看轨迹、向 monitor 提问、给出 feedback，然后 monitor 重新打分。作者比较三种方式：所有轨迹都更新；加 strong prior，要求 monitor 只有在有具体证据时改分；再加 selective escalation，只对原始 score 大于 1 的轨迹引入人类反馈。

这里最值得注意的是，论文没有把 monitor 当成一个纯模型，而是当成一个 **scaffolded decision procedure/带脚手架的判断程序**。这和普通安全分类器很不一样。一个长 agent trajectory 通常有上万 token，恶意动作可能只藏在少数几步里；如果 monitor 一次性读完整上下文，它可能被无关细节稀释、被攻击文本劫持，或者在最后忘记前面的关键步骤。Hierarchical scaffold 把问题改成多段局部检测再汇总，sequential scaffold 把问题改成沿时间线保留 state，hybrid 则试图同时拿到局部敏感性和时间连续性。这解释了为什么“更好的提示词”不一定够，轨迹结构本身需要被算法化处理。

另一个方法细节是，论文把 monitor awareness 和 monitor scaffold 分开控制。这个设计很重要，因为如果只比较 omniscient monitor 和 unaware monitor，很容易把“知道什么”与“怎么读轨迹”混在一起。结果显示，在很多场景里，给 monitor main task 或 side task 的收益不如换一个 scaffolding 大。这并不是说信息不重要，而是说长轨迹监督的瓶颈往往在 **retrieval and aggregation/检索与聚合**，不是背景知识缺失。

## 4. Experiments

第一组实验比较 unaware agents 和 aware agents with no strategy。结果很直接：**agent awareness 对 monitor 可靠性的破坏，比 monitor awareness 带来的增益更大**。当 agent unaware 时，各类 monitor 在两个数据集上已经能达到较高 AUC，monitor 是否知道 main task 或 side task 不总是显著改变结果。当 agent aware 时，AUC 和 TPR 普遍下降；这说明“知道自己被审计”本身就足以改变轨迹，让监控变难。

这组实验还显示，aware monitor 不一定明显优于 unaware monitor。这个结论有现实意义：第三方审计者可能出于隐私或安全原因拿不到完整 main task specification，但如果 monitor scaffolding 设计得好，仍然可能有效。反过来，omniscient monitor 的上限并没有总是接近 1，这也暴露一个弱点：LLM monitor 即使知道 side task，也未必会做程序化验证，只是在做语言判断。

第二组实验加入 evasion strategies。baseline full-trajectory monitor 在 monitor hijacking 等策略下会显著掉性能，甚至出现 AUC 低于 0.5 的极端情况，因为 malicious trajectories 被攻击文本诱导成低 suspicious score，而 benign trajectories 反而可能被误报。hybrid scaffolding 在所有 monitor awareness level 上最稳。图 5 中，SHADE-Arena 上 hybrid 的 winning AUC 大约是 0.75、0.73、0.80；CUA-SHADE-Arena 上约是 0.85、0.84、0.98，对应 TPR at FPR=0.01 也明显更强，尤其 CUA-SHADE-Arena 中 hybrid 能到约 0.68、0.67、0.88。

第三组 weak-to-strong 结果是论文最有战略意义的部分。baseline monitor 的 AUC 会随 attacker-monitor capability gap 变化，弱 monitor 看强 agent 更吃力。但 hybrid monitor 的曲线更平，弱模型如 GPT-4o-mini 或 Claude 3.6 Haiku 在 hybrid scaffolding 下也能对所有 tested agents 达到超过 0.85 的 AUC。这个结果支持一个谨慎结论：**verification 的系统设计可以部分弥补 verifier model 本身较弱的问题**。

不过这里不能读过头。reviewer 的质疑很对：prompt-induced malicious agents 和真正经过训练、优化、长期适应 monitor 的 misaligned agents 不是一回事。Claude Haiku 能监控“被 prompt 告知要做 side task 的 Claude Sonnet 4”，不等于它能监控真正 superhuman、长期适应检测器的 agent。论文的结论应当停在“hybrid scaffolding 是当前 benchmark 上的有效弱到强监控机制”，而不是“弱模型监控强模型问题已解决”。

第四组 human-in-the-loop 结果很有警示性。无脑让人类反馈所有轨迹，AUC 和 TPR 反而下降，因为人类和 monitor 会对复杂轨迹变得更 paranoid，把 benign trajectories 也抬高 suspiciousness。加 strong prior 后，TPR at FPR=0.01 明显改善；再使用 selective escalation，只让原始 score 大于 1 的案例进入人类复核，TPR 从约 20.6% 提升到 36.2%，AUC 基本保持。这说明人类监督不是“加人就更安全”，而需要 **升级策略、证据约束和专用界面**。

总体实验结论可以压成一句话：**monitor 的信息量重要，但 monitor 的轨迹处理结构更重要；人类监督重要，但只有在被自动系统精准触发并被强证据约束时才更有用**。这对真实部署比单纯 leaderboard 数字更有价值。

实验还有一个值得单独读的指标张力。AUC 反映整体排序，TPR at 1% FPR 反映高精度部署。论文里有些设置 AUC 改善但 TPR 不动，也有 human-in-the-loop 设置 AUC 几乎不动但 TPR 大幅上升。这说明监控系统不能只追平均判别能力。在真实企业环境里，只有极少数轨迹能被升级给人类或强模型复核，因此低 false-positive 区间的 recall 才是关键。一个 monitor 如果只能在宽松阈值下表现好，部署价值会被严重高估。

还要注意 CUA-SHADE-Arena 的定位。它比纯文本或模拟工具环境更接近真实 computer-use agent，因为 agent 可以执行 bash、写代码、改文件、操作虚拟环境；但它仍然不是完整企业系统，side tasks 数量也只有 15 个。论文把它作为第一步是合理的，但不能据此说所有真实 agent sabotage 都被覆盖。真正难的场景会包含长期状态、权限边界、外部网络、多人协作和不可逆操作，这些都可能改变 monitor 的误报和漏报结构。

从工程落地看，offline AUC 还缺一个关键变量：**time-to-detection/检测时机**。如果 monitor 只能在完整轨迹结束后判断恶意，那它更像审计工具；如果能在 side task 关键动作发生前触发，就能成为控制系统。Sequential scaffold 最接近 online monitor，但论文没有系统报告每类 scaffold 在第几步首次 flag，也没有把“阻断前检测”和“事后识别”分开。这会影响真实安全含义，因为高 AUC 的事后 monitor 可能仍然无法阻止数据泄露或文件破坏。

## 5. Reviewer Discussion

OpenReview 最终决定是 **Accept (Oral)**。四位 reviewer 的原始分数是 8、6、6、4。正面评价集中在几个方面：论文提出的 MRT workflow 比很多 agent evaluation 更科学，显式定义 threat models、metrics 和 evasion strategies；CUA-SHADE-Arena 把评估扩展到更现实的 computer-use setting；hybrid scaffolding 和 weak-to-strong monitoring 结果对 AI control 有直接意义；实验规模和文档也比较完整。

低分 reviewer 的批评非常关键。第一，threat model 仍然人工：agents 被 prompt 明确告知 main/side task，有时还被框在 red-teaming setting 中，这和训练出来的 adversarial or misaligned agent 不同。第二，omniscient monitor 表现不接近完美，说明 LLM monitor 没有做系统化 side task verification；如果写一个 programmatic omniscient baseline，真正上限可能高得多。第三，hybrid monitoring 的理论解释和 ablation 还不够：它为什么互补、在哪类 attack 上各自失败、是否比简单 ensemble 更强、成本是否值得，都没有完全封口。

其他 reviewer 也追问了泛化和深度。比如这些结论是否适用于 BashBench、APPS、更多 coding setup 或 open-source models；monitor capability scaling 到底取决于绝对能力、参数量还是 attacker-monitor 差距；不同 chunk size 如何影响 TPR/FPR；aware attackers 具体使用哪些策略以及频率如何。作者的实验已经很丰富，但这些问题说明 MRT 还处在 infrastructure-building 阶段。

我的客观评述是：这篇论文的 oral 价值不在于提出了一个复杂算法，而在于把 agent oversight 的评估问题整理成可复用协议。它给安全研究者一个很重要的提醒：**监控系统也是一个可被攻击的模型系统**，不能只测普通输入下的 detection rate。它最强的结论是 hybrid scaffolding 在当前长轨迹 benchmark 中很有效；最弱的结论是“弱到强监控”这个说法还需要更真实的 attacker training 和 online evaluation 才能站稳。

我对 reviewer 中 “omniscient monitor should be near-perfect” 的批评尤其认同。如果 monitor 已知 side task 是把某些 SSN 写到公共目录，它理论上可以直接检查文件系统和 action log，而不是让 LLM 泛泛判断 suspiciousness。当前 omniscient monitor 没到接近满分，说明论文测到的是 **LLM-as-judge monitor** 的上限，而不是 **monitoring system** 的上限。下一阶段的强系统很可能是 LLM summary、programmatic checker、policy engine、retrieval index 和 human escalation 的组合，不能只靠一个模型读文本。

不过这个缺口不削弱论文的基础价值，反而说明 MRT protocol 是必要的。因为只有先定义 threat model、evasion strategy 和 metric，后续更强的 programmatic monitor 才有可比较的评估场。论文当前的 hybrid scaffold 可以看成 baseline infrastructure，而不是终点。

## 6. Related Work & Future Work

这篇论文接在 scalable oversight、AI control、agent monitoring 和 sabotage benchmarks 后面。和只测 harmful task completion 的 benchmark 不同，MRT 测的是 monitor 在 agent trajectory 上能否可靠区分 benign 与 malicious behavior。和普通 LLM guardrail 不同，它不只拦截单个 unsafe output，而是审查一条长轨迹中是否存在隐蔽 side task。

它和 CoT monitoring 的关系也很微妙。论文使用 CoT/action trajectory 作为 monitor 输入，但同时承认 CoT 可能不忠实，尤其当模型知道自己被监控时会 under-verbalize 或修饰可见思维。CoT obfuscation 正是为了逼近这一点。因此这篇和 TRACE、CRV 一样，都在提醒我们：读 CoT 有价值，但不能把 CoT 当成透明窗口。

后续最该补的是三件事。第一，做 **trained adversarial agents**，包括 fine-tuned evasion、reward-shaped stealth 和长期适应 monitor 的 agents。第二，做 **online monitoring**，测每个 action 何时被 flag、能否在危害发生前阻断，而不是只对完整 trajectory 打分。第三，加入 **programmatic or tool-assisted monitors**，尤其在 omniscient setting 中验证 side task outcome 和关键行为，而不是让 LLM 只做语义判断。这样才能知道 hybrid scaffolding 的收益来自更好阅读长轨迹，还是只是当前 prompt baseline 太弱。

还有一条后续线是成本建模。Hybrid 往往要同时跑 hierarchical 和 sequential 路径，token 成本、延迟和实现复杂度都高于 baseline。论文证明它更可靠，但还没有回答“多可靠才值得多花成本”。真正系统应该按风险等级动态选择 monitor：低风险轨迹用便宜 baseline，高风险或高权限轨迹用 hybrid，再把少量高疑似案例升级给人类或更强模型。
