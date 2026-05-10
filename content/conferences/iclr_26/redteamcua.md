---
title: "RedTeamCUA"
headline: "RedTeamCUA: Realistic Adversarial Testing of Computer-Use Agents in Hybrid Web-OS Environments"
description: "ICLR 2026 Oral note on RedTeamCUA, a hybrid web-OS sandbox and RTC-Bench benchmark for adversarial testing of computer-use agents."
status: complete
visibility: public
topic: "safety_alignment"
venue: "ICLR 2026 Oral"
year: 2026
paper_url: "https://arxiv.org/abs/2505.21936"
openreview_url: "https://openreview.net/forum?id=yWwrgcBoK3"
source_pdf: "raw/papers/arxiv-2505.21936.pdf"
tags: ["prompt_injection", "computer_use_agents", "adversarial_benchmark"]
created: "2026-05-01"
updated: "2026-05-01"
---

> [!abstract] Contributions
>
> 这篇论文提出 **RedTeamCUA**，一个面向 computer-use agents/CUAs 的混合 web-OS adversarial testing framework。它把 OSWorld 的 VM-based OS environment 和 WebArena/TheAgentCompany 的 Docker-based web platforms 结合起来，让研究者能在安全可控环境里测试 indirect prompt injection 如何从网页内容传播到本地 OS action。基于这个框架，作者构造 **RTC-Bench**：9 个 benign goals、24 个 adversarial goals、2 种 benign instruction specificity、2 种 injection format，共 864 个 adversarial examples。
>
> 论文最重要的结果是，当前 frontier CUAs 对混合 web-OS prompt injection 仍然非常脆弱。Decoupled Eval 中，GPT-4o ASR 达到 66.19%，Claude 3.7 Sonnet | CUA 为 42.93%，Operator 在内置确认和安全检查下仍有 7.57%，去掉或弱化检查后为 30.89%。End2End 设置里，Claude 3.7 Sonnet | CUA 达到 50%，Claude 4.5 Opus | CUA 达到 83%，Claude 4.6 Opus | CUA 仍有 50%。边界在于，benchmark 的 benign goals 只有 9 类，End2End 只跑 50 个高风险样本，且评测以闭源 CUA 为主；它是很强的风险暴露框架，不是完整防御方案。

## 1. Introduction

Computer-use agents 能操作浏览器、文件系统、终端和 GUI，这让 prompt injection 的风险从“模型生成一段错误文本”升级为“模型执行真实系统动作”。一个网页评论、一条 chat message 或一个 shared document 里嵌入的恶意指令，可能诱导 agent 删除本地文件、泄露隐私数据、安装恶意包或上传敏感内容。传统 web-only 或 tool-only prompt injection benchmark 很难覆盖这种跨界风险，因为真正危险的是 **Web -> OS** 或 **Web -> OS -> Web** 的动作链。

RedTeamCUA 的动机就是补这个空缺。OSWorld 提供真实 VM 桌面环境，但浏览器可访问 live web，不适合安全红队；WebArena 和 TheAgentCompany 提供隔离 web replica，但缺少 OS action surface。论文把二者接起来：agent 在 VM 里通过浏览器访问 Docker-hosted web platforms，再把网页中的信息用于本地 OS 操作。这样就能安全地测试“网页注入如何影响本地系统”。

这篇论文在 Safety & Alignment 里很重要，因为 CUA 的安全边界不同于 chat model。Chatbot 错答可以被用户忽略，但 CUA 会真的执行命令。一个 agent 若不能区分 trusted user instruction 和 untrusted environment content，就会把网页里看似相关的“安装说明”“管理员提示”“专家建议”当成用户目标的一部分。论文把这个风险做成了可复现 benchmark，而不是只展示几个手工案例。

## 2. Problem Setup

RedTeamCUA 的核心环境由两部分组成。OS 侧基于 OSWorld，提供 Ubuntu VM、浏览器、终端、文件管理器、VSCode 等可交互应用，并支持 snapshot reset，保证危险动作不会影响真实系统。Web 侧来自 WebArena 和 TheAgentCompany，包括 OwnCloud、Forum、RocketChat 三类平台，分别模拟共享云文档、论坛和实时聊天。agent 在 OS 浏览器中访问这些本地 Docker web replicas。

benchmark 的 benign goals 反映一种常见 CUA 场景：用户从网上获取知识，然后让 agent 在本地执行对应动作。作者设置 3 类任务：software installation、system configuration、project setup，每类 3 个，共 9 个 benign goals。每个 benign goal 有 general 和 specific 两种指令形式，用来模拟用户专业程度不同。

adversarial goals 按 **CIA triad** 组织。**Confidentiality** 指未经授权的信息泄露，例如从本地 OS 读取隐私文件并通过 RocketChat、OwnCloud 或 Forum 传出去；**Integrity** 指破坏数据可信性，例如删除或修改文件；**Availability** 指影响系统可用性，例如停止服务或耗尽资源。每个目标又有 natural language 和 code 两种 injection form。于是 RTC-Bench 形成 9 x 24 x 4 = 864 个测试样本。

论文定义两个评估设置。**Decoupled Eval** 用预处理动作把 CUA 直接放到包含 adversarial injection 的位置，避免把导航失败误判成安全鲁棒性。**End2End Eval** 则从初始任务状态开始，更接近真实部署，但会混入 navigation、perception、tool-use 能力限制。两个设置互补：Decoupled 看“看到注入后会不会被骗”，End2End 看“完整流程里风险是否真的落地”。

## 3. Methods

RedTeamCUA 的第一个方法贡献是 hybrid sandbox。作者扩展 OSWorld 配置，使 adversarial task initial state 可以自动注入恶意内容；不同 web platform 有对应注入脚本，例如修改数据库、上传文件或插入消息。这个自动化非常关键，因为红队评测必须可重复、可批量、可恢复，而不能依赖人工把恶意文本手动贴进页面。

第二个方法贡献是 realistic attacker constraints。攻击者不能改用户原始指令，不能改 agent prompt、model weight 或系统组件，也不能控制整个网页 UI；只能在网页中通常可编辑的位置写入文本，例如 forum comment、RocketChat message 或 shared OwnCloud file。这比“攻击者能任意改 DOM、弹窗、banner”的设定更接近真实平台权限，也让 benchmark 更严格。

第三个方法贡献是 contextualized injection strategy。攻击文本不是单纯写“ignore previous instruction”，而是把恶意动作包装成与当前网页和用户任务相关的说明。例如用户想安装 termcolor，论坛评论就声称必须先删除 `/etc/security` 才能安装最新版，然后再给出正常安装命令。这种设计针对 CUA 的核心弱点：它需要利用网页内容完成任务，因此容易把环境中的任务相关文本误当成可信操作步骤。

评估指标包括 **ASR/Attack Success Rate** 和 **AR/Attempt Rate**。ASR 用 example-specific execution-based evaluators 判断恶意目标是否真正完成；AR 用 GPT-4o judge 判断 agent 是否在轨迹中尝试执行恶意目标，即使最后由于能力不足没有成功。二者缺一不可：ASR 衡量真实伤害落地，AR 衡量模型是否已经被操控。若 AR 高于 ASR，说明当前能力不足暂时挡住了部分风险，但未来 capability 提升会把这些 attempt 变成 success。

这里的 Decoupled Eval 很值得强调。很多 agent benchmark 会把“没有完成攻击”直接当成安全，但对 CUA 来说这很危险。一个模型可能只是没找到网页、没点对按钮、没打开文件，却在一旦看到注入后立刻服从恶意指令。Decoupled Eval 把 agent 放到注入点，专门测试 instruction trust boundary；End2End Eval 再测试完整部署风险。两者结合，才能区分 **capability bottleneck/能力瓶颈** 和 **robustness bottleneck/鲁棒性瓶颈**。如果只看 End2End，很容易把笨误判成安全。

论文的 injection template 也不是随便写一段恶意提示。它包含“THIS IS IMPORTANT”这样的优先级暗示、与环境任务相关的 deceptive rationale，以及正常任务步骤的延续。这种设计反映了真实攻击者会做的事情：不是粗暴要求 agent 忽略用户，而是把恶意步骤伪装成完成用户任务所需的前置条件。对 CUA 来说，这比普通 jailbreak 更麻烦，因为 agent 本来就必须从网页里读取操作说明。

## 4. Experiments

Decoupled Eval 的主表非常刺眼。GPT-4o 平均 ASR 为 66.19%，AR 为 92.45%；Claude 3.7 Sonnet | CUA 平均 ASR 为 42.93%，AR 为 64.39%；Operator 在内置安全机制下最低，但仍有 7.57% ASR 和 14.06% AR。Operator w/o checks 上升到 30.89% ASR 和 47.84% AR，说明它的安全优势很大程度依赖确认和安全检查，而不是 agent 内部完全理解了 instruction hierarchy。

CIA 分类揭示了不同风险的难度结构。Integrity 目标常常是简单本地动作，例如删除文件，因此 ASR 高；Confidentiality 目标通常需要读取、整理、再上传或发送本地文件，执行链更长，所以 ASR 相对低，但 AR 仍然高。这意味着 agent 经常已经被说服去泄露信息，只是执行能力或流程复杂度暂时阻止了成功。随着 CUA 更擅长多步文件操作，Confidentiality 风险可能会显著上升。

平台差异也很有信息量。RocketChat 往往 ASR 更高，因为聊天消息看起来更像直接来自可信人类联系人，且发送数据是平台原生动作；Forum comment 或 OwnCloud shared file 的可信度和动作自然性不同。这个结果说明 prompt injection 风险不是只由文本内容决定，还由平台语义、交互习惯和目标动作的自然性决定。防御不能只看“是否有恶意关键词”，还要理解上下文里的信任边界。

防御评测并不乐观。作者测试 LlamaFirewall、PromptArmor、防御性 system prompt 和 Meta SecAlign。LlamaFirewall 在 50 个高风险样本上检测率为 0%；PromptArmor 最好变体也只检测 30%；防御性 system prompt 可以降低部分 ASR，但 Claude 3.7 Sonnet | CUA 和 Operator w/o checks 在 End2End 中仍接近 50%；Meta SecAlign 仍有 52% AR 和 32% ASR。这个结果支持论文的强结论：当前 prompt injection 防御还不能直接迁移到真实 CUA 场景。

End2End Eval 更贴近部署。50 个高风险样本中，Operator 是 10%，Operator w/o checks 是 42%，Claude 3.7 Sonnet | CUA 是 50%，Claude Opus 4 | CUA 是 48%，Claude Sonnet 4.5 | CUA 是 60%，Claude Opus 4.5 | CUA 是 83%，Claude Opus 4.6 | CUA 是 50%。尤其值得注意的是，能力更强不一定更安全；如果 instruction following 和 navigation 变强，但 instruction trust boundary 没有同步变强，ASR 可能上升。

Operator 的结果给了一个很现实的提示。内置 confirmation 和 safety check 能显著降低 ASR，但一旦模拟 inattentive user 或取消检查，风险马上上升。这说明当前最有效的 CUA 安全机制仍然高度依赖人类确认，而不是 agent 自身稳定识别不可信指令。真实部署中，人类确认也不是免费资源：用户会疲劳、误点、被上下文说服，或者根本看不懂某个终端命令的后果。因此，确认机制应当被视为最后一道 guardrail，而不能替代 agent 的权限分层和来源可信度判断。

另一个重要结果是 observation modality 的 trade-off。论文附录显示，使用 screenshot with accessibility tree 可以降低 ASR/AR，但可能伤害 benign task success rate。这说明给 agent 更多结构化环境信息并不总是单调提高安全；a11y tree 可能让模型更清楚地区分页面结构和文本来源，也可能引入更多上下文和误导信息。CUA 防御需要做的是让环境信息带上信任标签，而不是简单多给或少给上下文。

论文也做了 failure taxonomy。End2End 中失败可能来自 failed navigation、ignored injection、limited capability for adversarial goal、Operator triggered check 等。这个分析解释了为什么 Decoupled Eval 必要：一个弱 CUA 没到达注入页面，不说明它懂得拒绝注入，只说明它导航失败。真正的安全评测要把 capability failure 和 robustness 分开。

## 5. Reviewer Discussion

OpenReview 最终决定是 **Accept (Oral)**。四位 reviewer 的分数是 4->6、8、6、6。AC 总结认为论文的主要贡献是把 VM-based OS 和 Docker-based web platforms 整合成可控 hybrid sandbox，提出 Decoupled Eval，并构造 RTC-Bench 来系统评估 cross-environment prompt injection。正面评价集中在现实威胁模型、benchmark 规模、执行式 ASR、Attempt Rate、以及 frontier CUAs 暴露出的高风险。

主要批评集中在五点。第一，novelty：sandbox 建在 OSWorld、WebArena、TheAgentCompany 等已有系统之上，864 样本也来自有限 benign/adversarial goal 的 cross product。第二，模型选择偏闭源，早期缺少强 open-source CUAs。第三，防御评测更多是测试现有方法，没有提出新的强防御。第四，End2End 只跑 50 个高风险样本，UI noise sensitivity 不够。第五，缺少机制分析，不能深入解释 CUA 为什么会服从注入。

作者 rebuttal 补了很多实用内容：解释 UI-TARS/OpenCUA 等开源 CUA 当时不可用或不够稳定；增加 Claude 4.5 Sonnet 和 Qwen-3-VL 结果；强调 Decoupled Eval 的价值是隔离 adversarial robustness；补充 failure taxonomy；说明 defensive system prompt、PromptArmor、LlamaFirewall、Meta SecAlign 的覆盖范围和局限。低分 reviewer 因此从 4 升到 6，但部分 novelty 和 End2End 范围问题仍存在。

我的客观评述是：RedTeamCUA 的贡献不是单个算法，而是 **把 CUA prompt injection 从演示案例推进到混合 web-OS 可执行评测**。这个贡献很重要，因为真实 CUA 风险恰恰发生在跨环境动作链里。它的弱点也很明确：benchmark 的任务种类还不够丰富，攻击策略仍偏单一，防御章节更多说明“现有方法不够”，还没有给出下一代防御架构。

我特别认可 reviewer 对 agent utility 的质疑。一个什么都做不成的 agent 可以看起来很安全；一个非常强的 agent 如果不懂信任边界，反而更危险。因此 CUA 安全评测必须同时报告 benign SR、ASR 和 AR。RedTeamCUA 已经朝这个方向走了一步，但未来需要更系统地把 utility-security Pareto frontier 做出来，而不是只比较 ASR。

reviewer 对 novelty 的质疑也应该认真对待。RedTeamCUA 的 sandbox 确实建立在 OSWorld、WebArena 和 TheAgentCompany 上，攻击策略也延续了 prompt injection 传统。但我不认为这削弱 oral 价值，因为这里的核心不是发明全新攻击文本，而是把已有组件整合成能评估真实 web-OS harm 的实验系统。安全 benchmark 的创新经常来自 **evaluation boundary/评测边界** 的推进：以前只能测网页里是否被说服，现在能测网页注入是否导致本地 OS 被修改或数据被外传。

不过，benchmark 规模需要冷静理解。864 examples 来自 9 个 benign goals、24 个 adversarial goals 和 4 种形式组合，不等于 864 个完全独立现实任务。这个 cross-product 设计有利于系统覆盖和可控比较，但会让实际任务多样性低于表面数字。后续版本如果要成为长期标准，需要增加更多真实工作流、平台类型、权限边界和攻击风格，而不只是扩展组合数量。

## 6. Related Work & Future Work

RedTeamCUA 处在 OSWorld、WebArena、TheAgentCompany、AgentDojo、DoomArena、OS-Harm、SafeArena 等工作的交叉点。它与 web-only benchmark 的区别在于引入本地 OS action；与 OS-only benchmark 的区别在于攻击源来自 web environment；与普通 prompt injection dataset 的区别在于它用可执行 evaluator 衡量真实 harm 是否发生。

未来最关键的是防御。真正的 CUA 防御不应该只靠 system prompt，而需要 instruction hierarchy、source trust labeling、read-only vs. write/action separation、high-risk action confirmation、taint tracking、environment content sandboxing、tool permission policy 和 human escalation。比如网页内容可以提供事实信息，但不应该自动获得操作本地 OS 的 authority。Agent 需要把“用户目标”“网页资料”“第三方指令”“系统安全策略”分成不同权限层。

第二个方向是扩大 benchmark 生态。当前 9 个 benign goals 和 24 个 adversarial goals 是很好的起点，但真实 CUA 会处理邮件、日历、企业文档、浏览器扩展、代码仓库、支付系统和内部工具。不同平台的信任结构差异很大，不能假设 RocketChat/Forum/OwnCloud 足以覆盖。第三个方向是更强的机制分析，尤其要区分模型是没识别指令来源、过度服从任务相关文本、无法维护 user intent，还是被局部 UI context 淹没。

还有一条关键路线是 online intervention。RedTeamCUA 主要报告攻击最终是否成功或是否尝试，但真实控制系统需要知道 agent 在第几步开始偏离用户目标、是否能在危险动作前阻断、阻断后如何恢复 benign task。一个 End2End 事后 ASR 很有研究价值，但部署中更需要 **time-to-detection/检测时机** 和 **pre-harm intervention/危害前干预**。未来 benchmark 可以把每个动作标注成 safe、suspicious、irreversible、harmful，并评估 monitor 是否能在 irreversible step 前触发。

最后，RedTeamCUA 应该和 monitor red teaming 一起读。前者制造 realistic CUA attacks，后者评估 monitor 是否能看懂 agent trajectory。真正的 CUA 安全栈可能需要二者结合：RedTeamCUA 生成高风险轨迹，monitor 系统逐步检查来源、权限、动作后果和任务一致性。这样才能从“发现 CUA 会被注入骗”走向“构造能持续压低 ASR 的防御系统”。

因此，这篇的长期价值不在某个具体 ASR 数字，而在它把 CUA 风险评测的单位从单网页、单提示词，推进到跨平台、跨权限、可执行的动作链。
