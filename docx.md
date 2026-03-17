# IDEA 1  Beyond Hindsight: Adaptive ExOPD for Self-Distilled Agent Improvement
# 相关论文

### 1）《Aligning Language Models from User Interactions》

这篇的核心是：把一段交互写成 `(x, y, o)`，其中 `x` 是上下文，`y` 是模型原回答，`o` 是用户后续消息。然后用同一个模型在看到 `o` 之后形成一个 **hindsight policy**，再比较 `π(y_i | x,o,y_<i)` 和 `π(y_i | x,y_<i)` 的对数比值，把它当成 token-level advantage，回灌到原策略里。这是非常标准的“**模型看到后验信息后，自己教自己**”。论文也明确把这个方法称为 **SDPO from User Interactions**，并强调它能支持 alignment、personalization 和 continual adaptation。([arXiv](https://arxiv.org/pdf/2603.12273))

这篇对你 idea 的启发是：

**自蒸馏不一定需要外部 teacher，它可以来自“同一模型在更多上下文下的更优后验行为”。** 这就给“agent 自进化”提供了最自然的数据闭环：agent 与用户/环境交互 -> 获得后续反馈 -> hindsight teacher -> 再蒸馏回当前 policy。([arXiv](https://arxiv.org/pdf/2603.12273))

### 2）《OpenClaw-RL: Train Any Agent Simply by Talking》

这篇把上面的思想推进到了 **agent RL**。它不是简单依赖最终 reward，而是把 `s_{t+1}` 里的信息抽成一个 **hint**，构造 enhanced teacher context，然后定义 token-level OPD advantage：


$
A_t = \log \pi_{\text{teacher}}(a_t|s_{\text{enhanced}}) - \log \pi_\theta(a_t|s_t).
$


同时，它还把这种 OPD advantage 和 binary/verifiable reward 组合成加权优势：

[

A_t = w_{\text{binary}} r_{\text{final}} + w_{\text{opd}}(\log \pi_{\text{teacher}}(a_t|s_{\text{enhanced}})-\log \pi_\theta(a_t|s_t)).

]

作者明确说 binary RL 提供 broad coverage，OPD 提供 high-resolution per-token correction。

这篇对你 idea 最重要的价值是：

**它把 self-/hindsight distillation 从“聊天模型对话纠偏”推进到了“具环境状态转移的 agent”**。也就是说，你的“自进化 agent”不必局限于纯文本聊天；可以是 tool-use、GUI、SWE agent，只要下一状态里含有可提炼的纠偏信号，就可以自蒸馏。([arXiv](https://arxiv.org/pdf/2603.10165v1))

### 3）《Self-Distillation Enables Continual Learning》

这篇是你整个设想的“稳定性地基”。它提出 **SDFT**：学生采样自己的 on-policy 轨迹，但 teacher 不是外部专家，而是“**同一模型在 demonstrations 条件下**”形成的 teacher distribution；训练目标是 student 和 demonstration-conditioned teacher 的 reverse-KL。论文还把它解释成一种隐式 IRL：内在 reward 可以写成 teacher 与 student 的 log-prob shift。更关键的是，它实验上强调 **continual learning 可行，而且 EMA teacher 比 frozen base 或 current student 更稳**。([arXiv](https://arxiv.org/pdf/2601.19897))

这篇给你的最关键启发是两点：

第一，**自蒸馏并不天然导致灾难性遗忘，反而可能比离线 SFT 更适合 continual learning**。([arXiv](https://arxiv.org/pdf/2601.19897))

第二，**teacher 的参数化很关键**。他们默认用 EMA teacher，因为直接用当前 student 当 teacher 会不稳定，而 frozen base 又跟不上学习进度。这个结论对“自进化”非常重要——否则你很容易把系统做成自我确认偏差放大器。([arXiv](https://arxiv.org/pdf/2601.19897))

### 4）《Learning beyond Teacher: Generalized On-Policy Distillation with Reward Extrapolation》

这篇是把 OPD 从“模仿 teacher”提升成“**可控地超越 teacher**”。它先把 OPD重写成一个 KL-constrained RL 目标，再引入第三个 reference model `π_ref` 和 reward scaling factor `λ`，得到 G-OPD：

[

J_{\text{G-OPD}}(\theta)=\mathbb{E}\left[\lambda \log \frac{\pi^*(y|x)}{\pi_{ref}(y|x)} - D_{KL}(\pi_\theta || \pi_{ref})\right].

]

论文明确说 `λ` 控制 reward 与 KL 的相对权重；并指出当 `λ>1` 时，就是 **reward extrapolation / ExOPD**。此外，在 strong-to-weak distillation 里，他们又进一步讨论 **reward correction**：当 reference 选成 teacher 的 pre-RL variant 一类更合适的模型时，效果还能继续提升。

这篇对你 idea 的决定性意义在于：

**如果只做普通 self-distillation，你通常只是在更稳地吸收已有能力；而 ExOPD 给了一个“向 teacher 之上外推”的控制杆。** 这正是“self-improvement”走向“self-evolution”的那一脚油门。([arXiv](https://arxiv.org/pdf/2602.12125))

---




## 一、研究背景与动机

近年来，大语言模型驱动的智能体（agent）在多轮对话、工具调用、代码执行、终端操作与复杂任务规划等场景中取得了显著进展。然而，这类智能体在真实交互过程中仍面临一个核心问题：**如何利用交互后自然产生的反馈信号持续改进自身能力，而不依赖昂贵的人类标注、额外奖励模型或更强外部教师模型**。

现有研究已经表明，模型在看到额外上下文后，往往能够生成比原始输出更合理的分布。例如，用户后续追问、纠错信息、工具执行结果、环境状态变化等，都可能为模型提供“事后更优”的条件信息。基于这一观察，近期工作提出可以将“带有后见信息的模型”视为一种**条件化教师（conditional teacher）**，并将其输出分布蒸馏回当前策略，从而实现直接从用户交互或环境交互中学习。

另一方面，关于在线蒸馏（on-policy distillation）的最新研究进一步指出，蒸馏并不必然局限于“模仿教师”。通过对奖励项进行适度外推（reward extrapolation），学生模型在理论上有可能突破原教师分布所定义的上界，从而实现“**超越教师（beyond teacher）**”的学习效果。

基于上述进展，本研究拟探索一个新的问题：**在不依赖额外强教师模型的前提下，是否能够仅利用智能体自身在后见提示（hindsight hints）条件下形成的自教师分布，并结合外推式在线蒸馏，实现持续的自增强学习？**

该问题同时具有理论意义与现实价值。理论上，它涉及“自蒸馏是否足以形成稳定能力提升闭环”这一尚未充分回答的问题；实践上，它有望降低智能体在线学习系统对外部监督器、专用奖励模型和高成本人工反馈的依赖。

---

## 二、研究问题

本研究拟回答以下核心问题：

> **在真实智能体交互过程中，能否将由下一状态信号（next-state signals）提取出的后见提示作为条件信息，使当前策略模型自身充当教师；并通过外推式在线蒸馏，使智能体在持续交互中实现超越原始自身能力的改进？**
> 

围绕这一核心问题，本文进一步细化为以下几个子问题：

1. **后见提示条件下的自教师分布，是否系统性优于原始策略分布？**
2. **在噪声较大的交互反馈场景中，基于自教师的在线蒸馏是否能够稳定工作？**
3. **当引入 reward extrapolation 后，性能提升究竟来源于真实能力增强，还是仅仅来源于输出长度增加、策略偏移放大或局部启发式过拟合？**
4. **持续迭代更新后，该方法是否能够带来可累积的能力提升，而不是自我强化偏差或造成旧能力遗忘？**

---

## 三、相关研究基础

### 3.1 从用户交互与后见信息中学习

已有工作表明，用户后续追问、纠错和补充信息本身可以作为学习信号。模型在看到这些后见信息后，往往能够在相同输入上生成更合理的响应，因此可以将“后见条件下的模型分布”蒸馏回原模型。这类方法证明了：**无须显式偏好标注，真实交互本身就可能包含有效监督信息。**

与此同时，面向智能体训练的研究进一步指出，交互后的下一状态信号不仅包含评估性信息，也可能包含指导性信息。若能从这些信号中提炼出简洁的纠正提示（hint），则可构造出一种后见提示条件下的策略分布，用于进行 token-level 的在线蒸馏。

### 3.2 自蒸馏与持续学习

另一类相关研究表明，自蒸馏不仅可以用于压缩模型，也可以用来缓解持续学习中的灾难性遗忘。其核心思想在于：**让同一个模型在更丰富条件下产生教师分布，再利用该分布对当前策略进行 on-policy 学习**。与纯 SFT 相比，这类方法更接近策略当前分布，通常更有利于稳定学习和保留既有能力。

### 3.3 超越教师的在线蒸馏

更进一步，近期关于广义在线蒸馏（generalized on-policy distillation）的研究提出：若将教师分布视作隐式奖励结构的一部分，则通过调整奖励与参考策略之间的权重关系，学生模型可能不仅逼近教师，而且能够在一定条件下**超过教师本身**。其中，reward extrapolation 为“如何从蒸馏走向超越”提供了一个重要算法视角。

### 3.4 现有研究的不足

尽管上述三类工作各自取得了进展，但它们之间仍存在一个明显空白：

- 用户交互学习工作通常强调**后见信息驱动的自蒸馏**，但不讨论“超越教师”；
- agent 学习工作提出了**后见提示引导的在线蒸馏**，但通常仍依赖专门的评分器、奖励模型或组合式训练；
- ExOPD 类工作展示了“**超越教师**”的可能性，但多在高质量领域教师或较干净任务环境下验证，尚未系统研究**噪声后见提示 + 自教师**这一更困难情形。

因此，本研究的目标不是简单重复已有方法，而是尝试回答一个尚未被直接验证的问题：

**当教师不再是外部更强模型，而仅仅是“后见提示条件下的模型自身”时，外推式在线蒸馏是否仍然有效，且是否能够支撑持续的智能体自增强？**

---

## 四、研究思路与方法

## 4.1 总体思路

本文拟提出一个**单模型自蒸馏在线学习框架**。在该框架中，模型在每轮智能体交互后接收环境返回的下一状态信号，例如用户反馈、工具输出、执行结果或错误信息。系统随后从该信号中提取后见提示，并利用同一模型在“带提示”和“不带提示”两种条件下对原动作序列进行重评分，从而构造教师—学生之间的 token-level 学习信号。进一步地，本文将引入 ExOPD 风格的 reward extrapolation，以探索模型是否可能超越其后见提示条件下的自教师分布。

该方法的关键思想可概括为：

1. **后见提示提供纠正方向**；
2. **同一模型在带提示条件下形成自教师分布**；
3. **学生模型在当前策略轨迹上进行在线蒸馏**；
4. **通过外推而非纯模仿，探索超越自教师的可能性**。

---

## 4.2 方法定义

设智能体在状态 (s_t) 下产生动作或回答 (a_t)，环境返回下一状态 (s_{t+1})。系统从 ((a_t, s_{t+1})) 中抽取一个后见提示 (h_t)，并构造增强状态：

[

s_t^+ = s_t \oplus h_t

]

其中，(\oplus) 表示将提示拼接至原始状态上下文。

随后，使用同一基础模型分别计算：

- 原始条件下的 student 分布：
    
    [
    
    \log \pi_\theta(a_t \mid s_t)
    
    ]
    
- 带有后见提示条件下的 self-teacher 分布：
    
    [
    
    \log \pi_\theta(a_t \mid s_t^+)
    
    ]
    

据此，可定义基础自蒸馏优势信号为：

[

A_t^{self} = \log \pi_\theta(a_t \mid s_t^+) - \log \pi_\theta(a_t \mid s_t)

]

这一直观地表示：若原动作中的某些 token 在后见提示条件下被模型赋予更高概率，则说明这些 token 与修正后的意图更一致，反之则应被削弱。

为探索“超过自教师”而非“仅模仿自教师”，本文进一步引入外推式蒸馏。设 (\pi_{ref}) 为参考策略，可取初始策略、前一轮策略或滑动平均策略，则构造外推目标：

[

A_t^{exo} = \lambda \cdot \log \pi_\theta(a_t \mid s_t^+) - \log \pi_{ref}(a_t \mid s_t)

]

其中 (\lambda > 1) 为外推系数。

当 (\lambda = 1) 时，该方法退化为标准的自教师在线蒸馏；

当 (\lambda > 1) 时，模型将沿着教师偏好的方向继续推进，从而在理论上具有“超越教师”的可能性。

---

## 4.3 方法实现的三个层次

为避免研究结论过于脆弱，本文不直接仅做单一版本，而计划实现以下三个版本：

### （1）Self-OPD

仅使用后见提示条件下的自教师分布进行在线蒸馏，不引入外推，即 (\lambda = 1)。

此版本用于回答最基础问题：

**后见提示条件下的自教师，是否本身就是一个有效学习信号？**

### （2）Self-ExOPD

在 Self-OPD 基础上引入 (\lambda > 1) 的外推式训练。

此版本用于回答关键问题：

**在自教师设定下，reward extrapolation 是否仍然能够提升性能？**

### （3）Hybrid：Binary RL + Self-ExOPD

保留粗粒度的 evaluative signal（如成功/失败、偏好方向等），同时结合 token-level 的自教师外推蒸馏。

此版本的动机是：已有 agent 学习研究表明，粗粒度评估与细粒度蒸馏通常是互补关系。若完全舍弃评估性信号，仅依赖提示驱动蒸馏，可能导致训练覆盖范围不足或学习信号不稳定。因此，Hybrid 版本将用于测试：

**自教师外推是否应被视作独立方法，还是更适合作为 agent RL 框架中的增强模块。**

---

## 五、核心假设

本文的研究建立在以下几个假设之上。需要强调的是，这些假设并非都已被直接验证，其中部分仅由已有文献间接支持，仍需本研究通过实验检验。

### 假设 H1：后见提示条件下的自教师优于原始策略

即，模型在看到从下一状态中提取出的提示后，对原动作的条件分布更接近高质量行为分布。

这是本研究最基本的可行性前提。

### 假设 H2：自教师分布与原策略分布足够接近

若两者过于偏离，则在线蒸馏可能导致学习不稳定。自蒸馏之所以有机会稳定，部分原因在于教师和学生共享相同参数基础与表示空间。

### 假设 H3：适度外推可以带来超过自教师的收益

该假设来源于 ExOPD 的理论启发，但是否适用于“噪声提示 + 自教师”场景，目前尚未有直接证据。

### 假设 H4：持续迭代不会积累错误强化

如果模型同时负责解释反馈、生成提示、构造教师分布并更新自身，则存在自我强化偏差的风险。本文需要验证：这种闭环是否会带来稳定能力提升，还是仅仅放大原有偏差。

---


## 七、实验设计

## 7.1 任务选择

### （1）多轮对话/个人助理纠错任务

主要考察模型从用户后续纠正中学习格式约束、回答方式、风格一致性和显式需求修正的能力。

### （2）工具调用或代码执行任务

主要考察模型从工具返回结果、执行报错、测试反馈中提取后见提示，并提升后续动作质量的能力。

这样的任务设置具有两个优点：

一是与研究问题高度相关；

二是实验成本可控，适合进行大量消融与稳定性分析。

---

## 7.2 基线方法

为避免结论失真，本文计划至少设置以下基线：

1. **SFT on corrected/hindsight data**
    
    将修正后的样本直接作为监督数据进行微调。
    
2. **Binary RL only**
    
    只使用粗粒度评估信号，不做 token-level 蒸馏。
    
3. **OpenClaw-style OPD**
    
    使用提示引导的在线蒸馏，但不做自教师外推。
    
4. **Self-OPD**
    
    本文提出的基础版本，不带外推。
    
5. **Self-ExOPD**
    
    本文提出的外推版本。
    
6. **Binary RL + Self-ExOPD**
    
    混合版本，测试互补性。
    
7. **Optional stronger-teacher baseline**
    
    若资源允许，可使用较强模型或离线校正器构造小规模上限对照。
    

---

## 7.3 关键消融实验

本文将重点开展以下消融：

### （1）外推系数 (\lambda) 的影响

测试 (\lambda \in {1.0, 1.1, 1.25, 1.5}) 等不同设置，分析性能、稳定性和输出长度的变化趋势。

### （2）参考策略 (\pi_{ref}) 的选择

比较使用：

- 初始策略；
- 上一轮 checkpoint；
- 指数滑动平均策略（EMA）。
    
    分析不同参考策略对稳定性和遗忘的影响。
    

### （3）提示质量过滤机制

比较：

- 不过滤；
- 仅保留高置信提示；
- 置信度感知的外推强度控制。
    
    该部分也可能进一步发展为本文的一个附加创新模块。
    

### （4）同模型 judge 与独立 judge 的对比

用于分析“单模型闭环”的收益与风险，尤其关注错误相关性是否会影响训练效果。

### （5）在线与离线训练方式对比

比较在相同交互数据上，离线 replay 式训练与逐轮在线更新的差异。

---

## 7.4 评价指标

本文将从多个维度评估方法效果，而不局限于单一任务成功率。

### 主要效果指标

- 任务成功率 / Pass@k / 完成率
- 对话纠错后采纳率
- 工具调用正确率
- 代码执行通过率

### 稳定性与机制指标

- 提示采纳率（hint acceptance ratio）
- teacher-student logprob gap
- token-level advantage 分布
- 训练过程中的 KL 演化
- 策略熵与多样性变化

### 能力保持指标

- 旧任务保留率
- OOD 泛化性能
- 通用指令能力是否退化

### 混淆因素控制指标

- 输出长度
- 响应延迟
- 工具调用次数
    
    这些指标非常重要，因为如果不控制长度与预算，性能提升可能只是“输出更长、更保守”的假象。
    

---


### Idea 1 数据集

| 论文 | 用途 | 数据集名称 | 数据集大小（论文中报告） | 数据集内容简介 |
| --- | --- | --- | --- | --- |
| Aligning Language Models from User Interactions | 训练 | **WildFeedback** | 约 **20,000** 个对话；其中约 **6,000** 个只有单轮 prompt-response；实际训练使用其余约 **14,000** 个对话，构成约 **50,000** 个 interaction tuples | WildFeedback 是 WildChat 的一个人工筛选子集，保留带有隐式反馈信号的真实用户对话，如不满、纠错、重写请求等；论文用它来做 SDPO 训练。 ([arXiv](https://arxiv.org/html/2603.12273v1)) |
| Aligning Language Models from User Interactions | 训练 | **WildChat** | 论文用于鲁棒性实验时随机抽样约 **14,000** 个对话，约 **50,000** 个 interaction tuples | 真实用户与外部模型（如 GPT-3.5 Turbo、GPT-4）的对话集合；论文把它作为“未筛选真实交互”的对照训练源。 ([arXiv](https://arxiv.org/html/2603.12273v1)) |
| Aligning Language Models from User Interactions | 评测 | **AlpacaEval 2.0** | 论文未明确报告样本数 | 指令跟随 / 对齐评测基准；论文用其默认设置与 GPT-4 Turbo 评判配置。 ([arXiv](https://arxiv.org/html/2603.12273v1)) |
| Aligning Language Models from User Interactions | 评测 | **IFEval** | 论文未明确报告样本数 | 指令遵循评测；论文报告 prompt-level loose 指标。 ([arXiv](https://arxiv.org/html/2603.12273v1)) |
| Aligning Language Models from User Interactions | 评测 | **ArenaHard-v2** | 论文未明确报告样本数 | 用于 hard prompts、creative writing，以及文中提到的 math / coding 相关综合评测。 ([arXiv](https://arxiv.org/html/2603.12273v1)) |
| Aligning Language Models from User Interactions | 评测 | **MMLU-Pro** | 论文未明确报告样本数 | 知识与推理评测；论文使用推荐的 chain-of-thought 5-shot 设置。 ([arXiv](https://arxiv.org/html/2603.12273v1)) |
| Aligning Language Models from User Interactions | 个性化实验 | **TL;DR dataset** | 论文未明确报告样本数 | 论文在受控摘要任务中，用 TL;DR prompts 模拟“简洁/随意/初学者友好”等风格偏好。 ([arXiv](https://arxiv.org/html/2603.12273v1)) |
| Aligning Language Models from User Interactions | 个性化实验 | **HelpSteer2** | 论文未明确报告样本数；在线实验共 **1,500** 次用户交互，每 **500** 次引入一种新偏好 | 更广泛的真实世界 prompts，用于连续个性化实验，偏好是互补而非互斥的。 ([arXiv](https://arxiv.org/html/2603.12273v1)) |

| 论文 | 用途 | 数据集名称 | 数据集大小（论文中报告） | 数据集内容简介 |
| --- | --- | --- | --- | --- |
| OpenClaw-RL | 个人 agent 模拟任务 | **GSM8K** | 论文未给固定离线规模；这是在线交互设置，文中报告“学生场景”约 **36** 次 problem-solving interactions 后已出现明显改进 | 在 personal-agent 模拟中，作业题目来自 GSM8K；模型扮演学生，用 OpenClaw 辅助做作业，同时学习避免“AI 味太重”的回答风格。 ([arXiv](https://arxiv.org/html/2603.10165v1)) |
| OpenClaw-RL | General agent 训练（terminal） | **SETA RL data** | 论文未明确报告样本数 | 终端 agent 的 RL 训练数据。论文未进一步展开条目规模。 ([arXiv](https://arxiv.org/html/2603.10165v1)) |
| OpenClaw-RL | General agent 训练（GUI） | **OSWorld-Verified** | 论文未明确报告样本数；GUI 评测直接在训练集上进行，但排除了 chrome 和 multi-app tasks | GUI 智能体任务数据，用于图形界面交互 agent 的 RL 训练与评测。 ([arXiv](https://arxiv.org/html/2603.10165v1)) |
| OpenClaw-RL | General agent 训练（SWE） | **SWE-Bench-Verified** | 论文未明确报告样本数 | 软件工程修复任务基准，用于 SWE agent。 ([arXiv](https://arxiv.org/html/2603.10165v1)) |
| OpenClaw-RL | General agent 训练（tool-call） | **DAPO RL data** | 论文未明确报告样本数 | 工具调用 agent 的 RL 训练数据。 ([arXiv](https://arxiv.org/html/2603.10165v1)) |
| OpenClaw-RL | General agent 评测（tool-call） | **AIME 2024** | 论文未明确报告样本数 | 工具调用 agent 的评测集。这里比较特别：训练用 DAPO RL data，评测用 AIME 2024。 ([arXiv](https://arxiv.org/html/2603.10165v1)) |

| 论文 | 用途 | 数据集名称 | 数据集大小（论文中报告） | 数据集内容简介 |
| --- | --- | --- | --- | --- |
| Self-Distillation Enables Continual Learning | 技能学习 | **SciKnowEval（Chemistry L-3 subset）** | 采用约 **75%/5%/20%** 的 train/val/test 切分；论文未给出总条数 | 本科层次科学推理、多选问答；论文用 GPT-4o 为训练样本构造 demonstrations。 ([arXiv](https://arxiv.org/pdf/2601.19897)) |
| Self-Distillation Enables Continual Learning | 技能学习 | **ToolAlpaca** | 沿用原作者 train-test split；论文未给出总条数 | 将工具/API 规格与用户请求映射为正确工具调用的任务。 ([arXiv](https://arxiv.org/pdf/2601.19897)) |
| Self-Distillation Enables Continual Learning | 技能学习 / 医疗推理 | **HuatuoGPT-o1** | 训练只用英文问题，约 **20,000** 个样本；评测从可验证题集中随机采样 **1,000** 题 | 医疗/临床推理数据；训练使用 stage 1 的 SFT 数据，评测使用 stage 2 的可验证问题。 ([arXiv](https://arxiv.org/pdf/2601.19897)) |
| Self-Distillation Enables Continual Learning | 知识注入 | **2025 自然灾害 Wikipedia 语料（作者自建）** | 约 **200K tokens**；随后生成 QA 数据集，规模约为原始语料的 **5 倍左右**；未给出精确 QA 条数 | 从 2025 年自然灾害条目构造的新知识语料，用于测试模型能否将“知识注入”内化到参数中，而非仅记住问答模板。 ([arXiv](https://arxiv.org/pdf/2601.19897)) |

| 论文 | 用途 | 数据集名称 | 数据集大小（论文中报告） | 数据集内容简介 |
| --- | --- | --- | --- | --- |
| Learning beyond Teacher | 训练（数学 RL / distillation） | **DeepMath（过滤后子集）** | 过滤后选取 **57K** 个样本（difficulty ≥ 6） | 数学推理训练数据；论文用它训练 math domain teacher，并且 distillation 数据与 RL 数据相同。 ([arXiv](https://arxiv.org/html/2602.12125v2)) |
| Learning beyond Teacher | 训练（代码 RL / distillation） | **Eurus-RL-Code** | **25K** 个样本 | 代码生成训练数据；同样用于训练 code domain teacher，蒸馏数据与 RL 数据相同。 ([arXiv](https://arxiv.org/html/2602.12125v2)) |
| Learning beyond Teacher | 评测（数学） | **AIME 2024** | 论文未明确报告样本数 | 竞赛级数学推理评测。 ([arXiv](https://arxiv.org/html/2602.12125v2)) |
| Learning beyond Teacher | 评测（数学） | **AIME 2025** | 论文未明确报告样本数 | 竞赛级数学推理评测。 ([arXiv](https://arxiv.org/html/2602.12125v2)) |
| Learning beyond Teacher | 评测（数学） | **HMMT25 (February)** | 论文未明确报告样本数 | 竞赛级数学推理评测。 ([arXiv](https://arxiv.org/html/2602.12125v2)) |
| Learning beyond Teacher | 评测（数学） | **HMMT25 (November)** | 论文未明确报告样本数 | 竞赛级数学推理评测。 ([arXiv](https://arxiv.org/html/2602.12125v2)) |
| Learning beyond Teacher | 评测（代码） | **HumanEval+** | 论文未明确报告样本数 | 代码生成评测集。 ([arXiv](https://arxiv.org/html/2602.12125v2)) |
| Learning beyond Teacher | 评测（代码） | **MBPP+** | 论文未明确报告样本数 | Python 编程题代码生成评测集。 ([arXiv](https://arxiv.org/html/2602.12125v2)) |
| Learning beyond Teacher | 评测（代码） | **LiveCodeBench（v6 only, Feb 2025–May 2025）** | 论文未明确报告样本数 | 更贴近时序、低污染的代码生成评测集；论文只使用 v6 的指定时间窗。 ([arXiv](https://arxiv.org/html/2602.12125v2)) |
