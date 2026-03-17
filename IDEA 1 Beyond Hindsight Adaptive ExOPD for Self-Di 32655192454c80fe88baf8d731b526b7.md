# IDEA 1  Beyond Hindsight: Adaptive ExOPD for Self-Distilled Agent Improvement

### idea 1 — self distillation  &  learning beyond teacher  —— AI评价

这个 idea **是成立的，而且这四篇放在一起，几乎正好拼出一条完整路线**：

- 论文 1 给了你“**从真实交互中做 hindsight self-distillation**”的在线数据来源；
- 论文 2 给了你“**在 agent 环境里，把下一状态/用户后续话语变成 token-level hindsight signal**”的工程化办法；
- 论文 3 给了你“**自蒸馏可以支持 continual learning，而不是只会遗忘**”的核心论据；
- 论文 4 给了你“**仅仅蒸馏 teacher 不够，应该通过 ExOPD 的 λ>1 去做 reward extrapolation，从而可能超过 teacher**”的关键机制。([arXiv](https://arxiv.org/pdf/2603.12273))

所以你提的“**提出一个自蒸馏框架，采用 ExOPD，让 agent 产生自进化效果**”这件事，我的判断是：

**很有研究价值，而且方法论上是自洽的；但要把它从‘self-improvement’做成真正可信的‘self-evolution’，关键不在 ExOPD 本身，而在‘新信息从哪来’、‘reference 怎么设’、以及‘何时允许超过 teacher’。** ([arXiv](https://arxiv.org/pdf/2603.12273))

---

## 我先把四篇文章压成一条技术主线

### 1）《Aligning Language Models from User Interactions》

这篇的核心是：把一段交互写成 `(x, y, o)`，其中 `x` 是上下文，`y` 是模型原回答，`o` 是用户后续消息。然后用同一个模型在看到 `o` 之后形成一个 **hindsight policy**，再比较 `π(y_i | x,o,y_<i)` 和 `π(y_i | x,y_<i)` 的对数比值，把它当成 token-level advantage，回灌到原策略里。这是非常标准的“**模型看到后验信息后，自己教自己**”。论文也明确把这个方法称为 **SDPO from User Interactions**，并强调它能支持 alignment、personalization 和 continual adaptation。([arXiv](https://arxiv.org/pdf/2603.12273))

这篇对你 idea 的启发是：

**自蒸馏不一定需要外部 teacher，它可以来自“同一模型在更多上下文下的更优后验行为”。** 这就给“agent 自进化”提供了最自然的数据闭环：agent 与用户/环境交互 -> 获得后续反馈 -> hindsight teacher -> 再蒸馏回当前 policy。([arXiv](https://arxiv.org/pdf/2603.12273))

### 2）《OpenClaw-RL: Train Any Agent Simply by Talking》

这篇把上面的思想推进到了 **agent RL**。它不是简单依赖最终 reward，而是把 `s_{t+1}` 里的信息抽成一个 **hint**，构造 enhanced teacher context，然后定义 token-level OPD advantage：

[

A_t = \log \pi_{\text{teacher}}(a_t|s_{\text{enhanced}}) - \log \pi_\theta(a_t|s_t).

]

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

## 所以，你这个 idea 的一个清晰表述可以是

我会把它概括成：

**Self-Extrapolative On-Policy Distillation for Agent Evolution**

或者更学术一点：

**Agent Continual Self-Evolution via Hindsight Self-Distillation and Reward Extrapolation**

核心思想是：

1. agent 与用户/环境在线交互，产生 on-policy 轨迹；
2. 从后续用户消息、下一状态、工具执行结果、verifier、PRM 中提取 hindsight 信息；
3. 用同一模型或其 EMA/reference 版本，在增强上下文下形成 hindsight teacher；
4. 用 token-level OPD/SDPO 形式得到 dense advantage；
5. 再用 ExOPD 的 `λ>1` 对这部分 implicit reward 做外推，而不是只保守蒸馏；
6. 通过 continual replay / EMA teacher / trust region 控制，让 agent 不只是“纠错”，而是逐步形成新的、更强的策略。([arXiv](https://arxiv.org/pdf/2603.12273))

---

## 这个 idea 为什么有吸引力

### 第一层：它补了 RLVR / outcome reward 的短板

传统 agent RL 常见问题是 reward 太稀疏，只在 episode 末尾才知道成败。OpenClaw-RL 明确想做的就是把 next-state 信号转成 token-level advantage，因为这类 dense correction 比单纯 scalar reward 更细。你的方案如果把 self-distillation 和 ExOPD 结合，就等于同时拿到了：

- **稀疏 reward 的 correctness 信号**
- **hindsight signal 的局部纠偏能力**
- **ExOPD 的 beyond-teacher 外推能力**。

### 第二层：它天然适合 continual learning

论文 3 的核心贡献，就是说明 self-distillation 可以把 demonstrations 变成 on-policy learning signal，从而缓解 off-policy SFT 的遗忘问题。agent 自进化本质就是 continual learning；如果你不用这类 on-policy self-distillation，系统大概率会在新任务上变强、老能力上崩掉。([arXiv](https://arxiv.org/pdf/2601.19897))

### 第三层：它允许“超越当前自己”

论文 1 和 2 都是在说：**模型看到后续信息后，自己其实知道更好的做法**。论文 4 则进一步说：即便 teacher 已经比 student 强，也不必停在 teacher 上，可以通过 `λ>1` 做 extrapolation。把两者叠起来，就是一种很自然的“**模型用自己 hindsight 的更优后验作为 teacher，再用外推继续拉升**”的机制。([arXiv](https://arxiv.org/pdf/2603.12273))

---

## 但这里有一个必须说清的理论边界

**ExOPD 不能凭空创造信息。**

这是这个 idea 最容易被质疑的地方。

如果你的 hindsight teacher 只是“同一个模型 + 同一份信息 + 同样的错误先验”，那 `λ>1` 只是把已有偏好 shift 放大；它可能带来更强的行为偏移，但不等于真的获得新知识。真正的“自进化”至少要依赖下面三类新增信息之一：

- **用户后续反馈 / preference shift**，像论文 1；
- **环境转移 / tool result / verifier / PRM**，像论文 2；
- **demonstration / solved trajectory / archived better behavior**，像论文 3。 ([arXiv](https://arxiv.org/pdf/2603.12273))

所以更严谨地说，这个 idea 不是“闭门造车式 self-improvement”，而是：

**通过在线交互获取新增监督，再用 self-distillation + extrapolation 将其内化和放大。** ([arXiv](https://arxiv.org/pdf/2603.12273))

---

## 我觉得最好的框架，不是“纯 self-distill”，而是三模型结构

如果让我设计，我会用：

- **Student / Actor**：当前在线 agent `π_θ`
- **Hindsight Teacher**：同架构 EMA 版本 `π_ema`，在 enhanced context 下打分
- **Reference Model**：较慢更新的 anchor `π_ref`，用于 KL 约束和 ExOPD reward 基线

为什么这么分？

因为论文 3 明确表明 EMA teacher 更稳；论文 4 明确说明 `π_ref` 的选取影响 implicit reward 形状；OpenClaw-RL 也说明 enhanced context teacher 不应直接等同于原 student。把 teacher 和 reference 分开，会比“一个模型扮演全部角色”稳很多。([arXiv](https://arxiv.org/pdf/2601.19897))

我会把 advantage 写成下面这种统一形式：

[

A_t = w_{env} \hat r_t^{env}

- w_{sd}\Big(\log \pi_{ema}(a_t|s_t,h_t)-\log \pi_\theta(a_t|s_t)\Big)
- w_{ex}(\lambda-1)\Big(\log \pi_{ema}(a_t|s_t,h_t)-\log \pi_{ref}(a_t|s_t)\Big)
    
    ]
    

这里：

- 第一项是环境/验证器奖励；
- 第二项是普通 self-distillation 的 hindsight correction；
- 第三项是 extrapolation term，表示“teacher 相对 reference 的超额 reward”被进一步放大。

这不是原文现成公式，而是基于论文 2 的混合优势、论文 3 的 self-distillation 结构、论文 4 的 G-OPD/ExOPD 写出的一个统一化设计。这个推导方向和三篇的原始目标是一致的，但这一步本身是我的综合建议。支持它的论文依据分别来自 token-level OPD、SDFT 的 teacher-student reverse-KL、以及 G-OPD 的 `λ log(π*/π_ref)` 结构。([arXiv](https://arxiv.org/pdf/2603.10165v1))

---

## 你这个 idea 里，最有 novelty 的点应该放在哪里

我觉得真正能发出来的 novelty，不是单纯一句“self-distillation + ExOPD”，而是下面三点里的至少一点：

### 1）把 ExOPD 从“teacher-student distillation”推广到“self-hindsight distillation”

论文 4 主要在 teacher-student on-policy distillation 语境里讨论 ExOPD。你如果把 teacher 换成 **self-hindsight teacher**，也就是：

[

\pi^* \leftarrow \pi_{ema}(\cdot|s,hindsight)

]

那就把论文 1/2/3 的 self-distillation 体系，和论文 4 的 extrapolation 机制真正焊接起来了。这个连接本身就有新意。([arXiv](https://arxiv.org/pdf/2603.12273))

### 2）设计“何时允许 extrapolate”的 gating

不是所有 hindsight signal 都该 `λ>1`。

如果用户反馈噪声大，或者下一状态本身有歧义，外推只会放大错误。OpenClaw-RL 已经在 hint quality filtering 上很谨慎；论文 3 也说明 teacher 参数化不稳会出问题。最自然的创新点是做一个 **confidence-gated ExOPD**：

- verifier / judge 置信度高
- hindsight teacher 熵低
- teacher-reference gap 稳定
- 多次采样一致

满足这些条件时，才让 `λ>1`；否则退回 `λ=1` 或更小。这个 gating 在我看来非常关键。论文 2 已经给了 hint filtering 的动机，论文 4 已经告诉你 reward correction / ref 选择会影响效果。

### 3）把“自进化”定义成可测的长期提升，而不是单轮纠错

论文 1 更偏 alignment/personalization，论文 2 更偏 agent RL，论文 3 更偏 continual learning，论文 4 更偏 beyond-teacher distillation。你如果做这条线，实验必须证明：

- 新任务会持续变强
- 老任务不掉
- 同一环境分布上长期 rollouts 的 success rate 上升
- 不是仅仅变得更 verbose、更 exploit verifier、更讨好 judge

这里最该借鉴的是论文 3 的 continual-learning protocol。([arXiv](https://arxiv.org/pdf/2601.19897))

---

## 我最担心的三个坑

### 坑 1：自举偏差会被 ExOPD 放大

普通 self-distillation 已经有自我确认偏差风险；ExOPD 再把 reward scale 拉大，错误 hint、错误 hindsight、错误 verifier 都会被放大。论文 4 在 strong-to-weak 里需要 reward correction，已经说明 implicit reward 本身不是总干净的。([arXiv](https://arxiv.org/pdf/2602.12125))

### 坑 2：真正的“超越 teacher”在 agent 里未必等于更优策略

在论文 4 的数学/代码基准里，超过 teacher 比较好定义；但 agent 环境中，超过 hindsight teacher 可能意味着：

- 采取更短路径；
- 学到投机动作；
- verifier hack；
- 或者只是更迎合 judge 模型。
    
    所以 agent 场景里，“beyond teacher”必须绑定真实环境 success，而不能只看 distillation loss 或 judge score。OpenClaw-RL 本身就很强调 verifiable outcomes。([arXiv](https://arxiv.org/pdf/2603.10165v1))
    

### 坑 3：如果 reference 选错，ExOPD 的方向会歪

论文 4 的一个关键观察是 `π_ref` 不是装饰，它决定了 implicit reward 的形状，而且 reward correction 在 strong-to-weak 里能进一步提升表现。你的 self-evolution 框架里，`π_ref` 如果就用当前 online actor，训练会非常飘；如果用过旧 base，又会过于保守。这里很可能需要“慢 EMA reference”或“能力边界 reference”。

---

## 如果让我写成论文，我会这样组织方法部分

### 方法名

**SEED: Self-Extrapolative Evolutionary Distillation for Agents**

### 数据

每一步交互保存：

[

(s_t, a_t, s_{t+1}, r_t^{env}, h_t, v_t)

]

其中 `h_t` 是从用户后续话语/next-state/tool results 中提取的 hindsight hint，`v_t` 是 verifier 或 judge 的可信度。这个思路直接对应论文 1 的 user follow-up 和论文 2 的 hint extraction。([arXiv](https://arxiv.org/pdf/2603.12273))

### Teacher

[

\pi_T = \pi_{ema}(\cdot|s_t \oplus h_t)

]

teacher 不是独立大模型，而是 EMA self-teacher，在增强上下文下打分。这样继承论文 3 的稳定性做法。([arXiv](https://arxiv.org/pdf/2601.19897))

### Reference

[

\pi_{ref} = \text{slow-moving anchor}

]

可以是 base、task-family checkpoint，或更慢 EMA。借鉴论文 4。

### Advantage

[

A_t^{seed} = \underbrace{\hat r_t^{env}}_{\text{verifiable reward}}

- \underbrace{\log \pi_T(a_t)-\log \pi_\theta(a_t)}_{\text{self-distillation}}
- \underbrace{\alpha_t(\lambda_t-1)\big(\log \pi_T(a_t)-\log \pi_{ref}(a_t)\big)}_{\text{confidence-gated extrapolation}}
    
    ]
    
    这里 `α_t` 由 verifier confidence / hint quality / ensemble agreement 决定。第一项来自环境，第二项来自 SDPO/SDFT/OPD，第三项来自 G-OPD。([arXiv](https://arxiv.org/pdf/2603.12273))
    

### 训练机制

- PPO / GRPO 风格更新 actor
- teacher 用 EMA
- ref 更慢更新
- replay buffer 保留老任务，防遗忘
    
    这部分和论文 2 的 RL agent setting、论文 3 的 continual learning 一脉相承。([arXiv](https://arxiv.org/pdf/2603.10165v1))
    

---

## 我对这个 idea 的总体评价

**这是个好 idea，而且不是空想；但论文价值取决于你怎么回答下面这个核心问题：**

> 自蒸馏 + ExOPD 到底是在“放大已有偏差”，还是在“把交互中新出现的信息更高效地内化为长期能力”？
> 

如果你的方法和实验能证明是后者，那这条线很强。

如果不能，你就会被审稿人认为只是把 hindsight imitation 和 reward scaling 拼在一起。这个边界非常关键。支撑这个判断的原因，正是四篇论文分别已经把 self-distillation、agent hindsight correction、continual learning 和 extrapolation 各自做到了比较清楚的程度。([arXiv](https://arxiv.org/pdf/2603.12273))

我个人最看好的落点不是“通用 AGI 自进化”这种大词，而是更具体一点：

**在有 verifier / tool feedback / user follow-up 的 agent setting 下，通过 confidence-gated self-ExOPD，实现长期在线改进且低遗忘的 agent continual learning。**

这个表述更稳，也更容易做出扎实实验。([arXiv](https://arxiv.org/pdf/2603.10165v1))

下一步我建议我们直接做两件事中的一件：

要么我帮你把这个 idea 写成一页 **research proposal / method sketch**；要么我直接给你列一版 **方法公式 + 训练流程 + 实验设计**。

### idea ai修改意见

从“能发论文”的角度，你这条线最该改的不是再加一个模块，而是**换一个更窄、更锋利的问题定义**。否则审稿人很容易说：这只是把 SDPO / agent hindsight / self-distillation / ExOPD 四篇工作做了工程拼接。

我会建议你把 idea 从：

**“自蒸馏 + ExOPD，实现 agent 自进化”**

改成下面三种里的一种，其中我最推荐第一种。

---

## 最推荐的改法

### 方向 A：把贡献点改成“何时应该外推”，而不是“可以外推”

也就是不要把 novelty 放在 **self-distillation + ExOPD** 这个组合本身，而要放在：

**Confidence-Gated Self-ExOPD for Continual Agent Learning**

核心问题变成：

> 在 agent 的 hindsight self-distillation 里，什么时候应该做 ExOPD 的 reward extrapolation，什么时候只该保守蒸馏？
> 

这会比原始版本强很多，因为前面几篇已经分别说明了：

- 可以直接从用户交互做 self-distillation；([arXiv](https://arxiv.org/pdf/2603.12273))
- 可以在 agent setting 里利用 next-state / hint 做 token-level hindsight correction；([arXiv](https://arxiv.org/pdf/2603.10165v1))
- self-distillation 可以支持 continual learning；([arXiv](https://arxiv.org/pdf/2601.19897))
- ExOPD 通过 `λ>1` 做 reward extrapolation，目标是 beyond teacher。([arXiv](https://arxiv.org/pdf/2602.12125))

所以“简单相加”不够新。

但“**提出一个判定机制，决定何时 extrapolate**”是新的，而且非常合理，因为 ExOPD 最大的风险本来就是会放大错误信号，这正是 agent 里最痛的点。

你可以让 `λ` 不再是常数，而变成：

[

\lambda_t = 1 + g(c_t)

]

其中 `c_t` 来自：

- verifier confidence
- hindsight teacher 熵
- teacher-reference margin
- 多次 rollout 一致性
- 工具结果是否可验证

这样你的贡献就从“拼框架”变成了“**可控外推机制**”。

---

## 第二推荐的改法

### 方向 B：把问题改成“self-distillation 何时能真正超过 hindsight teacher”

这条线的标题可以是：

**Beyond Hindsight: Self-Extrapolative Distillation for Agent Improvement**

这里的重点不是“teacher-student distillation beyond teacher”，而是：

> 当 teacher 本身就是同一个模型在 hindsight 条件下形成的 posterior policy 时，怎样定义和实现真正的 beyond-teacher improvement？
> 

这比原始版本更学术，因为它把论文 4 的 ExOPD，迁移到了一个**更难、更新的 setting**：

teacher 不是外部强模型，而是 **self-hindsight teacher**。论文 1 已经给了 self-distillation from user interactions 的 hindsight form；你这里的创新是把论文 4 的 extrapolation 机制从 teacher-student setting 推广到 self-hindsight setting。([arXiv](https://arxiv.org/pdf/2603.12273))

这条线的好处是理论味更强。

坏处是审稿人会追问：“如果没有新增信息，凭什么 beyond teacher？”

所以你必须把“新增信息源”讲得非常明确：用户 follow-up、tool execution、environment transition、verifier outcome。这些都是前两篇明确依赖的 hindsight 信息来源。([arXiv](https://arxiv.org/pdf/2603.12273))

---

## 第三推荐的改法

### 方向 C：不要讲“自进化”，改讲“低遗忘在线 agent continual learning”

如果你想更稳、更容易投稿，我甚至建议把“自进化”这个表述弱化，因为它容易显得太大、太泛。改成：

**Continual Agent Learning via Hindsight Self-Distillation with Reward Extrapolation**

这样论文会更像一个扎实的 continual learning / agent post-training paper，而不是一个容易被质疑概念过大的 AGI paper。

这个改法的优势是：

- 论文 1 已经证明 raw user interactions 可用于 continual adaptation；([arXiv](https://arxiv.org/pdf/2603.12273))
- 论文 3 已经证明 self-distillation 能支持 continual learning；([arXiv](https://arxiv.org/pdf/2601.19897))
- 论文 2 提供了 agent 的 hindsight token-level advantage 机制；([arXiv](https://arxiv.org/pdf/2603.10165v1))
- 论文 4 则给你“不是只吸收，而是适度外推”的 knob。([arXiv](https://arxiv.org/pdf/2602.12125))

这条线会让你的论文更“像研究论文”，更少 marketing 味。

---

## 我会怎么改你的核心 claim

你现在的 claim 大概是：

> 通过自蒸馏 + ExOPD，agent 可以自进化。
> 

这个 claim 太大，也太容易被打。

我建议改成下面这种更能过审的版本：

### 版本 1：机制型

> We study when hindsight self-distillation should extrapolate beyond the self-teacher in continual agent learning, and propose a confidence-gated ExOPD objective that improves online adaptation while preserving prior capabilities.
> 

### 版本 2：设定型

> We introduce self-extrapolative on-policy distillation, a framework where an agent learns from hindsight self-teachers and selectively extrapolates beyond them using verifiable interaction signals.
> 

### 版本 3：系统型

> We formulate continual agent improvement from deployment interactions as hindsight self-distillation with adaptive reward extrapolation, enabling stronger online learning under sparse environmental rewards.
> 

这三种都比“agent 自进化”更像论文语言。

---

## 从审稿视角看，你现在最缺的不是方法，而是“独立问题”

一个能发的论文，通常要让审稿人一眼看见下面三件事：

### 1）这是个新问题，不只是旧模块叠加

你可以把问题定义成：

**在 deployment-time interaction data 上，如何让 agent 从 hindsight self-feedback 中持续学习，同时避免把错误 hindsight 放大？**

这个问题就很独立，也天然导向你的 gating / adaptive extrapolation 设计。

### 2）这个问题确实存在

你需要在引言里点出一条明确张力：

- self-distillation 能从交互里学，但通常偏保守；论文 1、3 都是“吸收 hindsight 信息”，不是主动 beyond-teacher。([arXiv](https://arxiv.org/pdf/2603.12273))
- ExOPD 能 beyond teacher，但默认设置并不考虑 self-generated hindsight signal 的可靠性。([arXiv](https://arxiv.org/pdf/2602.12125))
- agent 场景的 hindsight signal 噪声很大，因为 next-state / tool result / verifier 可能局部误导。论文 2 已经说明要从状态中提 hint，而这本身就意味着 signal quality 是核心问题。([arXiv](https://arxiv.org/pdf/2603.10165v1))

于是你的切入点自然就是：

**self-generated hindsight signals need adaptive extrapolation rather than uniform extrapolation.**

### 3）你提出的机制正好解决这个问题

这就是前面说的 `adaptive λ` 或 gated ExOPD。

---

## 我建议你避免的两种写法

### 不要写法 1：大而空的“self-evolution”

这个词可以放在动机里，但不要放在主 claim。

因为它会立刻引来两个质疑：

- 你到底有没有真正获得新能力，还是只是在当前分布上做在线适配？
- 你的提升是不是 verifier hacking / preference overfitting / benchmark overfitting？

改成 “continual agent improvement” 或 “online capability accumulation” 会稳很多。

### 不要写法 2：把 novelty 写成“首次结合 A+B+C+D”

这种写法很危险。

尤其你列的四篇都很新，审稿人会非常敏感于“只是把 contemporaneous work 拼起来”。你必须把 novelty 落在**一个新的 decision problem**上，比如：

- adaptive extrapolation
- extrapolation under self-generated hindsight
- reliable beyond-hindsight learning
- forgetting-aware self-extrapolative distillation

---

## 真正值得做的方法改造

如果从方法层面只允许你改一刀，我建议是：

### 把常数 λ 改成“有依据的、分层的 λ”

比如：

# [A_t

A_t^{env}

+

A_t^{sd}

+

\alpha_t(\lambda_t-1)A_t^{extra}

]

其中：

- `A_t^{env}`：环境或 verifier 奖励
- `A_t^{sd}`：普通 hindsight self-distillation 项
- `A_t^{extra}`：teacher 相对 reference 的 extrapolation 项
- `α_t`：是否允许外推的 gate
- `λ_t`：外推强度

然后你把贡献写成：

1. 提出 self-hindsight ExOPD 框架
2. 提出 confidence-based extrapolation control
3. 在 continual agent learning 中证明它比纯 self-distillation 和固定 ExOPD 都更稳

这个结构很完整。

---

## 从“发论文”角度，实验也要跟着改

你的实验不要只问“效果有没有涨”，而要问三个更论文化的问题：

### 问题 1：为什么需要 extrapolation？

对比：

- pure RL
- pure hindsight self-distillation
- fixed-λ ExOPD
- adaptive-λ ExOPD

如果没有这组，你的论文没有方法必要性。

### 问题 2：为什么需要 gate？

要专门做 noisy hindsight / wrong hint / ambiguous verifier 的实验。

证明固定 `λ>1` 会放大噪声，而 gated 版本更稳。这个实验会非常加分，因为它直接对应你的新问题定义。

### 问题 3：为什么叫 continual improvement，而不是短期 overfit？

要测：

- 新任务增益
- 老任务保持
- 长 horizon online success
- 不同时间窗口下的能力轨迹

这点和论文 3 的 continual learning 脉络最一致。([arXiv](https://arxiv.org/pdf/2601.19897))

---

## 如果你想要一个最像论文标题的版本

我给你几个更能投稿的标题方向：

1. **Confidence-Gated Reward Extrapolation for Hindsight Self-Distillation in Continual Agent Learning**
2. **Beyond Hindsight: Adaptive ExOPD for Self-Distilled Agent Improvement**
3. **Reliable Self-Extrapolative Distillation from Deployment Interactions**
4. **Continual Agent Learning from User and Environment Feedback via Adaptive Self-ExOPD**

这几个标题都比“agent self-evolution”稳。

---

## 我最推荐你最后落成的版本

一句话总结：

**不要把论文写成“提出一个 self-distillation + ExOPD 的 agent 自进化框架”；要把它写成“研究在 noisy hindsight feedback 下，agent 何时应当超越 self-teacher，并提出一种 adaptive / confidence-gated ExOPD 机制，在 continual agent learning 中实现更强且更稳的在线改进”。**

这样改完以后：

- 问题更新
- 机制更清楚
- 风险点被你主动接住了
- 审稿人更难说你只是拼工作

下一步最值得做的是把这个改成一页 proposal：**problem statement、method、3 个核心实验、3 个 baseline、1 个主图**。

### proposal

下面是一版可直接用于组会、开题讨论或项目立项的**正式中文研究提案**。我保留了较强的学术写作风格，同时尽量把其中仍属**推测性**的部分写得谨慎，不把未验证结论写成既定事实。

---

# 研究提案

## 题目

**基于后见提示的外推式自蒸馏：面向智能体持续自增强的单模型在线学习框架**

英文可对应为：

**Extrapolated Self-Distillation from Hindsight Hints for Continual Agent Self-Improvement**

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

## 六、创新点

本研究的创新性主要体现在以下三个方面。

### 6.1 提出面向智能体在线学习的“后见提示自教师”框架

与依赖外部教师或奖励模型的方法不同，本文尝试将**模型自身在带后见提示条件下的分布**作为教师信号来源，从而构建单模型闭环学习体系。

### 6.2 将 reward extrapolation 引入自教师蒸馏场景

现有“超过教师”的研究主要在更干净、更强教师的设定下展开。本文将首次系统检验：**当教师只是模型自身的后见条件版本时，外推仍是否有效。**

### 6.3 研究持续自增强的有效性与边界

本文不仅关注是否“能提高一点性能”，还将重点分析：

- 何种提示质量下外推有效；
- 何种情况下会发生策略漂移或错误放大；
- 该机制是否能够支撑多轮持续提升。
    
    因此，本研究更接近一个**机制研究（mechanistic study）+ 方法研究（method study）**，而不是单纯提出一个工程技巧。
    

---

## 七、实验设计

## 7.1 任务选择

考虑到可用资源约为单机 8 卡 4090D、每卡显存约 50GB，本文不计划直接在大规模长时程通用智能体环境中训练，而是优先选择**中等复杂度且可控的智能体任务**，包括：

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

## 八、预期结果与贡献

本文预期可能得到以下几类结果中的一种或多种：

### 情形一：Self-ExOPD 明显优于 Self-OPD 和标准 OPD

若出现这一结果，则说明**即使教师只是后见提示条件下的模型自身，外推式蒸馏仍有可能带来 beyond-self 提升**。这将构成本文最强的正面结论。

### 情形二：Self-OPD 有效，但 Self-ExOPD 仅在高质量提示上有效

若出现这一结果，则说明自教师框架本身可行，但 ExOPD 的成功高度依赖提示质量与置信度控制。这类结论同样有价值，因为它能够揭示方法的适用边界。

### 情形三：Self-ExOPD 效果不稳定，甚至明显放大偏差

若出现这一结果，则说明“噪声后见提示 + 自教师 + 外推”这一组合存在根本风险。即使得到负结论，若分析充分，也具有研究价值，因为它能够澄清当前文献间接推理链条中的薄弱环节。

无论结果属于哪种情况，本文都预期在以下方面提供贡献：

1. 给出一种新的单模型 agent 在线学习框架；
2. 系统研究后见提示自教师蒸馏的有效性；
3. 首次在该场景下检验 ExOPD 的适用性；
4. 给出该方法成功与失败的边界条件。

---

## 九、风险与局限性

本研究存在若干不可忽视的风险。

### 9.1 提示质量不稳定

后见提示若提取不准，则整个自教师分布可能建立在错误理解之上，进而导致错误学习。

### 9.2 单模型闭环导致错误相关性

同一模型同时承担 judge、teacher 和 student 三种角色，可能导致“自己说服自己”的闭环偏差，降低训练信号的独立性与可信度。

### 9.3 外推可能放大噪声

ExOPD 的有效性依赖于教师方向本身是可靠的。若教师信号质量不足，(\lambda > 1) 可能只是放大错误。

### 9.4 持续训练可能引发漂移与遗忘

即使短期性能上升，也可能在长期更新后导致分布漂移、旧能力退化或生成风格畸变。

### 9.5 资源限制约束实验规模

在当前计算资源下，本研究更适合做机制验证型工作，而非大规模系统级工作。因此其实验范围需要刻意控制，否则很难在时间和资源内完成高质量验证。

---

## 十、可行性分析

## 10.1 概念可行性：中高

从概念上看，本研究有较清晰的文献链条支持：后见学习、自蒸馏与 ExOPD 可以被统一到一个合理框架中。因此，该方向并非无根据设想。

但需要强调的是，“后见提示自教师 + 外推 = 持续超越自身”这一完整命题尚未被现有文献直接证明，因此目前只能视为一个**有根据但未证实的研究假设**。

## 10.2 技术可行性：中

在 8×4090D 的资源条件下，采用 7B 级基础模型、LoRA/QLoRA、短轨迹 agent 任务和在线—离线混合训练策略，技术实现是可行的。

但若试图扩展到大规模长轨迹通用 agent，则工程开销和实验方差都可能超出可承受范围。

## 10.3 实验可行性：中

本课题可以在控制变量良好的小到中规模任务上验证清楚，适合形成一篇机制明确、实验扎实的论文。

其难点在于：必须设计足够严谨的基线与消融，避免结论停留在“组合方法略优于若干基线”这种说服力较弱的层面。

## 10.4 资源可行性：中

当前资源足以支持方法验证和中等规模实验，但不足以支撑大型系统论文。因此研究范围应聚焦于**机制清晰的小而精问题**，而非追求覆盖所有 agent 场景。

## 10.5 创新潜力：中高

若实验表明自教师外推在 agent setting 中稳定有效，本文将具有较高创新价值。

但若实现方式过于接近“现有方法简单拼接”，则创新性会被明显削弱。因此需要通过新的分析、机制或边界结论来增强论文独立性。

## 10.6 发表潜力：中

若只做基础方法拼接和有限实验，投稿顶会的说服力有限。

若能够在实验中清楚展示：

- 自教师的有效性；
- 外推的边界条件；
- 负面结果和稳定性分析；
    
    则具有较好的发表潜力，尤其适合作为 2026 年智能体学习方向的一篇问题导向型论文。
    

---

## 十一、审稿人视角下的潜在质疑

从审稿人角度看，本文可能会面临以下质疑：

### 11.1 方法是否只是简单组合？

审稿人可能认为本文只是将已有的 hindsight learning、自蒸馏和 ExOPD 三类方法做了拼接，而缺乏真正独立的算法创新。

### 11.2 自教师是否足够可靠？

若教师只是同一个模型在带提示条件下的自身分布，那么“性能提高”是否只是条件诱导下的局部修正，而非稳定能力提升？

### 11.3 外推收益是否真实？

若性能提升同时伴随输出变长、回答更保守或更倾向某些模板，则难以说明 ExOPD 带来的是真实智能增强。

### 11.4 单模型 judge 是否引入确认偏差？

同一模型同时生成提示、评估提示并学习自身，可能形成自我强化闭环，降低实验结论可信度。

### 11.5 结果是否仅在狭窄场景成立？

若方法只在简单对话纠错中有效，而在工具调用或代码任务中不稳，则其贡献可能被视为局部增量。

因此，若要使本文更具说服力，必须在实验中主动回应这些质疑，而不能只展示平均性能提升。

---

## 十二、研究计划与时间安排

考虑到目标为 **NeurIPS 2026**，且摘要注册与正式投稿时间较早，本文建议采用分阶段推进策略。

### 第一阶段：问题收敛与原型实现

目标是完成 Self-OPD 的最小可运行闭环，回答最基础问题：

**后见提示条件下的自教师是否显著优于原始策略？**

### 第二阶段：加入外推与关键消融

在确认自教师有效后，加入 ExOPD 机制，系统测试不同 (\lambda)、不同参考策略与不同提示过滤方案的影响。

### 第三阶段：扩展到第二类任务

在对话纠错任务上稳定后，将方法迁移到工具调用或代码执行任务，以验证其跨场景适用性。

### 第四阶段：完善论文叙事

围绕以下主线组织论文：

1. 自教师是否成立；
2. ExOPD 是否有效；
3. 何时有效，何时失败；
4. 持续训练是否稳定。

这样的叙事将明显优于单纯“提出一个新方法并报告平均提升”。

---

## 十三、结论

本文提出了一项面向智能体持续学习的新研究方向：**利用后见提示条件下的模型自身作为教师，并通过外推式在线蒸馏探索智能体的持续自增强能力**。与依赖外部强教师、显式奖励模型或人工标注的方法相比，该方向具有更强的自主性与潜在扩展性；同时，它也面临提示噪声、自我强化偏差、策略漂移与稳定性不足等显著风险。

总体而言，本研究方向具有**中等偏上的整体可行性和较好的创新潜力**，但要形成有竞争力的顶会论文，关键不在于“把几个方法拼起来”，而在于：

- 明确提出一个先前未被回答的研究问题；
- 通过严谨实验验证或证伪这一问题；
- 系统揭示方法有效与失败的边界。

若研究推进顺利，本文有望形成一篇围绕“**后见提示自教师蒸馏是否能够支持智能体持续自增强**”这一核心问题展开的完整论文；即便最终结论偏保守，只要分析充分，也仍具有发表价值。

---

如果你愿意，我下一步可以继续把这版提案扩展成两种实用版本之一：

**版本A：开题/组会PPT式提案提纲**，或者 **版本B：NeurIPS论文引言 + related work + method 初稿**。

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