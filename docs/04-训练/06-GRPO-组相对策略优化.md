# 06 - GRPO 组相对策略优化 train_grpo.py 全文

> 覆盖代码：`trainer/train_grpo.py` **全文 329 行**。
> GRPO（Group Relative Policy Optimization）是 DeepSeekMath/R1 提出的 PPO 简化版：**去掉 Critic，用「同一 prompt 采样一组答案、组内相对打分」代替价值函数**。本仓库同时实现了 CISPO 变体（`--loss_type` 切换）。
> 前置：[05-DPO](./05-DPO-直接偏好优化.md)（logp 抽取思想）、[10-rollout引擎](./10-rollout引擎.md)（生成后端）。

---

## 1. 原理

### 1.1 PPO 的回顾与 GRPO 的简化

PPO 的策略目标（clip 版）：

$$
\mathcal{L}^{CLIP} = -\mathbb{E}_t\Big[\min\big(r_t A_t,\; \text{clip}(r_t, 1\pm\epsilon)A_t\big)\Big],\qquad r_t = \frac{\pi_\theta(a_t|s_t)}{\pi_{old}(a_t|s_t)}
$$

PPO 需要 Critic 网络估计状态价值 V(s) 来算优势 A（GAE），critic 通常是同规模的另一个模型——**显存翻倍、训练慢**。

**GRPO 的替代**：对同一个 prompt x，采样 G 个回答 $y_1..y_G$（本仓库 G=num_generations=6）。用奖励模型给每个回答打分 $r_i$，然后**组内归一化**得到优势：

$$
A_i = \frac{r_i - \mathrm{mean}(r_1..r_G)}{\mathrm{std}(r_1..r_G)}
$$

直觉：**「这组回答里比平均好的就是正样本、比平均差的就是负样本」**——组均值天然扮演了 baseline（价值函数估计），无需 Critic。

### 1.2 KL 惩罚：k3 估计器

GRPO 在 token 级损失里加 KL（相对参考模型），用的是**无偏低方差**的 k3 估计：

$$
\hat{D}_{KL} = \frac{\pi_{\mathrm{ref}}(a)}{\pi_\theta(a)} - \log\frac{\pi_{\mathrm{ref}}(a)}{\pi_\theta(a)} - 1
$$

记 `kl_div = logπ_ref − logπ`，则估计 = `exp(kl_div) − kl_div − 1`。性质：
- kl_div=0 → 估计=0（正确，无散度）；
- 估计值恒 ≥0（凸函数性质，exp(z)≥1+z）；
- 只需采样到的 token 的概率（不像 k1 要算全词表）。

### 1.3 CISPO 变体（MiniMind 默认）

CISPO（MiniMax-M1 论文）改用：

$$
\mathcal{L}^{CISPO} = -\,\mathbb{E}_t\Big[\min\big(\text{clip}(r_t, \epsilon_{low}, \epsilon_{high})\cdot \mathrm{sg}(r_t) \cdot A_t \log\pi_\theta,\ \ldots\Big]
$$

本仓库实现（138-139 行）为：

```python
clamped_ratio = torch.clamp(ratio, max=args.epsilon_high).detach()
per_token_loss = -(clamped_ratio * advantages.unsqueeze(1) * per_token_logps - args.beta * per_token_kl)
```

- `clamped_ratio` = ratio 截到上界 ε_high=5 后 **detach**（当作常数权重）；
- 损失 = −(权重 × A × **logπ**) —— 梯度直接来自 logπ（而非 ratio 里隐含的 logπ），ratio 只起「权重」作用。
- 效果：**被 clip 掉的 token 仍能收到梯度**（PPO 的 min-clip 会让越界 token 梯度归零，长 CoT 训练里大量 token 长期被 clip → 学习停滞；CISPO 保留它们的加权梯度）。

### 1.4 奖励构成（reward shaping）

总奖励 = 规则奖励 + 奖励模型分数：

| 项 | 条件 | 分值 |
|----|------|------|
| 长度 | 回答 20~800 字符 | ±0.5 |
| 思考长度 | think 段 20~300 字符 | +1.0 / −0.5 |
| 思考闭合 | `</think>` 恰好 1 次 | +0.25 / −0.25 |
| 重复惩罚 | 3-gram 重复率 | −rep_penalty（≤0.5） |
| **RM 分数** | internlm2-1.8b-reward 打分 | ±3（已 clip） |

规则奖励把「格式正确」编码进奖励——**这就是 RLAIF 用 RM + 规则塑形引导出思考行为的方式**。

---

## 2. 逐行精读

### 2.1 rep_penalty：重复度惩罚（31-36 行）

```python
def rep_penalty(text, n=3, cap=0.5):
    cjk = ("\u3007\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uac00-\ud7af"
           "\U00020000-\U0002ee5f\U0002f800-\U0002fa1f\U00030000-\U0003347f")
    toks = re.findall(rf"[{cjk}]|[^\W{cjk}]+|[^\w\s]", text.lower())
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return min(cap, (len(grams) - len(set(grams))) * cap * 2 / len(grams)) if grams else 0.0
```

| 行 | 讲解 |
|----|------|
| 32-33 | CJK 字符的 Unicode 区间合集（日文假名、汉字扩展、韩文、CJK 扩展 A/B/C…）——正则字符类 |
| 34 | 三类 token 化：单个 CJK 字符 | 连续非 CJK 词（`[^\W{cjk}]+`：字母数字词）| 单个标点。中文按字、英文按词，粒度对齐 |
| 35 | 全部 n=3 元组；`grams` 总数 − 去重后数量 = **重复 gram 数** |
| 36 | 重复率 × cap×2 归一，封顶 cap=0.5。文本越啰嗦扣越多（从奖励里减） |

### 2.2 calculate_rewards（39-70 行）

```python
def calculate_rewards(prompts, responses, reward_model):
    rewards = torch.zeros(len(responses), device=args.device)

    with torch.no_grad():
        reward_model_scores = []
        batch_size = len(prompts)

        for i in range(batch_size):
            for j in range(args.num_generations):
                response_idx = i * args.num_generations + j
                response = responses[response_idx]
                prompt = prompts[i]

                pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
                matches = re.findall(pattern, prompt, re.DOTALL)
                messages = [{"role": role, "content": content.strip()} for role, content in matches]
                answer = response
                rewards[response_idx] += 0.5 if 20 <= len(response.strip()) <= 800 else -0.5
                if '</think>' in response:
                    thinking_content, answer_content = response.split('</think>', 1)
                    rewards[response_idx] += 1.0 if 20 <= len(thinking_content.strip()) <= 300 else -0.5
                    rewards[response_idx] += 0.25 if response.count('</think>') == 1 else -0.25
                    answer = answer_content.strip()
                rewards[response_idx] -= rep_penalty(answer)

                score = reward_model.get_score(messages, answer)
                reward_model_scores.append(score)

        reward_model_scores = torch.tensor(reward_model_scores, device=args.device)
        rewards += reward_model_scores

    return rewards
```

| 行 | 讲解 |
|----|------|
| 40 | 奖励容器：长度 = B×G（每个生成一条） |
| 46-48 | 双层循环遍历「第 i 个 prompt 的第 j 个生成」，全局下标 `i*G+j` 与 rollout 输出对齐 |
| 52-54 | **从 prompt 文本反解析出消息列表**：prompt 是 chat_template 渲染的字符串，用正则 `<|im_start|>(role)\s+(内容)<|im_end|>` 抽回结构化消息（奖励模型需要 role 结构）。`re.DOTALL` 让 `.` 匹配换行 |
| 56 | 长度规则分（见 §1.4 表） |
| 57-61 | 思考段规则分：**首个 `</think>` 前是思考、后是答案**（split(…, 1)）；思考过长/过短惩罚；多个 `</think>`（格式混乱）扣分 |
| 62 | 重复惩罚作用于**答案部分**（思考里重复推理链是正常的） |
| 64-65 | RM 打分（答案 + 反解出的对话历史）。逐条调用——RM 是 1.8B 模型，每 batch B×G 次推理，是训练的主要瓶颈之一 |
| 67-68 | RM 分数并入总奖励 |

### 2.3 grpo_train_epoch（73-198 行）——核心

**Step 1：prompt 编码（左 padding！）**

```python
    for step, batch in enumerate(loader, start=start_step + 1):
        prompts = batch['prompt']  # list[str], length B
        prompt_inputs = tokenizer(prompts, return_tensors="pt", padding=True, return_token_type_ids=False,
                                  padding_side="left", add_special_tokens=False).to(args.device)
        if args.max_seq_len:
            prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -args.max_seq_len:]
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -args.max_seq_len:]
```

| 行 | 讲解 |
|----|------|
| 75-78 | **左 padding** 是生成任务的硬要求：生成是在序列**尾部**续写，若右 padding，第 1 个新 token 会接在 pad 后面（pad 位置破坏 RoPE 相位与注意力语义）。左 pad 让所有样本的真实 token 都顶在右端。`add_special_tokens=False`：prompt 已是模板渲染好的（含 im_start 等），不再加 bos |
| 79-81 | 超长 prompt 从左侧截断（保留**最近**的 max_seq_len 个 token——对话尾巴更重要） |

**Step 2：rollout 采样**

```python
        rollout_result = rollout_engine.rollout(
            prompt_ids=prompt_inputs["input_ids"],
            attention_mask=prompt_inputs["attention_mask"],
            num_generations=args.num_generations,
            max_new_tokens=args.max_gen_len,
            temperature=0.8,
        )
        outputs = rollout_result.output_ids
        completion_ids = rollout_result.completion_ids
        completions = rollout_result.completions
        old_per_token_logps = rollout_result.per_token_logps.to(args.device).detach()
        prompt_lens = rollout_result.prompt_lens.to(args.device)
```

| 行 | 讲解 |
|----|------|
| 82-89 | 调用 rollout 引擎（torch 或 sglang，见 [10 篇](./10-rollout引擎.md)）：每个 prompt 复制 G 份采样，返回 output_ids `[B*G, P+R]`、completion_ids `[B*G, R]`、**逐 token logp**（`old_per_token_logps`：rollout 时刻策略的 logp，即 PPO 里的 π_old，**detach** 当常数——importance ratio 的分母） |

**Step 3：构造训练 mask**

```python
        full_mask = (outputs != tokenizer.pad_token_id).long()
        logp_pos = prompt_lens.unsqueeze(1) - 1 + torch.arange(completion_ids.size(1), device=args.device).unsqueeze(0)
        full_mask.scatter_(1, logp_pos + 1, rollout_result.completion_mask.to(args.device, dtype=full_mask.dtype))
```

| 行 | 讲解 |
|----|------|
| 94 | 粗 mask：非 pad 即 1（SGLang 引擎右 pad 时剔除 pad 区） |
| 95 | **logp 位置索引**：completion 的第 r 个 token（completion_ids[:, r]）在**完整序列里**的位置 = `prompt_len − 1 + r`——因为「预测第 r 个生成 token」用的是位置 prompt_len−1+r 的 logits（错位一步！位置 p 的 logits 预测 token p+1）。`logp_pos` 形状 `[B*G, R]` |
| 96 | `scatter_`：把 completion_mask（真实生成区=1）**按 logp_pos+1 的位置**写进 full_mask——修正 pad 误判（SGLang 的 pad 不会真等于 pad_id 的情况）与保证生成区掩码精确 |

**Step 4：奖励**

```python
        rewards = calculate_rewards(prompts, completions, reward_model).to(args.device)  # [B*num_gen]
```

**Step 5：策略与参考模型的 logp**

```python
        with autocast_ctx:
            res = model(outputs, attention_mask=full_mask)
            aux_loss = res.aux_loss if lm_config.use_moe else torch.tensor(0.0, device=args.device)
            per_token_logps = F.log_softmax(res.logits[:, :-1, :], dim=-1).gather(2, outputs[:, 1:].unsqueeze(-1)).squeeze(-1).gather(1, logp_pos)

        with torch.no_grad():
            ref_per_token_logps = F.log_softmax(ref_model(outputs, attention_mask=full_mask).logits[:, :-1, :], dim=-1).gather(2, outputs[:, 1:].unsqueeze(-1)).squeeze(-1).gather(1, logp_pos)
```

| 行 | 讲解 |
|----|------|
| 100-103 | **策略前向（可训练）**：对完整序列 outputs 前向；logits 去掉最后一行（`[:, :-1]`，对应预测下一个 token）；log_softmax 后 gather 抽「outputs[:, 1:]（每个位置的实际下一 token）」的 logp → `[B*G, P+R-1]` 全序列 logp；**再 gather(1, logp_pos)** 按位置索引只取生成区那 R 个 logp → `[B*G, R]`。两段 gather 是「先全量、再精选」，因为 logp_pos 的锚定错位无法用简单切片表达（prompt 长度可变） |
| 105-106 | 参考模型同款（no_grad、常数）。**ref 与 policy 用的同一 outputs**——KL 是逐 token 的 π_ref/π 比值 |

**Step 6（可选）：调试打印**

```python
        if args.debug_mode and is_main_process() and step % args.debug_interval == 0:
            ...  # 打印 prompt、每个生成的 completion、reward——人工抽查奖励函数是否合理
```

RL 训练极易「奖励函数错了但 loss 正常下降」，**必须定期肉眼看样本**。

**Step 7：组归一化优势**

```python
        grouped_rewards = rewards.view(-1, args.num_generations)  # [B, num_gen]
        mean_r = grouped_rewards.mean(dim=1).repeat_interleave(args.num_generations)  # [B*num_gen]
        std_r = grouped_rewards.std(dim=1, unbiased=False).repeat_interleave(args.num_generations)  # [B*num_gen]
        advantages = (rewards - mean_r) / (std_r + 1e-4)  # [B*num_gen]
```

| 行 | 讲解 |
|----|------|
| 123 | reshape 成 [B, G]——按 prompt 分组 |
| 124-125 | 组内均值/标准差（`unbiased=False` 除以 G 而非 G−1），`repeat_interleave` 摊回 [B*G]（与 rewards 对齐） |
| 126 | **GRPO 的灵魂**：组内标准化。+1e-4 防 G 个奖励全同时除零（此时优势全 0，该组不产生梯度——「组内无差别就没有学习信号」） |

**Step 8：completion mask 与 EOS 截断**

```python
        completion_pad_mask = rollout_result.completion_mask.to(args.device).bool()
        is_eos = (completion_ids == tokenizer.eos_token_id) & completion_pad_mask  # [B*num_gen, R]
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1) - 1, dtype=torch.long, device=args.device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        completion_mask = ((torch.arange(is_eos.size(1), device=args.device).expand(is_eos.size(0), -1) <= eos_idx.unsqueeze(1)) & completion_pad_mask).int()  # [B*num_gen, R]
```

| 行 | 讲解 |
|----|------|
| 128-132 | 生成区掩码精细化的目的：**EOS 之后（padding 阶段）不该有损失**。<br>• `is_eos`：哪些位置是真实（非 pad）的 EOS；<br>• `eos_idx`：每个样本**第一个** EOS 的位置（argmax 在 bool 上返回首个 True）；没有 EOS 的样本默认 R−1（最后一格）；<br>• `completion_mask`：位置 ≤ eos_idx 且非 pad → 1。EOS 本身被保留（要学习「何时停」），EOS 之后的填充被清零 |

**Step 9：损失**

```python
        kl_div = ref_per_token_logps - per_token_logps
        per_token_kl = torch.exp(kl_div) - kl_div - 1  # [B*num_gen, R]
        ratio = torch.exp(per_token_logps - old_per_token_logps)  # [B*num_gen, R]
        if args.loss_type == "cispo":
            clamped_ratio = torch.clamp(ratio, max=args.epsilon_high).detach()
            per_token_loss = -(clamped_ratio * advantages.unsqueeze(1) * per_token_logps - args.beta * per_token_kl)
        else:
            clipped_ratio = torch.clamp(ratio, 1 - args.epsilon, 1 + args.epsilon)
            per_token_loss1 = ratio * advantages.unsqueeze(1)
            per_token_loss2 = clipped_ratio * advantages.unsqueeze(1)
            per_token_loss = -(torch.min(per_token_loss1, per_token_loss2) - args.beta * per_token_kl)
        policy_loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1).clamp(min=1)).mean()
        loss = (policy_loss + aux_loss) / args.accumulation_steps  # scalar
        loss.backward()
```

| 行 | 讲解 |
|----|------|
| 134-135 | KL（k3 估计，见 §1.2） |
| 136 | **importance ratio** = π_θ / π_old：rollout（π_old）与本次更新（π_θ）之间策略已变，比例修正 off-policy 偏差 |
| 137-139 | CISPO 分支（见 §1.3）：clamp 上界 + detach 当权重；损失=权重×A×logπ − β·KL |
| 140-144 | **GRPO/PPO 分支**：clip 到 [1−ε, 1+ε]；`min(无 clip 版, clip 版)` 取保守项——A>0 时取 clip（不让 ratio 飙升过界；A<0 时取 ratio 但也不鼓励 ratio 跌破下界）。最终 −(reward 项 − β·KL)，**β·per_token_kl 作为正项被减去 = 惩罚偏离参考模型** |
| 145 | **mask 平均**：逐样本对有效 token 平均，再对 batch 平均（「每 token 平均贡献」，长回答不会稀释信号） |
| 146-147 | +aux_loss（MoE）、除累积步数、反传 |

**Step 10：更新与日志**

```python
        if step % args.accumulation_steps == 0 or step == iters:
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
```

| 行 | 讲解 |
|----|------|
| 149-154 | 更新三连。**注意 GRPO 用了真正的 scheduler**（CosineAnnealingLR，见 main），这里 `scheduler.step()` 按 optimizer step 递减；且**没有 scaler**——GRPO/PPO 路径不套 GradScaler（loss 已在 autocast 外稳定，且 ratio/logp 计算对 fp16 敏感） |

```python
        if step % args.log_interval == 0 or step == iters:
            ...  # Reward / KL_ref / Adv Std / Adv Mean / Actor Loss / Avg Response Len / LR
```

七个监控量：平均奖励、相对参考模型的 KL、优势的均值/方差（**Adv Std 若趋 0 说明组内无差异、学不到东西**——奖励太同质）、策略损失、平均回复长度（R1 训练里长度暴涨=废话膨胀信号）、学习率。

```python
        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            ...  # 保存 grpo_768.pth + checkpoints/grpo_768_resume.pth（scheduler 状态一并进 kwargs）
        if step % args.save_interval == 0 or step == iters: rollout_engine.update_policy(model)
```

- 存档同前（resume kwargs 带 scheduler）。
- **`rollout_engine.update_policy(model)`**：把最新权重同步给 rollout 引擎——torch 引擎只是换引用；SGLang 引擎会把权重写盘并通知远端服务器热加载（见 10 篇）。**RL 的闭环就在这一行：采样引擎必须用最新策略，否则 ratio 会越偏越大**。

### 2.4 main（201-329 行）

argparse 差异行（204-239 行）：

| 行 | 参数 | 默认 | 讲解 |
|----|------|------|------|
| 207 | lr | 3e-7 | RL 极小步长 |
| 218 | max_seq_len | 768 | prompt 上限 |
| 219 | max_gen_len | 1024 | 生成长度上限 |
| 221 | **num_generations** | 6 | 组大小 G |
| 222 | beta | 0.1 | KL 系数 |
| 223 | loss_type | **cispo** | grpo/cispo 二选一 |
| 224 | epsilon | 0.2 | PPO 下界 clip |
| 225 | epsilon_high | 5.0 | CISPO 上界 |
| 227 | reward_model_path | ../../internlm2-1_8b-reward | RM 路径 |
| 235-238 | rollout_engine/sglang_* | torch | 引擎选择 |
| 234 | thinking_ratio | 0.9 | 90% 的 prompt 开思考 |

main 流程差异（266-312 行）：

```python
    # ========== 5. 初始化模型和数据 ==========
    base_weight = args.from_weight
    model, tokenizer = init_model(lm_config, base_weight, device=args.device)      # policy
    ref_model, _ = init_model(lm_config, base_weight, device=args.device)           # reference
    ref_model = ref_model.eval().requires_grad_(False)
    reward_model = LMForRewardModel(args.reward_model_path, device=args.device, dtype=torch.float16)
    rollout_engine = create_rollout_engine(engine_type=..., policy_model=model, ...)
    train_ds = RLAIFDataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len, thinking_ratio=args.thinking_ratio)
    ...
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    loader_for_count = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler)
    iters = len(loader_for_count)
    total_optimizer_steps = math.ceil(iters / args.accumulation_steps) * args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_optimizer_steps, eta_min=args.learning_rate / 10)
```

| 行 | 讲解 |
|----|------|
| 269-274 | **三个模型**：policy（可训）、reference（冻结）、**外部 RM**（1.8B，占大头显存） |
| 276-285 | rollout 引擎工厂（10 篇详解） |
| 287 | RLAIFDataset：只出 prompt |
| 290-293 | **iters 的计数技巧**：真实 loader 用 SkipBatchSampler（长度会变），所以另建一个「计数用」loader 只为拿 `len()`；总 optimizer 步数 = epoch 迭代数/累积 × epochs → 给 CosineAnnealingLR 定 T_max，尾 lr=lr/10 |

```python
    if ckp_data: ...  # 恢复 model/optimizer/scheduler/进度
    if dist.is_initialized():
        # 同 train_ppo：RoPE buffer 各 rank 一致，每步广播纯属浪费
        model = DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False)
    rollout_engine.update_policy(model)
```

| 行 | 讲解 |
|----|------|
| 310 | `broadcast_buffers=False`：DDP 默认每步广播所有 buffer（保证各 rank 一致）。本模型的 `freqs_cos/freqs_sin` 是**非持久 buffer，由 config 确定性计算**——各 rank 本来就一致，每步广播纯属浪费带宽。**MoE 的 aux_loss 属性不是 buffer 所以不受影响** |
| 311 | 训练开始前把初始权重同步给引擎 |

训练循环（314-324 行）同骨架；`use_sglang=(args.rollout_engine=="sglang")` 传参（当前 grpo_train_epoch 的 use_sglang 形参未实际使用，引擎行为由 rollout_engine 内部决定）。

## 3. 训练闭环全景

```
prompt(B 条) ──左pad编码──→ rollout引擎(π_old)×G 采样 ──→ completions + old_logp
   │                                                        │
   │                    规则奖励 + RM 打分 → 组内归一化 → advantages
   │                                                        │
policy forward(完整序列) → π_θ logp ──┐                      │
ref forward(冻结) → π_ref logp ──┤→ ratio/k3-KL → CISPO或GRPO loss → backward → 更新
   │                                                        │
   └──────── 每 save_interval：update_policy 同步引擎 ←──────┘
```

## 4. 本篇小结

1. GRPO = PPO 去 Critic：**组内 (r−mean)/std 当优势**；k3 估计 KL；ratio 重要性采样。
2. CISPO：clip 的 ratio detach 当权重，梯度走 logπ——被 clip 的 token 也能学（长 CoT 关键）。
3. 奖励 = 规则（长度/思考格式/重复）+ RM 分数；`debug_mode` 肉眼验证奖励函数是 RL 的必修课。
4. 工程要点：**左 padding 生成**、`logp_pos` 错位索引、EOS 截断 mask、`update_policy` 同步采样引擎、DDP 关 buffer 广播。

下一篇：[07-PPO-近端策略优化](./07-PPO-近端策略优化.md)
