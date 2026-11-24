# 07 - PPO 近端策略优化 train_ppo.py 全文

> 覆盖代码：`trainer/train_ppo.py` **全文 450 行**。
> PPO（Proximal Policy Optimization, 2017）是 RLHF 的经典算法：**Actor（策略）+ Critic（价值）双模型**，用 GAE 计算优势、clip 目标限制单步更新。GRPO 是它的「去 Critic」简化版；本篇按完整版逐行精读。
> 前置：[05-DPO](./05-DPO-直接偏好优化.md)（logp）、[06-GRPO](./06-GRPO-组相对策略优化.md)（奖励函数/ratio/k3，本篇复用）。

---

## 1. 原理

### 1.1 Actor-Critic 框架

- **Actor**（π_θ）：策略模型，决定「下一步生成什么 token」——就是 MiniMind 主干。
- **Critic**（V_φ）：价值网络，估计「当前状态未来总奖励的期望」V(s)。本仓库把 Critic 做成 `MiniMindForCausalLM` 的子类，只换输出头。
- **优势** A = 实际回报 − 基线（V(s)）：比 Critic 预期「好多少」，正的加强、负的抑制。

### 1.2 GAE（Generalized Advantage Estimation）

对生成序列逐 token 建立折扣回报。定义 TD 残差：

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

GAE 用指数加权平均 TD 残差（λ 平衡偏差与方差）：

$$
A_t^{GAE} = \sum_{l \ge 0} (\gamma\lambda)^l\, \delta_{t+l}
$$

代码里用**反向递推**实现（省循环嵌套）：

```
lastgaelam = 0
for t in reversed(range(T)):
    delta_t = r_t + γ·V(t+1) − V(t)          # V(T) 之后视作 0（回合结束）
    lastgaelam = delta_t + γ·λ·lastgaelam     # 累积
    A[t] = lastgaelam
returns[t] = A[t] + V(t)                      # 价值学习目标
```

### 1.3 PPO 的两个 clip

**策略 clip**（1.1 已给公式）：ratio 超出 [1−ε, 1+ε] 就不再奖励更大的偏离。实现里写成 `max(−A·ratio, −A·clip(ratio))`，等价于 −min(A·ratio, A·clip(ratio))。

**价值 clip**：

$$
L^{VF} = \tfrac{1}{2}\max\big((V−R)^2,\; (\text{clip}(V, V_{old}±\epsilon_v) − R)^2\big)
$$

V 相对旧值 V_old 跑出 ±ε_v 时，超界部分不再计损失——critic 也别一步跳太远。

### 1.4 早停与「零损失闭环」

PPO 训练会在**同一批 rollout 数据上做多轮更新**（ppo_update_iters=2），策略可能偏离 π_old 太远，用 approx_kl 超阈值（early_stop_kl=0.25）触发早停。但 DDP 下**所有 rank 必须执行相同次数的 forward/backward**（梯度 all-reduce 是集体操作，某 rank 跳过会让别的卡死等）——所以早停时不 break，而是把 loss 乘 0.0 继续走完 forward-backward（梯度为 0，不实际更新）。这是分布式训练的经典坑，代码注释里点明了。

---

## 2. 逐行精读

### 2.1 CriticModel（37-50 行）

```python
# 自定义的Critic模型，继承自MiniMindLM
class CriticModel(MiniMindForCausalLM):
    def __init__(self, params):
        super().__init__(params)
        # lm_head 不参与 forward，仅靠 tie_word_embeddings 与 embed_tokens 共享权重，解绑后 DDP 会报未使用参数
        self.value_head = nn.Linear(params.hidden_size, 1)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        # 使用基础模型获取隐藏状态
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        hidden_states = self.model.norm(outputs[0])
        # 使用value_head获取价值估计
        values = self.value_head(hidden_states).squeeze(-1)
        return values
```

| 行 | 讲解 |
|----|------|
| 38-42 | 继承完整主干（embedding/8 层/norm 全继承），**加一个 `value_head`：[768→1]**。注释解释一个 DDP 陷阱：主干 forward 不再经过 lm_head（我们只要隐状态），而 lm_head 因权重绑定与 embedding 共享——DDP 会发现「这个参数在 forward 里没用上」而报错。靠绑定关系它实际仍在图里（embedding 即 lm_head），能过；若解绑就崩 |
| 44-50 | forward：只跑主干（不走 ForCausalLM 的 lm_head/loss），取**最终 norm 后**的隐状态（对齐主干输出的归一化分布），value_head 输出每个位置的标量价值 → squeeze 成 `[B, S]`。**Critic = 同架构主干 + 线性头**，初始化直接从 full_sft 权重拷贝（见 main）——价值估计一开始就「懂语言」 |

### 2.2 calculate_rewards（53-77 行）

与 GRPO 版**几乎逐行相同**，差异只有循环形式：PPO 每 prompt 只采 1 条（num_generations=1），循环是 `for i, (prompt, response) in enumerate(...)`、单奖励下标 `rewards[i]`。奖励构成（长度 ±0.5 / 思考长度 / 闭合 / rep_penalty / RM ±3）完全一致，不再重复（详见 GRPO 篇 §2.2）。

### 2.3 ppo_train_epoch（80-305 行）——核心

**Step 1：rollout（单样本）**

```python
def ppo_train_epoch(epoch, loader, iters, rollout_engine, ref_model, actor_scheduler, critic_scheduler, reward_model, start_step=0, wandb=None):
    actor_model.train()
    critic_model.train()
    grad_accum_step = 0

    for step, batch in enumerate(loader, start=start_step + 1):
        prompts = batch["prompt"]  # list[str], length B
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=args.max_seq_len,
                        padding_side="left").to(args.device)  # input_ids: [B, P], attention_mask: [B, P]
```

- 与 GRPO 相同的左 padding（右截断这里用 truncation=True）。

```python
        rollout_result = rollout_engine.rollout(
            prompt_ids=enc.input_ids, attention_mask=enc.attention_mask,
            num_generations=1, max_new_tokens=args.max_gen_len, temperature=0.8,
        )
        gen_out = rollout_result.output_ids
        completion_ids = rollout_result.completion_ids
        prompt_lens = rollout_result.prompt_lens.to(args.device)
        responses_text = rollout_result.completions
        old_resp_logp = rollout_result.per_token_logps.to(args.device)
        rewards = calculate_rewards(prompts, responses_text, reward_model)  # [B]
```

- **PPO 每 prompt 只采样 1 条**（GRPO 采 G 条）。`old_resp_logp`：π_old 的逐 token logp `[B, R]`。

**Step 2：mask 与 EOS 位置处理**

```python
        full_mask = (gen_out != tokenizer.pad_token_id).long()  # [B, P+R]
        labels = gen_out[:, 1:].clone()  # [B, P+R-1]
        B = len(prompts)
        resp_labels = completion_ids
        resp_idx = torch.arange(resp_labels.size(1), device=gen_out.device).unsqueeze(0)
        logp_pos = prompt_lens.unsqueeze(1) - 1 + resp_idx
        resp_pad_mask = rollout_result.completion_mask.to(args.device).bool()
        full_mask.scatter_(1, logp_pos + 1, resp_pad_mask.to(full_mask.dtype))
        resp_lengths = resp_pad_mask.sum(dim=1); valid_resp = resp_lengths > 0; eos_mask = resp_labels.eq(tokenizer.eos_token_id) & resp_pad_mask
        has_eos = eos_mask.any(dim=1); eos_pos = torch.argmax(eos_mask.int(), dim=1)
        resp_lengths = torch.where(has_eos, eos_pos + 1, resp_lengths).long().clamp(min=1)
        resp_policy_mask = ((resp_idx < resp_lengths.unsqueeze(1)) & resp_pad_mask).float()
        resp_value_mask = resp_policy_mask.clone()
```

| 行 | 讲解 |
|----|------|
| 118-119 | 全序列粗 mask；`labels` = 错位一步的目标（与模型 logits[:, :-1] 对齐——这次模型 forward 用 input_ids 全序列，不需要 labels 参数，自己 shift） |
| 121-125 | 与 GRPO 相同的 `logp_pos`（= prompt_len−1+r）与 scatter 修正（详见 GRPO 篇 Step 3） |
| 126-128 | **响应有效长度**：pad 长度 → 若有 EOS 则截到 EOS 位置 +1（含 EOS）→ clamp 至少 1 |
| 129-130 | `resp_policy_mask`：policy/value 共用的有效区掩码（EOS 前含 EOS）——比 GRPO 版多算了一步显式长度 |

**Step 3：旧价值 + GAE（no_grad 大块）**

```python
        with torch.no_grad():  # Rollout阶段只需推理获取old_logp和old_values，切断梯度省显存
            critic_for_rollout = critic_model.module if isinstance(critic_model, DistributedDataParallel) else critic_model
            values_seq = critic_for_rollout(input_ids=gen_out, attention_mask=full_mask)
            old_resp_values = values_seq.gather(1, logp_pos) * resp_value_mask
            
            ref_resp_logp = F.log_softmax(ref_model(input_ids=gen_out, attention_mask=full_mask).logits[:, :-1], dim=-1).gather(2, labels.unsqueeze(-1)).squeeze(-1).gather(1, logp_pos)
            token_rewards = torch.zeros_like(old_resp_logp)
            last_idx = resp_lengths - 1  # [B]
            token_rewards[torch.arange(B, device=args.device)[valid_resp], last_idx[valid_resp]] += rewards[valid_resp]  # 末尾加外部奖励

            gen_len = old_resp_values.size(1); lastgaelam = torch.zeros(B, device=args.device); advs_rev = []
            for t in reversed(range(gen_len)):
                nv = old_resp_values[:, t + 1] if t < gen_len - 1 else 0.0
                delta = token_rewards[:, t] + args.gamma * nv - old_resp_values[:, t]
                lastgaelam = delta + args.gamma * args.lam * lastgaelam
                advs_rev.append(lastgaelam)
            advantages = torch.stack(advs_rev[::-1], dim=1)  # [B, R]
            returns = advantages + old_resp_values  # [B, R]

            adv_mean = (advantages * resp_policy_mask).sum() / resp_policy_mask.sum().clamp(min=1)
            adv_var = ((advantages - adv_mean) ** 2 * resp_policy_mask).sum() / resp_policy_mask.sum().clamp(min=1)
            advantages = (advantages - adv_mean) * torch.rsqrt(adv_var + 1e-8) * resp_policy_mask
```

| 行 | 讲解 |
|----|------|
| 132-135 | **旧价值**：critic 对整个序列输出每位置价值 → `gather(1, logp_pos)` 只取生成区 `[B, R]`（×mask 清 pad）。no_grad：旧值是常数基线 |
| 137 | 参考模型 logp（同 GRPO 的两段 gather；KL 惩罚用） |
| 138-140 | **token 级奖励**：外部奖励是**整条回答一个分数**，只在**最后一个有效 token**（EOS 或末位）位置发放——中间 token 奖励 0。这是「回合制奖励」的标准稀疏化（γ=1.0 时 GAE 会把这个分数按 (γλ)^l 衰减回传到每个 token） |
| 142-149 | **GAE 反向递推**（§1.2 公式的逐行实现）：<br>• `nv`：下一位置的价值，最后一位用 0（回合结束无未来）；<br>• `delta`：TD 残差 = r + γV(t+1) − V(t)；<br>• `lastgaelam = delta + γλ·lastgaelam`：累积项；<br>• `advs_rev` 是倒序收集，`[::-1]` 翻正后 stack 成 `[B, R]`；<br>• `returns = A + V`：价值网络的学习目标（回报估计） |
| 151-153 | **优势全局标准化**（batch 内全部有效 token 一起归一）——注意与 GRPO 的「组内」不同：PPO 有 Critic 基线，不需要组对比，这里做的是传统 PPO 的 advantage whitening。γ=1、λ=0.95 默认值 |

**Step 4：minibatch 多轮更新**

```python
        mb_size = max(1, min(args.mini_batch_size, B))
        stop_ppo = False
        ...
        for ppo_epoch in range(args.ppo_update_iters):
            if stop_ppo:
                break
            b_inds = torch.randperm(B, device=args.device)
            for i in range(0, B, mb_size):
                inds = b_inds[i:i + mb_size]
                
                mb_values_seq = critic_model(input_ids=gen_out[inds], attention_mask=full_mask[inds])
                mb_resp_values = mb_values_seq.gather(1, logp_pos[inds])

                with autocast_ctx:
                    res = actor_model(input_ids=gen_out[inds], attention_mask=full_mask[inds])
                    aux_loss = res.aux_loss if lm_config.use_moe else torch.tensor(0.0, device=args.device)
                    mb_resp_logp = F.log_softmax(res.logits[:, :-1], dim=-1).gather(2, labels[inds].unsqueeze(-1)).squeeze(-1).gather(1, logp_pos[inds])
```

| 行 | 讲解 |
|----|------|
| 155-156 | minibatch 大小；初始化早停旗与日志累加器 |
| 164-167 | **同一批 rollout 数据复用 2 轮**（ppo_update_iters），每轮内再随机切 minibatch——off-policy 复用是 PPO 比 GRPO 数据效率高的点 |
| 171-172 | critic 前向取**新**价值（要梯度——critic 在被训练） |
| 174-177 | actor 前向取**新** logp（autocast 内） |

**Step 5：KL 早停**

```python
                log_ratio = mb_resp_logp - old_resp_logp[inds]

                # 可开关的诊断：观察首轮首个 minibatch 的 mb 与 old logp 差异。
                if args.debug_log_ratio and ppo_epoch == 0 and i == 0 and is_main_process():
                    ...  # 打印 |log_ratio| 统计与 ratio 范围——验证「首轮 ratio≈1」（若不≈1 说明 old_logp 与 forward 不一致，常见于 dropout/精度问题）
                approx_kl = (0.5 * (log_ratio ** 2) * resp_policy_mask[inds]).sum() / resp_policy_mask[inds].sum().clamp(min=1)
                
                # 同步各卡的 approx_kl，防止某卡 break 而其它卡继续导致 DDP 死锁
                approx_kl_val = approx_kl.detach().clone()
                if dist.is_initialized():
                    dist.all_reduce(approx_kl_val, op=dist.ReduceOp.AVG)
                    
                if approx_kl_val > args.early_stop_kl:
                    stop_ppo = True
```

| 行 | 讲解 |
|----|------|
| 179 | `log_ratio = log π_θ − log π_old`（ratio 的对数形式，数值稳定） |
| 193 | **approx_kl**：用 k1 二次近似估计 KL(π_θ‖π_old) ≈ E[(log_ratio)²/2]（mask 平均）。>0.25 → 策略已偏离过远 → 早停 |
| 196-200 | **DDP 死锁预防**：各卡数据不同 → 各卡的 approx_kl 不同 → 若各自独立 stop，某卡 break 后不再参与后续 all-reduce，其他卡永久阻塞。对策：all_reduce 求平均，**全体用同一个值判定**，步调一致 |

**Step 6：损失**

```python
                ratio = torch.exp(log_ratio)
                clipfrac = ((((ratio - 1.0).abs() > args.clip_epsilon).float() * resp_policy_mask[inds]).sum()
                            / resp_policy_mask[inds].sum().clamp(min=1))
                kl_ref_penalty = ((torch.exp(ref_resp_logp[inds] - mb_resp_logp) - (ref_resp_logp[inds] - mb_resp_logp) - 1.0)
                                  * resp_policy_mask[inds]).sum() / resp_policy_mask[inds].sum().clamp(min=1)
                policy_loss = ((torch.max(-advantages[inds] * ratio,
                                          -advantages[inds] * torch.clamp(ratio, 1.0 - args.clip_epsilon, 1.0 + args.clip_epsilon))
                               * resp_policy_mask[inds]).sum() / resp_policy_mask[inds].sum().clamp(min=1)
                               + args.kl_coef * kl_ref_penalty)
                value_loss = 0.5 * (torch.max((mb_resp_values - returns[inds]) ** 2,
                                              (torch.clamp(mb_resp_values, old_resp_values[inds] - args.cliprange_value,
                                                           old_resp_values[inds] + args.cliprange_value) - returns[inds]) ** 2)
                                    * resp_value_mask[inds]).sum() / resp_value_mask[inds].sum().clamp(min=1)
```

| 行 | 讲解 |
|----|------|
| 203 | ratio = exp(log_ratio) |
| 204-205 | **clipfrac**：被 clip 触碰的比例（诊断量——长期很高说明步长太大） |
| 206-207 | 相对**参考模型**的 KL：k3 估计（同 GRPO），mask 平均。这是 RLHF 特有的「贴住 SFT 起点」惩罚 |
| 208-211 | **策略 clip 损失**：`max(−A·ratio, −A·clip(ratio))` 等价于 `−min(A·ratio, A·clip(ratio))`（§1.3），再加 `kl_coef×kl_ref`（PPO1/PPO2 风格的 KL 加在损失里；β=0.02 很小） |
| 212-215 | **价值 clip 损失**（§1.3 公式）：新旧价值差超 ±0.2 的部分不计；乘 ½（方差口径）；用 resp_value_mask。 |

**Step 7：早停的零损失闭环 + 更新**

```python
                kl = approx_kl_val
                kl_ref = kl_ref_penalty.detach()

                # 早停时必须保证 forward-backward 闭环，故只截断 loss 不中断 DDP 通信
                if stop_ppo:
                    loss = (policy_loss + args.vf_coef * value_loss + aux_loss) * 0.0
                else:
                    loss = (policy_loss + args.vf_coef * value_loss + aux_loss) / args.accumulation_steps
                
                loss.backward()
```

| 行 | 讲解 |
|----|------|
| 221-224 | **总损失** = policy_loss + 0.5×value_loss（vf_coef）+ aux_loss。早停分支 ×0.0：保留完整计算图（DDP 需要本 rank 参与 all-reduce），但梯度全零 → 等效「跳过」，通信不中断。**×0.0 而不是 break**，就是 §1.4 的关键设计 |
| 226 | backward（早停时反传 0 梯度） |

```python
                if grad_accum_step % args.accumulation_steps == 0:
                    clip_grad_norm_(actor_model.parameters(), args.grad_clip)
                    clip_grad_norm_(critic_model.parameters(), args.grad_clip)
                    actor_optimizer.step()
                    critic_optimizer.step()
                    actor_scheduler.step()
                    critic_scheduler.step()
                    actor_optimizer.zero_grad()
                    critic_optimizer.zero_grad()
```

- 双模型各自裁剪/步进/清零；scheduler 按 optimizer step 走（CosineAnnealingLR）。
- 循环外（248-256 行）兜底：累积不满一轮的尾巴也要更新（`grad_accum_step % N != 0` 时补一次）——防止最后几步梯度被丢弃。

**Step 8：日志/保存/引擎同步**

```python
        if step % args.save_interval == 0 or step == iters: rollout_engine.update_policy(actor_model)

        if is_main_process():
            ...  # critic_loss / reward / approx_kl / kl_ref / clipfrac / avg_len / 双 lr
        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            actor_model.eval()
            ...  # 存 actor（ppo_actor_768.pth）+ resume 包（kwargs: scheduler, critic_model, critic_optimizer, critic_scheduler）
            actor_model.train()
            del actor_state

        del enc, gen_out, ... # 一次删十几个大张量
```

- **只保存 actor**（推理只要策略）；resume 包通过 lm_checkpoint 的 kwargs 带上 **critic 全家**（模型/优化器/调度器）——续训后 Critic 状态完整。
- **update_policy**：rollout 引擎换到新 actor（同 GRPO）。

### 2.4 main（308-450 行）

关键差异行：

```python
    # ========== 5. 初始化模型和数据 ==========
    actor_model, tokenizer = init_model(lm_config, base_weight, device=args.device)     # Actor
    ref_model, _ = init_model(lm_config, base_weight, device=args.device)               # Reference（冻结）
    ref_model = ref_model.eval().requires_grad_(False)
    moe_suffix = '_moe' if lm_config.use_moe else ''
    ckp = f'{args.save_dir}/{base_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
    state_dict = torch.load(ckp, map_location=args.device)
    critic_model = CriticModel(lm_config)
    critic_model.load_state_dict(state_dict, strict=False)
    critic_model = critic_model.to(args.device)
```

| 行 | 讲解 |
|----|------|
| 381-383 | Actor 与 Ref 同源（full_sft） |
| 384-389 | **Critic 独立构建**：CriticModel 空模型 → **直接加载 full_sft 权重**（`strict=False`：value_head 是新参数没在 ckp 里，随机初始化；主干全部对上）。Critic 起点 = SFT 模型，value_head 从零学 |

```python
    actor_optimizer = optim.AdamW(actor_model.parameters(), lr=args.learning_rate)        # 3e-7
    critic_optimizer = optim.AdamW(critic_model.parameters(), lr=args.critic_learning_rate)  # 5e-7
    ...
    mb_factor = max(1, math.ceil(args.batch_size / args.mini_batch_size))
    total_optimizer_steps = math.ceil(iters * args.epochs * args.ppo_update_iters * mb_factor / args.accumulation_steps)
    actor_scheduler = CosineAnnealingLR(actor_optimizer, T_max=total_optimizer_steps, eta_min=args.learning_rate / 10)
    critic_scheduler = CosineAnnealingLR(critic_optimizer, T_max=total_optimizer_steps, eta_min=args.critic_learning_rate / 10)
```

- 双优化器、**critic lr=5e-7 比 actor 大约一倍**（价值头是随机初始化，需要更快收敛；主干部分本来也会被稍快带动）。
- 总步数估算把 **ppo_update_iters 与 minibatch 因子**乘进去（同一批数据会被多次 optimizer step，scheduler 的 T_max 要覆盖真实步数）。

```python
    if dist.is_initialized():
        # freqs_cos/freqs_sin 各 rank 由 config 确定性算出，默认每步广播一次纯属浪费
        actor_model = DistributedDataParallel(actor_model, device_ids=[local_rank], broadcast_buffers=False)
        critic_model = DistributedDataParallel(critic_model, device_ids=[local_rank], broadcast_buffers=False)
```

- actor 与 critic **分别 DDP 包装**（各自 all-reduce）。

## 3. PPO vs GRPO 速查

| | PPO（本篇） | GRPO（上篇） |
|--|-------------|--------------|
| 基线 | Critic 的 V(s)（GAE） | 组内均值 |
| 模型数 | actor + critic + ref + RM（4 份） | policy + ref + RM（3 份） |
| 采样 | 每 prompt 1 条 | 每 prompt G=6 条 |
| 数据复用 | 2 轮 × minibatch | 单轮单批 |
| 优势归一 | 全 batch whitening | 组内 (r−μ)/σ |
| KL | 损失内加 k3（coef 0.02）+ 早停 approx_kl | 损失内加 k3（β=0.1） |
| 价值损失 | 有（clip 版） | 无 |

## 4. 本篇小结

1. **Critic = 主干 + value_head**，从 full_sft 初始化；forward 只取隐状态不打词表。
2. GAE 反向递推一行差分 `delta = r + γV(t+1) − V(t)` 累积成优势；稀疏奖励放在序列最后一个 token。
3. **双 clip**（策略 + 价值）+ **双保险 KL**（损失内 k3 + approx_kl 早停）。
4. 分布式关键设计：approx_kl **all-reduce 后统一判定**；早停用 **loss×0.0 保持 forward-backward 闭环**防 DDP 死锁。
5. resume 包用 kwargs 扩展携带 critic 全家——lm_checkpoint 的通用性设计在此兑现。

下一篇：[08-AgenticRL-智能体强化学习](./08-AgenticRL-智能体强化学习.md)
