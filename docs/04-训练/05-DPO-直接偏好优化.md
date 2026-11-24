# 05 - DPO 直接偏好优化 train_dpo.py 全文

> 覆盖代码：`trainer/train_dpo.py` **全文 219 行**。
> DPO（Direct Preference Optimization, 2023）是 RLHF 的「免 RL」替代：不需要奖励模型、不需要采样与策略梯度，直接用**偏好对数据 + 一个冻结的参考模型**做监督式更新。本篇先完整推导 DPO 损失，再逐行读代码。
> 骨架（import/混合精度/保存/DDP）与 [pretrain](./02-预训练-train_pretrain.md) 一致，不再重复。

---

## 1. 原理：从 RLHF 到 DPO 的推导

### 1.1 RLHF 的标准形式

RLHF 目标（KL 约束的奖励最大化）：

$$
\max_\pi \; \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi}\Big[r(x,y)\Big] - \beta\, \mathbb{D}_{\mathrm{KL}}\big[\pi(y|x)\,\|\,\pi_{\mathrm{ref}}(y|x)\big]
$$

- π 是策略模型（正在训练的）；π_ref 是参考模型（SFT 权重的冻结副本，作为「不许漂太远」的锚）；
- β 控制两者张力：β 大→贴紧参考模型，β 小→放飞。

该目标有**解析最优解**：

$$
\pi^*(y|x) = \frac{1}{Z(x)}\,\pi_{\mathrm{ref}}(y|x)\,\exp\!\Big(\frac{r(x,y)}{\beta}\Big)
$$

反解奖励：

$$
r(x,y) = \beta \log \frac{\pi^*(y|x)}{\pi_{\mathrm{ref}}(y|x)} + \beta \log Z(x)
$$

**关键一步**：把 r 代入Bradley-Terry 偏好模型（人类偏好 y_w ≻ y_l 的概率）：

$$
p(y_w \succ y_l) = \sigma\big(r(x,y_w) - r(x,y_l)\big)
$$

两个 y 的 r 相减时，**log Z(x)（配分函数，算不出来）被抵消**！于是奖励模型可以被「策略比值」直接替换：

$$
p(y_w \succ y_l \mid x) = \sigma\!\left(\beta \log\frac{\pi(y_w|x)}{\pi_{\mathrm{ref}}(y_w|x)} - \beta \log\frac{\pi(y_l|x)}{\pi_{\mathrm{ref}}(y_l|x)}\right)
$$

对这批偏好数据做极大似然，得到 **DPO 损失**：

$$
\mathcal{L}_{\mathrm{DPO}} = -\,\mathbb{E}\Big[\log \sigma\big(\beta (\underbrace{\log\tfrac{\pi(y_w|x)}{\pi_{\mathrm{ref}}(y_w|x)}}_{\text{chosen 优势}} - \underbrace{\log\tfrac{\pi(y_l|x)}{\pi_{\mathrm{ref}}(y_l|x)}}_{\text{rejected 优势}})\big)\Big]
$$

直觉：**拉大「chosen 相对参考模型的优势」与「rejected 相对参考模型的优势」之差**——把 chosen 推得更可能、rejected 推得更不可能，同时被 β·log(π/π_ref) 的锚拴住不漂移。

### 1.2 与 PPO/GRPO 的对比

| | DPO | GRPO/PPO |
|--|-----|----------|
| 需要奖励模型 | 否（偏好标签隐式给了奖励） | 是（RM 打分） |
| 需要采样 | 否（直接用数据） | 是（rollout） |
| 显存 | 双模型（policy+ref） | 双模型+（PPO 还有 critic） |
| 数据 | 离线偏好对 | 在线生成 |
| 表达力 | 受限于数据分布 | 可探索新行为 |

## 2. 逐行精读

### 2.1 logits_to_log_probs（25-31 行）

```python
def logits_to_log_probs(logits, labels):
    # logits shape: (batch_size, seq_len, vocab_size)
    # labels shape: (batch_size, seq_len)
    # log_probs shape: (batch_size, seq_len)
    log_probs = F.log_softmax(logits, dim=2)
    log_probs_per_token = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)
    return log_probs_per_token
```

| 行 | 讲解 |
|----|------|
| 29 | logits softmax 取 log → 每个 token 位置上**全词表的对数概率** `[B, S, V]` |
| 30 | `gather`：按 labels（把每个位置要查的 id 展成 `[B, S, 1]`）从 log_probs 里**抽出实际 token 的 logp** → `[B, S, 1]` → squeeze → `[B, S]`。<br>**gather 是「按下标取值」的可导查表**：梯度只流向被选中的那个 logit——这正是策略梯度需要的（其他 token 的概率变化不影响已选 token 的 logp） |

### 2.2 dpo_loss（34-50 行）

```python
def dpo_loss(ref_log_probs, policy_log_probs, mask, beta):
    # ref_log_probs 和 policy_log_probs 都是 shape: (batch_size, seq_len)
    ref_log_probs = (ref_log_probs * mask).sum(dim=1)
    policy_log_probs = (policy_log_probs * mask).sum(dim=1)

    # 将 chosen 和 rejected 数据分开
    batch_size = ref_log_probs.shape[0]
    chosen_ref_log_probs = ref_log_probs[:batch_size // 2]
    reject_ref_log_probs = ref_log_probs[batch_size // 2:]
    chosen_policy_log_probs = policy_log_probs[:batch_size // 2]
    reject_policy_log_probs = policy_log_probs[batch_size // 2:]

    pi_logratios = chosen_policy_log_probs - reject_policy_log_probs
    ref_logratios = chosen_ref_log_probs - reject_ref_log_probs
    logits = pi_logratios - ref_logratios
    loss = -F.logsigmoid(beta * logits)
    return loss.mean()
```

| 行 | 讲解 |
|----|------|
| 36-37 | **序列 logp 求和**：逐 token logp 乘 mask（assistant 区间=1）后按序列维求和 → 每条序列的总 logp（假设 token 独立，序列概率 = ∏p，logp = Σ logp）。mask 保证只算回复段 |
| 40-44 | 数据排布约定：DataLoader 的 batch 是 `[chosen 半批; rejected 半批]`（DPODataset 输出六键、train_epoch 里 cat 拼的），前半 chosen、后半 rejected，各取 `batch_size//2` |
| 46 | `pi_logratios = log π(y_w) − log π(y_l)`：**策略模型**里 chosen 与 rejected 的 logp 之差 |
| 47 | `ref_logratios`：参考模型的同款差值 |
| 48 | `logits = pi_logratios − ref_logratios`：展开就是 β 里面那个「优势差 − 优势差」（β 在下一行乘）。注意其中 **ref_logratios 是常数**（ref 冻结、no_grad 前向），梯度全来自 policy 项——但它**不能省略**：它把「绝对 logp 差」变成「相对参考模型的优势差」，这正是防漂移的锚 |
| 49 | `−log σ(β·logits)`：Bradley-Terry 下的负对数似然。`logsigmoid` 比 `log(sigmoid)` 数值稳定（大 logits 不溢出） |
| 50 | batch 平均 |

**导数直觉**：∂loss/∂logits = −β·σ(−β·logits)。当 chosen 明显优于 rejected（logits 很大）时 σ→0、梯度→0——**已经分对的不继续推**；分错的样本梯度最大。这是 logistic 损失的自我平衡性质。

### 2.3 train_epoch（53-119 行）

```python
def train_epoch(epoch, loader, iters, ref_model, lm_config, start_step=0, wandb=None, beta=0.1):
    start_time = time.time()

    for step, batch in enumerate(loader, start=start_step + 1):
        x_chosen = batch['x_chosen'].to(args.device)
        ...（六个键全部搬 GPU）
        x = torch.cat([x_chosen, x_rejected], dim=0)
        y = torch.cat([y_chosen, y_rejected], dim=0)
        mask = torch.cat([mask_chosen, mask_rejected], dim=0)
```

| 行 | 讲解 |
|----|------|
| 57-65 | 六键拆包搬设备；**chosen 与 rejected 沿 batch 维拼接**成 2B 的大 batch——一次 forward 同时算两类，ref/policy 各一次前向就够（而非四次） |

```python
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            with torch.no_grad():
                ref_outputs = ref_model(x)
                ref_logits = ref_outputs.logits
            ref_log_probs = logits_to_log_probs(ref_logits, y)
            
            outputs = model(x)
            logits = outputs.logits
            policy_log_probs = logits_to_log_probs(logits, y)
            
            dpo_loss_val = dpo_loss(ref_log_probs, policy_log_probs, mask, beta=beta)
            loss = dpo_loss_val + outputs.aux_loss
            loss = loss / args.accumulation_steps
```

| 行 | 讲解 |
|----|------|
| 67-69 | 与 pretrain 相同的手动余弦 lr（lr=4e-8 起步） |
| 71-74 | **参考模型前向**：`torch.no_grad()` 包住（省显存——不需要它的计算图）；同一个大 batch 过 ref。注意 DPODataset 已做好 shift（x=ids[:-1], y=ids[1:]），所以 logits 与 y 天然对齐，不需要模型内部 labels 的 shift |
| 75 | 抽 ref 的逐 token logp（no_grad 下是常数张量） |
| 77-79 | 策略模型前向（**要梯度**）+ 抽 logp |
| 81-83 | DPO 损失 + MoE aux loss + 累积除法。aux_loss 是 MoE 负载均衡（dense 为 0）——路由器也参与 DPO 更新 |

```python
        scaler.scale(loss).backward()

        if step % args.accumulation_steps == 0 or step == iters:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
```

与 pretrain 相同的 AMP/裁剪/更新五连（注意只更新 `model`——ref_model 从未进 optimizer）。

```python
        if step % args.log_interval == 0 or step == iters:
            ...
            current_dpo_loss = dpo_loss_val.item()
            current_aux_loss = outputs.aux_loss.item()
            ...
            Logger(f'... loss: ..., dpo_loss: ..., aux_loss: ..., learning_rate: ..., epoch_time: ...')
            if wandb: wandb.log({...})
```

日志分列：总 loss、dpo_loss、aux_loss。

```python
        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            ...  # 保存 dpo_768.pth（fp16 全模型）+ checkpoints/dpo_768_resume.pth（fp32 全状态）
            del state_dict

        del x_chosen, x_rejected, y_chosen, y_rejected, mask_chosen, mask_rejected, x, y, mask
        del ref_outputs, ref_logits, ref_log_probs, outputs, logits, policy_log_probs, loss
```

- 保存逻辑同 pretrain（save_weight=**dpo**）。
- 最后两行显式 del **两倍显存的大对象**（DPO 的 batch 是 2B×4096 token，双模型前向的 logits 都是巨型张量 `[8, 4095, 6400]` fp16 就约 400MB，及时释放关键）。

### 2.4 main 的差异（122-219 行）

argparse 差异行：

| 行 | 值 | 讲解 |
|----|-----|------|
| 126 | save_weight=**dpo** | 产物名 |
| 127 | epochs=**1** | 一轮足够（偏好信号在数据里，多轮易过拟合崩溃） |
| 128 | batch_size=**4**（每卡） | 实际 batch=8（chosen+rejected 拼接） |
| 128 | learning_rate=**4e-8**（help 注明 ≤5e-8 防遗忘） | DPO 的 lr 敏感区极窄：大了模型直接崩坏（输出乱码），因为损失鼓励「拉大 logp 差」而没有词级约束 |
| 138 | max_seq_len=**1024** | 对话更长 |
| 140 | data_path=**dpo.jsonl** | 偏好对数据 |
| 141 | from_weight=**full_sft** | 权重链 |
| 143 | beta=**0.15** | KL 锚强度（原论文 0.1） |

main 流程差异（173-185 行）：

```python
    # ========== 5. 定义模型和参考模型 ==========
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    Logger(f'策略模型总参数量：...')
    # 初始化参考模型（ref_model冻结）
    ref_model, _ = init_model(lm_config, args.from_weight, device=args.device)
    ref_model.eval()
    ref_model.requires_grad_(False)
    Logger(f'参考模型总参数量：...')
```

| 行 | 讲解 |
|----|------|
| 174-175 | 策略模型：加载 full_sft 权重，可训练 |
| 177-180 | **参考模型**：**从同一份 full_sft 权重再加载一份**（`args.from_weight` 相同）→ eval 模式（关 dropout）→ `requires_grad_(False)` 冻结全部参数。**ref 永远不更新**，它是 DPO 公式里的 π_ref 锚 |

```python
    train_ds = DPODataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    ...
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
```

- 只有 policy 进 optimizer（180 行）；DPODataset 提供 x/y/mask 六键。
- 训练循环（204-214 行）与 pretrain 相同（含 SkipBatchSampler 续训），传参多了 ref_model 与 args.beta。

## 3. 训练观察点

- **健康信号**：dpo_loss 缓慢下降；reward margin（logits）上升；生成文本仍然通顺。
- **崩溃信号**：loss 不降反升、输出乱码——lr 太大，降到 2e-8 重试。
- **经济性**：DPO 显存 ≈ 2×模型（policy+ref），但无 rollout、无 RM，比 PPO/GRPO 便宜得多——工业界 RLHF 阶段常用 DPO 打底。

## 4. 本篇小结

1. DPO = 用**偏好对 + 冻结参考模型**做监督式对齐；损失是 `-log σ(β·[(logπ_w−logπ_ref_w) − (logπ_l−logπ_ref_l)])`。
2. 实现三板斧：**成对拼接 forward（2B）→ gather 抽 logp → mask 求和后算优势差**。
3. ref_model 与 policy 同权重起点、永不更新；β 控制锚强度；lr 必须 ≤5e-8。
4. `logits_to_log_probs` 的 gather 是「可导查表」，梯度只流向被选 token——后续 GRPO/PPO 反复复用这个思想。

下一篇：[06-GRPO-组相对策略优化](./06-GRPO-组相对策略优化.md)
