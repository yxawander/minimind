# 09 - 知识蒸馏 train_distillation.py 全文

> 覆盖代码：`trainer/train_distillation.py` **全文 241 行**。
> 蒸馏：让**学生模型**同时学习「数据的答案（硬标签）」和「**教师模型的输出分布（软标签）**」——教师不仅告诉学生「正确答案是什么」，还告诉它「每个候选词的合理程度分布」，暗知识（dark knowledge）随分布一起传递。
> 默认场景：**MoE 教师（大）→ Dense 学生（小）**（脚本第 140 行注释），也可以是任意两个规格差。
> 骨架同 [pretrain](./02-预训练-train_pretrain.md)，核心差异在损失函数。

---

## 1. 原理

### 1.1 为什么软标签有用

数据标签「2**10 = 1024」只说「1024 是对的，其余全错」；教师分布可能是：

```
1024: 0.72   1023: 0.09   1025: 0.08   2048: 0.03   ...
```

「1023/1025 与正确答案很接近」「2048 是另一种常见误解」——这些**类间相似度信息**就是暗知识，学生能从中学到「数值相近的 token 语义相近」等结构，往往比纯硬标签收敛更快、泛化更好。

### 1.2 温度蒸馏（Hinton 2015）

直接用教师 logits 做 KL，分布太尖锐（大模型 logits 尺度大），暗知识全挤在 top-1。**温度 T** 平滑分布：

$$
p_i^{(T)} = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)},\qquad
\mathcal{L}_{distill} = T^2 \cdot \mathrm{KL}\big(p^{(T)}_{\text{teacher}} \,\|\, p^{(T)}_{\text{student}}\big)
$$

- T>1 → 分布变平，top-2/top-3 的相对关系被放大出来；
- **乘 T²** 补偿梯度量级（T 改变后 KL 的梯度缩放约 1/T²，Hinton 论文的推导，保证 T 可调而损失量级稳定）；
- T=1 时退化为普通 KL。

### 1.3 混合损失

$$
\mathcal{L} = \alpha \cdot \mathcal{L}_{CE}(\text{学生 vs 硬标签}) + (1-\alpha) \cdot \mathcal{L}_{distill}(\text{学生 vs 教师})
$$

α=0.5（默认）：一半跟数据走（保底正确性），一半跟教师走（学结构）。α=0 纯蒸馏、α=1 等价普通 SFT（教师白算）。

---

## 2. 逐行精读

### 2.1 distillation_loss（25-38 行）

```python
def distillation_loss(student_logits, teacher_logits, temperature=1.0, reduction='batchmean'):
    if student_logits.size(0) == 0 and reduction == 'batchmean':
        return student_logits.sum() * 0.0
    with torch.no_grad():
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1).detach()

    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)

    kl = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction=reduction
    )
    return (temperature ** 2) * kl
```

| 行 | 讲解 |
|----|------|
| 26-27 | 空输入守卫：loss_mask 筛完可能 0 行（整批样本 mask 全空）；`sum()*0.0` 返回「有计算图的 0」——**不能返回常数 0.0**，否则 autograd 报「张量不接图」或 DDP 因缺梯度挂掉（与 MoE 空专家补丁同一动机） |
| 28-29 | **教师概率**：logits/T → softmax → **显式 detach**（no_grad 也包了，双保险）——教师是目标不是被优化对象 |
| 31 | 学生侧用 log_softmax（KL 的对数部分），同样除温度 |
| 33-37 | `F.kl_div(input=log p_student, target=p_teacher)`：注意 PyTorch 的参数顺序是反直觉的——**第一个参数是学生的 log 概率，第二个是教师的概率**（KL(p‖q) 的 p 是 target）。`reduction='batchmean'`：对 batch 维求平均（KL 散度的数学正确平均方式；'mean' 会错除以元素数） |
| 38 | **乘 T²**（§1.2 的梯度补偿） |

### 2.2 train_epoch 的损失块（41-93 行）

```python
def train_epoch(epoch, loader, iters, teacher_model, lm_config_student, start_step=0, wandb=None, alpha=0.0, temperature=1.0):
    start_time = time.time()
    
    if teacher_model is not None:
        teacher_model.eval()
        teacher_model.requires_grad_(False)

    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        loss_mask = (labels[..., 1:] != -100).float()
```

| 行 | 讲解 |
|----|------|
| 44-46 | **教师进训练循环前**先 eval + 冻结（main 里也做过，双保险） |
| 48-51 | 数据仍是 SFTDataset 的 (input_ids, labels)；**loss_mask 从 labels 反推**：labels[1:] != -100 的位置=1——即「assistant 回复区」。**蒸馏也遵守 loss mask**：只在回答部分让学生对齐教师（prompt 部分教师和学生的分布差异没有学习意义） |

```python
        # 前向传播（学生模型）
        with autocast_ctx:
            res = model(input_ids)
            student_logits = res.logits[..., :-1, :].contiguous()

        # 教师模型前向传播（只在eval & no_grad）
        if teacher_model is not None:
            with torch.no_grad():
                teacher_logits = teacher_model(input_ids).logits[..., :-1, :].contiguous()
                vocab_size_student = student_logits.size(-1)
                teacher_logits = teacher_logits[..., :vocab_size_student]
```

| 行 | 讲解 |
|----|------|
| 57-59 | 学生前向（**不带 labels**——CE 损失自己算，因为还要 mask 加权），logits 去尾巴对齐 |
| 62-66 | 教师前向（no_grad）；**词表截齐**：若教师词表 > 学生（跨模型蒸馏的通用场景），教师 logits 截到学生词表长度——本仓库两个模型同 tokenizer 同 6400，此行是通用性保险 |

```python
        # ========== 计算损失 ==========
        # 1) Ground-Truth CE Loss
        shift_labels = labels[..., 1:].contiguous()
        loss_mask_flat = loss_mask.view(-1)
        ce_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction='none'
        )
        ce_loss_raw = torch.sum(ce_loss * loss_mask_flat) / (loss_mask_flat.sum() + 1e-8)
        if lm_config_student.use_moe: ce_loss = ce_loss_raw + res.aux_loss
        else: ce_loss = ce_loss_raw
```

| 行 | 讲解 |
|----|------|
| 70-77 | **CE 部分**：`reduction='none'` 拿到**逐 token** 损失（模型内置的 loss 是全序列平均，这里要 mask 加权，所以自己算）；乘 mask、除 mask 总和（**只在有效 token 上平均** +1e-8 防零）——比模型内置版本多出「mask 平均」的精确性 |
| 79-80 | MoE 学生加 aux_loss；dense 不加。注意此时 aux_loss 还没被除 accumulation（后面统一除） |

```python
        # 2) Distillation Loss
        if teacher_model is not None:
            distill_loss = distillation_loss(
                student_logits.view(-1, student_logits.size(-1))[loss_mask_flat == 1],
                teacher_logits.view(-1, teacher_logits.size(-1))[loss_mask_flat == 1],
                temperature=temperature
            )
        else:
            distill_loss = torch.tensor(0.0, device=args.device)

        # 3) 总损失 = alpha * CE + (1-alpha) * Distill
        loss = (alpha * ce_loss + (1 - alpha) * distill_loss) / args.accumulation_steps
```

| 行 | 讲解 |
|----|------|
| 83-90 | **蒸馏部分**：布尔索引 `[..., loss_mask_flat == 1]` 把**有效 token 的行**抽出来做 KL（prompt/pad 行不参与）。注意 mask 是 shift 后的长度（labels[...,1:] 与 logits[..., :-1] 天然对齐） |
| 93 | 三损失合成：α 混合 → 除累积步数。后续 backward/更新/日志/保存与 pretrain 骨架一致 |

日志行（104-122 行）把 ce/aux/distill 分列打印——蒸馏训练必须盯两个损失的相对变化（若 distill 不降而 ce 降，可能是教师词表/温度配置错了）。

### 2.3 main 差异（139-236 行）

argparse 特有参数（140-169 行）：

| 参数 | 默认 | 讲解 |
|------|------|------|
| student_hidden_size/layers | 768/8 | 学生规格 |
| teacher_hidden_size/layers | 768/8 | 教师规格（可设 1024/16 等更大） |
| student_use_moe / teacher_use_moe | **0 / 1** | 默认「MoE 教师 → Dense 学生」 |
| from_student_weight | full_sft | 学生起点 |
| from_teacher_weight | full_sft | 教师起点（官方用法：MoE 的 full_sft_moe 权重当教师） |
| **alpha** | 0.5 | CE 权重 |
| **temperature** | 1.5 | 蒸馏温度（推荐 1.0-2.0） |
| save_weight | full_dist | 产物名 |

```python
    lm_config_student = MiniMindConfig(hidden_size=args.student_hidden_size, num_hidden_layers=args.student_num_layers, use_moe=bool(args.student_use_moe))
    lm_config_teacher = MiniMindConfig(hidden_size=args.teacher_hidden_size, num_hidden_layers=args.teacher_num_layers, use_moe=bool(args.teacher_use_moe))
```

- **两个 config**：学生与教师独立配置；续训 checkpoint 按学生 config 查找。

```python
    # ========== 5. 定义学生和教师模型 ==========
    model, tokenizer = init_model(lm_config_student, args.from_student_weight, device=args.device)
    ...
    teacher_model, _ = init_model(lm_config_teacher, args.from_teacher_weight, device=args.device)
    teacher_model.eval()
    teacher_model.requires_grad_(False)
```

- 学生进 optimizer；教师 eval+冻结。**注意教师权重文件名按教师 config 生成**（如 `full_sft_768_moe.pth`）——MoE 教师要先用 `--use_moe 1` 训出来。

## 3. 数值例子：温度在做什么

教师 logits（某位置 top-5）：`[8.0, 6.5, 6.4, 3.0, 1.0]`

- T=1：softmax ≈ [0.83, 0.15, 0.14, 0.006, 0.0001] ——几乎只剩 top-1；
- T=1.5：÷1.5 后 ≈ [0.55, 0.25, 0.23, 0.02, 0.002] ——次优候选的份额被放大，暗知识显形。

## 4. 本篇小结

1. 损失 = **α·CE(硬标签) + (1−α)·T²·KL(教师‖学生)**，两者都只在 loss mask 区间计算。
2. `F.kl_div(log 学生, 教师)` 参数顺序易错；`reduction='batchmean'` 才是数学正确的 KL 平均；**乘 T²** 补梯度。
3. 空行守卫（`sum()*0.0`）与教师 detach 是两个必踩的坑。
4. 词表截齐（教师→学生长度）让跨 tokenizer/规格蒸馏也安全。

下一篇：[10-rollout引擎](./10-rollout引擎.md)
