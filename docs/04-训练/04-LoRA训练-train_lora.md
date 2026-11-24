# 04 - LoRA 微调 train_lora.py 全文

> 覆盖代码：`trainer/train_lora.py` **全文 179 行**。
> 骨架同 [train_full_sft](./03-监督微调-train_full_sft.md)/[train_pretrain](./02-预训练-train_pretrain.md)，核心差异：**挂载 LoRA、冻结主干、只存 LoRA 权重、禁用 torch.compile**。
> LoRA 模块本身的实现见 [01-模型架构/09-LoRA原理与实现](../01-模型架构/09-LoRA原理与实现.md)。

---

## 1. 与 full SFT 的差异总表

| 维度 | full_sft | train_lora |
|------|----------|------------|
| 可训练参数 | 全部（约 64M） | 只有 LoRA 旁路（约 3-4M，占比 <6%） |
| 主干权重 | 被更新 | **冻结** |
| 数据 | sft_t2t | 领域数据（默认 lora_medical） |
| lr / epochs | 1e-5 / 2 | **1e-4 / 10**（参数少、任务窄，可以更激进） |
| 产物 | 整个模型 fp16 | **只有 LoRA 权重的小文件**（save_lora） |
| torch.compile | 可开 | **强制关闭**（monkey-patch forward 不兼容编译） |
| 断点续训 | strict=True 恢复 | `strict=False` 恢复（见 §3.3） |

## 2. 逐行精读（差异部分）

### 2.1 import（19 行）

```python
from model.model_lora import save_lora, apply_lora
```

引入两个关键函数：apply_lora（挂旁路+换 forward）、save_lora（只存 LoRA）。

### 2.2 train_epoch 签名与裁剪对象（25-46 行）

```python
def train_epoch(epoch, loader, iters, lora_params, start_step=0, wandb=None):
```

- 多了 `lora_params` 参数：**可训练参数的列表**（不是全模型参数）。

```python
        if step % args.accumulation_steps == 0 or step == iters:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(lora_params, args.grad_clip)   # ← 只裁 LoRA 参数
```

- 裁剪只针对 `lora_params`——冻结的参数根本没有梯度，裁全模型也没错，但传列表语义更精确。

```python
        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            lora_save_path = f'{args.save_dir}/{args.lora_name}_{lm_config.hidden_size}{moe_suffix}.pth'
            # LoRA只保存LoRA权重
            save_lora(model, lora_save_path)
            lm_checkpoint(lm_config, weight=args.lora_name, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')
            model.train()
```

- 保存分两份：
  1. `save_lora`：只含 A/B/scaling 的**LoRA 小文件**（约几 MB）——部署用（apply_lora+load_lora 或 merge_lora）；
  2. `lm_checkpoint`：resume 包里存的是**整个模型的 state_dict**（含主干+LoRA）——续训用 fp32 全状态。注意 weight 用 `args.lora_name`，checkpoint 文件名形如 `lora_medical_768_resume.pth`。

### 2.3 argparse 新参数（71-74 行）

```python
    parser.add_argument("--lora_rank", default=16, type=int, help="LoRA rank")
    parser.add_argument("--lora_alpha", default=None, type=float, help="LoRA alpha（默认等于 rank，即 scaling=1）")
    parser.add_argument("--lora_name", type=str, default="lora_medical", help="LoRA权重名称(如lora_identity/lora_medical等)")
```

| 参数 | 讲解 |
|------|------|
| lora_rank=16 | 低秩维度 r：越大容量越大、参数越多。常用 8-64 |
| lora_alpha=None | 缩放 α；None 时 =rank → scaling=1 |
| lora_name | LoRA 权重的名字（一个基模可挂多个不同任务的 LoRA，切换即换文件） |

其余参数：epochs=10、batch_size=32、lr=1e-4、from_weight=**full_sft**（LoRA 微调通常站在 SFT 之后）、data_path=lora_medical.jsonl、max_seq_len=340。

### 2.4 main：挂载与冻结（121-139 行）

```python
    # ========== 5. 定义模型、应用LoRA、冻结非LoRA参数 ==========
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    apply_lora(model, rank=args.lora_rank, lora_alpha=args.lora_alpha)
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    lora_params_count = sum(p.numel() for name, p in model.named_parameters() if 'lora' in name)
    Logger(f"LLM 总参数量: {total_params / 1e6:.3f} M")
    Logger(f"LoRA 参数量: {lora_params_count / 1e6:.3f} M")
    Logger(f"LoRA 参数占比: {lora_params_count / total_params * 100:.2f}%")
    
    # 冻结非LoRA参数，收集LoRA参数
    lora_params = []
    for name, param in model.named_parameters():
        if 'lora' in name:
            param.requires_grad = True
            lora_params.append(param)
        else:
            param.requires_grad = False
```

| 行 | 讲解 |
|----|------|
| 122-123 | 先加载基模权重，**再** apply_lora——旁路 A/B 是新初始化的（A 高斯、B 零），主干带着 full_sft 的能力 |
| 126-130 | 参数统计打印。Dense 模型 r=16 时 LoRA ≈ 32 个矩阵 × (768×16+16×768) ≈ 3.1M，占比约 4-5% |
| 133-139 | **冻结/解冻**：键名含 'lora' 的参数 `requires_grad=True` 并收进列表；其余全部 False。**requires_grad=False 的参数**：backward 不为它计算/存储梯度，optimizer 也不该收到它（下一步只把 lora_params 给 AdamW） |

```python
    # ========== 6. 定义数据和优化器 ==========
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)   # ← 还是 SFT 格式数据
    ...
    optimizer = optim.AdamW(lora_params, lr=args.learning_rate)                     # ← 只优化 LoRA 参数
```

- 数据仍是 SFT 格式（领域对话）；**优化器只装 lora_params**——这是显存大幅下降的来源：AdamW 每参数要存 m/v 两个状态，64M 参数 vs 3M 参数差 20 倍状态显存。

### 2.5 续训恢复的 strict=False（149-150 行）

```python
    if ckp_data:
        model.load_state_dict(ckp_data['model'], strict=False)
```

- resume 包里的模型 state_dict 含主干+LoRA 全部键。**为什么要 strict=False**：如果 LoRA 结构参数（rank）与 checkpoint 不一致，或 checkpoint 保存时含某些不匹配键，严格模式会直接抛错；宽松模式允许部分加载。代价是键名打错也悄悄放过——续训后务必看 loss 是否连续。

### 2.6 torch.compile 禁用（157-159 行）

```python
    if args.use_compile == 1:
        args.use_compile = 0
        Logger('[LoRA] monkey-patch forward 与 torch.compile 不兼容，use_compile 已自动关闭')
```

- apply_lora 用 **monkey-patch 替换了 Linear.forward**（Python 层函数替换）。torch.compile 需要追踪模块结构生成静态图，被替换的 forward 带闭包变量（layer1/layer2），dyno 追踪会失败或行为异常。防御式处理：用户开了也自动关掉。

### 2.7 训练循环调用（172-174 行）

```python
            train_epoch(epoch, loader, len(loader) + skip, lora_params, start_step, wandb)
```

- 把 `lora_params` 传入（裁剪用）。其余与 pretrain 一致。

## 3. 训练前后怎么用 LoRA

```bash
# 训练
python trainer/train_lora.py --lora_name lora_medical --data_path ../dataset/lora_medical.jsonl

# 方式 A：推理时挂载（eval_llm.py）
python eval_llm.py --weight full_sft --lora_weight lora_medical

# 方式 B：合并进主干（convert_model.py）
python scripts/convert_model.py   # 调用 convert_merge_base_lora
```

## 4. 本篇小结

1. LoRA 训练 = SFT 骨架 + **apply_lora 挂旁路 + 按 `'lora' in name` 冻结主干** + **优化器只装旁路参数**。
2. 产物是 LoRA 小文件（可多任务并存）；续训包仍是全状态（strict=False 宽松加载）。
3. compile 被自动关闭——monkey-patch 与静态图编译冲突，这是 LoRA「旁路注入」的实现代价。
4. lr 1e-4 / 10 epochs 比全量微调激进：可训练参数少且任务窄，收敛需要更多 epoch。

下一篇：[05-DPO-直接偏好优化](./05-DPO-直接偏好优化.md)
