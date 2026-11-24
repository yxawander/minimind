# 03 - 监督微调 train_full_sft.py 全文

> 覆盖代码：`trainer/train_full_sft.py` **全文 165 行**。
> 与 [train_pretrain.py](./02-预训练-train_pretrain.md) 的骨架**逐行相同**（import/argparse/main 流程），本篇讲清「SFT 到底改了什么」+ 两处与 pretrain 不一样的细节。

---

## 1. 原理：SFT 与预训练的三点差异

| 维度 | 预训练 | SFT |
|------|--------|-----|
| 数据 | `{"text": ...}` 纯文本 | `{"conversations": [...]}` 多轮对话 |
| loss 范围 | **全文**（含 eos） | **仅 assistant 回复段**（含 `<|im_end|>`），mask 由 `SFTDataset.generate_labels` 生成 |
| 权重起点 | 随机（from_weight=none） | **pretrain 权重**（from_weight=pretrain） |
| lr | 5e-4，累积 8 | **1e-5**，累积 1 |
| max_seq_len | 340 | **768**（对话更长） |

「为什么只对回复段算 loss」在 [03-数据集/01](../03-数据集/01-数据集总览.md#3-核心机制loss-mask哪些-token-参与训练) 已讲：防止模型学成「复述用户输入的复读机」，并保护预训练已获得的语言能力（防灾难性遗忘）——lr 降到 1e-5 也是同一目的。

**"full_sft" 的 "full"**：全参数微调（对比 LoRA 只训旁路）。

## 2. 与 pretrain 逐行的差异清单

下面只列**代码实际不同**的行，其余行号一一对应、含义相同（见预训练篇）。

### 2.1 import 差异（18 行）

```python
from dataset.lm_dataset import SFTDataset        # ← 换成 SFTDataset（pretrain 是 PretrainDataset）
```

### 2.2 train_epoch 内部：零改动

前向、损失合成、反向、更新、日志、保存**全部相同**——因为 loss mask 的魔法已经在 `labels` 里（-100 处由 `F.cross_entropy(ignore_index=-100)` 自动跳过），训练循环无需感知。

### 2.3 argparse 差异（76-99 行）

| 行 | pretrain | full_sft | 为什么 |
|----|----------|----------|--------|
| 76 | "MiniMind Pretraining" | "MiniMind Full SFT" | 描述 |
| 78 | save_weight=pretrain | **save_weight=full_sft** | 产物名 `out/full_sft_768.pth` |
| 80 | batch_size=32 | **16** | 对话样本 768 token 更长，显存换 batch |
| 81 | lr=5e-4 | **1e-5** | 微调小步幅 |
| 85 | accumulation_steps=8 | **1** | 有效 batch 16 已经够（预训练需要大 batch 学统计规律，SFT 样本信号更强） |
| 91 | max_seq_len=340 | **768** | 对话+模板更长 |
| 94 | data_path=pretrain_t2t_mini | **sft_t2t_mini** | 换数据 |
| 95 | from_weight=none | **pretrain** | **站在预训练权重上继续**（权重链） |
| 98 | wandb_project=Pretrain | Full-SFT | 曲线归属 |

### 2.4 main 差异（126-128 行）

```python
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)   # ← from_weight='pretrain'：加载 out/pretrain_768.pth
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)     # ← 换数据集
```

其余（seed/config/autocast/wandb/续训恢复/compile/DDP/训练循环/清理）与 pretrain 完全一致，不再重复。

## 3. 一次 SFT step 的完整数据流（串联前后篇）

```
jsonl 样本
 └→ SFTDataset.__getitem__
     ├→ pre_processing_chat        （20% 概率加随机 system）
     ├→ create_chat_prompt         （apply_chat_template 渲染）
     ├→ post_processing_chat       （80% 概率删空思考段）
     ├→ tokenizer 编码 + pad 到 768
     └→ generate_labels            （assistant 区间=token，其余=-100）
          ↓ [B,768] × 2
train_epoch
     ├→ model(input_ids, labels)   → res.loss（已按 mask 平均）+ res.aux_loss
     ├→ scaler.scale(loss).backward()
     └→ 每 step：unscale → clip → step → zero_grad
          ↓
out/full_sft_768.pth （fp16）  +  checkpoints/full_sft_768_resume.pth（fp32 全状态）
```

## 4. 训练曲线上你会看到什么

- loss 从约 3~4 快速降到 1.5~2.0 量级（取决于数据）。
- **logits_loss 与 aux_loss 分列**：dense 模型 aux_loss 恒 0；MoE 模型若 aux_loss 不降反升说明路由失衡。
- SFT 的 loss 通常比预训练低（对话文本更规整），且下降更快——能力迁移自预训练。

## 5. 本篇小结

1. SFT = 预训练骨架 + **SFTDataset（loss mask 在数据侧完成）** + **pretrain 权重起点** + 更小 lr/batch。
2. 「full」指全参数微调；产物 `full_sft_768.pth` 是后续 DPO/GRPO/PPO/Agent/蒸馏的**共同起点**。
3. 读其他训练篇时，与本篇相同的行不再展开，直接看差异。

下一篇：[04-LoRA训练-train_lora](./04-LoRA训练-train_lora.md)
