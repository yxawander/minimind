# 02 - 预训练 train_pretrain.py 全文

> 覆盖代码：`trainer/train_pretrain.py` **全文 164 行**。
> 预训练是「学语言」的阶段：纯文本上下文，逐 token 预测，损失覆盖整段文本。
> 本文同时把所有训练脚本的**公共骨架**（混合精度/梯度累积/DDP/保存循环）讲透——SFT/LoRA/蒸馏篇只讲差异。

---

## 1. 原理：预训练在学什么

### 1.1 训练目标

给定一段文本 `t1..tn`，模型学习条件分布：

$$
\mathcal{L} = -\frac{1}{T}\sum_{t} \log P(t_{i} \mid t_{<i})
$$

即「看到前文，猜下一个词」。整段文本（含 eos）都参与损失。预训练后模型会「续写」，但不会「对话」——那是 SFT 的任务。

### 1.2 公共训练骨架的五个工程件

| 组件 | 一句话 | 详解位置 |
|------|--------|----------|
| **混合精度 AMP** | 前向 bf16/fp16，省显存提速；loss 缩放保 fp16 梯度 | 本文 §3.3 |
| **梯度累积** | 小显存模拟大 batch：N 个 micro-batch 梯度累加后统一更新 | 本文 §3.4 |
| **梯度裁剪** | 梯度范数压到上限 1.0，防梯度爆炸 | 本文 §3.5 |
| **DDP** | 多卡数据并行，梯度自动 all-reduce | trainer_utils 篇 §5 |
| **断点续训** | lm_checkpoint 双文件 + SkipBatchSampler | trainer_utils 篇 §7/§9 |

---

## 2. 逐行精读

### 2.1 文件头（1-21 行）

```python
import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import datasets  # noqa: F401  # Windows pyarrow/torch DLL conflict workaround (issue #771)
import argparse
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import PretrainDataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler

warnings.filterwarnings('ignore')
```

| 行 | 讲解 |
|----|------|
| 4-5 | 同 trainer_utils：包标记 + 仓库根目录入 path |
| 7 | **`import datasets` 空导入（noqa: F401）**：不是为了用它的功能，而是利用 import 顺序规避 Windows 下 pyarrow 与 torch 的 DLL 加载冲突（issue #771）。**必须最先 import**——所有 trainer 都原样保留这行 |
| 9 | `nullcontext`：CPU 训练时充当「空的 autocast 上下文」（见 3.3） |
| 14-15 | `optim`（AdamW）、DDP、DataLoader/DistributedSampler |
| 17-19 | 模型/数据/工具导入 |
| 21 | 全局屏蔽 warning（训练日志干净，但调试时建议注释掉） |

### 2.2 train_epoch：训练主循环（24-72 行）

```python
def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    start_time = time.time()
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
```

| 行 | 讲解 |
|----|------|
| 26 | `enumerate(loader, start=start_step+1)`：step 号从续训点继续（配合 SkipBatchSampler 跳批） |
| 27-28 | 数据搬 GPU |
| 29-31 | **手动学习率调度**：全局 step = `epoch×每epoch步数 + 本epoch内步号`；总步数 = `epochs×iters`；算出余弦 lr 后**直接写进 optimizer 的每个参数组**（不用 scheduler 对象，实现透明） |

```python
        with autocast_ctx:
            res = model(input_ids, labels=labels)
            loss = res.loss + res.aux_loss
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()
```

| 行 | 讲解 |
|----|------|
| 33-36 | **前向**（autocast 内自动半精度）：模型返回 MoeCausalLMOutput；总损失 = **主损失（交叉熵）+ aux_loss（MoE 负载均衡；dense 时为 0）**；除以累积步数——**这是梯度累积的关键**：N 次backward各积累 loss/N 的梯度，求和后恰好等于完整 batch 的平均梯度 |
| 38 | `scaler.scale(loss).backward()`：反向传播。scaler（fp16 模式）先把损失放大 GradScaler.init_scale 倍再反传，防止 fp16 梯度下溢为 0（bf16 模式 scaler enabled=False，是恒等直通） |

```python
        if step % args.accumulation_steps == 0 or step == iters:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)
```

| 行 | 讲解 |
|----|------|
| 40 | 每 N 步（或最后一个 step）执行一次**真更新** |
| 41 | `unscale_`：把梯度缩回去（除回放大倍数）——**必须在裁剪之前**，否则裁的是放大后的范数，阈值意义全变 |
| 42 | **梯度裁剪**：全部参数梯度的全局 L2 范数压到 1.0（超过就等比缩放）。LLM 训练标准操作，防个别 batch 的异常梯度把权重打飞 |
| 44-45 | `scaler.step`：若本轮**没有 inf/nan 梯度**才执行 optimizer.step()；有则跳过本轮更新。`scaler.update`：动态调整放大倍数（连续成功→增大，遇到 inf→减小） |
| 47 | `zero_grad(set_to_none=True)`：梯度清零且**置 None**（比置 0 省一次显存写，下次 backward 重新分配） |

```python
        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            current_logits_loss = current_loss - current_aux_loss
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: ..., logits_loss: ..., aux_loss: ..., lr: ..., epoch_time: ...min')
            if wandb: wandb.log({...})
```

| 行 | 讲解 |
|----|------|
| 49-57 | 每 100 步打印：<br>• `current_loss`：把除过 accumulation 的 loss **乘回去**（显示真实量级）；<br>• `aux_loss` 单列、`logits_loss` = 差值——监控 MoE 负载均衡损失是否在下降；<br>• `eta_min`：按已用时间/已完成步数外推剩余时间（分钟）；<br>• `wandb.log`：swanlab/wandb 记录曲线（只在主进程 init 过，见 main） |

```python
        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')
            model.train()
            del state_dict

        del input_ids, labels, res, loss
```

| 行 | 讲解 |
|----|------|
| 59-69 | 每 1000 步存档（仅主进程）：<br>1. 切 eval 模式（dropout 关闭，虽然默认 0，规范如此）；<br>2. 存**推理用 fp16 权重**到 `out/pretrain_768.pth`（直接可被 eval_llm/init_model 加载）；<br>3. 存**续训用完整状态**到 `checkpoints/`（fp32 模型+优化器+进度）；<br>4. 切回 train 模式；释放引用。<br>**存两份的原因**：推理只要权重；续训必须优化器状态。混成一个文件会浪费/混淆 |
| 71 | 每 step 结束显式 del——帮助 GC 及时回收 GPU 张量（长训练防内存碎片） |

### 2.3 main：初始化流水线（74-159 行）

```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    parser.add_argument("--save_dir", type=str, default="../out", ...)
    parser.add_argument('--save_weight', default='pretrain', ...)
    parser.add_argument("--epochs", type=int, default=2, ...)
    parser.add_argument("--batch_size", type=int, default=32, ...)
    parser.add_argument("--learning_rate", type=float, default=5e-4, ...)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", ...)
    parser.add_argument("--dtype", type=str, default="bfloat16", ...)
    parser.add_argument("--num_workers", type=int, default=8, ...)
    parser.add_argument("--accumulation_steps", type=int, default=8, ...)
    parser.add_argument("--grad_clip", type=float, default=1.0, ...)
    parser.add_argument("--log_interval", type=int, default=100, ...)
    parser.add_argument("--save_interval", type=int, default=1000, ...)
    parser.add_argument('--hidden_size', default=768, ...)
    parser.add_argument('--num_hidden_layers', default=8, ...)
    parser.add_argument('--max_seq_len', default=340, ...)
    parser.add_argument('--use_moe', default=0, ...)
    parser.add_argument('--seed', default=42, ...)
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_t2t_mini.jsonl", ...)
    parser.add_argument('--from_weight', default='none', ...)
    parser.add_argument('--from_resume', default=0, ...)
    parser.add_argument("--use_wandb", action="store_true", ...)
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain", ...)
    parser.add_argument("--use_compile", default=0, ...)
    args = parser.parse_args()
```

参数表（预训练特有值加粗）：

| 参数 | 默认 | 说明 |
|------|------|------|
| save_weight | pretrain | 权重文件前缀 |
| epochs | 2 | 预训练跑 2 轮（SFT 也是 2；LoRA 10；RL 1） |
| batch_size | 32 | 物理每卡 batch |
| **learning_rate** | **5e-4** | 预训练是「从随机开始」所以 lr 大；SFT 降到 1e-5（权重已好，小步微调）；DPO 4e-8（对齐阶段 lr 极小防崩） |
| dtype | bfloat16 | bf16 无需 loss scaling；fp16 会启用 GradScaler |
| accumulation_steps | **8** | 预训练有效 batch = 32×8 = 256；SFT 是 1 |
| max_seq_len | 340 | 预训练文本短；SFT 768 |
| from_weight | **none** | 预训练从头；SFT 默认 pretrain |
| use_moe / hidden_size / layers | 0 / 768 / 8 | 模型规格 |
| from_resume | 0 | 1 = 自动检测 checkpoints 续训 |
| use_compile | 0 | torch.compile 加速 |

```python
    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(args.seed + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
```

| 行 | 讲解 |
|----|------|
| 102-104 | DDP 初始化；多卡时 device 换成本 rank 的卡；种子 = seed+rank（各卡洗牌不同但确定） |
| 107-109 | 建输出目录；构建 config；**from_resume=1 时先查有无旧 checkpoint**（lm_checkpoint 读模式）——注意读取发生在构建数据集之前，因为续训要跳批 |

```python
    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. 配wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
```

| 行 | 讲解 |
|----|------|
| 112-114 | **混合精度上下文**：CPU 直接用 nullcontext（不套 autocast）；GPU 用 `torch.cuda.amp.autocast`——上下文内的矩阵乘/线性层自动用半精度算，norm/softmax 等敏感算子仍由实现内部保 fp32。整个 train_epoch 用 `with autocast_ctx:` 包前向 |
| 117-123 | 可视化：**import swanlab as wandb**（API 兼容替换）。只在主进程 init（其他 rank 不联网）；`id=wandb_id, resume='must'`：续训时接回原曲线而非开新图。run 名字里编码了超参数（可读性） |

```python
    # ========== 5. 定义模型、数据、优化器 ==========
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
```

| 行 | 讲解 |
|----|------|
| 126-129 | 模型加载（from_weight=none → 随机初始化）；数据集；DDP 时用 DistributedSampler 把数据**切成不重叠的片段**分给各卡 |
| 129 | **GradScaler 只在 fp16 时启用**（bf16 动态范围大不需要 loss scaling；enabled=False 时 scale/step 全是直通） |
| 130 | AdamW：所有参数（含 tied 的 embedding）。lr 会被 train_epoch 每步覆盖 |

```python
    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
```

| 行 | 讲解 |
|----|------|
| 133-139 | 续训三恢复：模型 fp32 权重、优化器动量、scaler 状态；起始位置 epoch/step。**顺序注意**：这里加载发生在 DDP 包装之前（第 7 步才包装）——权重进单模型再复制到各 rank，正确 |

```python
    # ========== 7. 编译和分布式包装 ==========
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])
```

| 行 | 讲解 |
|----|------|
| 142-144 | torch.compile：把模型图编译成融合 kernel（提速 10-30%，但首步编译慢、且与某些动态特性不兼容——LoRA 的 monkey-patch 就不兼容） |
| 145-146 | DDP 包装：此后 `model(x)` 会自动做前向 + 梯度同步；`model.module` 拿裸模型（保存时用） |

```python
    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(args.seed + epoch); indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0: 
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), 0, wandb)
```

| 行 | 讲解 |
|----|------|
| 150 | `train_sampler and train_sampler.set_epoch(epoch)`：**Python 短路技巧**——单卡时 sampler 是 None，表达式短路不执行；DDP 时 set_epoch 改变 shuffle 种子（否则每轮各卡拿到相同的数据切分） |
| 151 | **每轮重新播种+洗牌**：`seed+epoch` 保证「顺序确定且逐轮不同」——这正是 SkipBatchSampler 能复现顺序的前提 |
| 152 | 只在「续训的第一轮」跳批（后续轮从头） |
| 153 | **双通道采样**：DDP 用 DistributedSampler（自动分片），单卡用 `indices` 列表。SkipBatchSampler 统一在外层做跳批 |
| 154 | DataLoader 用 **batch_sampler**（直接成批 yield，batch_size 不再单独传）；`pin_memory=True`：锁页内存加速 CPU→GPU 拷贝 |
| 155-159 | skip>0 时总步数要加回 skip（iters 参数表示「本 epoch 的总步数」，enumerate 从 start_step+1 计数） |

```python
    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
```

| 行 | 讲解 |
|----|------|
| 162-164 | barrier 等所有 rank 结束（防止主进程先退出导致其他 rank 报错），再销毁进程组 |

---

## 3. 训练超参的直觉

| 阶段 | lr | batch(有效) | epochs | 为什么 |
|------|-----|------------|--------|--------|
| pretrain | 5e-4 | 256 | 2 | 从零学，可以大步走 |
| full_sft | 1e-5 | 16 | 2 | 权重已好，小步防「遗忘」 |
| lora | 1e-4 | 32 | 10 | 只训旁路小矩阵，lr 可以更大、轮数更多 |
| dpo | 4e-8 | 4 | 1 | 对齐目标是「微调偏好」，lr 稍大就灾难性遗忘 |
| grpo/ppo | 3e-7 | 2 | 1 | RL 的策略更新必须保守 |

## 4. 本篇小结

1. 预训练 loss = 全文本的交叉熵 + MoE aux loss；模型内部已做 shift。
2. **AMP 五连**：autocast 前向 → scale 后 backward → unscale → clip → 条件 step；bf16 时 scaler 是直通。
3. 梯度累积：`loss/N` 逐次 backward，`step % N == 0` 才更新，有效 batch = bs×N×卡数。
4. 断点续训四件套：fp32 resume 包、fp16 推理包、种子复现顺序、SkipBatchSampler 跳批。
5. 学习率手动余弦退火；保存时剥 DDP/compile 两层壳。

下一篇：[03-监督微调-train_full_sft](./03-监督微调-train_full_sft.md)
