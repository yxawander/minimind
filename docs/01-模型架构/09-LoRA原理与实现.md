# 09 - LoRA 原理与实现（model_lora.py 全文）

> 覆盖代码：`model/model_lora.py` **全文 71 行**：`LoRA` 类（6-20）、`apply_lora`（23-34）、`load_lora`（37-44）、`save_lora`（47-56）、`merge_lora`（59-70）。
> 配套训练脚本：[04-训练/04-LoRA训练](../04-训练/04-LoRA训练-train_lora.md)；配套合并工具：[05-推理与部署/05-模型格式转换](../05-推理与部署/05-模型格式转换-convert_model.md)。

---

## 1. 原理：LoRA 在解决什么问题

### 1.1 全量微调的痛点

微调一个大模型（比如把通用模型改成医疗领域）需要更新全部权重：显存要装下「参数 + 梯度 + AdamW 的一阶/二阶动量」（每参数约 16 字节 fp32），还要为每个下游任务存一份完整模型。对 64M 的 MiniMind 尚可，对百亿模型不可行。

### 1.2 LoRA 的核心假设与公式

LoRA（*LoRA: Low-Rank Adaptation of Large Language Models*, 2021）假设：**微调引起的权重变化量 ΔW 是低秩的**——任务适配信息可以压进一个很小的子空间。于是不直接学 ΔW，而是学两个小矩阵的乘积：

$$
W' = W + \Delta W = W + \frac{\alpha}{r} B A,\qquad A \in \mathbb{R}^{r \times d_{in}},\; B \in \mathbb{R}^{d_{out} \times r},\; r \ll \min(d_{in}, d_{out})
$$

以本仓库 rank=16 为例，对一个 768×768 的注意力投影：

| | 全量微调 | LoRA(r=16) |
|--|----------|-----------|
| 可训练参数 | 589,824 | 768×16 + 16×768 = **24,576**（4.2%） |

前向传播变成 `h = Wx + (α/r)·B(Ax)`——原权重 W **冻结不动**，LoRA 分支并联在旁边。

### 1.3 关键设计点

1. **初始化不对称**：A 用小随机高斯、B 用全 0。这样 BA=0，**训练起点模型与原模型完全一致**（不会一上来就破坏预训练能力）。若 A、B 都随机，初始 ΔW 就是非零随机扰动，效果差。
2. **缩放系数 α/r**：α（lora_alpha）控制 LoRA 分支的「音量」。缩放写成 α/r 使得「改 r 时更新幅度大致可比」；本仓库默认 `lora_alpha=None → alpha=rank → scaling=1`，即最朴素的 BA。
3. **只加在方阵 Linear 上**：`apply_lora` 只对 `in_features == out_features` 的层挂 LoRA（见下），即注意力的 q/k/v/o 投影与 norm 不动。lm_head 是 768→6400 非方阵，天然跳过——这是简化选择（原论文主要也只调注意力）。
4. **为何训练快**：优化器状态只存 LoRA 参数；主干 W 被 `requires_grad_(False)` 冻结后，**W 的参数梯度根本不会被计算/存储**（激活梯度仍会穿过 W 传给下游，但参数梯度这块大头上没了），显存与时间的最大开销消失。

---

## 2. 逐行精读

### 2.1 LoRA 模块（6-20 行）

```python
class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank, lora_alpha=None):
        super().__init__()
        self.rank = rank  # LoRA的秩（rank），控制低秩矩阵的大小
        scaling = (rank if lora_alpha is None else lora_alpha) / rank
        self.register_buffer("scaling", torch.tensor(scaling, dtype=torch.float32))
        self.A = nn.Linear(in_features, rank, bias=False)  # 低秩矩阵A
        self.B = nn.Linear(rank, out_features, bias=False)  # 低秩矩阵B
        # 矩阵A高斯初始化
        self.A.weight.data.normal_(mean=0.0, std=0.02)
        # 矩阵B全0初始化
        self.B.weight.data.zero_()

    def forward(self, x):
        return self.B(self.A(x)) * self.scaling
```

| 行 | 讲解 |
|----|------|
| 7-9 | `rank`：秩 r。注释点明它控制低秩矩阵大小 |
| 10 | **缩放系数**：`lora_alpha` 没传时用 rank 本身 → scaling=1；传了 α 则 scaling=α/r（比如 α=32、r=16 → 2.0） |
| 11 | 用 `register_buffer` 而非普通属性：① 能跟着 `model.to(device)` 走；② 存进 state_dict（保存 LoRA 权重时 scaling 一起带走，加载时不用再传 α）。buffer 不可训练 |
| 12 | **A：降维** `768 → 16`。`nn.Linear(in, rank)` 的权重矩阵形状是 `[rank, in]`，即实现 `A(x) = W_A·x`——数学上的 Aᵀ。用 Linear 包装省手写 matmul |
| 13 | **B：升维** `16 → 768`，同理 |
| 15 | A 高斯初始化 std=0.02（GPT 系惯例的「小随机」） |
| 17 | **B 全 0**——与 A 的非零组合出 ΔW=0（见原理 1.3 节）。**只有 B 置零即可**：若 A 置零而 B 随机，梯度 dL/dA = dL/dh·B=0，A 学不动；反过来 B 置零时 dL/dB ≠ 0（因为 A≠0），正常启动 |
| 19-20 | 前向：先降维再升维再缩放。数值顺序 `B(A(x))·s` 与数学 `(BA)x·s` 等价（矩阵乘结合律），但**先小后大**的计算量远小于先合成大矩阵——这正是 LoRA 省算力的点 |

### 2.2 apply_lora（23-34 行）——把 LoRA「焊」进现有模型

```python
def apply_lora(model, rank=16, lora_alpha=None):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.in_features == module.out_features:
            lora = LoRA(module.in_features, module.out_features, rank=rank, lora_alpha=lora_alpha).to(model.device)
            setattr(module, "lora", lora)
            original_forward = module.forward

            # 显式绑定
            def forward_with_lora(x, layer1=original_forward, layer2=lora):
                return layer1(x) + layer2(x)

            module.forward = forward_with_lora
```

| 行 | 讲解 |
|----|------|
| 24 | `named_modules()` 遍历全部子模块（包括嵌套的：8 层 × q/k/v/o × (MoE 专家)） |
| 25 | 筛选：是 Linear **且方阵**。命中：每层的 q_proj/k_proj/v_proj/o_proj（768×768）；MoE 模型再加 4 个专家的 12 条矩阵。**跳过**：lm_head（非方阵）、gate（768→4）、value_head（768→1） |
| 26 | 构造 LoRA 并 `.to(model.device)`——注意这行代码执行时模型可能还在 CPU，device 属性取的是模型当前所在 |
| 27 | 把 LoRA 挂到原层上作为子模块属性 `module.lora`。**关键**：因为是 setattr 到 nn.Module 上，LoRA 的参数会自动进入 `model.parameters()`/`named_parameters()`（名字形如 `...self_attn.q_proj.lora.A.weight`）——后面 train_lora 靠 `'lora' in name` 筛选可训练参数 |
| 28 | **先保存原 forward**（普通函数对象，不是 bound method 调用结果） |
| 31-32 | **monkey-patch 替换 forward**：新 forward = 原输出 + LoRA 输出。`layer1=original_forward, layer2=lora` 用**默认参数**在定义时把两个函数「钉」进闭包——这是 Python 闭包的经典坑规避：若写成 `return original_forward(x)+lora(x)`，循环里所有层共享同一个 `original_forward/lora` 变量名，最终全部指向**最后一层**的 LoRA（晚绑定 late binding）。默认参数绑定在 def 时刻求值，每层各拿各的，注释「显式绑定」说的就是这个 |
| 34 | 直接覆盖实例属性 forward。实例属性优先于类的 forward，调用 `module(x)` 时走 patched 版本 |

> 这个「旁路并联」设计的好处：**零侵入**。模型类一行不用改；去掉 LoRA 只需不调用 apply_lora。缺点：monkey-patch 与 `torch.compile` 不兼容（train_lora.py:158 专门检测并自动关闭 compile）。

### 2.3 load_lora（37-44 行）

```python
def load_lora(model, path):
    state_dict = torch.load(path, map_location=model.device)
    state_dict = {(k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items()}

    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            lora_state = {k.replace(f'{name}.lora.', ''): v for k, v in state_dict.items() if f'{name}.lora.' in k}
            module.lora.load_state_dict(lora_state, strict=False)
```

| 行 | 讲解 |
|----|------|
| 38 | 加载权重文件到模型设备 |
| 39 | **剥 DDP 前缀**：DDP 训练时 state_dict 键都带 `module.` 前缀，推理加载时要剥掉 |
| 41-42 | 再次遍历找**挂着 lora 的层**（前提：推理时同样调用了 apply_lora，结构才能对上） |
| 43 | 按「完整键名 = `{层名}.lora.{A/B}.weight`」把属于本层的键抽出来，并剥掉前缀得到 LoRA 模块自己的键（`A.weight`/`B.weight`/`scaling`） |
| 44 | `strict=False`：允许文件里有多余/缺失键不报错（宽容加载）。注意 scaling 是 buffer 也在 state_dict 里，会被一起恢复 |

### 2.4 save_lora（47-56 行）

```python
def save_lora(model, path):
    raw_model = getattr(model, '_orig_mod', model)
    state_dict = {}
    for name, module in raw_model.named_modules():
        if hasattr(module, 'lora'):
            clean_name = name[7:] if name.startswith("module.") else name
            lora_state = {f'{clean_name}.lora.{k}': v.cpu().half() if k.endswith('.weight') else v.cpu().float()
                          for k, v in module.lora.state_dict().items()}
            state_dict.update(lora_state)
    torch.save(state_dict, path)
```

| 行 | 讲解 |
|----|------|
| 48 | `getattr(model, '_orig_mod', model)`：**剥 torch.compile 包装**（compile 后的模型包在 `_orig_mod` 下）；`getattr` 三参形式没编译时返回模型本身 |
| 50-51 | 找所有挂 lora 的层 |
| 52 | 同样剥 `module.` 前缀（DDP 下 named_modules 也带前缀） |
| 53-54 | 组键 + 精度处理：**权重转 fp16**（省一半空间），**scaling 保持 fp32**（它只有 1 个数，且是 buffer 不参与训练，保持精度无妨）。`.cpu()` 先搬出 GPU 再存 |
| 55-56 | 汇总保存。产物就是一个只有 LoRA 参数的小文件（rank=16 时约 3-4MB，对比全模型 128MB） |

### 2.5 merge_lora（59-70 行）——合并回主干

```python
def merge_lora(model, lora_path, save_path):
    load_lora(model, lora_path)
    raw_model = getattr(model, '_orig_mod', model)
    state_dict = {k: v.cpu().half() for k, v in raw_model.state_dict().items() if '.lora.' not in k}
    for name, module in raw_model.named_modules():
        if isinstance(module, nn.Linear) and '.lora.' not in name:
            state_dict[f'{name}.weight'] = module.weight.data.clone().cpu().float()
            if hasattr(module, 'lora'):
                delta = (module.lora.B.weight.data.float() @ module.lora.A.weight.data.float()) * module.lora.scaling.float()
                state_dict[f'{name}.weight'] += delta.cpu()
            state_dict[f'{name}.weight'] = state_dict[f'{name}.weight'].half()
    torch.save(state_dict, save_path)
```

| 行 | 讲解 |
|----|------|
| 60 | 先把 LoRA 权重加载进模型（结构必须已 apply_lora） |
| 62 | 拷贝**非 LoRA** 的全部权重（`.lora.` 不在键名里的）为 fp16——embedding、norm、lm_head 等原样带走 |
| 63 | 遍历所有 Linear（`'.lora.' not in name` 确保只处理宿主层本身，不进 LoRA 内部） |
| 65 | 把宿主层权重**升 fp32** 备用（待会要做加法，fp16 直接加会有舍入损失） |
| 67-68 | **合并公式**：`ΔW = B(768×16) @ A(16×768) × scaling` → [768×768]，加到原权重上。这一步把两个小矩阵**物理相乘**还原成完整 ΔW——合并后模型结构与原模型完全一致，LoRA 消失 |
| 69 | 合并结果降回 fp16 存储 |
| 70 | 保存为标准基模结构权重（可被 init_model/convert 直接加载） |

> 合并的代价与收益：合并后**零额外推理开销**（没有旁路矩阵乘）、部署简单；代价是失去「卸载 LoRA 恢复原模型」的灵活性——通常对最终交付的模型才合并。

### 2.6 全文件数据流总结

```
apply_lora(model)                     训练前：挂旁路、换 forward
    ↓
train_lora：只训 'lora' 参数          训练中：主干冻结
    ↓
save_lora(model, path)                存：只有 A/B/scaling 的小文件
    ↓
    ├─ 推理：apply_lora + load_lora     运行时旁路生效
    └─ 交付：merge_lora                  W += B@A·s，旁路消失
```

## 3. 本篇小结

1. LoRA = 冻结 W，旁路学 `ΔW = (α/r)BA`；B 置零保证起点无损。
2. `apply_lora` 用 **setattr + monkey-patch forward（默认参数防晚绑定）** 实现零侵入注入；只对方阵 Linear 生效。
3. 保存/加载都以 `{层名}.lora.{键}` 命名，处理了 DDP `module.` 与 torch.compile `_orig_mod` 两种包装前缀。
4. `merge_lora` 把 `B@A×scaling` 物理加回主干，得到无旁路的独立模型。

下一篇：[02-分词器/01-BPE分词原理](../02-分词器/01-BPE分词原理.md)
