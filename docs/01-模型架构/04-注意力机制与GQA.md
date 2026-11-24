# 04 - 注意力机制与 GQA

> 覆盖代码：`model/model_minimind.py` **第 86-134 行**：`repeat_kv`（86-89）、`Attention` 类（91-134）。
> 前置阅读：[03-RoPE](./03-RoPE旋转位置编码.md)（Q/K 旋转）、[02-RMSNorm](./02-RMSNorm.md)（q/k norm）。

---

## 1. 原理

### 1.1 Scaled Dot-Product Attention 回顾

对每个注意力头（head 维 96）：

$$
\text{Attention}(Q,K,V) = \text{softmax}\!\Big(\frac{QK^\top}{\sqrt{d_k}} + \text{mask}\Big)V
$$

- $QK^\top$：每个 query 对每个 key 算相似度 → 得分矩阵 `[S_q, S_k]`。
- 除以 $\sqrt{d_k}$：防止点积随维度增长而过大，把 softmax 推入饱和区（梯度消失）。$d_k$=96，√96≈9.8。
- **mask**：Causal（因果）mask 把「未来位置」设为 −∞——softmax 后权重为 0，位置 t 只能看见 ≤t 的 token。这是语言模型「预测下一个词」的数学表达。
- 乘 V：按注意力权重对 value 加权求和。

### 1.2 MHA → MQA → GQA

多头注意力里，每个头独立算 Q/K/V：

- **MHA**（Multi-Head Attention）：Q 和 KV 头数相同（如 8/8）。效果最好，但推理时 **KV Cache 大小 ∝ KV 头数**——自回归生成每一步都要把历史 K/V 存着，MHA 的 Cache 显存和带宽是瓶颈。
- **MQA**（Multi-Query）：所有 Q 头共享 1 组 KV（8/1）。Cache 缩小 8 倍，但效果有损。
- **GQA**（Grouped-Query Attention）：折中——Q 头分组共享 KV。MiniMind/Qwen3 用 **8 Q 头 : 4 KV 头**，即每 2 个 Q 头共享 1 组 KV（`n_rep = 2`）。Cache 减半，效果接近 MHA。

```
Q 头:   q0 q1 | q2 q3 | q4 q5 | q6 q7     ← 8 个
KV 头:   k0     k1     k2     k3          ← 4 个
        (q0,q1)共享k0, (q2,q3)共享k1, ...
```

MiniMind 的配置：`num_attention_heads=8`、`num_key_value_heads=4`、`head_dim=96`。

### 1.3 QK-Norm（Qwen3 引入）

在 RoPE 之前对每个头的 q、k 向量（96 维）各做一次 RMSNorm。目的：**控制注意力 logits 的规模**。小模型训练中 q/k 的范数可能失控 → softmax 饱和 → 训练不稳；先归一化再旋转，让点积尺度有界。

### 1.4 Flash Attention（SDPA）与手工实现的双路径

PyTorch 内置 `F.scaled_dot_product_attention`（SDPA）在满足条件时自动走 **FlashAttention 内核**：不物化 `[S,S]` 的得分矩阵，用分块在线 softmax 把显存从 O(S²) 降到 O(S)，速度快数倍。MiniMind 在「理想条件」下走 SDPA，否则退回**手写逐步实现**（便于学习 + 处理复杂 mask）。

---

## 2. 逐行精读

### 2.1 repeat_kv（86-89 行）——GQA 的 KV 复制

```python
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1: return x
    return (x[:, :, :, None, :].expand(bs, slen, num_key_value_heads, n_rep, head_dim).reshape(bs, slen, num_key_value_heads * n_rep, head_dim))
```

| 行 | 讲解 |
|----|------|
| 86 | 输入 x 是 K 或 V：`[B, S_kv, kv_heads=4, head_dim=96]`；`n_rep` = 每组 KV 要复制几份给 Q 头用（8/4=2） |
| 88 | n_rep=1（MHA）时无需复制，直接返回 |
| 89 | 三步变形：<br>1. `x[:, :, :, None, :]` → `[B, S, 4, 1, 96]`（插入组内维）<br>2. `.expand(..., n_rep, ...)` → `[B, S, 4, 2, 96]`（**expand 不复制内存**，只是广播视图）<br>3. `.reshape` → `[B, S, 8, 96]`——此时 reshape 遇到非连续内存会触发实际拷贝。<br>输出头序：k0,k0,k1,k1,k2,k2,k3,k3——与 Q 头的分组顺序一一对应（q0,q1→k0；q2,q3→k1……） |

> 为什么 reshape 能保证组内相邻？expand 后第 2、3 维是 (kv_head, n_rep)，展平后同一 kv_head 的 n_rep 份拷贝天然相邻。

### 2.2 Attention.__init__（92-110 行）

```python
class Attention(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.num_key_value_heads = config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        self.n_local_heads = config.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = config.head_dim
        self.is_causal = True
```

| 行 | 讲解 |
|----|------|
| 94 | `num_key_value_heads=None` 时退化为 MHA（KV 头数=Q 头数）；这里 config 给 4 |
| 95-96 | 缓存 Q 头数（8）与 KV 头数（4）。名字里的 `local` 是沿袭 Llama 原始实现的叫法（在张量并行里每卡只持有部分头，此处单卡即全部） |
| 97 | `n_rep = 8 // 4 = 2`：每组 KV 服务 2 个 Q 头 |
| 98 | 头维 96 |
| 99 | `is_causal=True`：语言模型的注意力永远带因果 mask（本类只服务于 LM，写死） |

```python
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
```

| 行 | 讲解 |
|----|------|
| 100 | Q 投影：768 → 8×96=768（方阵）。输出会 reshape 成 8 个头 |
| 101 | **K 投影只到 4×96=384**：这是 GQA 省参数/省 Cache 的地方——K/V 的头数就是 KV 头数 |
| 102 | V 同 K |
| 103 | 输出投影：8×96=768 → 768，把 8 个头的拼接结果混合回残差流。四条投影全部 `bias=False`（现代 LLM 惯例） |

```python
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
```

| 行 | 讲解 |
|----|------|
| 104-105 | **QK-Norm**：注意归一化维度是 `head_dim=96`（头内），不是 hidden。每个头的 q 向量独立归一化。对比：Llama 没有 QK-Norm，Qwen3 有——所以 convert_model 转 Qwen3 格式时这两份权重能对上 |

```python
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and config.flash_attn
```

| 行 | 讲解 |
|----|------|
| 106 | softmax 之后的 dropout（只对注意力权重） |
| 107 | 输出投影后的 dropout（对子层输出） |
| 108 | dropout 概率缓存给 SDPA 用（SDPA 需要 float 参数而不是模块） |
| 109 | 走 SDPA 快路径的条件：PyTorch 有该函数 **且** config 开关打开。关掉开关即可回退手写路径（调试/兼容用） |

### 2.3 Attention.forward（111-134 行）——核心

```python
    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        bsz, seq_len, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
```

| 行 | 讲解 |
|----|------|
| 112 | 输入 `[B, S, 768]` |
| 113 | 三条投影。K/V 只有 384 维输出（GQA） |
| 114-116 | 拆头：Q → `[B, S, 8, 96]`；K/V → `[B, S, 4, 96]`。**view（不拆头）+ 逐头处理**是标准做法：`view` 只重解释内存，把 768 拆成 8×96 |

```python
        xq, xk = self.q_norm(xq), self.k_norm(xk)
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)
```

| 行 | 讲解 |
|----|------|
| 117 | QK-Norm：在最后一维 96 上归一化（norm 内部 mean(-1) 自动作用于此） |
| 118 | 拿到 RoPE 表切片 `(cos, sin)`，形状 `[S, 96]` |
| 119 | 旋转 q/k（详见 RoPE 篇）。**顺序：先 norm 再 RoPE**——Qwen3 的约定，权重迁移时保持一致 |

```python
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None
```

| 行 | 讲解 |
|----|------|
| 120-122 | **KV Cache 增量拼接**：推理时每步只算 1 个新 token 的 K/V（seq_len=1），把它**拼到缓存的旧 K/V 后面**（dim=1 是序列维）。拼完后 xk/xv 覆盖整个历史 `[B, past_len+1, 4, 96]`——注意力就能「看到」全部上文。训练时 past 为 None，跳过 |
| 123 | `use_cache=True` 时把（拼接后的）K/V 打包返回，供下一步当缓存。训练时不返回（省内存） |

```python
        xq, xk, xv = (xq.transpose(1, 2), repeat_kv(xk, self.n_rep).transpose(1, 2), repeat_kv(xv, self.n_rep).transpose(1, 2))
```

| 行 | 讲解 |
|----|------|
| 124 | 三步准备：<br>1. `repeat_kv` 把 K/V 从 4 头复制到 8 头（GQA→MHA 等价形式）；<br>2. `transpose(1,2)` 把 Q/K/V 从 `[B, S, heads, dim]` 转成 **`[B, heads, S, dim]`**——因为后面 `xq @ xk.transpose(-2,-1)` 要在「S×S」的二维上做矩阵乘，SDPA 也要求头维在前。<br>结果形状均为 `[B, 8, S, 96]` |

```python
        if self.flash and (seq_len > 1) and (not self.is_causal or past_key_value is None) and (attention_mask is None or torch.all(attention_mask == 1)):
            output = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=self.dropout if self.training else 0.0, is_causal=self.is_causal)
```

| 行 | 讲解 |
|----|------|
| 125 | **SDPA 快路径的 4 个前提**，逐个拆：<br>1. `self.flash`：config 允许；<br>2. `seq_len > 1`：单 token 增量步（seq_len=1）没有优化空间，且 SDPA 的 causal mask 语义是「Q 长度 = KV 长度」，增量步 Q 只有 1 行、KV 是 past_len+1，**不匹配**，必须走手写路径；<br>3. `(not causal) or (无缓存)`：同上——训练（无缓存）时 Q/KV 等长，is_causal=True 语义正确；<br>4. `attention_mask 全 1 或为 None`：SDPA 原生只支持「因果」或「无 mask」，任意 padding mask 它不管（padding mask 会和 causal 语义冲突），所以 batch 内全是真 token 时才走。<br>满足全部条件 → SDPA（内部自动 FlashAttention/高效内核）。`dropout_p` 只在训练时传 |
| 126 | 输出 `[B, 8, S, 96]` |

```python
        else:
            scores = (xq @ xk.transpose(-2, -1)).float() / math.sqrt(self.head_dim)
            if self.is_causal: scores[:, :, :, -seq_len:] += torch.full((seq_len, seq_len), float("-inf"), device=scores.device).triu(1)
            if attention_mask is not None: scores += (1.0 - attention_mask.to(scores.dtype).unsqueeze(1).unsqueeze(2)) * -1e9
            output = self.attn_dropout(F.softmax(scores, dim=-1).type_as(xq)) @ xv
```

| 行 | 讲解 |
|----|------|
| 128 | **手写路径第 1 步**：得分矩阵。`xq @ xk.transpose(-2,-1)`：`[B,8,S_q,96] @ [B,8,96,S_k]` → `[B, 8, S_q, S_k]`。`.float()`：升 fp32 算 softmax（防 bf16 精度问题）；`/√96` 缩放 |
| 129 | **因果 mask 的巧妙写法**：`torch.full((S_q,S_q), -inf).triu(1)` 生成右上三角 −∞ 矩阵（对角线及以上……准确说对角线以上；triu(1) 是严格上三角，对角线本身不受影响——**自己可以看见自己**）。关键在切片 `[..., -seq_len:]`：只加到得分矩阵的**最后 seq_len 列**上。<br>• 训练：S_k = S_q = seq_len，加满整张表，等价标准 causal mask；<br>• 增量解码：S_k = past_len+1（KV 长），S_q = 1（新 token）。它只能看见自己及之前——本来就不需要 mask（没有「未来」）。−seq_len 切片保证 mask 对齐到「最后 seq_len 个 key」，在 sliding window 场景也正确。<br>用 `+=` 到 scores 上而不是构造完整 mask 矩阵相加，省内存 |
| 130 | **padding mask**：`attention_mask` 是 HF 风格的 `[B, S]`，1=真 token、0=padding。`(1-mask)·(-1e9)` 把 padding 列的得分压到 −1e9（softmax 后≈0）。unsqueeze(1)/(2) 对齐到 `[B, 1, 1, S]` 广播到所有头和所有 query 行。注意增量步拼接的新位置全是 1（见 generate），所以这行只在**批量编码含 padding 的 prompt** 时起作用 |
| 131 | softmax（dim=-1，对 key 维归一）→ 降回原精度 → dropout → 乘 V：`[B,8,S_q,96] @ [B,8,S_k,96]` → `[B,8,S_q,96]` |

```python
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv
```

| 行 | 讲解 |
|----|------|
| 132 | 转回 `[B, S, 8, 96]` 再 reshape 合头 → `[B, S, 768]`（`-1` 自动算 8×96）。注意 `transpose` 后内存不连续，reshape 需要拷贝——所以这里不能省 |
| 133 | 输出投影回残差流维度 + resid dropout |
| 134 | 返回（注意力输出，本层 KV）。Block 里拿它做残差相加 |

---

## 3. 形状速查表（训练，B=4, S=340）

| 位置 | 张量 | 形状 |
|------|------|------|
| 输入 x | hidden | [4, 340, 768] |
| q_proj 后 | xq | [4, 340, 768] → view [4, 340, 8, 96] |
| k/v_proj 后 | xk/xv | [4, 340, 384] → view [4, 340, 4, 96] |
| RoPE 后 | xq/xk | [4, 340, 8, 96] / [4, 340, 4, 96] |
| repeat_kv 后 | xk/xv | [4, 340, 8, 96] |
| transpose 后 | 全部 | [4, 8, 340, 96] |
| scores | — | [4, 8, 340, 340] |
| output | — | [4, 8, 340, 96] → [4, 340, 768] |

增量解码时：S_q=1、S_k=past_len+1、scores=[B,8,1,past_len+1]。

## 4. 本篇小结

1. **GQA = K/V 投影到更少的头 + `repeat_kv` 复制回 Q 头数**；本仓库 8Q:4KV，n_rep=2。
2. QK-Norm 在 **RoPE 之前**、头内 96 维上做 RMSNorm，稳定注意力得分。
3. forward 有双路径：SDPA（训练/全 1 mask/非增量步）与手写（增量步、含 padding mask）；手写版的 causal mask 用 `[..., -seq_len:]` 切片技巧兼容「KV 比 Q 长」。
4. KV Cache 在本层完成「拼接 + 返回」；位置编码（RoPE）已经把绝对位置烧进 K，缓存里的 K 无需重算。
5. 全部投影无 bias；输出投影 + resid dropout 收尾。

下一篇：[05-FFN与SwiGLU.md](./05-FFN与SwiGLU.md)
