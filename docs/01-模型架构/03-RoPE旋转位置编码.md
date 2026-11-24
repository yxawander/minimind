# 03 - RoPE 旋转位置编码

> 覆盖代码：`model/model_minimind.py` **第 62-84 行**：`precompute_freqs_cis`（62-78）、`apply_rotary_pos_emb`（80-84）。
> YaRN 外推分支（64-73 行内的 `if rope_scaling is not None`）单独成篇：[07-YaRN长文本外推](./07-YaRN长文本外推.md)，本篇讲主干。

---

## 1. 原理：为什么需要位置编码，为什么是 RoPE

### 1.1 问题：Self-Attention 天生「无序」

注意力是对集合的操作——把 token 顺序打乱，输出只是相应置换。但语言显然依赖顺序（"猫吃鱼" ≠ "鱼吃猫"）。所以要**把位置信息注入** Q/K 向量。

三代方案：

| 方案 | 代表 | 问题 |
|------|------|------|
| 正弦绝对位置编码 | 原 Transformer | 只加在输入端，深层会被 FFN 冲淡 |
| 可学习位置嵌入 | GPT-2 | 长度固定，外推差 |
| **旋转位置编码 RoPE** | Llama / Qwen / MiniMind | 把位置变成**对 Q/K 的旋转**，天然相对 |

### 1.2 RoPE 的核心思想（数学推导）

目标：找一个变换 `f(q, m)`（m 是 token 的绝对位置），使得两个位置的 Q·K 内积**只依赖它们的相对距离 m−n**：

$$
\langle f(q, m),\; f(k, n)\rangle = g(q, k, m-n)
$$

RoPE 的解是：把 q、k 的每**相邻两个维度**组成一个二维平面，在该平面上把向量**旋转 m·θᵢ 角度**（θᵢ 是第 i 个平面专属的角速度）。二维情形下内积为：

$$
\langle R(m\theta)q,\; R(n\theta)k\rangle = q^\top R((m-n)\theta) k
$$

——旋转矩阵相乘等价于角度相减，相对位置 m−n 自动出现在内积里。这就是「**绝对位置编码的形式，相对位置编码的效果**」。

对 d 维向量，把 d/2 个二维平面各配一个角速度：

$$
\theta_i = \text{base}^{-2i/d},\qquad i = 0,1,\dots,d/2-1
$$

- 低维平面（i 小）θ 接近 1：转得快，编码**局部/细粒度**位置；
- 高维平面（i 大）θ 接近 0：转得慢，编码**全局/长程**位置。
- `base`（rope_theta）越大，高频分量越少、低频越丰富，长文本外推越好。MiniMind/Qwen 用 10⁶。

写成矩阵形式后可以合并成一个 2×2 复数乘法；工程上通常展开成「cos/sin 逐元素乘 + rotate_half」的形式（本仓库即如此）。

### 1.3 训练时与推理时的行为

- **训练**：一次前向整段序列，取第 0..S-1 行的 cos/sin。
- **推理（KV Cache 增量解码）**：每步只喂 1 个新 token，此时 RoPE 表必须从 `start_pos`（缓存长度）开始切——位置 m 直接查表第 m 行，缓存里的旧 K 已经带着正确的旋转，无需重算（见 01-整体架构 2.4 的 `start_pos`）。

---

## 2. 逐行精读

### 2.1 precompute_freqs_cis（62-78 行）——造 cos/sin 查找表

```python
def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6, rope_scaling: dict = None):
    freqs, attn_factor = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)), 1.0
```

| 行 | 讲解 |
|----|------|
| 62 | `dim`：**每个注意力头的维度**（96，不是 hidden_size！RoPE 逐头旋转）；`end`：表的最大位置数（32768）；`rope_base`：θ 底数；`rope_scaling`：YaRN 参数或 None |
| 63 | 频率序列 θᵢ：<br>• `torch.arange(0, dim, 2)` = [0, 2, 4, ..., 94]，共 48 个<br>• `[:dim//2]` 截断保险（dim 为奇数时也成立）<br>• `.float()/dim` = 2i/d<br>• `rope_base ** (...)` = 10^(6·2i/96)<br>• `1.0 / ...` = θᵢ = base^(−2i/d)<br>• `attn_factor` 是 YaRN 的幅度补偿因子，非外推时 =1.0。<br>结果 `freqs` 形状 `[48]`，从 1.0（转最快）单调递减到约 0.41（转最慢，10^(−12/8)？不必背，理解「单调递减」即可） |

> 修正一个直觉：θ₀=1 表示最低维平面每前进 1 个 token 转 1 弧度；θ₄₇≈10⁻⁶·...——转得极慢，负责长程。

```python
    if rope_scaling is not None: # YaRN: f'(i) = f(i)((1-γ) + γ/s), where γ∈[0,1] is linear ramp
        ...
        freqs = freqs * (1 - ramp + ramp / factor)
```

| 行 | 讲解 |
|----|------|
| 64 | YaRN 分支整体略过（详见 07-YaRN）。行内注释给出核心公式：对每个频率按 ramp 系数 γ 在「原频率 1」和「插值频率 1/factor」之间线性过渡 |

```python
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin
```

| 行 | 讲解 |
|----|------|
| 74 | 位置序列 t = [0, 1, ..., 32767]，形状 `[32768]` |
| 75 | `torch.outer(t, freqs)`：外积 → 矩阵 `[32768, 48]`，第 (m, i) 个元素 = **m·θᵢ**（位置 m 在第 i 平面的旋转角）。这就是「位置 × 频率 = 角度」 |
| 76 | cos 表：先 cos 得 `[32768, 48]`，再 `cat` 自己一遍 → **`[32768, 96]`**。为什么要复制拼接？看下面 rotate_half 的实现技巧 |
| 77 | sin 表同上。`* attn_factor`（YaRN 时非 1，用于补偿注意力熵；普通 RoPE =1 无影响） |
| 78 | 返回 `(cos, sin)` 两张表。模型 `__init__` 里把它们注册为非持久 buffer（见 01-整体架构 211-212 行） |

**为什么 cat 两次？** 配合 `rotate_half`：把 q 分成前后两半 `[q1, q2]`（各 48 维），cos/sin 复制成 `[cos, cos]` 后，`q*cos + rotate_half(q)*sin` 展开就是两个二维平面的旋转：

- 前半：`q1·cos − q2·sin`（rotate_half 后半部分为 −q2）
- 后半：`q2·cos + q1·sin`

这正是复数 `(q1 + i·q2) · e^{i·m·θ}` 的实部/虚部——**即二维旋转矩阵作用**。cat 复制 + rotate_half 是把「成对维度旋转」写成「整向量逐元素乘」的等价技巧，避免真正的复数运算，完全可向量化。

### 2.2 apply_rotary_pos_emb（80-84 行）——执行旋转

```python
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    def rotate_half(x): return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)
    q_embed = ((q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))).to(q.dtype)
    k_embed = ((k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))).to(k.dtype)
    return q_embed, k_embed
```

| 行 | 讲解 |
|----|------|
| 80 | 入参：q/k 形状 `[B, S, n_heads, head_dim]`；cos/sin 形状 `[S, head_dim]`（由调用处从表里切片）；`unsqueeze_dim=1` 表示在序列维前插入 |
| 81 | `rotate_half`：`[-x后半, x前半]`。例：x=[a,b,c,d] → [−c,−d,a,b]。配合逐元素乘 cos/sin 实现成对旋转（见 2.1 的推导） |
| 82 | q 旋转：`q·cos + rotate_half(q)·sin`，逐元素乘法 + 广播（cos 从 `[S, 96]` unsqueeze 成 `[1, S, 1, 96]`，能广播到 `[B, S, heads, 96]`），最后 `.to(q.dtype)` 降回原精度（cos/sin 表是 fp32） |
| 83 | k 同样旋转。**注意 v 不旋转**——位置信息只进 Q/K，V 只携带内容 |
| 84 | 返回旋转后的 q, k |

### 2.3 调用链与形状总览

```
MiniMindModel.forward:
    position_embeddings = (freqs_cos[start:start+S], freqs_sin[start:start+S])   # [S, 96]
        │
MiniMindBlock → Attention.forward:
    xq, xk = self.q_norm(xq), self.k_norm(xk)     # 先 QK-Norm（见 04 篇）
    cos, sin = position_embeddings                # [S, 96]
    xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)   # [B, S, 8, 96] / [B, S, 4, 96]
```

两个关键正确性点：

1. **cos/sin 按「绝对位置」切片**：训练从 0 切；KV Cache 推理从 `start_pos` 切。表是全局的，切片位置即 token 的绝对位置。
2. **Q 和 K 用同一张表**：正因为如此，点积里旋转角相减，得到相对位置——绝对编码的形式、相对编码的效果。

---

## 3. 实验直觉：为什么 θ 从 1 衰减到 ~0

把位置序列 [0..32767] 画在 48 个平面上：低维平面（θ≈1）上向量每 2π 个 token 转一整圈——像秒针，能区分相邻 token；高维平面（θ≈1e-6）转一圈需要上百万 token——像时针，编码「离段落开头多远」。注意力可以同时利用「局部精确匹配」和「长程粗对齐」。

## 4. 本篇小结

1. RoPE = **按位置旋转 Q/K**，内积自动带上相对位置；V 不转。
2. `precompute_freqs_cis` 只算一张「位置×频率」的 cos/sin 大表（[32768, 96]），构造时算一次，forward 切片。
3. `apply_rotary_pos_emb` 用 `q*cos + rotate_half(q)*sin` 等价实现成对维度旋转；fp32 表乘完降回 bf16。
4. 切片起点 `start_pos` 支撑 KV Cache 增量解码；YaRN 分支修改频率序列实现长文本外推（下一篇的下一篇详述）。

下一篇：[04-注意力机制与GQA.md](./04-注意力机制与GQA.md)
