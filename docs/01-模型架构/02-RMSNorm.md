# 02 - RMSNorm（归一化层）

> 覆盖代码：`model/model_minimind.py` **第 50-60 行** 的 `RMSNorm` 类。
> 它在模型里出现的位置：每个 Block 的两个 Norm（`input_layernorm`、`post_attention_layernorm`）、最终的 `model.norm`、注意力内部的 q/k norm（`self.q_norm/self.k_norm`，用法相同但维度不同）。

---

## 1. 原理：从 LayerNorm 到 RMSNorm

### 1.1 LayerNorm 回顾

对最后一个维度（特征维 d）做归一化，把每个 token 的特征向量拉成均值 0、方差 1：

$$
\text{LN}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \gamma + \beta,\qquad
\mu = \frac{1}{d}\sum_i x_i,\quad \sigma^2 = \frac{1}{d}\sum_i (x_i - \mu)^2
$$

- γ（scale）、β（bias）是可学习参数，维度都是 d。
- 作用：控制每层的数值尺度，稳定训练；让不同样本、不同位置的激活分布一致。

### 1.2 RMSNorm 做了什么减法

RMSNorm（Root Mean Square Normalization，来自论文 *Root Mean Square Layer Normalization*, 2019）发现：**均值中心化那一步（减 μ）对效果贡献很小，但计算不便宜**，于是去掉它、去掉 bias，只保留「按均方根缩放」：

$$
\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2 + \epsilon}} \odot \gamma
$$

| 对比项 | LayerNorm | RMSNorm |
|--------|-----------|---------|
| 减均值 | 是 | 否 |
| 分母 | 标准差 σ | 均方根 RMS |
| 可学习参数 | γ + β（2d 个） | 只有 γ（d 个） |
| 计算量 | 两次归约（均值、方差） | 一次归约（平方均值） |
| 使用者 | 原始 Transformer/BERT | **Llama / Qwen / MiniMind 等所有现代 LLM** |

去掉中心化的直观解释：Transformer 里向量经过残差累加后，其「均值分量」本身也携带信息；RMSNorm 只关心向量的**长度**（尺度），不强行把分布中心搬回原点，既省算力又少两个可学习矩阵。实测在 LLM 规模下与 LN 效果相当。

### 1.3 数值稳定与精度技巧

归一化公式里有除法和平方和——在 fp16/bf16 下 `Σx²` 可能溢出（768 个数平方和轻松超过 fp16 上限 65504）。所以实现中普遍采用「**升 fp32 归一化，再降回原精度**」的策略，MiniMind 也这么做（见 60 行 `x.float()` / `.type_as(x)`）。

---

## 2. 逐行精读

```python
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return (self.weight * self.norm(x.float())).type_as(x)
```

| 行 | 讲解 |
|----|------|
| 50 | 继承 `nn.Module`。注意没有用 `torch.nn.RMSNorm`（老版本 torch 没有），手写保证兼容 |
| 51-52 | `dim` 是归一化的特征维大小；`eps` 默认 1e-5，但实际构造时传的是 `config.rms_norm_eps=1e-6` |
| 53 | ε 存下来备用 |
| 54 | **可学习缩放向量 γ**，初始化为全 1（即初始时是「纯归一化」，不改变数值）。`nn.Parameter` 让它进入参数列表、参与训练。注意只有 γ，没有 β（原理 1.2 节） |
| 56-57 | `norm()` 是纯数学部分：<br>• `x.pow(2)` 逐元素平方 → `[B, S, d]`<br>• `.mean(-1, keepdim=True)` 对最后一维求均值 → `[B, S, 1]`（`keepdim=True` 保留维度，能直接广播回去）<br>• `+ self.eps` 防止除零（防止整条向量全 0 时梯度爆炸）<br>• `torch.rsqrt(...)` = 1/√(...)，比 `torch.sqrt` 后再相除少一步、数值更稳<br>• `x * rms`：逐元素除以 RMS（乘倒数）。结果就是「把向量缩放到均方根为 1」 |
| 59-60 | forward 分两步：<br>1. `self.norm(x.float())`——**先升 fp32** 做归一化（避免 bf16 下平方和溢出、除法精度损失）；<br>2. `self.weight * ...`——乘上可学习 γ（weight 本身是 fp32）；<br>3. `.type_as(x)`——**降回输入的原精度**（bf16/fp16）。<br>这就是「升 32 算、降 16 存」的标准模式，全仓库所有数值敏感计算（softmax、scores）都遵循它 |

### 2.1 形状追踪

以 attention 前的 norm 为例：输入 `x` 是 `[B, S, 768]` → norm 输出同形状，每行被独立归一化。以 **q/k norm** 为例（04-注意力机制详解）：q 形状 `[B, S, 8, 96]`，在 `[-1]` 即 96 维的**头内维度**上归一化——这是 Qwen3 引入的 QK-Norm，稳定注意力 logits 的尺度。

### 2.2 与训练的交互

- γ 全 1 初始化 + 归一化输出 ≈ 原向量/常数，保证训练初期网络行为接近线性，不引入扰动。
- ε=1e-6 加在**开方之前**（分母整体 +ε），而不是开方之后——两者数值含义不同，前者在 x 全零时给出 1/√ε 的大但有限值。

---

## 3. 本篇小结

1. RMSNorm = LayerNorm 去掉「减均值」和 bias，只保留「除以均方根 + 可学习缩放 γ」。
2. 归一化发生在**每个 Block 的两个子层之前**（Pre-Norm）以及最终 lm_head 之前；q/k 还各有一次头内归一化。
3. 实现要点：`keepdim=True` 维持广播、`rsqrt` 快而稳、**fp32 升降**防精度问题。
4. 可学习参数只有 `weight`（d 个标量），初始化为全 1。

下一篇：[03-RoPE旋转位置编码.md](./03-RoPE旋转位置编码.md)
