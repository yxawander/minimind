# 05 - FFN 与 SwiGLU

> 覆盖代码：`model/model_minimind.py` **第 136-146 行** 的 `FeedForward` 类。
> 它被两处使用：每个 dense Block 的 `mlp`（189 行）、以及 MoE 里每个专家（153 行，传入更小的 `moe_intermediate_size` 可选）。

---

## 1. 原理

### 1.1 FFN 在 Transformer 里的角色

注意力负责「token 之间交换信息」（横向），FFN 负责「每个 token 独立地深度加工」（纵向）。经验上，**模型的大部分参数和「知识存储」都在 FFN 里**——FFN 被普遍视作 key-value 记忆库。参数量对比：注意力四条投影 ≈ 4×d²，FFN ≈ 3×d×d_ff（SwiGLU 三条投影），d_ff≈3.2×d，FFN 占单层参数约 2/3。

### 1.2 从经典 FFN 到 SwiGLU

**经典两段式（GPT-2/BERT）**：

$$
\text{FFN}(x) = W_2\,\text{GELU}(W_1 x)
$$

**GLU 门控族**：引入一个「门」分支，让激活函数控制信息通路：

$$
\text{GLU}(x) = (W_1 x) \otimes \sigma(W_v x)
$$

**SwiGLU** = GLU 门控 + SiLU 激活（SiLU(x)=x·sigmoid(x)，也叫 swish）：

$$
\text{SwiGLU}(x) = \text{down}\Big(\text{SiLU}(\text{gate}(x)) \otimes \text{up}(x)\Big)
$$

- `gate_proj`：门分支，过 SiLU（输出 −∞..∞，负值被压向 0 但保留小负值，平滑可导）；
- `up_proj`：内容分支，恒等直通；
- 两者**逐元素相乘**（⊗）——门决定哪些维度放行；
- `down_proj`：投影回 d 维。

SwiGLU 论文（*GLU Variants Improve Transformer*, 2020）实测它在同参数量下优于 GELU-FFN，此后成为 LLM 标配（Llama/Qwen/MiniMind 全用）。代价是投影从 2 条变 3 条，所以中间维要缩一点保持参数量持平——这正是 Config 里 π 公式的来历。

### 1.3 intermediate_size 的 π 公式

```python
self.intermediate_size = kwargs.get("intermediate_size", math.ceil(hidden_size * math.pi / 64) * 64)
```

- 传统 FFN 隐层取 4×d；SwiGLU 三投影要 `3×d×d_ff` 参数，为与「4×d 经典 FFN」≈ 持平：`3×d×d_ff ≈ 2×d×4d → d_ff ≈ 8d/3 ≈ 2.67d`。实践中大家取 `d_ff = 2/3 × 4d` 附近，配 π≈3.14 也一样能对齐：`d_ff = π·d ≈ 3.14d`。
- `ceil(.../64)*64`：向上对齐到 64 的倍数——GPU tensor core 对 64 倍数维度的矩阵乘吞吐最好。
- 768：768×π/64 = 37.7 → 38 → **2432**。所以每层 FFN 三条矩阵：[768→2432]×2 + [2432→768]。

---

## 2. 逐行精读

```python
class FeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig, intermediate_size: int = None):
        super().__init__()
        intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
```

| 行 | 讲解 |
|----|------|
| 137-138 | 构造参数 `intermediate_size=None`：允许调用方覆盖中间维。**这个默认参数的存在正是为 MoE 服务的**——MOEFeedForward 构造每个专家时传 `intermediate_size=config.moe_intermediate_size`，让专家比 dense FFN 更窄（详见 06-MoE）。`None or config.intermediate_size`：没传就用 config 的 2432 |
| 139 | **门投影**：768 → 2432。命名对齐 HF（`gate_proj`），权重迁移到 Qwen3 时键名直接匹配 |
| 140 | **降维投影**：2432 → 768。声明顺序故意是 gate→down→up（Llama 源码风格），保持 state_dict 键序一致便于转换工具处理 |
| 141 | **升维投影**：768 → 2432。三条矩阵全部无 bias |
| 142 | `ACT2FN['silu']` 从 transformers 拿 SiLU 函数（`x·sigmoid(x)`）。用字符串配置而非硬编码，方便实验 gelu 等变体 |
| 145 | **一行完成 SwiGLU**，从内到外读：<br>1. `self.gate_proj(x)`：门分支 `[B,S,2432]`<br>2. `self.act_fn(...)`：SiLU 门控激活<br>3. `* self.up_proj(x)`：与直通的内容分支逐元素相乘（广播不涉及，形状相同）<br>4. `self.down_proj(...)`：降回 768，作为子层输出进入残差 |

### 2.1 形状追踪（单层，B=4, S=340）

```
x:                [4, 340, 768]
gate_proj(x):     [4, 340, 2432]
SiLU(...):        [4, 340, 2432]
up_proj(x):       [4, 340, 2432]
二者逐元素乘:      [4, 340, 2432]
down_proj(...):   [4, 340, 768]   → 回到 Block 做残差
```

### 2.2 参数量核算

单层 FFN：3 × 768 × 2432 ≈ 560 万。8 层 ≈ 4480 万。
单层注意力（GQA）：Q 768×768 + K/V 768×384×2 + O 768×768 + QK-norm ≈ 118 万。8 层 ≈ 946 万。
加 embedding（绑定后算一份）491 万 → 64M 级别，与官方标称一致。

---

## 3. 本篇小结

1. FFN = **门控前馈**：`down(SiLU(gate(x)) ⊙ up(x))`，三条无 bias 投影。
2. SwiGLU 相比 GELU-FFN 在同参数量下更优，是 Llama/Qwen 以来的事实标准。
3. `intermediate_size` 默认 `ceil(d×π/64)×64` = 2432，兼顾效果与 GPU 对齐。
4. `intermediate_size` 形参是为 MoE 专家「缩小版 FFN」准备的插槽。

下一篇：[06-MoE混合专家.md](./06-MoE混合专家.md)
