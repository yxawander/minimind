# 07 - YaRN 长文本外推

> 覆盖代码：`model/model_minimind.py` **第 31-39 行**（config 开关）、**第 64-73 行**（`precompute_freqs_cis` 里的 YaRN 分支）。
> 前置阅读：[03-RoPE旋转位置编码](./03-RoPE旋转位置编码.md)。

---

## 1. 原理：为什么模型「读不了」比训练更长的文本

模型训练时只见过 0~2047 的位置（`original_max_position_embeddings=2048`）。直接把 3 万位置喂给 RoPE，低维平面（θ≈1）的角度 m·θ 超出训练时见过的相位范围，注意力得分分布崩坏 → 输出乱码。

**位置编码外推/插值**试图解决「训练短、推理长」：

| 方法 | 思路 | 问题 |
|------|------|------|
| 直接外推 | 位置超了也硬算 | 相位没见过，崩 |
| 线性插值 PI | 位置 m → m/factor，把 32768 压进 2048 的训练范围 | **所有**频率被等比压缩，高频平面相位变化过慢，相邻 token 区分度下降，短文本能力受损 |
| NTK-aware | 调大底数（θ' = θ·factor^(d/(d−2))），低频多压、高频少压 | 手感好但无统一理论 |
| **YaRN** | 按「频率的重要性」**分区插值** + 注意力熵补偿 | 效果最好，Llama/Qwen 生态事实标准 |

### 1.1 YaRN 的两个核心机制

**① 分频段 ramp 插值**。定义 ramp 函数 γ(i)∈[0,1]（第 i 个频率平面的插值比例）：

$$
\theta_i' = \theta_i \cdot \big((1-\gamma_i) + \gamma_i / \text{factor}\big)
$$

- γ=0 → 频率不变（**高频端保持原样**，短距离语义不受损）；
- γ=1 → 完全插值到 1/factor（**低频端压缩**，把长距离纳入训练范围）；
- 中间用 ramp 平滑过渡，避免分段跳变。

γ 由**波长的经验界**决定：每个频率的「波长」= 2π/θᵢ（转一圈需要多少 token）。YaRN 用两个阈值 β_fast、β_slow（单位：token 数）：

- 波长 < β_fast（=32）的频率：周期太短，训练中早就见过完整圆周 → **无需插值**，γ=0；
- 波长 > β_slow（=1）×……的频率：周期极长，训练时连一小段弧都没转完 → 相位信息不可靠，**完全插值**，γ=1；
- 中间线性过渡。

**② 注意力熵补偿（attention_factor）**：插值后整体点积尺度变化，softmax 熵会漂移；YaRN 给注意力乘一个温度补偿因子（本仓库配置为 1.0，即未启用幅度补偿——只做频率插值的最简版）。

### 1.2 MiniMind 的用法

- **训练时**：`inference_rope_scaling=False`，`rope_scaling=None`，用原始 RoPE 训 2048/8192 长度。
- **推理时**：命令行 `--inference_rope_scaling` 打开 → factor=16 → 表达式上把 2048 扩到 32768。改的只是**那张 cos/sin 表**，模型权重零改动——这就是位置编码类方案的最大优点：外推不重训。

---

## 2. 逐行精读

### 2.1 config 部分（31-39 行）

```python
self.inference_rope_scaling = kwargs.get("inference_rope_scaling", False)
self.rope_scaling = {
    "beta_fast": 32, "beta_slow": 1, "factor": 16,
    "original_max_position_embeddings": 2048,
    "attention_factor": 1.0, "type": "yarn"
} if self.inference_rope_scaling else None
```

| 字段 | 含义 |
|------|------|
| `factor` | 插值因子：目标长度 = 2048 × 16 = 32768 |
| `original_max_position_embeddings` | 训练时见过的最大位置 2048 |
| `beta_fast` | 32：波长 < 32 token 的频率保持原频率 |
| `beta_slow` | 1：波长 > … 的频率完全插值（取 1 是极激进设置，几乎全部中低频都插值） |
| `attention_factor` | 注意力幅度补偿，1.0 = 不补偿 |
| `type: "yarn"` | 方法标识（对齐 HF 生态的 rope_scaling 字典格式） |

### 2.2 precompute_freqs_cis 的 YaRN 分支（64-73 行）

```python
def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6, rope_scaling: dict = None):
    freqs, attn_factor = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)), 1.0
    if rope_scaling is not None: # YaRN: f'(i) = f(i)((1-γ) + γ/s), where γ∈[0,1] is linear ramp
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048), rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0), rope_scaling.get("beta_slow", 1.0), rope_scaling.get("attention_factor", 1.0)
        )
        if end / orig_max > 1.0:
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
            low, high = max(math.floor(inv_dim(beta_fast)), 0), min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001), 0, 1)
            freqs = freqs * (1 - ramp + ramp / factor)
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin
```

| 行 | 讲解 |
|----|------|
| 63 | 原始频率 θᵢ（与普通 RoPE 相同），`attn_factor` 先置 1 |
| 65 | 进入 YaRN 分支。行内注释即核心公式：`f'(i) = f(i)·((1−γ) + γ/factor)` |
| 66-68 | 从字典解包 5 个超参（给默认值兜底） |
| 69 | 只有目标长度 `end` **超过**训练长度才需要插值；推理如果没超长，走原始 RoPE 更安全 |
| 70 | **「转数」→「频率下标」的换算函数** `inv_dim(b)`：求「在训练窗口 orig_max 内恰好完整旋转 b 圈」的频率下标 i。<br>YaRN 论文的原生定义用的是**转数**而非波长：位置 orig_max 处的相位 = orig_max·θᵢ，令它等于 2π·b（转 b 圈）→ `θᵢ = 2πb/orig_max`；<br>代入 `θᵢ = base^(−2i/d)` 取对数解出 i：`i = d·ln(orig_max/(2πb)) / (2·ln(base))`——正是代码表达式。<br>**i 越小 → 频率越高 → 在训练窗口里转的圈数越多**。β_fast=32 表示「训练中已转过 32 圈以上的高频平面」，β_slow=1 表示「整个训练期只转 1 圈的低频平面」 |
| 71 | `low = inv_dim(β_fast)`：波长 32 token 对应的下标（波长更短、下标更小的全部**不插值**）；`high = inv_dim(β_slow)`：波长 1 token 之下界（几乎全部中低频都**完全插值**）。`max(...,0)`/`min(...,dim//2-1)` 夹到合法区间。注意 β_slow=1 时 `inv_dim(1)` 很大 → `high` 被夹到 47（最后一个下标），即**几乎整段频率都参与插值**——这是 MiniMind 选择的激进配置 |
| 72 | ramp = γ(i)：把下标从 low→high 线性映射到 0→1，clamp 截断（<low 为 0，>high 为 1）；`max(high-low, 0.001)` 防除零。γ=0 → 行 73 里系数 = 1（原频率）；γ=1 → 系数 = 1/factor（完全插值） |
| 73 | **逐频率混合**：`freqs = θᵢ·((1−γᵢ) + γᵢ/factor)`，正是一开始注释的公式。之后 cos/sin 表的构造（74-78 行）与普通 RoPE 完全一致——**YaRN 只改这张频率表，模型其余部分零改动** |
| 76-77 | `* attn_factor`：YaRN 论文的注意力熵补偿乘子（本仓库=1.0，占位） |

### 2.3 调用时机

- 模型 `__init__`（210 行）：`rope_scaling=config.rope_scaling` → 训练时 None，原始表。
- 推理脚本 `eval_llm.py/serve_openai_api.py` 传 `--inference_rope_scaling` → config 打开 → 表按 YaRN 重建 → 同一份权重直接读长文本。
- 338 行那个 `if self.freqs_cos[0,0]==0` 的 meta 设备重算分支同样会带上当前 rope_scaling——加载 transformers 5.x 格式时也保持外推配置。

### 2.4 与 NTK/PI 的直观对比（48 个频率视角）

```
频率下标 i:   0 ──────────── low(≈?) ──────────── high(=47)
波长:        2 token ……… 32 token ……………… 超长
YaRN 系数:   1.0(不动) ────线性过渡──── 1/16(压 16 倍)
```

- PI：所有 i 一律 ÷16（高频被压坏）；
- NTK：底数换大，近似「低频多压高频少压」，但无显式 ramp；
- **YaRN：显式给出每个频率的插值比例**，兼顾短程与长程。

## 3. 本篇小结

1. 外推问题 = 训练没见过的 RoPE 相位；YaRN 用「**分频段 ramp 插值**」：高频不动、低频压 1/factor、中间线性过渡。
2. `inv_dim(b)` 把「波长阈值（β_fast/β_slow）」换算成频率下标界 low/high，决定 ramp 的位置。
3. 全部改动只发生在 cos/sin **频率表**上（64-73 行），模型权重与 forward 完全不变——所以可以推理期一键开启。
4. MiniMind 配置：factor=16（2048→32768）、β_slow=1（几乎全域插值）、attention_factor=1（不做熵补偿，最简版）。

下一篇：[08-生成解码策略.md](./08-生成解码策略.md)
