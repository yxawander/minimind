# 06 - MoE 混合专家

> 覆盖代码：`model/model_minimind.py` **第 148-181 行** 的 `MOEFeedForward` 类。
> 它在 `MiniMindBlock.__init__`（189 行）里替换 dense FFN：`self.mlp = MOEFeedForward(config) if config.use_moe`。
> 前置阅读：[05-FFN与SwiGLU](./05-FFN与SwiGLU.md)（专家就是缩小版 FeedForward）。

---

## 1. 原理

### 1.1 MoE 要解决什么问题

密集模型：每个 token 都经过**全部**参数。MoE（Mixture of Experts）思路：放 N 个「专家」（各自是独立 FFN），由一个**路由器**为每个 token 挑 k 个专家。效果：

- **总参数量**扩大（知识容量大，4 专家 × 各自 FFN）；
- **计算量**只按激活的 k 个专家算（MiniMind：N=4、k=1 → 每个 token 只算 1 个专家的 FFN）。

这就是「198M-A64M」的含义：总参 198M，每 token 激活约 64M（激活部分约等于 dense 版 64M 的 FFN 开销）。

### 1.2 路由的数学

对 token 向量 x（768 维）：

1. **打分**：`scores = softmax(gate(x))`，gate 是 [768→N] 的线性层，softmax 后是 N 维概率分布。
2. **选专家**：`topk(scores, k)` 取概率最大的 k 个，得到权重 w 与下标 idx。
3. **加权混合**：`y = Σ_i w_i · Expert_{idx_i}(x)`。

k>1 时（如 Qwen3-MoE 的 top-2 之外情形）权重需要**归一化**（`norm_topk_prob`：把 k 个权重除以它们的和，重新变成凸组合）。**k=1 时归一化无意义（单个权重归一化后恒为 1）**——但梯度也不能断，MiniMind 用了一个很巧的**直通估计器（straight-through）**，见 2.4 节。

### 1.3 负载均衡 aux loss

纯按概率路由会「赢者通吃」：少数专家早期略优 → 被路由得更多 → 进一步更强 → 其余专家饿死。标准对策是加一个**可微的负载均衡辅助损失**（Switch Transformer 提出）：

$$
\text{aux} = \alpha \cdot N \cdot \sum_{i=1}^{N} f_i \cdot p_i
$$

- `f_i` = 实际被路由到专家 i 的 token 比例（one-hot 统计，不可导）；
- `p_i` = 路由器给专家 i 的平均概率（可导）；
- 二者内积在**均匀分布时最小**（1/N × 1/N 求和 = 1/N），某专家过热时 f_i、p_i 同时大 → 惩罚大；
- 乘 N 使量纲稳定；乘系数 α（`router_aux_loss_coef`=5e-4）控制强度。

aux loss 在每层 MoE forward 里算好、挂到模块属性上，由 `MiniMindModel.forward`（236 行）汇总，trainer 里 `res.loss + res.aux_loss` 合入总损失（不乘 accumulation，直接同量级相加）。

---

## 2. 逐行精读

```python
class MOEFeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList([FeedForward(config, intermediate_size=config.moe_intermediate_size) for _ in range(config.num_experts)])
        self.act_fn = ACT2FN[config.hidden_act]
```

| 行 | 讲解 |
|----|------|
| 151 | 缓存 config（forward 里要读 `num_experts_per_tok`、`norm_topk_prob`、`router_aux_loss_coef`） |
| 152 | **路由器**：[768 → 4] 的线性层，输出每个专家的 logits（无 bias）。参数极少（3072 个），但它是 MoE 的灵魂 |
| 153 | **N=4 个专家**：每个都是标准 SwiGLU `FeedForward`，但中间维传 `moe_intermediate_size`（默认=dense 的 2432；官方 MoE 版会配更小值）。`nn.ModuleList` 确保全部专家参数被注册 |
| 154 | act_fn 其实没在 forward 里用（激活在专家内部）——冗余属性，无害 |

```python
    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.view(-1, hidden_dim)
```

| 行 | 讲解 |
|----|------|
| 157-158 | 输入 `[B, S, 768]` → 摊平成 `[B×S, 768]`。**摊平是因为路由是逐 token 的**：batch/序列位置不影响路由决策，摊平后统一处理。后面专家也是按「选中的 token 子集」计算，摊平让索引简单 |

```python
        scores = F.softmax(self.gate(x_flat), dim=-1)
        topk_weight, topk_idx = torch.topk(scores, k=self.config.num_experts_per_tok, dim=-1, sorted=False)
```

| 行 | 讲解 |
|----|------|
| 159 | 路由打分：`[N_tokens, 768] @ [768, 4]` → `[N_tokens, 4]`，softmax 成概率分布（每个 token 一行，和为 1） |
| 160 | `topk`：取概率最大的 k 个。返回 `topk_weight` `[N_tokens, k]`（选中的概率值）、`topk_idx` `[N_tokens, k]`(专家编号)。`sorted=False`：不按大小排序，省一步（后面按专家索引分组，顺序无所谓） |

```python
        if self.config.norm_topk_prob:
            if self.config.num_experts_per_tok > 1: 
                topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
            else:
                top1 = torch.topk(F.softmax(self.gate(x_flat.detach()), dim=-1), k=1, dim=-1, sorted=False)[0]
                topk_weight = top1 - top1.detach() + 1.0 # k=1: 1.0 forward on purpose, gradient via straight-through
```

| 行 | 讲解 |
|----|------|
| 161 | 进入权重处理分支 |
| 162-163 | **k>1**：把 k 个权重归一化成凸组合（除以它们的和；+1e-20 防零）。例如 top-2 概率 [0.7, 0.2] → [0.78, 0.22] |
| 164 | **k=1（本仓库默认）**：走直通估计器。先把 x **detach** 重新过一遍 gate + softmax + topk，拿到 `top1`（只取权重，不取索引） |
| 165 | **本行的魔术**：`top1 - top1.detach() + 1.0`。<br>• **前向数值**：`top1 - top1 + 1 = 1.0`——k=1 时权重就该是 1（没有第二个专家分摊），且 detach 版的 top1 与原 top1 数值完全相同，所以前向**精确等于 1.0**，不引入数值误差；<br>• **反向梯度**：`-top1.detach()` 是常数（梯度为 0），`+1.0` 是常数，于是 `d(topk_weight)/d(gate) = d(top1)/d(gate)`——**梯度原封不动流向路由器**！<br>如果直接写 `topk_weight = 1.0`，路由器完全收不到梯度（专家输出的梯度绕过了 gate）；如果直接用 `top1` 作权重，前向就变成「按概率加权」而不是「硬路由」，改变了 k=1 的语义。直通估计器 = **前向硬路由、反向有梯度**，两全其美。注意 detach 的那次 gate 前向是为了拿「同样的 top1 值」但切断其对 autograd 图的影响——其实可以直接 `topk_weight = topk_weight - topk_weight.detach() + 1.0`，多算一次前向只是实现风格 |

```python
        y = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            mask = (topk_idx == i)
            if mask.any():
                token_idx = mask.any(dim=-1).nonzero().flatten()
                weight = topk_weight[mask].view(-1, 1)
                y.index_add_(0, token_idx, (expert(x_flat[token_idx]) * weight).to(y.dtype))
            elif self.training:
                y[0, 0] += 0 * sum(p.sum() for p in expert.parameters())
```

| 行 | 讲解 |
|----|------|
| 167 | 输出容器：与输入同形状的零张量 `[N_tokens, 768]` |
| 168 | 逐专家循环（4 次） |
| 169 | `mask = (topk_idx == i)`：`[N_tokens, k]` 布尔矩阵，True 表示「该 token 的第几个选中专家是 i」 |
| 170 | `mask.any(dim=-1)`：该 token 是否用了专家 i（k 维任一为 True）→ `[N_tokens]`；`.nonzero().flatten()` 拿到被选中 token 的**全局下标列表** |
| 171 | `topk_weight[mask]`：布尔索引取出这批 token 对应的权重值（形状 [选中数×k 中等于 i 的那些]），view 成列向量 |
| 172 | **核心计算**：<br>1. `x_flat[token_idx]`：只把**被路由到 i 的 token** 抽出来（按专家分桶，每个专家只算自己的 token——这就是稀疏激活）；<br>2. `expert(...)`：过该专家的 SwiGLU；<br>3. `* weight`：乘路由权重；<br>4. `.to(y.dtype)`：专家在 autocast 下输出可能是 fp32，转回；<br>5. `y.index_add_(0, token_idx, ...)`：**按下标累加**回输出——因为一个 token 可能被多个专家处理（k>1），累加才能完成 Σ 混合。`index_add_` 是可导 scatter-add（重复下标梯度自动累加） |
| 174-175 | **DDP 兼容补丁**：训练中若某专家**一个 token 都没被分到**（mask 全 False），它的参数不参与本次前向 → DDP 同步梯度时会发现「该专家的参数没有梯度」，直接报错挂掉。对策：`y[0,0] += 0 × Σ参数`——数值上是加 0（不影响结果），但计算图里包含了这些参数，DDP 就能拿到全零梯度。这是 MoE + DDP 的经典坑，值得记住 |

```python
        if self.training and self.config.router_aux_loss_coef > 0:
            load = F.one_hot(topk_idx, self.config.num_experts).float().mean(0)
            self.aux_loss = (load * scores.mean(0)).sum() * self.config.num_experts * self.config.router_aux_loss_coef
        else:
            self.aux_loss = scores.new_zeros(1).squeeze()
        return y.view(batch_size, seq_len, hidden_dim)
```

| 行 | 讲解 |
|----|------|
| 176 | 只在训练且系数>0 时计算 aux loss |
| 177 | `load = f_i`：`F.one_hot(topk_idx, N)` → `[N_tokens, k, N]`，`.mean(0)` → **每个专家被选中的平均比例** `[N]`（k=1 时即「token 分给各专家的占比」）。one_hot 由整数下标构成、本身没有梯度通路，所以在乘法中天然是**常数项**——可导性全部由另一因子 `scores.mean(0)` 提供 |
| 178 | `scores.mean(0)` = `p_i`：路由器平均概率 `[N]`（**可导**，梯度流向 gate）。`(load * p).sum()` = 内积 `Σ f_i·p_i`——均匀时最小（=1/N）；乘 N 归一量纲；乘系数 α=5e-4 控制强度。存到 `self.aux_loss`，等 `MiniMindModel.forward`（236 行）来汇总 |
| 179-180 | 推理时或系数为 0：给一个 **dtype/设备一致的 0**（`scores.new_zeros(1).squeeze()`），保证上层 `res.loss + res.aux_loss` 不因 None/类型不匹配而崩 |
| 181 | 摊平的结果 reshape 回 `[B, S, 768]`，作为 Block 的 mlp 输出进入残差 |

---

## 3. 数值实验：路由在干什么

假设某 token 的 gate 输出 logits = [2.1, 0.3, −0.7, 0.1]：

```
softmax → scores = [0.72, 0.12, 0.04, 0.09]  （4 专家概率，和=1）
top-1   → topk_idx=[0], topk_weight=[0.72]
直通    → 实际乘的权重 = 1.0（前向），梯度经 0.72 流向 gate
输出    → y = 1.0 × Expert_0(x)
```

某 batch 里若 90% 的 token 都流向 Expert_0：`f = [0.9, 0.03, 0.03, 0.04]`，`p ≈ [0.65, ...]`，`Σ f·p ≈ 0.6`，而均匀时 `Σ f·p = 1/4 = 0.25` → aux loss 明显偏大 → 梯度推着 gate 把概率摊平。

## 4. 与 dense 版的成本对比

| | Dense FFN | MoE（N=4, k=1） |
|--|-----------|-----------------|
| 每 token FFN 矩阵乘 | 1 套 [768×2432×2+…] | 1 个专家（+router 一次 [768×4]） |
| 参数量 | ≈5.6M/层 | ≈5.6M×4 + router/层 |
| 显存 | 小 | 4 倍 FFN 参数 |
| 效果 | — | 同算力下容量更大；但训练更难稳（路由漂移） |

## 5. 本篇小结

1. **路由三步**：softmax 打分 → topk 选专家 → 加权混合（index_add_ 累加）。
2. **k=1 直通估计器**：`w - w.detach() + 1.0` 让前向权重恒为 1、梯度照常流向路由器——本仓库最精巧的一行。
3. **负载均衡**：`aux = α·N·Σ f_i·p_i`，f 是实际负载（常数）、p 是路由概率（可导），逼着 gate 摊平。
4. 两个工程坑：空专家触发 DDP 报错（用 `0×Σ参数` 挂图）；aux_loss 必须给 dtype 一致的 0 兜底。
5. 专家 = 缩小版 SwiGLU（`moe_intermediate_size` 可调），路由器只是一个 [768→4] 线性层。

下一篇：[07-YaRN长文本外推.md](./07-YaRN长文本外推.md)
