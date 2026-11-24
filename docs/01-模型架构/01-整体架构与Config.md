# 01 - 整体架构与 Config（model_minimind.py 上半部）

> 覆盖代码：`model/model_minimind.py` 的 **第 1-9、10-45（Config）、183-199（Block）、201-237（Model）、239-258（ForCausalLM）行**。
> 组件级代码（RMSNorm/RoPE/Attention/FFN/MoE）各有独立文档，本篇只讲「总装」。
> 生成相关（generate 方法，262-293 行）见 [08-生成解码策略](./08-生成解码策略.md)。

---

## 1. 原理：Decoder-only Transformer 的总装

### 1.1 整体数据流

MiniMind 是一个标准的 **Decoder-only Transformer**（和 GPT/Llama/Qwen 同构）。一段 token 序列进来，经过如下流水线：

```
input_ids [B, S]                        # B=batch，S=序列长度
   │
   ▼
embed_tokens: Embedding(6400, 768)      # token → 768 维向量，[B, S, 768]
   │
   ▼
┌───────────── 8 × MiniMindBlock ─────────────┐
│  x = x + Attention(RMSNorm(x))              # 残差 + 注意力
│  x = x + FFN(RMSNorm(x))                    # 残差 + 前馈网络（dense=SwiGLU / MoE）
└──────────────────────────────────────────────┘
   │
   ▼
norm: RMSNorm(768)                      # 最后一个 LayerNorm
   │
   ▼
lm_head: Linear(768, 6400)              # 只对需要的位置算（logits_to_keep）
   │
   ▼
logits [B, S, 6400] → 与 labels 做交叉熵 → loss
```

值得注意的三个设计：

1. **Pre-Norm 结构**：LayerNorm 放在子层（注意力/FFN）**之前**而不是之后（`x + Sub(RMSNorm(x))`）。这是 GPT-2 以后的主流做法——残差主路保持「干净」的恒等通路，训练更稳定，深层模型不 Pre-Norm 基本训不动。
2. **权重绑定（weight tying）**：输入 embedding 矩阵和输出 lm_head 矩阵**共享同一份权重**。词表 6400 × 隐藏维 768 ≈ 491 万参数，对小模型来说占比很大，绑定后省一半，且两个矩阵学到的语义空间天然一致。
3. **RoPE 表提前算好**：位置编码表 `freqs_cos/freqs_sin` 在模型构造时就按最大位置 32768 一次性算完，注册为非持久 buffer（不存进 checkpoint，不参与梯度），forward 时按需切片。

### 1.2 三个类的分工

```
MiniMindForCausalLM          ← transformers 的"壳"：加载/保存/生成接口 + lm_head + loss
    └── MiniMindModel        ← 纯变换器主体：embedding + N 层 Block + final norm + RoPE 表
            └── MiniMindBlock × 8   ← 单层：Attention + FFN + 两个 RMSNorm
```

这个三明治结构是照着 HF `LlamaModel/LlamaForCausalLM` 的习惯搭的：`MiniMindModel` 只管「把 token 向量变成上下文向量」，`ForCausalLM` 在外面套 lm_head 和损失计算。

### 1.3 为什么继承 PreTrainedModel 和 GenerationMixin

```python
class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
```

- `PreTrainedModel`（HF transformers 的基类）带来：`config_class` 机制、`save_pretrained/from_pretrained`（保存成 config.json + safetensors）、权重初始化体系（`post_init`）、与 `AutoModelForCausalLM` 的自动注册。
- `GenerationMixin` 带来 HF 的生成辅助接口约定。不过 MiniMind **自己重写了 `generate()`**（262 行起），所以实际上主要用到的是前者的保存/加载能力。

---

## 2. 逐行精读

### 2.1 导入（1-4 行）

```python
import math, torch, torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import MoeCausalLMOutputWithPast
```

| 行 | 讲解 |
|----|------|
| 1 | `math` 用于取整/π；`F` 是函数式算子（softmax、cross_entropy 等散用） |
| 2 | `nn` 提供模块基类（`nn.Module`/`nn.Linear`/`nn.Embedding`） |
| 3 | `ACT2FN` 是 transformers 的**激活函数名字典**：`ACT2FN['silu']` 返回 SiLU 函数。用它而不是硬编码，是为了让 config 里 `hidden_act` 字符串可配置 |
| 4 | 继承 transformers 三大基础设施：`PretrainedConfig`（配置容器）、`PreTrainedModel`（模型基类）、`GenerationMixin`（生成接口） |
| 5 | `MoeCausalLMOutputWithPast` 是 HF 的标准输出容器，带 `loss/logits/past_key_values/aux_loss` 字段——Keras 风格的 dataclass 返回值，方便上层按名取用 |

### 2.2 MiniMindConfig（10-45 行）

```python
class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"
    def __init__(self, hidden_size=768, num_hidden_layers=8, use_moe=False, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.use_moe = use_moe
```

| 行 | 讲解 |
|----|------|
| 11 | `model_type = "minimind"`：这是 HF 生态的「身份证」。保存成 `config.json` 时会写入 `"model_type": "minimind"`，配合 `AutoModelForCausalLM` + `trust_remote_code` 或 `register_for_auto_class`（convert_model.py 里用到）就能从目录自动找到这个类 |
| 12-13 | 构造参数给了三个最常改的显式参数：隐藏维 768、层数 8、是否 MoE。其余全走 `**kwargs` |
| 14-16 | 前三个参数存成实例属性。`PretrainedConfig` 的 `__setattr__` 会在内部注册，保证 `to_dict()/save_pretrained` 能序列化它们 |

```python
        self.dropout = kwargs.get("dropout", 0.0)
        self.vocab_size = kwargs.get("vocab_size", 6400)
        self.bos_token_id = kwargs.get("bos_token_id", 1)
        self.eos_token_id = kwargs.get("eos_token_id", 2)
        self.flash_attn = kwargs.get("flash_attn", True)
        self.num_attention_heads = kwargs.get("num_attention_heads", 8)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 4)
        self.head_dim = kwargs.get("head_dim", self.hidden_size // self.num_attention_heads)
```

| 行 | 讲解 |
|----|------|
| 17 | dropout 默认 0——LLM 训练一般不开 dropout（正则靠数据量与早停），保留参数是为了实验 |
| 18 | 词表 6400，与 `model/tokenizer.json` 的 BPE 词表严格一致（这是自训词表，详见 [02-分词器/01-BPE分词原理](../02-分词器/01-BPE分词原理.md)） |
| 19-20 | `bos=1 / eos=2`：对照 tokenizer 配置，0=`<|endoftext|>`、1=`<|im_start|>`、2=`<|im_end|>`。这里把 **`<|im_start|>` 当 BOS、`<|im_end|>` 当 EOS**——chat 模板里每条消息由它们包裹，生成到 `<|im_end|>` 即停止 |
| 21 | `flash_attn=True`：允许 Attention 走 PyTorch SDPA 快速路径（见 04-注意力机制） |
| 22 | Q 头 8 个 |
| 23 | KV 头 4 个 → **GQA**（8 个 Q 头每 2 个共享 1 组 KV），省 KV Cache 显存、提速度（详见 04-注意力机制与GQA） |
| 24 | 每头维度 = 768/8 = 96。单独存 `head_dim` 是现代模型惯例（Qwen3 也是），允许头维不整除 hidden |

```python
        self.hidden_act = kwargs.get("hidden_act", 'silu')
        self.intermediate_size = kwargs.get("intermediate_size", math.ceil(hidden_size * math.pi / 64) * 64)
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768)
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)
        self.rope_theta = kwargs.get("rope_theta", 1e6)
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", True)
        self.inference_rope_scaling = kwargs.get("inference_rope_scaling", False)
```

| 行 | 讲解 |
|----|------|
| 25 | 激活函数名，默认 SiLU（SwiGLU 的 σ 部分就用它） |
| 26 | FFN 中间维度公式：`ceil(hidden×π/64)×64`。π≈3.1416 是 **SwiGLU 论文的经验缩放**——用 SwiGLU 时把 FFN 维度乘 π≈2/3 再对齐 64 倍数，让参数量与传统 GELU-FFN 相当。768×π/64=37.7 → ceil=38 → 38×64=2432 |
| 27 | 最大位置 32768：RoPE 预计算表的长度、生成时的位置上限 |
| 28 | RMSNorm 的 ε，防止除零 |
| 29 | RoPE 底数 θ=10⁶：频率序列 1/θ^(2i/d)，θ 越大低频分量越丰富、外推性越好（详见 03-RoPE） |
| 30 | 权重绑定开关，默认开（见 1.1 节设计点 2） |
| 31 | 推理期 YaRN 外推开关：训练时**永远关**（训练用原始 RoPE），只在推理要处理超长文本时打开（详见 07-YaRN） |

```python
        self.rope_scaling = {
            "beta_fast": 32, "beta_slow": 1, "factor": 16,
            "original_max_position_embeddings": 2048,
            "attention_factor": 1.0, "type": "yarn"
        } if self.inference_rope_scaling else None
```

| 行 | 讲解 |
|----|------|
| 32-39 | YaRN 超参数字典：factor=16 表示把 2048 的训练长度插值扩展到 16×2048=32768；beta_fast/beta_slow 控制插值 ramp 的过渡区（详见 07-YaRN）。**只有 `inference_rope_scaling=True` 时才有值**，否则 `None`（走原始 RoPE） |

```python
        ### MoE specific configs (ignored if use_moe = False)
        self.num_experts = kwargs.get("num_experts", 4)
        self.num_experts_per_tok = kwargs.get("num_experts_per_tok", 1)
        self.moe_intermediate_size = kwargs.get("moe_intermediate_size", self.intermediate_size)
        self.norm_topk_prob = kwargs.get("norm_topk_prob", True)
        self.router_aux_loss_coef = kwargs.get("router_aux_loss_coef", 5e-4)
```

| 行 | 讲解 |
|----|------|
| 41 | 路由专家数 4 |
| 42 | 每 token 激活的专家数 k=1（**top-1 路由**，Qwen3-MoE 风格）。k>1 时按 softmax 归一化加权；k=1 时走直通估计器（详见 06-MoE） |
| 43 | MoE 专家的 FFN 中间维，默认与 dense 相同 |
| 44 | top-k 加权是否归一化（k>1 时才有意义） |
| 45 | 负载均衡辅助损失的系数 5e-4（06-MoE 详解） |

### 2.3 MiniMindBlock（183-199 行）——单层 Decoder Block

```python
class MiniMindBlock(nn.Module):
    def __init__(self, layer_id: int, config: MiniMindConfig):
        super().__init__()
        self.self_attn = Attention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)
```

| 行 | 讲解 |
|----|------|
| 185 | `layer_id` 传进来但**没用到**——保留它是为了和 Llama/Qwen 的参数命名对齐（它们的层里可能做 RoPE 缩放或层间差异化），这样权重文件在生态间迁移时键名不乱 |
| 186 | 注意力子层，命名为 `self_attn`（对齐 HF Llama 命名，转换成 Qwen3 格式时键名才对得上） |
| 187 | 注意力**前**的 RMSNorm |
| 188 | 注意力**后**（FFN 前）的 RMSNorm |
| 189 | FFN：`use_moe=False` 用普通 SwiGLU，`True` 用 MoE 版本——一行完成架构切换 |

```python
    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings,
            past_key_value, use_cache, attention_mask
        )
        hidden_states += residual
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value
```

| 行 | 讲解 |
|----|------|
| 192 | 保存输入作为残差 |
| 193-196 | Pre-Norm：先 `input_layernorm` 归一化再进注意力。注意归一化的结果是注意力的**输入**，残差加的是**未归一化**的原值。返回 `present_key_value` 是本层更新后的 KV Cache |
| 197 | 残差 1：`x = x + Attn(Norm(x))` |
| 198 | 残差 2：`x = x + FFN(Norm(x))`。FFN 的 norm 是另一个独立的 `post_attention_layernorm` |
| 199 | 返回（新的隐状态，本层 KV）。张量形状全程 `[B, S, 768]` |

> 为什么用 `+=`（in-place）？因为残差加法不改变计算图依赖关系，原地操作省一次张量分配。但对需要梯度的张量做 in-place 要小心 autograd 报错——这里 `hidden_states` 在注意力输出后已无其他消费者，安全。

### 2.4 MiniMindModel（201-237 行）——变换器主体

```python
class MiniMindModel(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([MiniMindBlock(l, config) for l in range(self.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        freqs_cos, freqs_sin = precompute_freqs_cis(dim=config.head_dim, end=config.max_position_embeddings, rope_base=config.rope_theta, rope_scaling=config.rope_scaling)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)
```

| 行 | 讲解 |
|----|------|
| 205 | vocab/层数缓存成属性（纯方便） |
| 206 | **词嵌入**：查表操作，`input_ids` 里的整数 → 768 维向量。参数矩阵 `[6400, 768]` |
| 207 | 嵌入后的 dropout（默认 p=0，实际不生效） |
| 208 | N=8 层 Block。必须用 `nn.ModuleList` 而不是 Python list，否则子模块不会被注册（参数、`.to(device)` 都会漏） |
| 209 | **最后一个** RMSNorm：输出到 lm_head 之前做最终归一化 |
| 210 | 预计算 RoPE 表。注意传入的 `dim=config.head_dim`（96），不是 hidden_size——**RoPE 是逐头做旋转的**，频率数 = head_dim/2 = 48 |
| 211-212 | 注册为 buffer 且 `persistent=False`：① buffer 会跟着 `model.to(device)` 移动；② `persistent=False` 表示**不写入 state_dict/checkpoint**——表可以随时从 config 重算，存进权重文件反而浪费且可能因版本不一致而错 |

```python
    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, **kwargs):
        batch_size, seq_length = input_ids.shape
        if hasattr(past_key_values, 'layers'): past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
```

| 行 | 讲解 |
|----|------|
| 214 | 前向入口。`past_key_values` 是 KV Cache（列表，每层一个 `(k, v)` 二元组）；`use_cache` 决定是否返回新 Cache |
| 215 | 记下 batch 和当前步的序列长度 |
| 216 | **兼容性补丁**：HF 新版会把 `past_key_values` 包装成 `DynamicCache` 之类的缓存对象（有 `.layers` 属性）。本实现的 Cache 是裸 tuple 列表，遇到包装对象直接丢弃（一次性重算，不增量）。这只在混用 HF 生成工具时才会触发 |
| 217 | `past_key_values or [...]`：传入 None/空时，给每层一个 None 占位。注意 Python 短路——传了合法列表就原样用 |
| 218 | **起点位置**：如果第 0 层的 Cache 存在，其 shape `[B, past_len, kv_heads, head_dim]` 的第 1 维就是「已缓存的长度」。增量解码时新 token 的 RoPE 相位要从 `past_len` 开始切，这正是行 224 切片的依据 |

```python
        hidden_states = self.dropout(self.embed_tokens(input_ids))
        # Recompute RoPE buffers lost during meta-device init (transformers>=5.x)
        if self.freqs_cos[0, 0] == 0:
            freqs_cos, freqs_sin = precompute_freqs_cis(dim=self.config.head_dim, end=self.config.max_position_embeddings, rope_base=self.config.rope_theta, rope_scaling=self.config.rope_scaling)
            self.freqs_cos, self.freqs_sin = freqs_cos.to(hidden_states.device), freqs_sin.to(hidden_states.device)
        position_embeddings = (self.freqs_cos[start_pos:start_pos + seq_length], self.freqs_sin[start_pos:start_pos + seq_length])
```

| 行 | 讲解 |
|----|------|
| 219 | 查表 + dropout。`hidden_states` 形状 `[B, S, 768]` |
| 220-223 | **meta-device 修复**：`transformers>=5.x` 用 `from_pretrained` 加载时会把模型先建在 meta 设备（占位、不分配内存）再搬真权重，而 **非持久 buffer 在 meta 设备上被清零丢失**。检测方法：`freqs_cos[0,0]==0`——位置 0 的 cos 必然是 cos(0)=1，等于 0 说明表丢了，就当场重算。这是模型第 220 行注释点名的 transformers 5.x 兼容点 |
| 224 | 组装位置编码：从 `start_pos` 切出本步需要的 `seq_length` 行 cos/sin。训练时 start_pos=0 全序列；增量解码时只切最后一小段。把 cos/sin 打包成 tuple 传递，对齐了 Qwen3 的接口风格（HF 新版也是传 `position_embeddings` 元组） |

```python
        presents = []
        for layer, past_key_value in zip(self.layers, past_key_values):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)
        hidden_states = self.norm(hidden_states)
        aux_loss = sum([l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)], hidden_states.new_zeros(1).squeeze())
        return hidden_states, presents, aux_loss
```

| 行 | 讲解 |
|----|------|
| 225-233 | 顺序过 N 层，每层喂同样的位置编码（cos/sin 表已按行内绝对位置排好）、各自的旧 Cache；收集每层的新 Cache 到 `presents` |
| 235 | 最终 RMSNorm |
| 236 | **MoE 负载均衡损失汇总**：把每一层 MoE 前向时挂在模块上的 `aux_loss`（MOEFeedForward.forward 里算好存为属性，见 06-MoE）全部加起来。`isinstance(l.mlp, MOEFeedForward)` 保证 dense 模型不会报错（dense 层的 mlp 没有该属性）。`hidden_states.new_zeros(1).squeeze()` 是「与 hidden_states 同设备同 dtype 的 0」，保证空列表求和时结果类型正确、可反传 |
| 237 | 返回三元组：最终隐状态、全部层 KV Cache、MoE 辅助损失 |

### 2.5 MiniMindForCausalLM（239-258 行）——外壳与损失

```python
class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MiniMindConfig
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    def __init__(self, config: MiniMindConfig = None):
        self.config = config or MiniMindConfig()
        super().__init__(self.config)
        self.model = MiniMindModel(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        if self.config.tie_word_embeddings: self.model.embed_tokens.weight = self.lm_head.weight
        self.post_init()
```

| 行 | 讲解 |
|----|------|
| 240 | `config_class` 告诉 HF：本模型用哪种 config 类反序列化（from_pretrained 时用） |
| 241 | `_tied_weights_keys` 声明绑定关系：保存时不重复存 lm_head，加载时自动把 lm_head 指回 embedding |
| 243 | config 缺省兜底为默认配置（hidden 768、8 层） |
| 244 | **必须先设 `self.config` 再调 `super().__init__`**——PreTrainedModel 的初始化会读 self.config |
| 245-246 | 变换器主体 + 输出投影 `[768 → 6400]`，无 bias（现代 LLM 惯例：lm_head/FFN/QKV 全部去 bias） |
| 247 | **执行绑定**：直接让两个模块共享同一个 `nn.Parameter` 对象。之后无论梯度更新哪一个，另一个同步变化（同一块内存）。注意在**load_state_dict 之前**做绑定，加载权重时 embedding 的值会同时「填充」两个名字 |
| 248 | `post_init()`：触发 HF 的权重初始化流程（按 `_init_weights` 钩子初始化所有子模块；本类没写 `_init_weights`，用 nn 默认初始化） |

```python
    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, logits_to_keep=0, labels=None, **kwargs):
        hidden_states, past_key_values, aux_loss = self.model(input_ids, attention_mask, past_key_values, use_cache, **kwargs)
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
```

| 行 | 讲解 |
|----|------|
| 250 | forward 签名对齐 HF 习惯。`logits_to_keep`：只对最后 N 个位置算 lm_head（省算力省显存）。**训练传 0** → `slice(-0, None)` 等价于 `slice(0, None)` 即全部位置（Python 的 `-0` 就是 0 的巧用）；**RL 采样时**传具体数字（如 `n_keep+1`）只算末尾若干位置 |
| 251-252 | 先跑主体；再按 `logits_to_keep` 切片。int 时取「倒数第 N 个起」，list 时按索引取 |
| 253 | 投影到词表得到 logits `[B, S, 6400]`。注意这里**没有 softmax**——训练用 `F.cross_entropy`（内部自带 log_softmax），推理时自己 softmax |

```python
        loss = None
        if labels is not None:
            x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
            loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
        return MoeCausalLMOutputWithPast(loss=loss, aux_loss=aux_loss, logits=logits, past_key_values=past_key_values, hidden_states=hidden_states)
```

| 行 | 讲解 |
|----|------|
| 256 | **causal LM 的错位一步预测**：位置 t 的 logits 要预测 token t+1，所以输入对齐 `logits[..., :-1, :]`（去掉最后一个位置的预测），标签对齐 `labels[..., 1:]`（去掉第一个 token——它没有上文）。这就是著名的 **shift** 操作。`.contiguous()` 是因为切片后内存不连续，后面 `.view()` 要求连续内存 |
| 257 | 摊平成 `[B×(S-1), vocab]` vs `[B×(S-1)]` 做交叉熵；`ignore_index=-100`：标签为 -100 的位置（padding、非 assistant 部分）不计损失。**平均方式**：默认 reduction='mean' 是对所有非忽略 token 求平均（注意不是按 batch 平均，也不是按序列平均） |
| 258 | 打包返回。`aux_loss` 放在输出里，trainer 里用 `res.loss + res.aux_loss` 合成总损失（见 04-训练/02） |

### 2.6 一个重要细节：labels 里的 -100 从哪来

- 预训练：`PretrainDataset` 把 padding 位置标成 -100（见 03-数据集/02）。
- SFT：非 assistant 回复部分全标 -100 → 模型只学「怎么回答」。
- DPO/RL：不再用 labels 参数，而是自己取 logits 算 log prob。

这个 -100 约定让**同一行 cross_entropy 同时实现了 padding 屏蔽和 loss mask**，是整个仓库数据-模型接缝的关键设计。

---

## 3. 与 Qwen3 的对应关系

| MiniMind | Qwen3 | 说明 |
|----------|-------|------|
| `MiniMindConfig` | `Qwen3Config` | 字段几乎一一对应；convert_model.py 据此互转 |
| `MiniMindModel` | `Qwen3Model` | embed/layers/norm 结构相同 |
| `MiniMindBlock` | `Qwen3DecoderLayer` | Pre-Norm + Attn + FFN 相同 |
| `Attention` | `Qwen3Attention` | GQA + q/k norm 相同 |
| `MOEFeedForward` | `Qwen3MoeSparseMoeBlock` | top-1 路由 + aux loss 思路相同 |
| `generate()` | `Qwen3ForCausalLM.generate` | MiniMind 手写简化版，采样参数语义一致 |

正因为逐组件对齐，`scripts/convert_model.py` 才能把训练出的权重**无损加载进官方 Qwen3 类**（strict=True），直接接入 vllm/ollama/llama.cpp 生态。

---

## 4. 本篇小结（务必记住的 5 件事）

1. 总装顺序：**Embedding → 8×[LN+Attn+残差 / LN+FFN+残差] → LN → lm_head**，Pre-Norm。
2. **shift loss**：`logits[:-1]` 对 `labels[1:]`，`ignore_index=-100` 一行实现 padding/loss-mask 屏蔽。
3. **权重绑定**：embed_tokens 与 lm_head 共享参数，`_tied_weights_keys` 负责保存/加载时的自洽。
4. **KV Cache 是每层一个 (k,v) 的裸列表**；`start_pos` 决定 RoPE 相位起点，这是增量解码正确性的关键。
5. RoPE 表是非持久 buffer，meta 设备加载会丢 → forward 里检测全 0 就地重算（transformers 5.x 兼容）。

下一篇：[02-RMSNorm.md](./02-RMSNorm.md)
