# 05 - 模型格式转换 convert_model.py 全文

> 覆盖代码：`scripts/convert_model.py` **全文 145 行**。
> 把训练产物 `.pth` 权重在三种格式间转换，并附带 LoRA 合并与模板格式工具：
>
> ```
> MiniMind .pth  ──①──→  MiniMind transformers 目录（config.json + pytorch_model.bin，trust_remote_code 用）
>                ──②──→  官方 Qwen3 架构目录（vllm/ollama/llama.cpp 直接可用）
>                ──③──←  transformers 目录 → .pth（逆转换）
>                ──④──→  基模+LoRA 合并 → 新 .pth
> ```

---

## 1. 为什么格式重要

- **.pth**：训练侧原生格式，单文件、加载快，但只有自定义类能读。
- **MiniMind transformers 目录**：`AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)` 能加载（config.json 里 `model_type: minimind` + `auto_map`），带 tokenizer 一起走。
- **Qwen3 架构目录**：权重键名/结构与官方 `Qwen3ForCausalLM` 完全一致（模型对齐的回报，见 01 篇 §3），**无需自定义代码**即可进 vllm/ollama/llama.cpp 生态。

## 2. 逐行精读

### 2.1 文件头（1-14 行）

```python
import os, sys, json
__package__ = "scripts"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import transformers
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM, Qwen3Config, Qwen3ForCausalLM, Qwen3MoeConfig, Qwen3MoeForCausalLM
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_lora import apply_lora, merge_lora
warnings.filterwarnings('ignore', category=UserWarning)
```

- 引入 **Qwen3 四件套**（Config/ForCausalLM 及 MoE 版）——转换目标类；`transformers.__version__` 用于 5.x 分支判断。

### 2.2 convert_torch2transformers_minimind（16-36 行）

```python
def convert_torch2transformers_minimind(torch_path, transformers_path, dtype=torch.float16):
    MiniMindConfig.register_for_auto_class()
    MiniMindForCausalLM.register_for_auto_class("AutoModelForCausalLM")
    lm_model = MiniMindForCausalLM(lm_config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load(torch_path, map_location=device)
    lm_model.load_state_dict(state_dict, strict=False)
    lm_model = lm_model.to(dtype)  # 转换模型权重精度
    model_params = sum(p.numel() for p in lm_model.parameters() if p.requires_grad)
    print(f'模型参数: {model_params / 1e6} 百万 = {model_params / 1e9} B (Billion)')
    lm_model.save_pretrained(transformers_path, safe_serialization=False)
    tokenizer = AutoTokenizer.from_pretrained('../model/')
    tokenizer.save_pretrained(transformers_path)
```

| 行 | 讲解 |
|----|------|
| 17-18 | **auto_map 注册**：`register_for_auto_class` 把类的信息写进保存目录的 config.json（`auto_map: {"AutoModelForCausalLM": "modeling_minimind.MinimindForCausalLM"}` 之类机制），配合 `trust_remote_code=True` 让**别人机器上**也能 `from_pretrained` 自动找到 MiniMind 类。注意这要求目录里有模型代码文件（trust_remote_code 场景）或仓库在 path 上 |
| 19-23 | 空模型 → 加载 .pth（**strict=False**：fp16 保存的权重少 buffer/部分键也能装）→ 转 dtype |
| 26 | `safe_serialization=False`：存成 `pytorch_model.bin` 而非 safetensors（老生态兼容） |
| 27-28 | **tokenizer 一起打包**：从 `model/` 拷贝 tokenizer.json/config——加载模型时同目录 AutoTokenizer 即用 |

```python
    # ======= transformers-5.0的兼容低版本写法 =======
    if int(transformers.__version__.split('.')[0]) >= 5:
        tokenizer_config_path, config_path = os.path.join(transformers_path, "tokenizer_config.json"), os.path.join(transformers_path, "config.json")
        json.dump({**json.load(open(tokenizer_config_path, 'r', encoding='utf-8')), "tokenizer_class": "PreTrainedTokenizerFast", "extra_special_tokens": {}}, open(tokenizer_config_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
        config = json.load(open(config_path, 'r', encoding='utf-8'))
        config['rope_theta'] = lm_config.rope_theta; config['rope_scaling'] = None; del config['rope_parameters']
        json.dump(config, open(config_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print(f"模型已保存为 Transformers-MiniMind 格式: {transformers_path}")
```

| 行 | 讲解 |
|----|------|
| 29-36 | **transformers 5.x 修正**：新版 save_pretrained 会把 tokenizer_class 写成 5.x 新类名、config 里把 RoPE 写成新的 `rope_parameters` 结构——**老版本 transformers（本仓库要求的 4.57.6）读不了**。手术：tokenizer_class 强制改回 `PreTrainedTokenizerFast`、补 `extra_special_tokens`、config 里 rope_theta 保留旧键、rope_scaling 置 None、**删掉 rope_parameters**——让目录双向兼容 4.x/5.x |

### 2.3 convert_torch2transformers：转 Qwen3 格式（39-96 行）

```python
def convert_torch2transformers(torch_path, transformers_path, dtype=torch.float16):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load(torch_path, map_location=device)
    common_config = {
        "vocab_size": lm_config.vocab_size,
        "hidden_size": lm_config.hidden_size,
        "intermediate_size": lm_config.intermediate_size,
        "num_hidden_layers": lm_config.num_hidden_layers,
        "num_attention_heads": lm_config.num_attention_heads,
        "num_key_value_heads": lm_config.num_key_value_heads,
        "head_dim": lm_config.hidden_size // lm_config.num_attention_heads,
        "max_position_embeddings": lm_config.max_position_embeddings,
        "rms_norm_eps": lm_config.rms_norm_eps,
        "rope_theta": lm_config.rope_theta,
        "tie_word_embeddings": lm_config.tie_word_embeddings
    }
```

| 行 | 讲解 |
|----|------|
| 43-54 | **公共 config 字段**：MiniMind 与 Qwen3 的参数一一对应地搬过去。`head_dim` 重新计算（Qwen3 也支持独立 head_dim） |

```python
    if not lm_config.use_moe:
        qwen_config = Qwen3Config(**common_config, use_sliding_window=False, sliding_window=None)
        qwen_model = Qwen3ForCausalLM(qwen_config)
    else:
        qwen_config = Qwen3MoeConfig(**common_config, num_experts=..., num_experts_per_tok=..., moe_intermediate_size=..., norm_topk_prob=...)
        qwen_model = Qwen3MoeForCausalLM(qwen_config)
        # ======= transformers-5.0的兼容低版本写法 =======
        if int(transformers.__version__.split('.')[0]) >= 5:
            new_sd = {k: v for k, v in state_dict.items() if 'experts.' not in k or 'gate.weight' in k}
            for l in range(lm_config.num_hidden_layers):
                p = f'model.layers.{l}.mlp.experts'
                new_sd[f'{p}.gate_up_proj'] = torch.cat([torch.stack([state_dict[f'{p}.{e}.gate_proj.weight'] for e in range(lm_config.num_experts)]), torch.stack([state_dict[f'{p}.{e}.up_proj.weight'] for e in range(lm_config.num_experts)])], dim=1)
                new_sd[f'{p}.down_proj'] = torch.stack([state_dict[f'{p}.{e}.down_proj.weight'] for e in range(lm_config.num_experts)])
            state_dict = new_sd

    qwen_model.load_state_dict(state_dict, strict=True)
```

| 行 | 讲解 |
|----|------|
| 56-62 | Dense：直接 `Qwen3Config` 建官方模型。键名逐层对得上（q_proj/k_proj/…/gate_proj/up_proj/down_proj/embed_tokens/lm_head 都在 01 篇按 Llama/Qwen 命名）→ `strict=True` **严格加载验证结构一致性**（这一步本身就是「结构对齐」的自检） |
| 63-79 | **MoE 的权重重排（5.x 分支）**：transformers 5.x 的 Qwen3-MoE 把每个专家的 gate/up 合并成一条 `gate_up_proj`（[E, 2*I, H]）、down 堆成 [E, H, I]。三步：<br>1. 丢掉旧的逐专家 gate/up 键（保留 mlp.gate.weight——路由器）；<br>2. `torch.stack([gate_proj for e in experts])` → [E, I, H]，同法 up → cat dim=1 → **[E, 2I, H]** 的 gate_up_proj；<br>3. down 直接 stack → [E, H, I]。<br>4.x 版本 Qwen3-MoE 本来就是分离的 gate/up/down，无需重排 |

```python
    qwen_model = qwen_model.to(dtype)
    qwen_model.save_pretrained(transformers_path)
    ...
    tokenizer = AutoTokenizer.from_pretrained('../model/')
    tokenizer.save_pretrained(transformers_path)
    if int(transformers.__version__.split('.')[0]) >= 5: ... # 同 2.2 的 config/tokenizer 手术
```

- 精度转换、保存、tokenizer 打包、5.x 修正（与 2.2 同款）。

### 2.4 convert_transformers2torch：逆转换（99-102 行）

```python
def convert_transformers2torch(transformers_path, torch_path):
    model = AutoModelForCausalLM.from_pretrained(transformers_path, trust_remote_code=True)
    torch.save({k: v.cpu().half() for k, v in model.state_dict().items()}, torch_path)
    print(f"模型已保存为 PyTorch 格式: {torch_path}")
```

- from_pretrained（本地类）→ state_dict 全转 fp16 CPU → 存 .pth。反向路径闭合成环。

### 2.5 convert_merge_base_lora：LoRA 合并（105-112 行）

```python
def convert_merge_base_lora(base_torch_path, lora_path, merged_torch_path, lora_rank=16):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lm_model = MiniMindForCausalLM(lm_config).to(device)
    state_dict = torch.load(base_torch_path, map_location=device)
    lm_model.load_state_dict(state_dict, strict=False)
    apply_lora(model=lm_model, rank=lora_rank)
    merge_lora(lm_model, lora_path, merged_torch_path)
```

- 三步编排（函数实现在 model_lora.py，见 01-模型架构/09）：加载基模 → 挂 LoRA 结构 → `merge_lora`（内部先 load LoRA 权重再 `W += B@A×scaling` 存新 .pth）。`lora_rank` 必须**与训练时一致**（决定 A/B 形状）。

### 2.6 chat_template 双向工具（115-125 行）

```python
def convert_jinja_to_json(jinja_path):
    with open(jinja_path, 'r') as f: template = f.read()
    escaped = json.dumps(template)
    print(f'"chat_template": {escaped}')

def convert_json_to_jinja(json_file_path, output_path):
    with open(json_file_path, 'r') as f: config = json.load(f)
    template = config['chat_template']
    with open(output_path, 'w') as f: f.write(template)
    print(f"模板已保存为 jinja 文件: {output_path}")
```

- jinja 文件 ↔ tokenizer_config.json 里的 chat_template 字符串互转（**json.dumps 转义**：模板含大量引号换行，嵌入 JSON 必须转义）。给「想单独编辑模板再放回 config」的工作流用。

### 2.7 main（128-144 行）

```python
if __name__ == '__main__':
    lm_config = MiniMindConfig(hidden_size=768, num_hidden_layers=8, max_seq_len=8192, use_moe=False)

    # convert torch to transformers
    torch_path = f"../out/full_sft_{lm_config.hidden_size}{'_moe' if lm_config.use_moe else ''}.pth"
    transformers_path = '../minimind-3'
    convert_torch2transformers(torch_path, transformers_path)

    # # merge lora
    # base_torch_path = ...
    # lora_path = ...
    # merged_torch_path = ...
    # convert_merge_base_lora(base_torch_path, lora_path, merged_torch_path)

    # convert_transformers2torch(transformers_path, torch_path)
    # convert_json_to_jinja('../model/tokenizer_config.json', '../model/chat_template.jinja')
    # convert_jinja_to_json('../model/chat_template.jinja')
```

| 行 | 讲解 |
|----|------|
| 129 | **lm_config 是模块级全局**——各转换函数引用它（与训练侧超参必须一致：hidden 768/8 层才装得进 out/full_sft_768.pth） |
| 132-134 | 默认执行：把 `out/full_sft_768.pth` 转成 **Qwen3 架构目录 `minimind-3/`**（web_demo/ollama/vllm 即用）。注意走的是 convert_torch2transformers（Qwen 版）而不是 _minimind 版 |
| 136-144 | 注释掉的其他流程（合并 LoRA / 逆转换 / 模板互转）按需打开 |

## 3. 本篇小结

1. 三格式：**.pth（训练）↔ MiniMind 目录（trust_remote_code）↔ Qwen3 目录（生态原生）**；Qwen3 转换的 `strict=True` 加载同时是结构对齐自检。
2. MoE 权重重排（5.x）：逐专家 gate/up/down → **stack+cat 出 [E, 2I, H] 的 gate_up_proj**。
3. transformers 5.x 的 config/tokenizer 手术让产物**双向兼容 4.x/5.x**。
4. LoRA 合并 = apply_lora 结构 + merge_lora 权重合成；rank 必须与训练一致。

下一篇：[06-工具调用评测-eval_toolcall](./06-工具调用评测-eval_toolcall.md)
