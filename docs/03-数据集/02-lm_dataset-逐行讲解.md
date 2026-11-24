# 02 - lm_dataset.py 逐行讲解

> 覆盖代码：`dataset/lm_dataset.py` **全文 261 行**：5 个 Dataset 类（Pretrain/SFT/DPO/RLAIF/AgentRL）+ 2 个公共预处理函数。
> 原理前置：[01-数据集总览](./01-数据集总览.md)、[02-分词器/chat_template 精读](../02-分词器/02-训练分词器-逐行讲解.md)。

---

## 1. 导入与全局（1-7 行）

```python
from torch.utils.data import Dataset
import torch
import json
import os
import random
from datasets import load_dataset, Features, Sequence, Value
os.environ["TOKENIZERS_PARALLELISM"] = "false"
```

| 行 | 讲解 |
|----|------|
| 1-5 | 标准 PyTorch Dataset 基类 + torch/json/os/random。`random` 用于概率性数据增强 |
| 6 | HF `datasets` 库：`load_dataset('json', ...)` 高效加载 jsonl（支持懒加载/内存映射，比逐行 json.loads 快且省内存） |
| 7 | 关闭 HF tokenizers 的多线程并行。**必须在做 fork 的 DataLoader worker 场景下关掉**，否则 tokenizers 的 Rust 线程池与 fork 冲突，偶发死锁/告警。用环境变量方式在任何 import 之前设置 |

---

## 2. pre_processing_chat：随机 system prompt（9-29 行）

```python
def pre_processing_chat(conversations, add_system_ratio=0.2):
    # tool use 数据完整保留不做处理
    if any(conv.get('tools') for conv in conversations): return conversations

    SYSTEM_PROMPTS = [
        "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
        "你是minimind，一个小巧但有用的语言模型。",
        "你是一个专业的AI助手，请提供有价值的回答。",
        "你是minimind，请尽力帮助用户解决问题。",
        "你是一个可靠的AI，请给出准确的回答。",
        "You are a helpful AI assistant.",
        "You are minimind, a lightweight intelligent assistant.",
        "You are a friendly chatbot. Please answer the user's questions carefully.",
        "You are a knowledgeable AI. Try your best to provide accurate information.",
        "You are minimind, a small but useful language model."
    ]
    # 概率性添加system
    if conversations[0].get('role') != 'system':
        if random.random() < add_system_ratio:
            return [{'role': 'system', 'content': random.choice(SYSTEM_PROMPTS)}] + conversations
    return conversations
```

| 行 | 讲解 |
|----|------|
| 10 | 带工具的样本（任一消息有 `tools` 字段）**原样返回**——工具对话的 system 段承载工具说明，不能乱动 |
| 12-24 | 10 个候选 system prompt（5 中 5 英），覆盖「通用助手/自称 minimind/友好/专业」等风格 |
| 26-28 | 若第一条不是 system：以 20% 概率**随机挑一条**插到最前面。<br>**为什么要随机加**：推理时用户端可能带也可能不带 system prompt，若 SFT 数据全是无 system 格式，模型遇到带 system 的输入会分布外漂移。训练时以 20% 比例混入，让模型对「有无 system」都稳健；随机换内容则避免把某一句 system 背得过死 |
| 29 | 已有 system 的样本不动 |

---

## 3. post_processing_chat：空思考段处理（31-38 行）

```python
def post_processing_chat(prompt_content, empty_think_ratio=0.2, remove_empty_think=None):
    # 以80%概率移除空思考标签
    if '<think>\n\n</think>\n\n' in prompt_content:
        if remove_empty_think is None:
            remove_empty_think = random.random() > empty_think_ratio
        if remove_empty_think:
            prompt_content = prompt_content.replace('<think>\n\n</think>\n\n', '')
    return prompt_content
```

| 行 | 讲解 |
|----|------|
| 32 | 目标串：模板给 assistant 预填的**空思考占位** `<think>\n\n</think>\n\n`（chat_template 块⑤的产物） |
| 33-35 | 删除概率默认 80%（`random.random() > 0.2`）。`remove_empty_think` 参数允许调用方**显式控制**（DPODataset 传同一布尔值给 chosen/rejected，保证成对样本删除行为一致！） |
| 36-37 | `replace` 把空思考段整段删掉——这条样本就变成「无思考直答」格式 |

**为什么这么做（自适应思考的核心）**：训练数据里 80% 删空思考、20% 保留，模型同时见过两种 assistant 格式 → 推理时 `open_thinking` 开或关都在分布内。若 100% 删，模型会「忘了」怎么在 `<think>` 里写内容。

---

## 4. PretrainDataset（40-58 行）

```python
class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset('json', data_files=data_path, split='train')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        tokens = self.tokenizer(str(sample['text']), add_special_tokens=False, max_length=self.max_length - 2, truncation=True).input_ids
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        return input_ids, labels
```

| 行 | 讲解 |
|----|------|
| 45 | 懒加载 jsonl 为 Arrow 数据集（内存映射，几 GB 数据也能秒开）；`split='train'` 取全部 |
| 52 | 编码文本：`add_special_tokens=False` **不要 tokenizer 自动加的 bos/eos**（下一步手动加，位置可控）；`max_length-2` + `truncation=True` 先截到 338（给 bos/eos 留位），保证拼完不超过 340 |
| 53 | 手动包裹：`[bos] + tokens + [eos]`——「这是一段完整文本」的信号 |
| 54 | **padding**：不足 340 的尾部补 pad_token_id（`<|endoftext|>`）。注意这里不训练「拼接过文本」——预训练一般应拼接长文档（packing），本项目简化为逐条独立样本 |
| 55-56 | 转 long 张量；`labels = input_ids.clone()`：预训练的标签就是输入本身（shift 在模型 forward 里做：`logits[:-1]` vs `labels[1:]`） |
| 57 | **padding 屏蔽**：把 pad 位置的标签改成 -100（cross_entropy 的 `ignore_index`）。注意一个边界细节：pad 值恰好等于 eos 后面的部分全变 -100，但**文本末尾真正的 eos 保留了**——模型学到「文本以 eos 结束」，生成时才会停 |
| 58 | 返回 `(input_ids, labels)`，都 `[340]`。DataLoader 会自动堆成 `[B, 340]` |

---

## 5. SFTDataset（61-123 行）

### 5.1 初始化（62-69 行）

```python
class SFTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        features = Features({'conversations': [{'role': Value('string'), 'content': Value('string'), 'reasoning_content': Value('string'), 'tools': Value('string'), 'tool_calls': Value('string')}]})
        self.samples = load_dataset('json', data_files=jsonl_path, split='train', features=features)
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids
```

| 行 | 讲解 |
|----|------|
| 66 | **显式 features schema**：声明 conversations 内每条消息的字段类型。这是对脏数据的防线——样本里某条消息缺字段/类型不一时，Arrow 不会抛错而是填 None（配合后续 `dict(message)`/`get()` 兜底）。五个字段对应 chat_template 用到的全部属性 |
| 67 | 按 schema 加载 |
| 68 | **bos_id 锚点**：`<|im_start|>assistant\n` 的 token 序列（注意模板渲染的 assistant 开头正是这串）。编码结果是一个 list，作为「回复段起点」的标记 |
| 69 | **eos_id 锚点**：`<|im_end|>\n`——「回复段终点」（含收尾换行）。用 token 序列而非字符串匹配，是因为编码后的 input_ids 是 token 列表，锚点必须也在 token 空间比较 |

### 5.2 create_chat_prompt（74-89 行）

```python
    def create_chat_prompt(self, conversations):
        messages = []
        tools = None
        for message in conversations:
            message = dict(message)
            if message.get("role") == "system" and message.get("tools"):
                tools = json.loads(message["tools"]) if isinstance(message["tools"], str) else message["tools"]
            if message.get("tool_calls") and isinstance(message["tool_calls"], str):
                message["tool_calls"] = json.loads(message["tool_calls"])
            messages.append(message)
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, tools=tools
        )
```

| 行 | 讲解 |
|----|------|
| 75-76 | 重建消息列表；`tools` 挖出来单独传 |
| 78 | `dict(message)`：把 HF Arrow 的记录转成普通 dict（getattr 语义更友好，也避免污染原数据） |
| 79-80 | system 消息上的 `tools` 字段（JSON 字符串）**解析回列表**——`apply_chat_template` 的 `tools=` 参数要求列表。只在 system 上找（模板约定） |
| 81-82 | `tool_calls` 字段同理：数据文件里存的是字符串化 JSON，模板需要 dict。**注意数据集 schema 里 tool_calls 是 string**，所以必须转 |
| 84-89 | 渲染成训练文本（`tokenize=False` 先拿字符串，后面统一编码；`add_generation_prompt=False`：训练要完整对话，不需要「assistant 待命」提示） |

### 5.3 generate_labels：loss mask 核心（91-107 行）

```python
    def generate_labels(self, input_ids):
        labels = [-100] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    labels[j] = input_ids[j]
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return labels
```

算法本质：**在 token 序列里做「起点锚点 + 终点锚点」的区间标记**，支持一条样本里的**多轮** assistant 回复（多轮对话每个 assistant 段都标）。

| 行 | 讲解 |
|----|------|
| 92 | 全部初始化 -100（不训练） |
| 93-106 | 主循环：`i` 是扫描指针。<br>• 行 95：**滑动窗口匹配** `input_ids[i:i+len(bos_id)] == bos_id`——列表切片相等即整段匹配 bos 锚点（`<|im_start|>assistant\n`）；<br>• 行 96-101：从锚点后开始找 eos 锚点 `<|im_end|>\n` 的**首次**出现位置 `end`（内层 while 逐位推进）；<br>• 行 102-103：`[start, end+len(eos_id))` 区间的标签设为**原 token 值**（即「参与训练」），上限夹到 max_length 防越界。注意终点**包含** `<|im_end|>\n` 本身——模型要学习「何时结束回复」；<br>• 行 104：指针跳过整个已处理区间（不回头重复匹配）；找不到 eos 时跳到末尾；<br>• 行 105-106：没匹配到 bos 锚点就 +1 继续扫。 |
| 返回 | labels：assistant 回复区间=原 token，其余=-100 |

**边界情况**：bos 锚点匹配是「贪婪找到的每处」——user 消息里若原文含「assistant\n」字样不会被误匹配，因为 bos_id 是 `<|im_start|>assistant\n` 的**完整 token 组合**，普通文本几乎不可能产生相同 token 序列。

### 5.4 __getitem__（109-123 行）

```python
    def __getitem__(self, index):
        sample = self.samples[index]
        conversations = pre_processing_chat(sample['conversations'])
        prompt = self.create_chat_prompt(conversations)
        prompt = post_processing_chat(prompt)
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        labels = self.generate_labels(input_ids)
        # # === 调试打印 ===（注释块，可解开查看 token 对齐）
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)
```

| 行 | 讲解 |
|----|------|
| 111-113 | 三步加工：随机 system → 渲染 → 随机删空思考 |
| 114 | 编码 + **超长截断**（`[:max_length]`）。注意：截断可能把 eos 锚点切碎——generate_labels 里「找不到 eos」分支会把剩余段全标为训练区，可接受 |
| 115 | 右侧补 pad。注意 pad 恰好等于 `<|endoftext|>`（0 号），而 bos 锚点匹配从左往右扫，pad 区不可能含 bos → pad 全是 -100，天然屏蔽 |
| 116 | 生成 labels |
| 117-121 | 注释掉的调试代码：解开可打印每个位置「输入 X → 预测 Y → label」的对照表，是理解 shift 对齐的最佳工具（建议学习时打开跑一遍） |
| 122 | 返回 `(input_ids, labels)` |

---

## 6. DPODataset（125-197 行）

```python
class DPODataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=4096):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids
        self.samples = load_dataset('json', data_files=file_path, split='train')
```

| 行 | 讲解 |
|----|------|
| 126 | max_length 默认 4096（偏好数据往往更长） |
| 130 | pad id 兜底逻辑（tokenizer 一定有 pad=0，此处防御性写法） |
| 131-132 | 与 SFT 相同的 bos/eos 锚点 |
| 133 | 加载 chosen/rejected 数据 |

### __getitem__（138-179 行）

```python
    def __getitem__(self, index):
        sample = self.samples[index]
        chosen = sample['chosen']  # 是一个 list，里面包含若干 {role, content}
        rejected = sample['rejected']  # 同上
        chosen_prompt = self.tokenizer.apply_chat_template(chosen, tokenize=False, add_generation_prompt=False)
        rejected_prompt = self.tokenizer.apply_chat_template(rejected, tokenize=False, add_generation_prompt=False)
        remove_empty_think = random.random() > 0.2
        chosen_prompt = post_processing_chat(chosen_prompt, remove_empty_think=remove_empty_think)
        rejected_prompt = post_processing_chat(rejected_prompt, remove_empty_think=remove_empty_think)
```

| 行 | 讲解 |
|----|------|
| 142-148 | chosen / rejected 各自渲染成完整对话文本 |
| 149-151 | **同一个布尔值**控制两者的空思考删除——偏好对必须格式一致，否则比较不公平（一个删了一个没删，分布差异会污染偏好信号） |

```python
        chosen_encoding = self.tokenizer(chosen_prompt, truncation=True, max_length=self.max_length, padding='max_length')
        ...
        chosen_loss_mask = self.generate_loss_mask(chosen_input_ids)
        ...
        x_chosen = torch.tensor(chosen_input_ids[:-1], dtype=torch.long)
        y_chosen = torch.tensor(chosen_input_ids[1:], dtype=torch.long)
        mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype=torch.long)
```

| 行 | 讲解 |
|----|------|
| 152-157 | 两条分别编码：截断 + **padding='max_length'**（右侧补齐到 4096） |
| 159-168 | loss_mask（assistant 区间=1）→ **这里自己做 shift**（模型 forward 不带 labels，logits 全量返回）：`x = ids[:-1]`（预测输入）、`y = ids[1:]`（预测目标）、mask 同步 `[:-1] → [1:]`?——准确说 `mask[1:]`：位置 t 的 loss mask 对应目标 y[t]=ids[t+1]，掩码与 y 同长度同偏移。x/y/mask 三者长度 4095 |
| 171-178 | 返回 dict：x/y/mask × chosen/rejected 六个键。train_dpo 把它们在 batch 维拼接后一次 forward |

### generate_loss_mask（180-197 行）

与 SFTDataset.generate_labels **完全同构**，唯一区别：区间内标 **1**（而不是复制 token 值），区间外标 **0**——0/1 浮点掩码便于 DPO 里做 `(logp * mask).sum()` 加权求和。

---

## 7. RLAIFDataset（199-229 行）

```python
class RLAIFDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024, thinking_ratio=0.5):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.thinking_ratio = thinking_ratio  # 按概率开启 thinking
        self.samples = load_dataset('json', data_files=jsonl_path, split='train')
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        conversations = pre_processing_chat(conversations)
        use_thinking = random.random() < self.thinking_ratio
        return self.tokenizer.apply_chat_template(
            conversations[:-1], tokenize=False, open_thinking=use_thinking, add_generation_prompt=True
        )

    def __getitem__(self, index):
        sample = self.samples[index]
        prompt = self.create_chat_prompt(sample['conversations'])
        return {'prompt': prompt, 'answer': ""}
```

| 行 | 讲解 |
|----|------|
| 200-204 | `thinking_ratio`：以该概率开启思考（GRPO/PPO 默认 0.9——RL 阶段重点训练思考能力）。bos/eos 锚点此处定义了但没在本类使用（rollout 由模型自己生成，不需要文本锚点匹配）——冗余属性 |
| 213 | 也做随机 system 增强 |
| 214-219 | **`conversations[:-1]` 丢弃最后一条 assistant**（标准答案不能喂给模型，否则作弊）→ `add_generation_prompt=True`（尾部加「assistant 待命」）→ `open_thinking=use_thinking` 按 50%/90% 概率决定要不要让模型思考。RL 训练的 prompt 就绪 |
| 225-228 | 返回 dict：prompt 文本 + 空 answer 占位（接口统一）。真正的 rollout/reward 在 trainer 里做 |

---

## 8. AgentRLDataset（230-258 行）

```python
class AgentRLDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.samples.append(json.loads(line.strip()))

    def __len__(self):
        return len(self.samples)

    def parse_conversations(self, conversations):
        messages = []
        tools = None
        for message in conversations:
            message = dict(message)
            if message.get("role") == "system" and message.get("tools"):
                tools = json.loads(message["tools"]) if isinstance(message["tools"], str) else message["tools"]
            messages.append(message)
        return messages[:-1], tools

    def __getitem__(self, index):
        sample = self.samples[index]
        messages, tools = self.parse_conversations(sample['conversations'])
        return {'messages': messages, 'tools': tools, 'gt': sample['gt']}
```

| 行 | 讲解 |
|----|------|
| 236-238 | **手动逐行读 json**（不用 load_dataset）——因为这里的样本含嵌套结构（tools 字符串、gt 列表），HF Arrow 的类型推断反而添乱；agent 数据量小，内存读无妨 |
| 243-251 | 解析：挖出 `tools`（JSON 字符串→列表）；`messages[:-1]` 同样**丢掉最后的 assistant 轨迹**（参考轨迹不喂模型） |
| 253-257 | 返回三元组：截断后的消息列表、工具定义、标准答案 `gt`。**注意返回的是原始结构而非编码文本**——多轮工具调用的 rollout 由 train_agent.py 每步动态渲染，Dataset 只负责供料 |

---

## 9. 五个 Dataset 一览表

| 类 | 输出 | 给谁用 | 关键点 |
|----|------|--------|--------|
| PretrainDataset | (input_ids, labels) | train_pretrain | bos/eos 包裹，pad→-100 |
| SFTDataset | (input_ids, labels) | train_full_sft / lora / distillation | 锚点定位 assistant 段做 mask；多轮支持 |
| DPODataset | 六键 dict（x/y/mask × chosen/rejected） | train_dpo | 成对一致处理；自己 shift |
| RLAIFDataset | {'prompt', 'answer': ""} | train_grpo / ppo | 只出 prompt；thinking_ratio 随机 |
| AgentRLDataset | {'messages', 'tools', 'gt'} | train_agent | 原始结构，rollout 时动态渲染 |

## 10. 本篇小结

1. **锚点匹配算法**（bos_id 搜索 → eos_id 搜索 → 区间标记）是 SFT/DPO 共用的 loss mask 核心，支持多轮。
2. 三个概率性增强各有用途：system 随机化（分布稳健）、空思考 80% 删除（自适应思考）、DPO 成对一致（公平比较）。
3. RL 类 Dataset 只出 prompt/结构，生成与打分交给 trainer——数据与训练职责分离。
4. 模型 forward 的 shift（`logits[:-1]` vs `labels[1:]`）在 Pretrain/SFT 由模型做，在 DPO 由数据集做——读训练代码时要分清。

下一篇：[04-训练/01-训练基础设施-trainer_utils](../04-训练/01-训练基础设施-trainer_utils.md)
