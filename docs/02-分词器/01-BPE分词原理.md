# 01 - BPE 分词原理

> 本篇是 [02-训练分词器-逐行讲解](./02-训练分词器-逐行讲解.md) 的原理篇。模型看到的世界不是文字，而是 token id——**分词器的质量直接决定模型的语言能力上限**。

---

## 1. 为什么需要分词器

神经网络的输入必须是数字。把文本变成数字序列的方案从粗到细：

| 方案 | 例子 | 问题 |
|------|------|------|
| 按字符 | 「你」「好」 | 序列超长（中文尚可、英文灾难），注意力 O(S²) 爆炸 |
| 按词 | "hello" | 词表爆炸（几十万），未登录词无法处理 |
| **子词（BPE 等）** | "unbelievable" → "un" + "believ" + "able" | 折中：高频词整词化、低频词字符化，**任何文本都可编码** |

现代 LLM（GPT/Qwen/Llama/MiniMind）几乎都用 **BPE（Byte-Pair Encoding）** 或其变体。

## 2. BPE 算法

### 2.1 训练阶段（建词表）

从「字符级初始词表」开始，**迭代地把语料中最常见的相邻符号对合并成新符号**：

```
语料统计（初始，全是字符）:
  low  ×5   →  l o w
  lower×2   →  l o w e r
  newest×6  →  n e w e s t
  widest×3  →  w i d e s t

第 1 步: 统计相邻对 → ("e","s") 出现 9 次最多 → 合并 → 词表 + "es"
第 2 步: ("es","t") 9 次 → 合并 "est"
第 3 步: ("l","o") 7 次 → 合并 "lo"
第 4 步: ("lo","w") 7 次 → 合并 "low"
...
直到词表达到 vocab_size（MiniMind: 6400）
```

每一步产出一个**合并规则（merge）**，规则按产出顺序编号——这个序号就是优先级。词表 = 初始字符 + 所有 merge 结果。

### 2.2 编码阶段（文本 → id）

对一段新文本：

1. **预切分**（pre-tokenization）：按规则把文本切成小片段（MiniMind 用 ByteLevel 预切分，见第 3 节）；
2. 每个片段转成字节序列（初始符号）；
3. **按 merge 优先级从高到低贪心合并**：能套用的最早规则先套用；
4. 查词表得到 id。

例：词表含 "low" 合并规则时，"lowest" 编码为 `low + est`（低频词被拆开）——这就是「高频整词、低频拆分」的弹性。

### 2.3 MiniMind 词表的三个决定

- **vocab_size=6400**：极小词表。代价是压缩率低（中文约 1 token≈1.5~1.7 字符，英文更碎），换来 embedding 矩阵小（6400×768≈4.9M，占 64M 模型的 7.7%）与 softmax 便宜。**词表大小是小模型算力的预算分配问题**——vocab 每大一分，embedding/输出层和 lm_head 计算都要涨价。
- **BPE + ByteLevel**（GPT-2 风格）。
- **36 个预留 special token**：控制流标记。

## 3. ByteLevel：字节级 BPE

### 3.1 为什么要在「字节」层面做 BPE

普通 BPE 从「Unicode 字符」出发，两个隐患：

1. 词表必须收录所有出现的字符，中文/日文/emoji 各占数千个词表位；
2. 训练语料没见过的字符（生僻 Unicode）**无法编码**。

**ByteLevel BPE**（GPT-2 方案）：先把文本 UTF-8 编码成**字节流**（任何文本都是 0-255 的字节序列），把每个字节映射成一个**可打印的 Unicode 代理字符**（如空格→`Ġ`），再在这个 256 个「字节字符」的字母表上做 BPE 合并。效果：

- 初始词表固定 256，天然覆盖一切语言/二进制/emoji；
- 多字节的中文字符（UTF-8 占 3 字节）作为整体高频出现后会被 merge 成整 token。

### 3.2 `pre_tokenizers.ByteLevel(add_prefix_space=False)`

HuggingFace tokenizers 的 ByteLevel 预切分器做两件事：

1. **按空格切块**（GPT-2 正则风格）：`" hello world"` → `[" hello", " world"]`——空格归属**后词开头**（英文单词+前置空格成为统计单元，这是英文压缩率的关键）；
2. 字节→代理字符映射。

`add_prefix_space=False`：**不要**在每个文本开头自动加空格（GPT-2 默认会加，对中文场景没必要且会污染对话格式）。

### 3.3 解码与流式

`decoders.ByteLevel()` 是逆映射：token → 代理字符 → 字节 → UTF-8 解码。流式输出时若一个中文字符的 3 个字节分属 3 个未完整的 token，解码会出现乱码占位符 `\ufffd`——train_tokenizer.py 的 eval 部分专门测试了「字节缓冲」处理（见逐行讲解篇）。

## 4. Special Token 设计（36 个）

MiniMind 的 special tokens 布局（详见逐行讲解篇代码）：

| 组 | Token | 用途 |
|----|-------|------|
| 核心（0/1/2） | `<|endoftext|>` `<|im_start|>` `<|im_end|>` | pad / **BOS（消息开始+角色）** / **EOS（消息结束，生成停止符）** |
| 多模态占位 | `<|vision_*|>` `<|image_pad|>` `<|audio_*|>` … | 为视觉/音频模型（MiniMind-V/O）预留 |
| TTS 占位 | `<tts_pad>` 等 | 语音预留 |
| 工具与思考 | `<tool_call>` `</tool_call>` `<tool_response>` `</tool_response>` `<think>` `</think>` | Tool Call 与 reasoning 协议 |
| 缓冲 | `<|buffer1|>`…`<|buffer9|>` | 凑满 36：将来加新标记时**词表 id 不位移**——special token 的 id 一旦在已发布模型间变动，新旧 checkpoint 全部不兼容，所以宁可先占位 |

注意 `<tool_call>` 这一组在 tokenizer.json 里被后处理成 `special=False`（非特殊 token）：它们是**普通文本 token**——模型在正文里也可能「说出」这几个字面量，不应被 tokenizer 强行拦截成特殊语义。

## 5. chat_template：分词器生态的另一半

tokenizer_config.json 里最长的字段是 `chat_template`——一段 Jinja2 模板，负责把 `[{role, content}, ...]` 消息列表渲染成模型实际看到的训练/推理文本：

```
<|im_start|>system
你是…<|im_end|>
<|im_start|>user
你好<|im_end|>
<|im_start|>assistant
<think>

</think>

你好！…<|im_end|>
```

**它决定了「学习目标怎么切」**：SFT 的 loss mask（只学 assistant 段）就是靠在渲染结果里定位 `<|im_start|>assistant\n` 与 `<|im_end|>\n` 实现的（见 03-数据集/02）。模板的逐块精读放在训练分词器篇里，因为它作为字符串写在那个脚本里。

## 6. 本篇小结

1. BPE = 「**统计高频相邻对并迭代合并**」的贪心词表构建 + 同规则优先级的贪心编码。
2. **ByteLevel** 让词表天然覆盖任意语言，中文 1 token≈1.5~1.7 字。
3. 词表 6400 是小模型的算力预算选择；36 个 special token 含 9 个 buffer，为生态扩展保 id 稳定。
4. `chat_template` 与分词器是「一体两面」：模板渲染的文本里含特殊标记，训练数据的 loss mask、推理的停止符都依赖它。

下一篇：[02-训练分词器-逐行讲解](./02-训练分词器-逐行讲解.md)
