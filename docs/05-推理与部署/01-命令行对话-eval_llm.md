# 01 - 命令行对话 eval_llm.py 全文

> 覆盖代码：`eval_llm.py`（仓库根目录）**全文 99 行**。
> 虽然名字带 eval，它其实是**对话试用脚本**：加载训好的权重，跟模型打字聊天（支持自动批量测试与手动输入），顺带测生成速度。真正的基准评测（C-Eval 等）走 README 里的独立工具。

---

## 1. 原理：本地权重 vs transformers 格式的双模加载

MiniMind 有两种「成品形态」：

| 形态 | 目录特征 | 加载方式 |
|------|----------|----------|
| 原生 torch 权重 | `out/full_sft_768.pth`（单文件 .pth） | `MiniMindForCausalLM(config)` + `load_state_dict` |
| transformers 格式 | `minimind-3/`（config.json+safetensors，由 convert_model 转出） | `AutoModelForCausalLM.from_pretrained` |

eval_llm 用「路径末段是否叫 `model`」来区分（这是仓库约定：`model/` 是分词器+默认 .pth 的家）。

## 2. 逐行精读

### 2.1 导入（1-11 行）

```python
import os, time, argparse, random, warnings
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_lora import *
from trainer.trainer_utils import setup_seed, get_model_params
warnings.filterwarnings('ignore')
```

| 行 | 讲解 |
|----|------|
| 7 | `TextStreamer`：HF 的流式打印器——每生成一个 token 就解码打印（内部含字节缓冲处理多字节字符，分词器篇讲过 `\ufffd` 机制） |
| 9 | `from ... import *`：LoRA 的四个函数（apply/load/save/merge）全进来（用在哪见 init_model） |
| 10 | setup_seed（全源种子）、get_model_params（参数量打印） |

### 2.2 init_model（13-31 行）

```python
def init_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)
    if os.path.basename(os.path.normpath(args.load_from)) == 'model':
        model = MiniMindForCausalLM(MiniMindConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            use_moe=bool(args.use_moe),
            inference_rope_scaling=args.inference_rope_scaling
        ))
        moe_suffix = '_moe' if args.use_moe else ''
        ckp = f'./{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'
        model.load_state_dict(torch.load(ckp, map_location=args.device), strict=True)
        if args.lora_weight != 'None':
            apply_lora(model, rank=args.lora_rank)
            load_lora(model, f'./{args.save_dir}/{args.lora_weight}_{args.hidden_size}{moe_suffix}.pth')
    else:
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)
    get_model_params(model, model.config)
    return model.half().eval().to(args.device), tokenizer
```

| 行 | 讲解 |
|----|------|
| 14 | tokenizer 从同一目录加载（`model/` 里有 tokenizer 配置；transformers 目录里是 convert_model 转存的那套） |
| 15 | `os.path.basename(os.path.normpath(...))`：规范化路径后取末段——`./model/` 和 `model` 都会变成 `'model'`，**避免尾斜杠破坏判断** |
| 16-23 | **原生分支**：空模型（可带 YaRN 外推开关）→ 按命名规则拼权重路径 → `strict=True` 严格加载（推理时不允许键不匹配——与训练续训的 strict=False 相反，宁可报错不可静默错配） |
| 24-27 | **LoRA 挂载**：`--lora_weight lora_medical` 时挂旁路、载权重（rank 必须与训练一致，否则形状对不上） |
| 29 | **transformers 分支**：`trust_remote_code=True` 允许执行目录里 model_type 指向的自定义代码（MiniMindConfig/ForCausalLM 是 register 过的） |
| 30-31 | 打参数量；**`.half().eval().to(device)`**：推理用 fp16（显存减半、速度更快）+ eval 模式（关 dropout） |

### 2.3 main 的参数（33-51 行）

```python
    parser.add_argument('--load_from', default='model', ...)
    parser.add_argument('--save_dir', default='out', ...)
    parser.add_argument('--weight', default='full_sft', ...)
    parser.add_argument('--lora_weight', default='None', ...)
    parser.add_argument('--lora_rank', default=16, ...)
    parser.add_argument('--hidden_size', default=768, ...)
    parser.add_argument('--num_hidden_layers', default=8, ...)
    parser.add_argument('--use_moe', default=0, ...)
    parser.add_argument('--inference_rope_scaling', default=False, action='store_true', ...)
    parser.add_argument('--max_new_tokens', default=8192, ...)
    parser.add_argument('--temperature', default=0.85, ...)
    parser.add_argument('--top_p', default=0.95, ...)
    parser.add_argument('--open_thinking', default=0, ...)
    parser.add_argument('--historys', default=0, ...)
    parser.add_argument('--show_speed', default=1, ...)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', ...)
```

要点：

| 参数 | 讲解 |
|------|------|
| weight | 权重前缀：pretrain/full_sft/dpo/ppo_actor/grpo/agent——**换一个权重就能对比各阶段效果** |
| inference_rope_scaling | store_true 开关：推理期 YaRN 外推（07 篇） |
| open_thinking | 自适应思考开关（模板的 open_thinking 参数） |
| historys | 携带几轮历史（**必须偶数**——一轮=Q+A 两条消息；0=无记忆单轮） |

### 2.4 对话主循环（53-93 行）

```python
    prompts = [
        '你有什么特长？', '为什么天空是蓝色的', '请用Python写一个计算斐波那契数列的函数',
        '解释一下"光合作用"的基本过程', '如果明天下雨，我应该如何出门',
        '比较一下猫和狗作为宠物的优缺点', '解释什么是机器学习', '推荐一些中国的美食'
    ]
    
    conversation = []
    model, tokenizer = init_model(args)
    input_mode = int(input('[0] 自动测试\n[1] 手动输入\n'))
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    prompt_iter = prompts if input_mode == 0 else iter(lambda: input('💬: '), '')
    for prompt in prompt_iter:
        setup_seed(random.randint(0, 31415926))
        if input_mode == 0: print(f'💬: {prompt}')
        conversation = conversation[-args.historys:] if args.historys else []
        conversation.append({"role": "user", "content": prompt})
        if 'pretrain' in args.weight:
            inputs = tokenizer.bos_token + prompt
        else:
            inputs = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True, open_thinking=bool(args.open_thinking))
```

| 行 | 讲解 |
|----|------|
| 69 | **两种输入源**：自动模式直接遍历 8 个测试题；手动模式 `iter(lambda: input(...), '')` 造一个「输入空串就停」的迭代器（第二个参数是哨兵值） |
| 71 | **每个 prompt 换一个随机种子**——同一问题多次运行答案不同（可控的随机性） |
| 73-74 | 历史窗口裁剪（偶数条）+ 追加本轮 user 消息 |
| 75-78 | **两种 prompt 形态**：pretrain 权重没学过对话模板，直接拼 `bos_token + 文本` 续写；其他权重走 chat_template（add_generation_prompt=True 尾部补 assistant 提示 + open_thinking 开关） |

```python
        inputs = tokenizer(inputs, return_tensors="pt", truncation=True).to(args.device)

        print('🧠: ', end='')
        st = time.time()
        generated_ids = model.generate(
            inputs=inputs["input_ids"], attention_mask=inputs["attention_mask"],
            max_new_tokens=args.max_new_tokens, do_sample=True, streamer=streamer,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            top_p=args.top_p, temperature=args.temperature, repetition_penalty=1
        )
        response = tokenizer.decode(generated_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
        conversation.append({"role": "assistant", "content": response})
        gen_tokens = len(generated_ids[0]) - len(inputs["input_ids"][0])
        print(f'\n[Speed]: {gen_tokens / (time.time() - st):.2f} tokens/s\n\n') if args.show_speed else print('\n\n')
```

| 行 | 讲解 |
|----|------|
| 80 | 编码（`truncation=True` 默认按 model_max_length 截）；mask 由 tokenizer 生成 |
| 84-89 | 调**手写 generate**（01-模型架构/08 篇逐行讲过）：streamer 让 token 边生成边打印；采样参数透传 |
| 90 | **切出回复**：`generated_ids[0][prompt_len:]` 丢掉 prompt 部分，skip_special_tokens 去掉 `<|im_end|>` 等标记 |
| 91 | 回复进历史（下一轮就是上下文） |
| 92-93 | 速度统计：生成 token 数 ÷ 耗时。TextStreamer 已经边生成边打印了内容，这里补一个速率 |

### 2.5 入口（95-99 行）

```python
if __name__ == "__main__":
    try:
        main()
    except (EOFError, KeyboardInterrupt):
        print()
```

- 捕获 Ctrl+C/EOF 干净退出（手动模式下 Ctrl+D 结束输入）。

## 3. 典型用法

```bash
# 用 full_sft 对话
python eval_llm.py --weight full_sft

# 对比 RL 前后
python eval_llm.py --weight full_sft
python eval_llm.py --weight grpo

# 挂 LoRA、开思考
python eval_llm.py --weight full_sft --lora_weight lora_medical --open_thinking 1

# 长文本外推
python eval_llm.py --inference_rope_scaling
```

## 4. 本篇小结

1. 双模加载：`.pth`（strict=True + 可选 LoRA）或 transformers 目录（AutoModel）。
2. prompt 组装分水岭：**pretrain 权重走 bos+文本，其余走 chat_template**——权重与数据形态必须匹配。
3. TextStreamer 流式打印 + generate 手写采样 + 速度统计，一条命令完成对话试用。

下一篇：[02-OpenAI兼容API服务-serve_openai_api](./02-OpenAI兼容API服务-serve_openai_api.md)
