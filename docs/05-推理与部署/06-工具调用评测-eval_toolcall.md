# 06 - 工具调用评测 eval_toolcall.py 全文

> 覆盖代码：`scripts/eval_toolcall.py` **全文 241 行**。
> 功能：**工具调用（Tool Call）能力评测/演示**——给模型一组工具与测试题，观察它能否选对工具、传对参数、消化结果、继续作答。支持两种后端：`local`（本地模型直接 generate）与 `api`（OpenAI 兼容接口，如 ollama）。
> 它同时是**学习 Tool Call 全链路**的最佳样本：schema 注入 → 模型输出解析 → 执行 → 结果回传 → 多轮循环。

---

## 1. 原理：Tool Call 的完整回路

```
                    ┌──────────────────────────────────────┐
   TOOLS schema ──→ │ system 段（chat_template 的 <tools> 块）│
                    └──────────────┬───────────────────────┘
                                   ▼
              user 问题 ──→ 模型生成 ──→ <tool_call>{"name":...,"arguments":...}</tool_call>
                                   │                    │
                                   │            parse_tool_calls 解析
                                   │                    ▼
                                   │            execute_tool 执行 → 结果 JSON
                                   │                    │
                                   ▼                    ▼
        下一轮 messages = [system, user, assistant(带tool_call), tool(结果)] → 再生成
                                   │
                          无 tool_call 输出 → 结束（文本回答）
```

关键点：**模型永远只产出文本**；`<tool_call>` 是文本里的格式约定，解析/执行/回传全在客户端代码里。

## 2. 逐行精读

### 2.1 工具定义（18-40 行）

```python
TOOLS = [
    {"type": "function", "function": {"name": "calculate_math", "description": "计算数学表达式的结果，支持加减乘除、幂运算、开方等", "parameters": {"type": "object", "properties": {"expression": {"type": "string", "description": "数学表达式，如123+456、2**10、sqrt(144)"}}, "required": ["expression"]}}},
    ...共 8 个工具
]
```

| 工具 | 参数 | 执行结果 |
|------|------|----------|
| calculate_math | expression | **真算**（safe_math_eval） |
| get_current_time | timezone（默认 Asia/Shanghai） | **真时钟**（datetime.now） |
| random_number | min/max | 真 randint |
| text_length | text | 真字符数/词数 |
| unit_converter | value/from_unit/to_unit | 固定 0.621371 系数（km→miles 占位） |
| get_current_weather | location/unit | 固定 22°C/晴 |
| get_exchange_rate | from/to_currency | 固定 7.15 |
| translate_text | text/target_language | 固定 hello world |

```python
MOCK_RESULTS = {
    "calculate_math": lambda args: {"result": str(safe_math_eval(args.get("expression", "0")))},
    "get_current_time": lambda args: {"datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), ...},
    ...
}
TOOL_MAP = {t["function"]["name"]: t for t in TOOLS}

def get_tools(names):
    return [TOOL_MAP[n] for n in names]
```

| 行 | 讲解 |
|----|------|
| 29-38 | 名字 → 执行 lambda 的映射表（与 train_agent 的 MOCK_RESULTS 结构一致，但这里时间/随机是**真值**，天气/汇率/翻译是占位——评测关心的是「调用行为正确性」不是数值） |
| 40-43 | `TOOL_MAP` + `get_tools(names)`：按名字子集取 schema——**测试时只暴露题目相关工具**，考验模型「在有限候选中选择」的能力 |

### 2.2 测试用例（45-54 行）

```python
TEST_CASES = [
    {"prompt": "帮我算一下 256 乘以 37 等于多少", "tools": ["calculate_math", "get_current_time"]},
    {"prompt": "现在几点了？", "tools": ["get_current_time", "random_number"]},
    {"prompt": "帮我把100公里换算成英里", "tools": ["unit_converter", "calculate_math"]},
    {"prompt": "帮我生成一个1到1000的随机数，然后计算它的平方", "tools": ["random_number", "calculate_math", "text_length"]},   # ← 两连调用
    {"prompt": "北京今天天气怎么样？", "tools": ["get_current_weather", "get_current_time"]},
    {"prompt": "查一下美元兑人民币汇率", "tools": ["get_exchange_rate", "get_current_time"]},
    {"prompt": "把'你好世界'翻译成英文", "tools": ["translate_text", "text_length"]},
    {"prompt": "What is the weather in Tokyo? Also convert 30 celsius to fahrenheit.", "tools": ["get_current_weather", "unit_converter", "get_current_time"]},  # ← 英文+复合
]
```

- 每题只给 2-3 个候选工具（**干扰项设计**：如「几点了」给个 random_number 干扰）；第 4 题考**链式调用**（随机数 → 平方需要先执行再看结果）；第 8 题考英文 + 复合任务。

### 2.3 init_model（57-67 行）

与 eval_llm.py 的 init_model **相同**（model 目录判断、strict=True、fp16、eval、参数量打印），少 LoRA 分支（评测不挂 LoRA）。

### 2.4 解析函数（70-97 行）

```python
def parse_tool_calls(text):
    matches = re.findall(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL)
    calls = []
    for m in matches:
        try:
            calls.append(json.loads(m.strip()))
        except Exception:
            pass
    return calls

def parse_tool_call_from_text(content):
    pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
    matches = re.findall(pattern, content, re.DOTALL)
    if not matches:
        return None
    tool_calls = []
    for i, match in enumerate(matches):
        try:
            data = json.loads(match)
            tool_calls.append({
                "id": f"call_{i}",
                "function": {"name": data.get("name", ""), "arguments": json.dumps(data.get("arguments", {}), ensure_ascii=False)}
            })
        except Exception:
            pass
    return tool_calls if tool_calls else None
```

| 函数 | 用途 | 差异 |
|------|------|------|
| `parse_tool_calls` | **local 后端**：模型原文抽 `<tool_call>…</tool_call>` → 原始 dict（name/arguments 平铺） | 宽松匹配 `.*?` |
| `parse_tool_call_from_text` | **api 后端的兜底**：服务端没把 tool_calls 结构化时（如 ollama 未启用原生工具协议），从 content 文本里抽并转成 **OpenAI 结构**（id/function.name/function.arguments 字符串） | 正则要求 `{...}` 内芯；输出对齐 SDK 的对象属性访问（run_case 里 `tc.id/tc.function.name`） |

### 2.5 execute_tool（99-112 行）

```python
def execute_tool(call, arguments=None):
    name = call.get("name", "") if isinstance(call, dict) else call
    try:
        raw_args = call.get("arguments", {}) if isinstance(call, dict) else arguments
        args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
    except Exception:
        args = {}
    fn = MOCK_RESULTS.get(name)
    if not fn:
        return {"error": f"未知工具: {name}"}
    try:
        return fn(args)
    except Exception as e:
        return {"error": f"工具执行失败: {str(e)[:80]}"}
```

- **双形态入参**：dict（含 name+arguments）或裸名字符串+arguments——兼容两种后端的调用习惯；arguments 字符串/字典都接。
- 三层防御：解析失败→空参数、未知工具→error dict、执行异常→截断 80 字符的 error。**错误也回传给模型**（养成模型从报错恢复的行为）。

### 2.6 generate：local 后端（115-130 行）

```python
def generate(model, tokenizer, messages, tools, args):
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, tools=tools, open_thinking=False)
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True).to(args.device)
    st = time.time()
    print('🧠: ', end='')
    generated_ids = model.generate(
        inputs["input_ids"], attention_mask=inputs["attention_mask"],
        max_new_tokens=args.max_new_tokens, do_sample=True, streamer=streamer,
        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
        top_p=args.top_p, temperature=args.temperature
    )
    response = tokenizer.decode(generated_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
    ...
    return response
```

- 与 eval_llm 的生成段相同（**open_thinking 固定 False**——评测专注工具行为）；`tools` 传给模板的 `<tools>` 块；返回完整原始文本（含 `<tool_call>` 字面量，供 parse_tool_calls 解析）。

### 2.7 chat_api：api 后端（133-174 行）

```python
def chat_api(client, messages, tools, args, stream=True):
    response = client.chat.completions.create(
        model=args.api_model, messages=messages, tools=tools,
        stream=stream, temperature=args.temperature,
        max_tokens=8192, top_p=args.top_p
    )
```

- **`tools=tools` 直接进 SDK**：OpenAI 协议的原生工具传递（服务端负责把 schema 塞进模板——serve_openai_api 正是这么做的）。

```python
    if not stream:
        choice = response.choices[0]
        content = choice.message.content or ""
        tool_calls = choice.message.tool_calls
        if not tool_calls:
            tool_calls = parse_tool_call_from_text(content)
        print(f'🧠: {content}')
        return content, tool_calls
```

| 行 | 讲解 |
|----|------|
| 139-146 | 非流式：先取**结构化** `message.tool_calls`（SDK 对象列表）；没有则**文本兜底解析**——两路取一，兼容「服务端支持/不支持原生工具协议」两种情况 |

```python
    print('🧠: ', end='', flush=True)
    content, tool_calls = "", None
    for chunk in response:
        delta = chunk.choices[0].delta
        if delta.content:
            print(delta.content, end="", flush=True); content += delta.content
        if delta.tool_calls:
            if tool_calls is None: tool_calls = []
            for tc_chunk in delta.tool_calls:
                idx = tc_chunk.index if tc_chunk.index is not None else len(tool_calls)
                while len(tool_calls) <= idx:
                    tool_calls.append({"id": "", "function": {"name": "", "arguments": ""}})
                if tc_chunk.id:
                    tool_calls[idx]["id"] += tc_chunk.id
                if tc_chunk.function:
                    if tc_chunk.function.name:
                        tool_calls[idx]["function"]["name"] += tc_chunk.function.name
                    if tc_chunk.function.arguments:
                        tool_calls[idx]["function"]["arguments"] += tc_chunk.function.arguments
    print()
    if not tool_calls:
        tool_calls = parse_tool_call_from_text(content)
    return content, tool_calls
```

| 行 | 讲解 |
|----|------|
| 149-170 | **流式 tool_calls 累积**——这是与 chat_api.py（纯对话客户端）最大的不同。OpenAI 流式协议把一个工具调用**拆成多个 chunk**：id/name/arguments 字符串分段到达。累积算法：<br>1. `tc_chunk.index` 定位是第几个工具（没有 index 就追加）；<br>2. `while len(tool_calls) <= idx`：按需补齐空槽（乱序到达也安全）；<br>3. id/name/arguments 逐字段 **字符串拼接**（`+=`）——chunk 内是片段不是整体；<br>4. 无结构化 tool_calls 时同样文本兜底。 |

### 2.8 run_case：单题多轮回路（177-199 行）

```python
def run_case(prompt, tools, args, model=None, tokenizer=None, client=None):
    messages = [{"role": "user", "content": prompt}]
    while True:
        if args.backend == 'local':
            content = generate(model, tokenizer, messages, tools, args)
            tool_calls = parse_tool_calls(content)
        else:
            content, tool_calls = chat_api(client, messages, tools, args, stream=bool(args.stream))
        if not tool_calls:
            break
        tool_calls = [{
            "id": tc.id if hasattr(tc, 'id') else tc.get("id", ""),
            "name": tc.function.name if hasattr(tc, 'function') else tc["function"]["name"],
            "arguments": tc.function.arguments if hasattr(tc, 'function') else tc["function"]["arguments"]
        } for tc in tool_calls] if args.backend == 'api' else tool_calls
        messages.append({"role": "assistant", "content": content} if args.backend == 'local' else {"role": "assistant", "content": content, "tool_calls": [{"id": tc["id"], "type": "function", "function": {"name": tc["name"], "arguments": tc["arguments"]}} for tc in tool_calls]})
        for tc in tool_calls:
            name = tc["name"]
            arguments = tc["arguments"]
            print(f'📞 [Tool Calling]: {name} | args={arguments}')
            result = execute_tool(tc if args.backend == 'local' else name, arguments)
            print(f'✅ [Tool Called]: {json.dumps(result, ensure_ascii=False)}')
            messages.append({"role": "tool", "content": json.dumps(result, ensure_ascii=False)} if args.backend == 'local' else {"role": "tool", "content": json.dumps(result, ensure_ascii=False), "tool_call_id": tc["id"]})
```

| 行 | 讲解 |
|----|------|
| 178-186 | 生成 → 解析；**没有工具调用 = 结束**（while True 的唯一出口） |
| 187-191 | api 后端的 SDK 对象 → 统一 dict（`hasattr(tc,'id')` 区分 SDK 对象/文本解析 dict——parse_tool_call_from_text 返回的元素就是普通 dict） |
| 192 | **assistant 消息回填**：local 直接存文本（下轮模板把 `<tool_call>` 字面量渲染回去）；**api 需要显式 tool_calls 结构**（OpenAI 协议要求 assistant 轮带上它声明的调用，随后 tool 轮用 tool_call_id 关联） |
| 193-199 | 逐个执行工具、打印（📞/✅ emoji 分行）、**tool 消息入栈**（api 版带 tool_call_id 关联——协议完整性；local 版纯文本即可，模板会把 tool 消息渲染成 `<tool_response>` user 段） |

### 2.9 main（202-241 行）

```python
    parser.add_argument('--backend', default='local', choices=['local', 'api'], ...)
    parser.add_argument('--api_base_url', default="http://localhost:11434/v1", ...)
    parser.add_argument('--api_model', default='jingyaogong/minimind-3:latest', ...)
    parser.add_argument('--stream', default=1, type=int, ...)
    ...
    model = tokenizer = client = None
    if args.backend == 'local': model, tokenizer = init_model(args)
    else: client = OpenAI(api_key=args.api_key, base_url=args.api_base_url)

    input_mode = int(input('[0] 自动测试\n[1] 手动输入\n'))

    cases = [{"prompt": case["prompt"], "tools": get_tools(case["tools"]), "tool_names": case["tools"]} for case in TEST_CASES] if input_mode == 0 else iter(lambda: {"prompt": input('💬: '), "tools": TOOLS, "tool_names": [...]}, {"prompt": "", ...})
    for case in cases:
        if not case["prompt"]: break
        setup_seed(random.randint(0, 31415926))
        if input_mode == 0:
            print(f'📦 可用工具: {case["tool_names"]}\n')
            print(f'💬: {case["prompt"]}')
        run_case(case["prompt"], case["tools"], args, model=model, tokenizer=tokenizer, client=client)
        print('\n' + '-' * 50 + '\n')
```

| 行 | 讲解 |
|----|------|
| 222-224 | 后端二选一初始化（local 模型 / OpenAI client） |
| 228 | 自动模式用 TEST_CASES（**每题按名字子集取工具**）；手动模式给全量 TOOLS、空 prompt 哨兵结束 |
| 231-236 | 随机种子逐题刷新；打印可用工具清单（评测者能直观看到「模型拿到的候选」）；分隔线 |

## 3. 一次运行的样子

```
📦 可用工具: ['calculate_math', 'get_current_time']
💬: 帮我算一下 256 乘以 37 等于多少
🧠: <tool_call>
{"name": "calculate_math", "arguments": {"expression": "256*37"}}
</tool_call>
📞 [Tool Calling]: calculate_math | args=...{"expression": "256*37"}
✅ [Tool Called]: {"result": "9472"}
🧠: 256 乘以 37 等于 9472。
```

第二轮模型把工具结果整合成自然语言答案且不再调用工具 → run_case 的 while 退出。

## 4. 本篇小结

1. **Tool Call 全链路六步**：schema 注入 → 生成 → 解析（正则/结构化双路）→ 执行（防御式）→ 结果回传（local 文本 / api 带 tool_call_id）→ 循环直到无调用。
2. **流式 tool_calls 累积**是 OpenAI 协议客户端的必备技能：index 定位 + 空槽补齐 + 字段字符串拼接。
3. 两后端一套逻辑：local 看 `<tool_call>` 字面量，api 看结构化消息——`hasattr`/`isinstance` 双形态兼容贯穿始终。
4. 评测设计细节：每题工具子集（干扰项）、链式调用题、英文复合题。

---

**docs 目录到此覆盖了仓库全部代码**（model / dataset / trainer / scripts / eval_llm.py / requirements / 数据格式 / chat_template）。建议从 [docs/README.md](../README.md) 的学习路线开始按序阅读，或直接进入 [00-项目总览](../00-项目总览.md) 复习全景。
