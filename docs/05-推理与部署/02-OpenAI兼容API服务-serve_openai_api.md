# 02 - OpenAI 兼容 API 服务 serve_openai_api.py 全文

> 覆盖代码：`scripts/serve_openai_api.py` **全文 277 行**。
> 用 FastAPI 实现一个**协议上兼容 OpenAI** 的 `/v1/chat/completions` 端点：任何 OpenAI SDK 客户端（含 ollama/Open-WebUI/FastGPT）都能直接连。亮点是**流式输出的实时解析状态机**：token 逐个产出时，`<think>` 段实时分流到 `reasoning_content` 字段、`<tool_call>` 块聚合为 `tool_calls` 字段。
> 前置：[01-模型架构/08-生成解码策略](../01-模型架构/08-生成解码策略.md)（generate 的 streamer 回调）。

---

## 1. 原理

### 1.1 OpenAI Chat API 协议要点

请求：`POST /v1/chat/completions`，body 含 `model/messages/temperature/top_p/max_tokens/stream/tools`。响应两种：

- **非流式**：一次性 JSON（`choices[0].message`）；
- **流式 SSE**：`text/event-stream`，每行 `data: {json}\n\n`，每个 chunk 携带 `choices[0].delta`（增量内容）；流末尾 `data: [DONE]`（本实现未发送 DONE，兼容大多数客户端）。

MiniMind 的三个协议扩展字段（对齐 DeepSeek/Qwen 风格）：

- `delta.reasoning_content`：思考内容（独立于 content 展示）；
- `delta.tool_calls`：工具调用（数组，含 index/id/function.name/arguments）；
- `finish_reason: "tool_calls"`：提示客户端「该执行工具了」。

### 1.2 生成线程 + 队列的流水线

```
主线程: FastAPI handler → while True: queue.get() → 解析 → yield SSE
生成线程: model.generate(..., streamer=CustomStreamer(queue))
              └ CustomStreamer.on_finalized_text: queue.put(text)
```

生成是**阻塞循环**，必须放独立线程；主线程从队列消费、立刻 yield 给 HTTP 客户端——两线程通过 `Queue` 解耦，生成不停、传输不断。`stream_end=True` 时放 `None` 哨兵，主线程据此退出。

### 1.3 边流边解析的困难

模型输出是**逐 token 的字符流**，而解析目标是结构化段：

```
<think>思考…</think>\n\n<tool_call>{"name":...}</tool_call>正文
```

困难有三：① 标记 `<tool_call>` 可能**跨 chunk 被劈开**（一次只到 `<tool_c`）；② 标记内外的内容要**分流到不同字段**；③ `<tool_call>` 到 `</tool_call>` 之间的内容要**攒齐再 JSON 解析**。解决法：状态机 + 「标记前缀保持」缓冲（见 §2.4）。

---

## 2. 逐行精读

### 2.1 文件头与 FastAPI（1-25 行）

```python
import argparse, json, re, os, sys
__package__ = "scripts"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time, torch, warnings, uvicorn
from threading import Thread
from queue import Queue
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_lora import apply_lora, load_lora
warnings.filterwarnings('ignore')
app = FastAPI()
```

| 行 | 讲解 |
|----|------|
| 12-13 | 路径修补（scripts 子目录脚本的标准动作，同 trainer） |
| 17-18 | `Thread/Queue` 生成-消费流水线组件；`StreamingResponse` 把生成器变成 SSE 响应 |
| 25 | 全局 `app` 对象；**model/tokenizer/device 是模块级全局变量**（init_model 后被赋值），handler 直接引用——单进程单模型的极简写法 |

### 2.2 init_model（28-47 行）

与 eval_llm.py 的同名函数**几乎一致**（双模加载+LoRA+half+eval），差异两处：

- 权重路径用 `../{save_dir}`（脚本在 scripts/ 下，相对仓库根多一层 `../`）；
- 构造 config 时**多传 `max_seq_len=args.max_seq_len`**（API 服务可配置上下文长度，默认 8192）。
- 47 行同样返回 `model.half().eval().to(device)`。注意 `device` 这里用的是模块级变量（main 里 `device = args.device` 提前赋值）。

### 2.3 ChatRequest：请求模型（50-68 行）

```python
class ChatRequest(BaseModel):
    model: str
    messages: list
    temperature: float = Field(default=0.7, ge=0.0, allow_inf_nan=False)
    top_p: float = 0.92
    max_tokens: int = 8192
    stream: bool = True
    tools: list = Field(default_factory=list)
    open_thinking: bool = False
    chat_template_kwargs: dict = None
    
    def get_open_thinking(self) -> bool:
        """兼容多种方式开启 thinking"""
        if self.open_thinking:
            return True
        if self.chat_template_kwargs:
            return self.chat_template_kwargs.get('open_thinking', False) or \
                   self.chat_template_kwargs.get('enable_thinking', False)
        return False
```

| 行 | 讲解 |
|----|------|
| 51-52 | Pydantic 模型：**自动校验请求体**（字段缺失用默认值、类型错报 422）。`model`/`messages` 是必填 |
| 53 | temperature 校验：≥0 且不能是 inf/nan（Field 的验证器） |
| 57 | `default_factory=list`：可变默认值的 Pydantic 正确写法（直接 `=[]` 会共享实例） |
| 59 | 客户端传的任意工具列表原样透传给模板 |
| 61-68 | **思考开关的三级兼容**：顶层 `open_thinking` → `chat_template_kwargs.open_thinking` → `chat_template_kwargs.enable_thinking`（Qwen 的叫法）。同一开关在不同客户端叫不同名字，都接住 |

### 2.4 CustomStreamer（71-81 行）

```python
class CustomStreamer(TextStreamer):
    def __init__(self, tokenizer, queue):
        super().__init__(tokenizer, skip_prompt=True, skip_special_tokens=True)
        self.queue = queue
        self.tokenizer = tokenizer

    def on_finalized_text(self, text: str, stream_end: bool = False):
        self.queue.put(text)
        if stream_end:
            self.queue.put(None)
```

| 行 | 讲解 |
|----|------|
| 71-72 | 继承 HF `TextStreamer`，重写唯一的钩子 |
| 73-74 | `skip_prompt=True`：不吐 prompt 部分；`skip_special_tokens=True`：`<|im_end|>` 等被过滤——**注意 `<tool_call>`/`<think>` 不是 special token**（分词器篇的 JSON 手术），会正常流出来，解析才有意义 |
| 77-80 | 每当内部字节缓冲攒齐一段可解码文本，HF 父类调用 `on_finalized_text`：文本入队；**`stream_end=True`（生成结束）时额外放 `None` 哨兵**——主线程 `queue.get()` 拿到 None 就退出循环 |

### 2.5 parse_response：整段解析（83-102 行）

```python
def parse_response(text):
    reasoning_content = None
    think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if think_match:
        reasoning_content = think_match.group(1).strip()
        text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)
    elif '</think>' in text:
        parts = text.split('</think>', 1)
        reasoning_content = parts[0].strip()
        text = parts[1].strip() if len(parts) > 1 else ''
    tool_calls = []
    for i, m in enumerate(re.findall(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL)):
        try:
            call = json.loads(m.strip())
            tool_calls.append({"id": f"call_{int(time.time())}_{i}", "type": "function", "function": {"name": call.get("name", ""), "arguments": json.dumps(call.get("arguments", {}), ensure_ascii=False)}})
        except Exception:
            pass
    if tool_calls:
        text = re.sub(r'<tool_call>.*?</tool_call>', '', text, flags=re.DOTALL)
    return text.strip(), reasoning_content, tool_calls or None
```

**非流式路径**用：整段生成完后一次性解析。两个 think 分支：

| 分支 | 场景 | 处理 |
|------|------|------|
| 完整 `<think>…</think>` | 正常生成 | 抽出思考内容；从正文删除（含尾随空白） |
| 只有 `</think>` | 模型开了思考但**被 EOS 截断**在思考中（没写出闭合） | `</think>` 之前全是思考，之后是正文 |

| 行 | 讲解 |
|----|------|
| 94-99 | tool_calls：`<tool_call>…</tool_call>` 内容 JSON 解析 → OpenAI 格式（id 用时间戳+序号、arguments 再序列化成字符串——协议要求 arguments 是字符串） |
| 100-101 | 解析成功的块从正文里删掉 |
| 102 | 返回三元组（正文, 思考, tool_calls 或 None） |

### 2.6 generate_stream_response：流式状态机（105-198 行）——全文件核心

```python
def generate_stream_response(messages, temperature, top_p, max_tokens, tools=None, open_thinking=False):
    try:
        new_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, tools=tools or None, open_thinking=open_thinking)
        inputs = tokenizer(new_prompt, return_tensors="pt", truncation=True).to(device)

        queue = Queue()
        streamer = CustomStreamer(tokenizer, queue)

        def _generate():
            try:
                model.generate(inputs.input_ids, max_new_tokens=max_tokens, do_sample=temperature > 0, temperature=temperature,
                               top_p=top_p, attention_mask=inputs.attention_mask, pad_token_id=tokenizer.pad_token_id,
                               eos_token_id=tokenizer.eos_token_id, streamer=streamer)
            except Exception as e:
                queue.put({"error": str(e)})
                queue.put(None)

        Thread(target=_generate).start()
```

| 行 | 讲解 |
|----|------|
| 107-108 | 模板渲染（tools 透传给模板的 `<tools>` 块；open_thinking 决定 `<think>\n` 还是空思考占位）+ 编码 |
| 111-113 | 生成闭包：异常时往队列塞**错误 dict + None 哨兵**（主线程把错误原样 yield 给客户端——不让连接悬挂） |
| 130 | 线程启动后本函数继续往下走消费循环 |

**状态机变量**（132-136 行）：

```python
        pending = ""
        in_tool_call = False
        tool_count = 0
        thinking_ended = not bool(open_thinking)
        strip_newlines = False
```

| 变量 | 含义 |
|------|------|
| pending | 尚未「放行」的字符缓冲（等标记攒齐） |
| in_tool_call | 当前正在 `<tool_call>…</tool_call>` 块内（内容暂存不外发） |
| thinking_ended | 已经过了思考段（True 后正文才发往 content 字段；False 时发往 reasoning_content）。**open_thinking=False 时初始就是 True**（没有思考段） |
| strip_newlines | 刚结束思考段 → 吃掉紧随的换行（模板约定 `\n</think>\n\n` 后正文才开始） |

**主消费循环**（138-190 行）逐块读：

```python
        while True:
            text = queue.get()
            if text is None:
                break
            if isinstance(text, dict):
                yield json.dumps(text, ensure_ascii=False)
                continue
            pending += text
```

- None → 生成结束；dict → 错误原样透传；否则缓冲累计。

```python
            while pending:
                if strip_newlines:
                    pending = pending.lstrip('\n')
                    if not pending:
                        break
                    strip_newlines = False
                if in_tool_call:
                    end = pending.find('</tool_call>')
                    if end < 0:
                        break
                    end += len('</tool_call>')
                    block, pending = pending[:end], pending[end:]
                    _, _, tool_calls = parse_response(block)
                    if tool_calls:
                        call = tool_calls[0]
                        call['index'] = tool_count
                        call['id'] = f"call_{int(time.time())}_{tool_count}"
                        tool_count += 1
                        yield json.dumps({"choices": [{"delta": {"tool_calls": [call]}}]}, ensure_ascii=False)
                    else:
                        yield json.dumps({"choices": [{"delta": {"content": block}}]}, ensure_ascii=False)
                    in_tool_call = False
                    continue
```

| 行 | 讲解 |
|----|------|
| 148-152 | 吃换行：`</think>` 后的空行不进正文；缓冲空了就等下个 chunk |
| 153-169 | **tool_call 块内状态**：找闭合标记；没找到就 break 等更多字符（**块内容不外发**，防止半个 JSON 被客户端误解）；攒齐整块后 parse_response 解析 → 以 `delta.tool_calls` 增量发出（index/id 按 tool_count 编号）；解析失败则当普通内容发（容错）；退出块状态 |

```python
                marker = '<tool_call>' if thinking_ended else '</think>'
                field = 'content' if thinking_ended else 'reasoning_content'
                pos = pending.find(marker)
                if pos < 0:
                    keep = next((n for n in range(len(marker) - 1, 0, -1)
                                 if pending.endswith(marker[:n])), 0)
                    end = len(pending) - keep
                    if end:
                        yield json.dumps({"choices": [{"delta": {field: pending[:end]}}]}, ensure_ascii=False)
                        pending = pending[end:]
                    break
                if pos:
                    yield json.dumps({"choices": [{"delta": {field: pending[:pos]}}]}, ensure_ascii=False)
                pending = pending[pos:]
                if thinking_ended:
                    in_tool_call = True
                else:
                    pending = pending[len(marker):]
                    thinking_ended = True
                    strip_newlines = True
```

这是状态机的核心（**安全外发算法**）：

| 行 | 讲解 |
|----|------|
| 171-172 | 根据阶段选「下一个要找的标记」：思考期找 `</think>`（思考结束后找 `<tool_call>`）；对应的输出字段（reasoning_content / content） |
| 173-181 | **没找到标记时**——关键难点：pending 末尾可能是标记的**前缀**（如 `…<tool_c`），直接全发出去，之后接上的 `all>` 就永远拼不成完整标记、解析失败。`keep` 用生成器+next 找「pending 结尾与 marker 前缀的最长公共匹配长度」（从 len(marker)−1 递减试）；`end = len(pending)−keep`：**安全部分**（keep 之前的）按当前字段发出，**尾缀留在 pending** 等下一 chunk 拼接验证。`break` 回外层等新数据 |
| 183-184 | 找到了标记：标记**之前**的内容发出（当当前字段）；pending 重置到标记起点 |
| 185-190 | 标记处理：思考结束后遇到 `<tool_call>` → 进入 in_tool_call 状态（块内容暂存）；思考期遇到 `</think>` → 剥掉标记本身、`thinking_ended=True`、打开吃换行开关 |

**收尾**（192-195 行）：

```python
        if pending:
            field = 'content' if thinking_ended else 'reasoning_content'
            yield json.dumps({"choices": [{"delta": {field: pending}}]}, ensure_ascii=False)
        yield json.dumps({"choices": [{"delta": {}, "finish_reason": "tool_calls" if tool_count else "stop"}]}, ensure_ascii=False)
```

- 循环退出后把残余 pending 一次性发掉（生成正常结束时 pending 里只有无标记风险的尾巴）；
- **finish_reason**：发过工具调用 → `"tool_calls"`（客户端应执行工具再回传），否则 `"stop"`（EOS）。

**状态机走一遍实例**：假设 open_thinking=True，流依次产出：

```
"<think>\n"        → 找 </think> 没找到；keep=0 全部发往 reasoning_content？ 
```

等等——逐 chunk 实际是 HF streamer 按可解码边界给的文本。设流为：`["<think", ">\n我", "想想", "</think>", "\n\n<tool", "_call>", '{"name":"get_current_time"}', "</tool", "_call>"]`：

1. `"<think"`：找 `</think>` 未命中；末尾 `<think` 是……不是 `</think>` 的前缀（首字符 `<` 匹配 `</` 失败——`</think>` 开头是 `</`，而 `<think` 的第 2 字符是 `t`），keep=0，全发到 **reasoning_content**（此刻 thinking_ended=False）。字段里出现了 `<think` 字样——客户端 UI 对此有容错（或接受轻微噪声），这正是「开始标记不参与解析」的取舍：**解析只锚定结束标记**。
2. `">\n我"`：仍无 `</think>` 前缀风险 → 全发 reasoning_content。
3. `"想想"`：同上。
4. `"</think>"`：pos=0（pending 开头就是标记）→ 无前置内容可发 → 剥标记，thinking_ended=True，strip_newlines=True。
5. `"\n\n<tool"`：先 lstrip 掉 `\n\n`；找 `<tool_call>` 未命中，但末尾 `<tool` 是前缀（n=5 命中）→ keep=5，发 `""`?——不，end=0 无内容可发，break。
6. `"_call>"`：pending=`<tool_call>` → pos=0 → in_tool_call=True。
7. JSON 串：in_tool_call 中、无 `</tool_call>` → 暂存。
8. `"</tool"`+`"_call>"`：凑齐闭合 → 整块 parse → `delta.tool_calls` 发出。

### 2.7 路由端点（201-257 行）

```python
@app.post("/v1/chat/completions")
def chat_completions(request: ChatRequest):
    try:
        if request.stream:
            return StreamingResponse(
                (f"data: {chunk}\n\n" for chunk in generate_stream_response(
                    messages=request.messages, temperature=request.temperature, top_p=request.top_p,
                    max_tokens=request.max_tokens, tools=request.tools, open_thinking=request.get_open_thinking()
                )),
                media_type="text/event-stream"
            )
```

| 行 | 讲解 |
|----|------|
| 205-215 | **流式**：`StreamingResponse` 包一个生成器——生成器表达式给每个 chunk 前面加 `data: `、后面加空行（**SSE 帧格式**），media_type 生命为 event-stream。FastAPI 会边迭代边推送 |

```python
        else:
            ...  # 非流式：apply_chat_template → generate（无 streamer，no_grad）→ decode 切 prompt
            content, reasoning_content, tool_calls = parse_response(answer)
            message = {"role": "assistant", "content": content}
            if reasoning_content: message["reasoning_content"] = reasoning_content
            if tool_calls: message["tool_calls"] = tool_calls
            return {
                "id": f"chatcmpl-{int(time.time())}", "object": "chat.completion", "created": int(time.time()),
                "model": "minimind",
                "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if tool_calls else "stop"}]
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

| 行 | 讲解 |
|----|------|
| 226 | `with torch.no_grad():` 包住生成 |
| 236 | `generated_ids[0][inputs["input_ids"].shape[1]:]` 切掉 prompt |
| 237-242 | 整段 parse_response → 组装 message（按需附加 reasoning_content/tool_calls） |
| 243-255 | 按协议拼完整响应（id/object/created/choices） |
| 256-257 | 任何异常 → 500（FastAPI 转 JSON 错误） |

### 2.8 main（260-277 行）

```python
    parser.add_argument('--load_from', default='../model', ...)
    ...  # 与 eval_llm 相同的参数集，多一个 max_seq_len=8192
    args = parser.parse_args()
    device = args.device
    model, tokenizer = init_model(args)
    uvicorn.run(app, host="0.0.0.0", port=8998)
```

- `device = args.device` **先赋给模块级变量**（init_model/handler 都引用它）；uvicorn 起服务在 8998 端口（与 SGLang RL 端口同一习惯数字）。

## 3. 客户端接线

```bash
python scripts/serve_openai_api.py          # 起服务（默认连 out/full_sft_768.pth）
# 另一个终端：
python scripts/chat_api.py                  # OpenAI SDK 客户端
# 或任何 OpenAI 兼容工具：
curl http://localhost:8998/v1/chat/completions -d '{"model":"minimind","messages":[{"role":"user","content":"你好"}],"stream":true}'
```

## 4. 本篇小结

1. **协议兼容 = Pydantic 请求模型 + SSE 增量流 + message/delta 三字段（content/reasoning_content/tool_calls）**。
2. 流式解析是「**生成线程 → Queue → 状态机 → SSE**」流水线；状态机核心是**标记前缀保持**（keep 算法）与 tool_call 块攒齐再解析。
3. 思考开关三级兼容（顶层/kwargs 两种字段名）；finish_reason 区分 stop/tool_calls。
4. 错误传递路径：生成线程异常 → 队列 dict → 客户端可见，连接不悬挂。

下一篇：[03-OpenAI客户端-chat_api](./03-OpenAI客户端-chat_api.md)
