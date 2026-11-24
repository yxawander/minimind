# 03 - OpenAI 客户端 chat_api.py 全文

> 覆盖代码：`scripts/chat_api.py` **全文 47 行**。
> 一个极简的 **OpenAI SDK 客户端**：连接任何 OpenAI 兼容服务（默认 ollama 的 11434 端口），实现多轮对话 + 流式打印（思考内容灰色、正文白色）。
> 与 [02-serve_openai_api](./02-OpenAI兼容API服务-serve_openai_api.md) 是「服务端 ↔ 客户端」配对，也可以单独连 ollama 部署的 minimind。

---

## 1. 逐行精读

```python
from openai import OpenAI

client = OpenAI(
    api_key="sk-123",
    base_url="http://localhost:11434/v1"
)
```

| 行 | 讲解 |
|----|------|
| 1 | OpenAI 官方 Python SDK |
| 3-6 | **`base_url` 重定向**：SDK 默认打 api.openai.com；改 base_url 后全部请求打到本地。`11434` 是 **ollama** 的默认端口——本地用 ollama 部署 minimind（`ollama run jingyaogong/minimind-3`）后，这个脚本直接可用。api_key 本地服务不校验，随便填 |

```python
stream = True
conversation_history_origin = []
conversation_history = conversation_history_origin.copy()
history_messages_num = 0  # 必须设置为偶数（Q+A），为0则不携带历史对话
```

| 行 | 讲解 |
|----|------|
| 7 | 流式开关（False 走一次性返回分支） |
| 8-9 | 消息总账本；`.copy()` 只是演示意图（浅拷贝 list，元素 dict 仍共享——单线程下无碍） |
| 10 | 历史窗口：0=无记忆。**偶数约束**：历史必须是完整问答对，奇数会把 user 消息截成「孤问」 |

```python
while True:
    query = input('[Q]: ')
    conversation_history.append({"role": "user", "content": query})
    response = client.chat.completions.create(
        model="minimind-local:latest",
        messages=conversation_history[-(history_messages_num or 1):],
        stream=stream,
        temperature=0.8,
        max_tokens=2048,
        top_p=0.8,
        extra_body={"chat_template_kwargs": {"open_thinking": True}, "reasoning_effort": "medium"} # 思考开关
    )
```

| 行 | 讲解 |
|----|------|
| 12-13 | 交互循环；user 消息入账本 |
| 14-22 | 发起请求：<br>• `model`：本地部署的模型 tag（ollama 命名风格；对 serve_openai_api 而言服务端不校验名字，任意值都行）；<br>• `messages=conversation_history[-(history_messages_num or 1):]`：**切片窗口**——history=0 时取 `[-1:]`（只发当前问题）；history=4 时取最近 4 条（2 轮）。`or 1` 处理 0 的陷阱（`[-0:]` 是全量切片！）；<br>• `temperature/max_tokens/top_p`：采样参数透传；<br>• **`extra_body`**：SDK 透传非标准字段的通道——`chat_template_kwargs.open_thinking=True` 被服务端的 `get_open_thinking()` 兼容逻辑识别（serve 篇三级兼容之一）；`reasoning_effort` 是给「支持努力等级的服务端」的提示（ollama 的 gpt-oss 风格字段），MiniMind serve 不处理，无害 |

```python
    if not stream:
        if not response.choices or response.choices[0].message is None:
            raise ValueError("LLM returned empty or filtered response")
        assistant_res = response.choices[0].message.content
        print('[A]: ', assistant_res)
    else:
        print('[A]: ', end='', flush=True)
        assistant_res = ''
        for chunk in response:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta is None:
                continue
            r = getattr(delta, 'reasoning_content', None) or ""
            c = delta.content or ""
            if r:
                print(f'\033[90m{r}\033[0m', end="", flush=True)
            if c:
                print(c, end="", flush=True)
            assistant_res += c
```

| 行 | 讲解 |
|----|------|
| 23-27 | **非流式分支**：判空防御（部分服务在触发内容过滤时返回空 choices）；取 `message.content` 打印 |
| 28-40 | **流式分支**：`for chunk in response` 迭代 SSE（SDK 已解析成对象）：<br>• `chunk.choices` 可能为空列表（某些中间 chunk，如 role 帧）→ 跳过；<br>• `delta.reasoning_content`：**getattr 防御式取属性**——普通 OpenAI 返回没有该字段，getattr 默认 None 不抛错。思考内容用 ANSI 码 `\033[90m…\033[0m` 包裹打印成**灰色**；<br>• `delta.content`：正文白色原样打印（`flush=True` 立刻刷出，无延迟）；<br>• `assistant_res += c`：**只累积正文**——回传给服务端做下一轮上下文时用 content（若把思考也塞进去，下一轮 prompt 会重复思考段） |

```python
    conversation_history.append({"role": "assistant", "content": assistant_res})
    print('\n\n')
```

| 行 | 讲解 |
|----|------|
| 45-46 | assistant 回复入账本（多轮记忆的数据结构维护）；空行分隔轮次 |

## 2. 思考内容去哪了（数据流图）

```
服务端 SSE:  delta.reasoning_content(思考) + delta.content(正文)
                    │                        │
              灰色打印                  白色打印 + 累积
                                             │
conversation_history: [user Q, assistant content(仅正文), user Q2, ...]
                                             │
下一轮 messages=history 窗口 → 模型看到干净问答对
```

## 3. 本篇小结

1. `OpenAI(base_url=...)` 三行接任何兼容服务（ollama / serve_openai_api / FastGPT 网关）。
2. `history_messages_num or 1` 的切片窗口规避 `[-0:]` 全量陷阱。
3. `extra_body.chat_template_kwargs.open_thinking` 是思考开关的标准通路（对应 serve 端三级兼容）。
4. 流式渲染：**reasoning_content 灰色、content 白色**；多轮历史只回传 content。

下一篇：[04-WebUI-web_demo](./04-WebUI-web_demo.md)
