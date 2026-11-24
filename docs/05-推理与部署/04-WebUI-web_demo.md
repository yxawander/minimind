# 04 - WebUI web_demo.py 全文

> 覆盖代码：`scripts/web_demo.py` **全文 425 行**。
> Streamlit 实现的聊天界面：**模型热扫描、多语言、思考折叠、多轮工具调用**。运行方式 `streamlit run scripts/web_demo.py`。
> 它加载的是 **transformers 格式模型目录**（不是 .pth），并自带一套工具执行器。

---

## 1. 原理：Streamlit 的执行模型

Streamlit **每次用户交互（发送消息/改参数）都从头重新执行整个脚本**，靠 `st.session_state` 字典跨执行保存状态。本脚本处处体现这个模型：

- 模型加载用 `@st.cache_resource` 缓存（否则每次交互重载模型）；
- 历史消息列表存 `st.session_state.messages`；
- 「先重绘全部历史 → 再处理新输入」的渲染顺序。

生成采用「**线程 + TextIteratorStreamer**」：generate 在后台线程跑，主线程 for 循环消费流式文本，每收到一段就更新页面占位符——与 serve 的 Queue 流水线同构，只是消费端从 SSE 换成 Streamlit 组件。

## 2. 逐行精读

### 2.1 页面与样式（16-72 行）

```python
st.set_page_config(page_title="MiniMind", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
        .stButton button { border-radius: 50% !important; width: 32px !important; ... }
        .stMainBlockContainer > div:first-child { margin-top: -50px !important; }
        ...
    </style>
""", unsafe_allow_html=True)
```

| 行 | 讲解 |
|----|------|
| 16 | 页面标题 + 折叠侧栏起步 |
| 18-72 | **CSS 注入**（`unsafe_allow_html=True` 允许原始 HTML）：把操作按钮改成小圆点、微调容器边距——纯美化，无逻辑 |

### 2.2 多语言文本表（74-108 行）

```python
device = "cuda" if torch.cuda.is_available() else "cpu"

LANG_TEXTS = {
    'zh': {'settings': '模型设定调整', 'history_rounds': '历史对话轮次', 'max_length': '最大生成长度',
           'temperature': '温度', 'thinking': '思考', 'tools': '工具', 'language': '语言',
           'send': '给 MiniMind 发送消息', 'disclaimer': 'AI 生成内容可能存在错误，请仔细核实',
           'think_tip': '自适应思考，目前多轮对话或Tool Call共存时思考不稳定', 'tool_select': '工具选择（最多4个）'},
    'en': {...同样 11 个 key 的英文版}
}

def get_text(key):
    lang = st.session_state.get('lang', 'en')
    return LANG_TEXTS.get(lang, {}).get(key, LANG_TEXTS['zh'].get(key, key))
```

| 行 | 讲解 |
|----|------|
| 74 | 全局 device（模型搬到哪） |
| 77-104 | 中英两套 UI 文案 |
| 106-108 | 取词函数：按 session 语言取，缺失回退中文（再缺失回退 key 本身） |

### 2.3 工具定义与执行器（110-150 行）

```python
TOOLS = [
    {"type": "function", "function": {"name": "calculate_math", ...}},
    ...共 8 个：calculate_math/get_current_time/random_number/text_length/unit_converter/get_current_weather/get_exchange_rate/translate_text
]

TOOL_SHORT_NAMES = {
    'calculate_math': '数学', 'get_current_time': '时间', 'random_number': '随机',
    'text_length': '字数', 'unit_converter': '单位', 'get_current_weather': '天气',
    'get_exchange_rate': '汇率', 'translate_text': '翻译'
}
```

- 与 train_agent/eval_toolcall 同款的 OpenAI schema（复用 SFT 见过的格式）；`TOOL_SHORT_NAMES` 是侧栏勾选框的显示名。

```python
def execute_tool(tool_name, args):
    import datetime
    try:
        if tool_name == 'calculate_math':
            return {"result": safe_math_eval(args.get('expression', '0'))}
        elif tool_name == 'get_current_time':
            tz = args.get('timezone', 'Asia/Shanghai')
            return {"result": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        elif tool_name == 'random_number':
            return {"result": random.randint(args.get('min', 0), args.get('max', 100))}
        elif tool_name == 'text_length':
            return {"result": len(args.get('text', ''))}
        elif tool_name == 'unit_converter':
            return {"result": f"{args.get('value', 0)} {args.get('from_unit', '')} = ? {args.get('to_unit', '')}"}
        elif tool_name == 'get_current_weather':
            return {"result": f"{args.get('city', 'Unknown')}: 晴, 7~10°C"}
        elif tool_name == 'get_exchange_rate':
            return {"result": f"1 {args.get('from_currency', 'USD')} = 7.2 {args.get('to_currency', 'CNY')}"}
        elif tool_name == 'translate_text':
            return {"result": f"[翻译结果]: hello world"}
        return {"result": "Unknown tool"}
    except Exception as e:
        return {"error": str(e)}
```

- 硬编码模拟执行器：**只有 calculate_math 是真算的**（safe_math_eval，AST 白名单见 trainer_utils 篇）；时间是**真时钟**；其余返回固定占位（单位换算/汇率/翻译返回假结果——WebUI 的定位是演示交互流程而非真工具）。异常兜底 `{"error": ...}`。

### 2.4 process_assistant_content：渲染格式化（153-199 行）

这个函数把模型的原始输出文本转成 HTML 展示，处理三种元素：

```python
def process_assistant_content(content, is_streaming=False):
    # 处理tool_call标签，格式化显示
    if '<tool_call>' in content:
        def format_tool_call(match):
            try:
                tc = json.loads(match.group(1))
                name = tc.get('name', 'unknown'); args = tc.get('arguments', {})
                return f'<div style="background: rgba(80, 110, 150, 0.20); ...">ToolCalling<b>{name}</b>: {json.dumps(args, ensure_ascii=False)}</div>'
            except:
                return match.group(0)
        content = re.sub(r'<tool_call>(.*?)</tool_call>', format_tool_call, content, flags=re.DOTALL)
```

| 行 | 讲解 |
|----|------|
| 155-164 | `<tool_call>…</tool_call>` → 蓝色卡片（**HTML 注入防御**：正则用非贪婪 `.*?` 且内容 json.loads 成功才渲染——不把模型输出的任意字符串直接拼进 HTML 属性之外的地方；失败时保留原文） |

```python
    # 流式生成且开启思考时，一开始就放到折叠里
    if is_streaming and st.session_state.get('enable_thinking', False) and '</think>' not in content and '<think>' not in content:
        m = re.search(r'(\n\n(?:我是|您好|你好)[^\n]*)', content)
        if m and m.start(1) > 5:
            ...  # 以「我是/您好/你好」开头的新段落为界，把前面内容当思考折叠
        elif len(content) > 5:
            ...  # 全部当「思考中...」折叠
```

| 行 | 讲解 |
|----|------|
| 166-175 | **流式期间无标记的启发式**：模型开了思考但还没输出 `</think>` 时（标记还没到），**猜**哪些是思考——若正文出现「我是/您好/你好」开头的段落（回答的典型开场白），把之前内容折进「已思考」；否则整段都当「思考中...」。这是为了流式时视觉不跳变，等真正的 think 标签到齐后走下面的精确分支 |

```python
    if '<think>' in content and '</think>' in content:
        def format_think(match):
            think_content = match.group(2)
            if think_content.replace('\n', '').strip():  # 不是全换行
                return f'<details open ...>已思考<div ...>{think_content.strip()}</div></details>'
            return ''
        content = re.sub(r'(<think>)(.*?)(</think>)', format_think, content, flags=re.DOTALL)
```

| 行 | 讲解 |
|----|------|
| 177-183 | **完整思考段**：`<details open>` 折叠框（HTML 原生可折叠），内容灰字；**空思考**（只有换行）整段删除——空占位不显示 |

```python
    if '<think>' in content and '</think>' not in content:
        ...  # 思考进行中：'思考中...' 折叠框，内容倒序排列（flex-direction: column-reverse，新内容视觉在顶）
    if '<think>' not in content and '</think>' in content:
        ...  # 只有 </think>：模型思考未闭合就 EOS，'...' 之前全当思考折叠
    return content
```

| 行 | 讲解 |
|----|------|
| 185-189 | 不完整段（`<think>` 后还没 `</think>`）→「思考中...」框；`column-reverse` 让滚动时最新内容贴着可视区 |
| 191-197 | 只有闭合标记（思考被截断）→ 开头全部当思考 |

四种分支覆盖 think 标记的全部生命周期。

### 2.5 模型加载与会话工具（202-241 行）

```python
@st.cache_resource
def load_model_tokenizer(model_path):
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = model.half().eval().to(device)
    return model, tokenizer

def clear_chat_messages():
    del st.session_state.messages
    del st.session_state.chat_messages

def init_chat_messages():
    if "messages" in st.session_state:
        for i, message in enumerate(st.session_state.messages):
            if message["role"] == "assistant":
                st.markdown(process_assistant_content(message["content"]), unsafe_allow_html=True)
            else:
                st.markdown(f'<div ...背景色气泡...>{message["content"]}</div>', unsafe_allow_html=True)
    else:
        st.session_state.messages = []
        st.session_state.chat_messages = []
    return st.session_state.messages

def regenerate_answer(index):
    st.session_state.messages.pop()
    st.session_state.chat_messages.pop()
    st.rerun()
```

| 行 | 讲解 |
|----|------|
| 202-213 | **缓存加载**：`@st.cache_resource` 让模型只加载一次（跨脚本重执行共享）。**trust_remote_code**：transformers 目录的 minimind 类；`.half().eval()` 推理精度 |
| 216-219 | 清空对话（删两个 state key；直接 del——Streamlit 会 KeyError 式重建） |
| 221-235 | **重绘历史**：assistant 消息过 process_assistant_content（含 think/tool 卡片），user 消息渲染成右侧深色气泡。**为什么有 init_chat_messages 又在 main 里重复一遍重绘逻辑**：main 的重绘在「新输入处理之前」，本函数是给外部（如 regenerate）复用的入口——轻微重复但保证任意执行路径都先画历史 |
| 237-240 | 重新生成：弹出最后一条问答，`st.rerun()` 重跑脚本触发重新回答 |

### 2.6 模型热扫描与侧栏（243-303 行）

```python
script_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATHS = {}
for d in sorted(os.listdir(script_dir), reverse=True):
    full_path = os.path.join(script_dir, d)
    if os.path.isdir(full_path) and not d.startswith('.') and not d.startswith('_'):
        if any(f.endswith(('.bin', '.safetensors', '.pt')) or os.path.exists(os.path.join(full_path, 'model.safetensors.index.json')) for f in os.listdir(full_path) if os.path.isfile(os.path.join(full_path, f))):
            MODEL_PATHS[d] = [d, d]
if not MODEL_PATHS:
    MODEL_PATHS = {"No models found": ["", "No models"]}
```

| 行 | 讲解 |
|----|------|
| 244-252 | **动态发现模型**：扫描 `scripts/` 下的子目录，凡含权重文件（.bin/.safetensors/.pt 或分片索引）的目录都算可用模型——把 `minimind-3/`（convert_model 的产物）放进 scripts/ 即被识别。倒序排序（目录名新的在前） |

```python
selected_model = st.sidebar.selectbox('Model', list(MODEL_PATHS.keys()), index=0)
model_path = MODEL_PATHS[selected_model][0]
slogan = f"我是 {MODEL_PATHS[selected_model][1]}，有什么可以帮你的？" if st.session_state.get('lang', 'en') == 'zh' else ...

lang_options = {'中文': 'zh', 'English': 'en'}
current_lang = st.session_state.get('lang', 'en')
lang_label = st.sidebar.radio('Language / 语言', list(lang_options.keys()), index=lang_index, horizontal=True)
if lang_options[lang_label] != current_lang:
    st.session_state.lang = lang_options[lang_label]
    st.rerun()
```

- 模型选择下拉框、slogan、语言单选（切换时改 state + **st.rerun() 立即重渲染**）。

```python
st.session_state.history_chat_num = st.sidebar.slider(get_text('history_rounds'), 0, 8, 0, step=2)
st.session_state.max_new_tokens = st.sidebar.slider(get_text('max_length'), 256, 8192, 8192, step=1)
st.session_state.temperature = st.sidebar.slider(get_text('temperature'), 0.6, 1.2, 0.90, step=0.01)
...
st.session_state.enable_thinking = st.sidebar.checkbox(get_text('thinking'), value=False, help=get_text('think_tip'))
st.session_state.selected_tools = []
with st.sidebar.expander(get_text('tools')):
    st.caption(get_text('tool_select'))
    selected_count = sum(1 for tool in TOOLS if st.session_state.get(f"tool_{tool['function']['name']}", False))
    for tool in TOOLS:
        name = tool['function']['name']
        short_name = TOOL_SHORT_NAMES.get(name, name)
        checked = st.checkbox(short_name, key=f"tool_{name}", disabled=(selected_count >= 4 and not st.session_state.get(f"tool_{name}", False)))
        if checked and len(st.session_state.selected_tools) < 4:
            st.session_state.selected_tools.append(name)
```

| 行 | 讲解 |
|----|------|
| 273-275 | 三个滑杆：历史轮数（步长 2 保证偶数问答对）、最大生成长度、温度（0.6-1.2） |
| 280 | 思考开关 checkbox |
| 281-290 | **工具选择器**：每工具一个 checkbox（key=`tool_{name}` 持久勾选态）；**最多选 4 个**——已满 4 个时其余 checkbox 置 disabled（`selected_count >= 4 and not 已勾选`），勾选结果收进 `selected_tools`。限制原因：工具定义写进 system 段占 token，太多会把 64M 小模型的上下文挤爆 |

### 2.7 main：对话与多轮工具循环（316-421 行）

```python
def main():
    model, tokenizer = load_model_tokenizer(model_path)

    if "messages" not in st.session_state:
        st.session_state.messages = []        # 展示用历史
        st.session_state.chat_messages = []   # 发给模型的消息（含 system/tool 结构）
```

| 行 | 讲解 |
|----|------|
| 319-322 | **两套历史**：`messages` 是展示层（user/assistant 文本）；`chat_messages` 是模型输入层（含 system、tool 消息、tool_calls 结构）——分层是因为展示不需要工具细节，模型输入需要 |

```python
    prompt = st.chat_input(key="input", placeholder=get_text('send'))

    if hasattr(st.session_state, 'regenerate') and st.session_state.regenerate:
        prompt = st.session_state.last_user_message
        regenerate_index = st.session_state.regenerate_index
        delattr(st.session_state, 'regenerate')
        ...  # 重新生成：用上一条 user 消息当作本次输入
```

- `st.chat_input`：底部输入框，有新消息时返回文本；regenerate 分支把「重新生成」事件转换成重放上一条 user 消息。

```python
    if prompt:
        st.markdown(...气泡...)                                     # 立刻显示用户消息
        messages.append({"role": "user", "content": prompt})
        st.session_state.chat_messages.append({"role": "user", "content": prompt})

        placeholder = st.empty()                                     # 流式占位组件

        random_seed = random.randint(0, 2 ** 32 - 1)
        setup_seed(random_seed)                                      # 本地版 setup_seed（306-313 行重定义了一份）

        tools = [t for t in TOOLS if t['function']['name'] in st.session_state.get('selected_tools', [])] or None
        sys_prompt = [] if tools else [{"role": "system", "content": "你是MiniMind，一个乐于助人、知识渊博的AI助手。请用完整且友好的方式回答用户问题。"}]
        st.session_state.chat_messages = sys_prompt + st.session_state.chat_messages[-(st.session_state.history_chat_num + 1):]
        template_kwargs = {"tokenize": False, "add_generation_prompt": True}
        if st.session_state.get('enable_thinking', False):
            template_kwargs["open_thinking"] = True
        if tools:
            template_kwargs["tools"] = tools
        new_prompt = tokenizer.apply_chat_template(st.session_state.chat_messages, **template_kwargs)
```

| 行 | 讲解 |
|----|------|
| 354-356 | **system prompt 的取舍**：选了工具 → 不加默认 system（模板会自己渲染 tools 段）；没选 → 注入固定中文 system。历史裁剪 `[-(history+1):]`：history 轮 + 当前 1 条 |
| 357-362 | 模板参数按需添加（open_thinking/tools 只在开启时传——与训练数据的条件分布对齐） |

```python
        inputs = tokenizer(new_prompt, return_tensors="pt", truncation=True).to(device)

        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = {
            "input_ids": inputs.input_ids,
            "max_length": inputs.input_ids.shape[1] + st.session_state.max_new_tokens,
            ...
            "streamer": streamer,
        }

        Thread(target=model.generate, kwargs=generation_kwargs).start()

        answer = ""
        for new_text in streamer:
            answer += new_text
            placeholder.markdown(process_assistant_content(answer, is_streaming=True), unsafe_allow_html=True)
```

| 行 | 讲解 |
|----|------|
| 364-380 | TextIteratorStreamer + 后台线程 generate（与 serve 的 Queue 同构，Streamlit 官方推荐姿势）。注意用的是 **max_length**（input+预算）而不是 max_new_tokens |
| 382-385 | 流式渲染：每段新文本 → 累积 answer → **process_assistant_content(is_streaming=True)** 转换 → 更新 placeholder（页面实时刷新） |

**多轮工具调用循环**（387-417 行）：

```python
        full_answer = answer
        for _ in range(16):
            tool_calls = re.findall(r'<tool_call>(.*?)</tool_call>', answer, re.DOTALL)
            if not tool_calls:
                break
            st.session_state.chat_messages.append({"role": "assistant", "content": answer})
            tool_results = []
            for tc_str in tool_calls:
                try:
                    tc = json.loads(tc_str.strip())
                    result = execute_tool(tc.get('name', ''), tc.get('arguments', {}))
                    st.session_state.chat_messages.append({"role": "tool", "content": json.dumps(result, ensure_ascii=False)})
                    tool_results.append(f'<div style="...绿色卡片...">ToolCalled<b>{tc.get("name","")}</b>: {json.dumps(result, ensure_ascii=False)}</div>')
                except:
                    pass
            full_answer += "\n" + "\n".join(tool_results) + "\n"
            placeholder.markdown(process_assistant_content(full_answer, is_streaming=True), unsafe_allow_html=True)
            new_prompt = tokenizer.apply_chat_template(st.session_state.chat_messages, **template_kwargs)
            inputs = tokenizer(new_prompt, return_tensors="pt", truncation=True).to(device)
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            generation_kwargs["input_ids"] = inputs.input_ids
            generation_kwargs["attention_mask"] = inputs.attention_mask
            generation_kwargs["max_length"] = inputs.input_ids.shape[1] + st.session_state.max_new_tokens
            generation_kwargs["streamer"] = streamer
            Thread(target=model.generate, kwargs=generation_kwargs).start()
            answer = ""
            for new_text in streamer:
                answer += new_text
                placeholder.markdown(process_assistant_content(full_answer + answer, is_streaming=True), unsafe_allow_html=True)
            full_answer += answer
        answer = full_answer
```

| 行 | 讲解 |
|----|------|
| 388-390 | 本轮输出里有 `<tool_call>` → 进入工具循环；没有 → 结束 |
| 392-401 | 工具消息入 chat_messages（assistant 带 tool_calls 结构的轮 + 逐个 tool 结果消息）；绿色结果卡片拼进展示文本 |
| 404-416 | **用更新后的 chat_messages 重新渲染模板 → 再起一轮流式生成**；展示时把 `full_answer + answer` 一起过格式化（卡片保留）。循环上限 16 次（防死循环——模型理论上可以永远调工具） |
| 417-420 | 循环结束，最终文本入两个历史 |

## 3. 与 serve_openai_api 的对比

| | web_demo | serve_openai_api |
|--|----------|------------------|
| 消费方式 | Streamlit placeholder 直渲染 | SSE 给任意客户端 |
| 解析方式 | 渲染时**整体**正则（不逐 chunk 状态机） | 流式状态机 |
| 工具循环 | 16 轮上限，前端执行 | 不执行（finish_reason 交客户端） |
| 模型来源 | transformers 目录热扫描 | 参数指定 |

## 4. 本篇小结

1. Streamlit 的「**脚本重执行 + session_state**」模型决定了：cache_resource 缓存模型、双历史分层（展示/输入）。
2. 流式 = 线程 + TextIteratorStreamer；**process_assistant_content 四分支**处理 think 标记全生命周期 + tool_call 卡片。
3. 工具循环在前端完成（模型→执行→拼上下文→再生成），上限 16 轮；最多 4 个工具防上下文爆炸。
4. 模型目录热扫描让「转换好的 transformers 模型文件夹」即放即用。

下一篇：[05-模型格式转换-convert_model](./05-模型格式转换-convert_model.md)
