# 08 - Agentic RL 智能体强化学习 train_agent.py 全文

> 覆盖代码：`trainer/train_agent.py` **全文 491 行**。
> 这是仓库最新的亮点：**多轮工具调用场景下的在线强化学习**。模型在 rollout 中真实地「生成 → 调工具 → 看结果 → 再生成」，整个交互轨迹作为一条序列做 GRPO/CISPO 更新。
> 核心难点：**多轮拼接后 token 级损失怎么算**（观察区 token 屏蔽）。本篇把整套机制讲透。

---

## 1. 全景：和单轮 RL（GRPO）差在哪

```
单轮 GRPO:  prompt → 一次生成 → 打分 → 更新
Agent RL:   prompt → 生成(可能含 <tool_call>)
                     → 执行工具 → 结果作为 tool 消息拼回上下文
                     → 继续生成 …… 最多 max_turns=3 轮
            整条「prompt + 多轮生成 + 工具观察」序列一起打分、一起更新
```

三个新增课题：

1. **工具环境**：工具定义、模拟执行器、参数校验（§2）。
2. **多轮 rollout 的 token 记账**：哪些 token 是模型生成的（要训练）、哪些是工具观察/模板注入的（不训练）（§4）。
3. **可验证奖励**：有标准答案（gt）时按「工具用对几个 + 答案对几个」打分，不依赖 RM（§5）。

---

## 2. 工具与模拟环境（40-92 行）

### 2.1 TOOLS：工具 schema（41-48 行）

```python
TOOLS = [
    {"type": "function", "function": {"name": "calculate_math", "description": "计算数学表达式", "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}},
    ...共 6 个：calculate_math / unit_converter / get_current_weather / get_current_time / get_exchange_rate / translate_text
]
```

- **OpenAI function-calling 格式**（type/function/name/parameters），与 chat_template 里 `<tools>` 块的渲染约定一致——模型在 SFT 阶段见过的就是这种格式。
- 工具列表会随消息一起渲染进 system 段，模型据此「知道」能调什么。

### 2.2 模拟数据与执行器（50-65 行）

```python
WEATHER_DATA = {"北京": ("28°C", "晴"), ...15 城}
TIME_DATA = {"Asia/Shanghai": "2025-03-07 14:30:00", ...}
EXCHANGE_DATA = {("USD", "CNY"): 7.21, ...8 组}
TRANSLATE_DATA = {("你好世界", "english"): "Hello World", ...6 组}
UNIT_DATA = {"km_miles": 0.621371, ...8 组}
```

- 硬编码的**确定性模拟环境**：同样的输入永远同样的输出——RL 需要可复现的奖励信号，真实工具 API 会引入噪声。

```python
MOCK_RESULTS = {
    "calculate_math": lambda args: {"result": str(safe_math_eval(args.get("expression", "0")))},
    "unit_converter": lambda args: {"result": round(float(args.get("value", 0)) * UNIT_DATA.get(f"{...}_{...}", 1), 4)},
    "get_current_weather": lambda args: (lambda w: {"city": ..., "temperature": w[0], "humidity": "65%", "condition": w[1]})(WEATHER_DATA.get(args.get("location"), ("22°C", "晴"))),
    "get_current_time": lambda args: {"datetime": TIME_DATA.get(args.get("timezone", "Asia/Shanghai"), "2025-03-07 14:30:00"), ...},
    "get_exchange_rate": lambda args: {"from": ..., "to": ..., "rate": EXCHANGE_DATA.get((...), 1.0)},
    "translate_text": lambda args: {"translated_text": TRANSLATE_DATA.get((...), args.get("text", ""))},
}
```

| 行 | 讲解 |
|----|------|
| 59 | 数学工具用 [safe_math_eval](./01-训练基础设施-trainer_utils.md#11-safe_math_eval安全表达式求值181-209-行)（AST 白名单+防爆炸）真实求值——**数学答案是「真」的**，可验证 |
| 60 | 单位换算查表：`f"{from}_{to}"` 拼 key（`km_miles`），查不到默认系数 1 |
| 61 | 天气：内层 lambda 取 `WEATHER_DATA`（缺城市兜底 22°C），外层组装返回 dict。**模型算错天气数值没关系**——天气类任务考的是「会不会调对工具」，不是数值 |
| 62-64 | 时间/汇率/翻译同理，查表兜底 |

### 2.3 CHECK_ARGS：参数校验（68-75 行）

```python
CHECK_ARGS = {
    "calculate_math": lambda a: bool(a.get("expression")),
    "unit_converter": lambda a: a.get("value") is not None and a.get("from_unit") and a.get("to_unit"),
    ...
}
```

- 每个工具一个「参数齐全性」lambda——奖励函数用 `valid_call_count` 统计**格式正确的调用**（§5），缺参数的调用不算合格。

### 2.4 解析与执行（78-92 行）

```python
def parse_tool_calls(text):
    calls = []
    for m in re.findall(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL):
        try: call = json.loads(m.strip())
        except: continue
        if isinstance(call, dict): calls.append(call)
    return calls

def execute_tool(name, args):
    fn = MOCK_RESULTS.get(name)
    if not fn: return None
    try:
        return fn(args)
    except Exception:
        return None
```

- `parse_tool_calls`：正则抽 `<tool_call>…</tool_call>` 内容 → JSON 解析（**失败静默跳过**——模型输出格式错误不能让训练崩，只让奖励变低）；只收 dict。
- `execute_tool`：查表执行，任何异常返回 None（奖励侧记为失败）。

---

## 3. 多轮 rollout（95-184 行）——本篇核心

### 3.1 rollout_single：单条轨迹（95-161 行）

```python
def rollout_single(rollout_engine, tokenizer, messages, tools, max_turns=3, max_new_tokens=256, thinking_ratio=0.5, device="cuda"):
    all_outputs = []
    prompt_ids = None
    response_ids = []        # 模型生成 + 工具观察 的 token 总账本
    response_mask = []       # 1=模型生成（要训练），0=观察/模板（不训练）
    response_old_logps = []  # 与 response_ids 对齐的 π_old logp（观察区填 0.0）
    final_context = ""
    unfinished = False
    open_thinking = random.random() < thinking_ratio
    for turn in range(max_turns):
        context = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, tools=tools, open_thinking=open_thinking)
        if prompt_ids is None:
            prompt_ids = tokenizer(context, add_special_tokens=False)["input_ids"]
        input_ids = torch.tensor([prompt_ids + response_ids], device=device)
        rollout_result = rollout_engine.rollout(
            prompt_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            num_generations=1,
            max_new_tokens=max_new_tokens,
            temperature=0.8,
        )
```

| 行 | 讲解 |
|----|------|
| 103 | 思考开关**整条轨迹共享**（第一轮决定，后续轮保持一致——多轮中途换格式会让模型困惑） |
| 104-107 | 每轮重新渲染完整上下文；**第一轮**把渲染结果编码为 `prompt_ids`（之后不再变——它是损失归属里的「prompt 区」） |
| 108 | 喂给引擎的是 `prompt_ids + response_ids`——**历史生成与观察全部在输入里**，模型每次都看着完整轨迹续写 |
| 109-115 | 单条采样（num_generations=1；组对比由 rollout_batch 复制轨迹实现） |

```python
        valid_len = int(rollout_result.completion_mask[0].sum().item())
        new_ids = rollout_result.completion_ids[0, :valid_len].tolist()
        new_logps = rollout_result.per_token_logps[0, :valid_len].tolist()
        if len(new_ids) != len(new_logps):
            raise RuntimeError(...)
        new_text = rollout_result.completions[0]
        all_outputs.append(new_text)
        response_ids.extend(new_ids)
        response_mask.extend([int(t != tokenizer.eos_token_id) for t in new_ids])
        response_old_logps.extend(new_logps)
        final_context = context + new_text
```

| 行 | 讲解 |
|----|------|
| 116-120 | 取有效生成（截掉 pad）与对应 logp，长度不一致直接抛错（防御引擎 bug） |
| 124 | **本段生成的 mask**：每个 token **除非它是 EOS** 否则记 1。EOS 记 0 的原因：EOS 之后（若引擎填了 pad 或后续轮的上下文接缝）不应有损失；EOS 本身由主循环的 eos 截断逻辑统一处理 |
| 125 | logp 账本同步追加 |
| 126 | 拼出「到本轮为止的完整文本」 |

```python
        calls = parse_tool_calls(new_text)
        if not calls:
            break
        unfinished = turn == max_turns - 1
        assistant_message = {"role": "assistant", "content": new_text}
        messages.append(assistant_message)
        for call in calls:
            name, raw = call.get("name", ""), call.get("arguments", {})
            if isinstance(raw, str):
                try: raw = json.loads(raw)
                except: raw = {}
            if not isinstance(raw, dict): raw = {}
            result = execute_tool(name, raw)
            result_str = (json.dumps(result, ensure_ascii=False) if result else '{"error": "tool not found"}')[:2048]  # 防止天文数字撑爆tokenizer
            messages.append({"role": "tool", "content": result_str})
```

| 行 | 讲解 |
|----|------|
| 127-129 | 本轮没有工具调用 → 轨迹自然结束（模型决定「直接回答」） |
| 130 | **打满轮数还在调工具** = 未完成（奖励侧扣分） |
| 131-141 | 逐个执行调用：arguments 兼容字符串/dict；执行结果 JSON 化，**截断到 2048 字符**（模型可能算出 9**9**9……即使 safe_math_eval 挡了大数，其他工具也可能返回长串，防上下文爆炸）；失败返回 `{"error": ...}`——**错误也是环境反馈**，模型要学会从报错恢复。tool 消息追加进 messages（chat_template 块④会把连续 tool 消息合并进一个 user 信封） |

**标记法：把「观察区」token 精确切出来**

```python
        marker = f"<|agent_observation_{id(messages)}_{len(response_ids)}|>"
        assistant_message["content"] += marker
        marked_context = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=not unfinished, tools=tools, open_thinking=open_thinking)
        assistant_message["content"] = new_text
        _, found, observation = marked_context.partition(marker)
        if not found:
            raise RuntimeError("chat template did not preserve the assistant content boundary")
        obs_delta = tokenizer(observation, add_special_tokens=False)["input_ids"]
        if new_ids and new_ids[-1] == tokenizer.eos_token_id and obs_delta[:1] == [tokenizer.eos_token_id]:
            obs_delta = obs_delta[1:]
        observe_context = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=not unfinished, tools=tools, open_thinking=open_thinking)
        response_ids.extend(obs_delta)
        response_mask.extend([0] * len(obs_delta))
        response_old_logps.extend([0.0] * len(obs_delta))
        final_context = observe_context
```

**这是全文件最精妙的 20 行**。问题：工具结果通过 chat_template 拼回上下文后，模板会在 assistant 内容后面注入**一堆结构性 token**（`<|im_end|>\n<|im_start|>user\n<tool_response>\n` + 结果 + `</tool_response><|im_end|>\n` + 下一轮的 `<|im_start|>assistant\n`）——这些 token **不是模型生成的**，不能算损失，怎么把它们精确切出来？

**解法（标记法 marker trick）**：

1. 在 assistant 内容尾部**临时**追加一个不可能冲突的唯一标记（`id(messages)` 是 Python 对象 id，`len(response_ids)` 是账本长度——双重保证唯一）；
2. 带 marker 渲染一次模板 → `partition(marker)` 把渲染文本切成「前 | marker | **observation**」——marker 之后的**全部内容**就是模板为「工具观察 + 下一轮生成提示」注入的部分；
3. 把 marker 从消息里拿掉（恢复原样），再**正常渲染**一次得到干净上下文；
4. observation 编码成 `obs_delta`，追加进账本：**mask=0（不训练）、logp=0.0（占位）**；
5. 边界修正：如果模型本轮以 EOS 结束，而观察区开头又渲染出 EOS（模板把 `</think>` 后的空内容 + im_end 组合出的），把重复的 EOS 去掉一个（行 151-152）——否则同一位置有两个 EOS token，与真实推理时的序列不一致。

> 为什么不用「对比两次渲染文本再找 diff」？因为文本 diff 在 token 级不可靠（BPE 边界漂移）。marker 是「渲染器内嵌的路标」，partition 保证切在**字符串**层面精确，再编码，无歧义。

```python
    final_output = all_outputs[-1] if all_outputs else ""
    prompt_ids = prompt_ids or []
    return final_output, final_context, prompt_ids, response_ids, response_mask, response_old_logps, list(all_outputs), unfinished
```

返回八元组：最终回复、最终上下文（调试用）、prompt token、**生成+观察的 token 账本**、mask、old logp、每轮输出、是否未完成。

### 3.2 rollout_batch（163-184 行）

```python
def rollout_batch(rollout_engine, tokenizer, messages_batch, tools_batch, num_gen, ...):
    ...
    for messages, tools in zip(messages_batch, tools_batch):
        for _ in range(num_gen):
            msgs_copy = [dict(m) for m in messages]
            completion, context, ... = rollout_single(rollout_engine, tokenizer, msgs_copy, tools, ...)
            all_*.append(...)
    return ...
```

- 每 prompt 复制 `num_gen` 份消息列表（**深拷贝每条消息**——rollout_single 会往 messages 里追加内容，必须隔离）逐条 rollout；结果按 [B×num_gen] 摊平收集。

---

## 4. 奖励函数（186-244 行）

### 4.1 validate_gt_in_text（187-190 行）

```python
def validate_gt_in_text(text, gt_list):
    text, text_num = str(text), str(text).replace(',', '')
    nums = [float(x) for x in re.findall(r'(?<![A-Za-z0-9_.])[-+]?\d+(?:\.\d+)?(?![A-Za-z0-9_]|\.\d)', text_num)]
    return {g for g in gt_list if (s := str(g).strip()) and (any(abs(float(s.replace(',', '')) - n) < 1e-6 for n in nums) if re.fullmatch(r'[-+]?\d+(?:\.\d+)?', s.replace(',', '')) else s.lower() in text.lower())}
```

一行难读，拆开是：**检查标准答案 gt 是否出现在回答文本里**——
- 从文本（先去掉千分位逗号）里抽**所有数字**（带边界断言，避免把 token id 里的数字误抓）；
- gt 是纯数字 → **数值匹配**（|gt−n|<1e-6，容忍 `1,024` vs `1024`、`42.0` vs `42`）；
- gt 不是纯数字 → **子串匹配**（大小写不敏感）；
- 返回**命中的 gt 集合**（奖励按命中比例给）。

### 4.2 calculate_rewards（192-244 行）

```python
def calculate_rewards(prompts, completions, gt_batch, tools_batch, num_gen, reward_model=None, device="cuda", turn_outputs_batch=None, unfinished_batch=None):
    rewards = torch.zeros(len(completions), device=device)
    for idx, response in enumerate(completions):
        reward, answer = 0.0, response
        sample_idx = idx // num_gen
        tools = tools_batch[sample_idx]
        turn_outputs = turn_outputs_batch[idx] if ... else [response]
        unfinished = unfinished_batch[idx] if ... else False
        turn_answers = [turn.split('</think>', 1)[-1].strip() if '</think>' in turn else turn.strip() for turn in turn_outputs]
        answer = turn_answers[-1] if turn_answers else response.strip()
        valid_names = {t['function']['name'] for t in tools} if tools else set()
        tool_calls = []
        for turn_answer in turn_answers: tool_calls.extend(parse_tool_calls(turn_answer))  # 解析tool调用
        reward -= 0.5 * sum(abs(turn.count('<tool_call>') - turn.count('</tool_call>')) for turn in turn_answers)  # 标签扣分
```

| 行 | 讲解 |
|----|------|
| 196-201 | 每条轨迹：找到所属样本（idx//num_gen）、逐轮把 `<think>` 段剥掉只留答案（`turn_answers`）、最终答案取**最后一轮** |
| 202-204 | 合法工具名集合（schema 里声明的）；跨**所有轮**解析工具调用 |
| 205 | **标签闭合扣分**：`<tool_call>` 与 `</tool_call>` 数量差不平衡的，每个差 0.5 分——培养格式洁癖 |

**无工具分支（直接回答）**（206-222 行）：

```python
        if not tool_calls:
            reward += 0.5 if 5 <= len(response.strip()) <= 800 else -0.5  # 长度分
            if '</think>' in response:
                think, answer = response.split('</think>', 1)
                reward += 1.0 if 20 <= len(think.strip()) <= 300 else -0.5  # 思考长度分
                reward += 0.25 if response.count('</think>') == 1 else -0.25  # 思考闭合分
                answer = answer.strip()
            if reward_model is not None:
                ...  # 从 prompt 正则反解 messages，RM 打分加进 reward
            reward -= rep_penalty(answer)
            rewards[idx] = max(min(reward, 3.0), -3.0)  # 总分Clip
```

- 与 GRPO 的规则奖励几乎一致（长度/思考/重复/RM）——**没有工具可调的题退化为普通 RLAIF**。

**有工具分支（可验证奖励）**（224-243 行）：

```python
        else:
            gt = gt_batch[sample_idx]
            valid_call_count = 0
            for tool_call in tool_calls:
                name, raw = ...  # 兼容字符串 arguments
                check = CHECK_ARGS.get(name)
                valid_call_count += int(bool(name in valid_names and check and check(raw)))
            tool_gap = abs(valid_call_count - len(gt)) + max(0, len(tool_calls) - valid_call_count)  # tool数差值
            reward += 0.5 if tool_gap == 0 else -0.5 * tool_gap  # tool对齐分
            
            final_text = "" if unfinished else (answer.split('</tool_call>')[-1] if '</tool_call>' in answer else answer)
            verified = validate_gt_in_text(final_text, gt) if gt else set()
            if gt: reward += 2.5 * len(verified) / len(gt)  # GT分
            if unfinished: reward -= 0.5  # 未完成扣分
            reward -= rep_penalty(final_text if final_text else answer)
            rewards[idx] = max(min(reward, 3.0), -3.0)  # 总分Clip
```

| 行 | 讲解 |
|----|------|
| 227-236 | **工具对齐分**：`valid_call_count` = 名字在 schema 里 **且** 参数齐全的调用数；`tool_gap` = |合格数−标准答案要求的调用数| + 多余的无效调用数。gap=0（数量恰好）+0.5，每差一个 −0.5——**鼓励「按需、正确地」调工具**，不鼓励刷调用次数 |
| 238-240 | **答案分（主奖励，权重 2.5）**：未完成的轨迹 final_text 置空（答案无效）；否则取最后一个 `</tool_call>` 之后的文本（最终总结）；GT 命中率 × 2.5——**不依赖 RM，可验证、无偏**。这是 agent RL 相比通用 RL 最稳的信号 |
| 241-243 | 未完成 −0.5；重复惩罚；**±3 clip** 统一量纲（与 RM 分数 clip 对齐，保证组内方差不会爆炸） |

---

## 5. rl_train_epoch：打包与损失（247-369 行）

### 5.1 变长轨迹打包（256-274 行）

```python
        prompts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True, tools=t) for m, t in zip(messages_batch, tools_batch)]
        packed_samples = []
        for p, r, m, old_lp in zip(prompt_ids_batch, response_ids_batch, response_masks_batch, response_old_logps_batch):
            ids = p + r
            mask = [0] * len(p) + m
            old_logps = [0.0] * max(len(p) - 1, 0) + old_lp
            if len(ids) > args.max_total_len:
                ids = ids[-args.max_total_len:]
                mask = mask[-args.max_total_len:]
                old_logps = old_logps[-(len(ids) - 1):]
            prompt_len = next((i for i, v in enumerate(mask) if v == 1), len(mask))
            packed_samples.append((ids, mask, prompt_len, old_logps))
        seq_lens = torch.tensor([len(ids) for ids, _, _, _ in packed_samples], device=args.device)
        max_len = seq_lens.max().item()
        input_ids = torch.tensor([ids + [tokenizer.pad_token_id] * (max_len - len(ids)) for ids, _, _, _ in packed_samples], device=args.device)
        prompt_lens = torch.tensor([prompt_len for _, _, prompt_len, _ in packed_samples], device=args.device)
        full_response_masks = torch.tensor([mask + [0] * (max_len - len(mask)) for _, mask, _, _ in packed_samples], device=args.device, dtype=torch.float32)
        old_per_token_logps = torch.tensor([old_logps + [0.0] * ((max_len - 1) - len(old_logps)) for _, _, _, old_logps in packed_samples], device=args.device, dtype=torch.float32)
        full_mask = (torch.arange(max_len, device=args.device).unsqueeze(0) < seq_lens.unsqueeze(1)).long()
```

| 行 | 讲解 |
|----|------|
| 256 | prompts 重新渲染（仅用于奖励函数反解 messages） |
| 258-267 | **逐轨迹拼装**：`ids = prompt + (生成+观察)`；`mask = prompt 区 0 + 账本 mask`；`old_logps = prompt 区 0 + 账本 logp`（**长度对齐妙处**：old_logps 少一位——它对应「预测下一个 token」的 logp，prompt 的第一位没有预测者，所以补 `len(p)−1` 个 0）。超长轨迹**左侧截断**（保留最近）；`prompt_len` 用第一个 mask=1 的位置回算 |
| 268-274 | 变长 → **右 pad 到 batch 内最长**；input_ids/prompt_lens/掩码/old_logps 全部转张量；`full_mask` 是「真 token 区」掩码（不含 pad） |

### 5.2 前向与损失（278-335 行）

```python
        rewards = calculate_rewards(...)

        with autocast_ctx:
            res = model(input_ids, attention_mask=full_mask)
            aux_loss = res.aux_loss if lm_config.use_moe else torch.tensor(0.0, device=args.device)
            logits = res.logits[:, :-1, :]
            per_token_logps = F.log_softmax(logits, dim=-1).gather(2, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            ref_per_token_logps = compute_per_token_logps(ref_model, input_ids, input_ids.size(1) - 1, attention_mask=full_mask)
```

| 行 | 讲解 |
|----|------|
| 278-282 | 策略前向：全序列 logits → shift（[:, :-1] 对 input_ids[:, 1:]）→ gather 出**全序列逐 token logp** `[B, max_len−1]`（agent 版不需要 logp_pos 精选——mask 已经标好哪些位置算） |
| 284-285 | 参考模型用 rollout_engine 里的 `compute_per_token_logps`（logits_to_keep 只算尾部，10 篇详解） |

```python
        completion_mask = full_response_masks[:, 1:]
        is_eos = (input_ids[:, 1:] == tokenizer.eos_token_id) & completion_mask.bool()
        eos_idx = torch.full((completion_mask.size(0),), completion_mask.size(1) - 1, device=args.device, dtype=torch.long)
        has_eos = is_eos.any(dim=1)
        eos_idx[has_eos] = is_eos.int().argmax(dim=1)[has_eos]
        pos = torch.arange(completion_mask.size(1), device=args.device).unsqueeze(0)
        completion_mask = completion_mask * (pos <= eos_idx.unsqueeze(1)).float()
        token_counts = completion_mask.sum(dim=1)
        valid_rows = token_counts > 0
```

- 与 GRPO 相同的 **EOS 截断**：掩码右移一位（对齐到「预测目标」坐标系）→ 找每个轨迹首个 EOS → EOS 之后清零。`valid_rows`：掩码全 0 的轨迹（异常情况）跳过。

```python
        grouped_rewards = rewards.view(-1, args.num_generations)
        mean_r = grouped_rewards.mean(dim=1).repeat_interleave(args.num_generations)
        std_r = grouped_rewards.std(dim=1, unbiased=False).repeat_interleave(args.num_generations)
        advantages = (rewards - mean_r) / (std_r + 1e-4)

        kl_div = ref_per_token_logps - per_token_logps
        per_token_kl = torch.exp(kl_div) - kl_div - 1
        ratio = torch.exp(per_token_logps - old_per_token_logps)
        if args.loss_type == "cispo":
            clamped_ratio = torch.clamp(ratio, max=args.epsilon_high).detach()
            per_token_loss = -(clamped_ratio * advantages.unsqueeze(1) * per_token_logps - args.beta * per_token_kl)
        else:
            ...  # GRPO clip 分支，与 train_grpo 逐行相同
        policy_loss = (((per_token_loss * completion_mask).sum(dim=1)[valid_rows] / token_counts[valid_rows].clamp(min=1)).mean()
                       if valid_rows.any() else per_token_loss.sum() * 0.0)
        loss = (policy_loss + aux_loss) / args.accumulation_steps
        loss.backward()
```

- **损失与 GRPO 完全同构**（组归一化优势 + k3 KL + CISPO/GRPO 二选一），差异只有两点：
  1. `old_per_token_logps` 里**观察区是 0.0**，而 `completion_mask` 观察区也是 0 → ratio 在观察区是 `exp(logp−0)` 但乘 0 掩码 → **不产生损失**——观察 token 的存在只影响「上下文」，不影响学习信号；
  2. `valid_rows` 过滤全零行（防 0/0）。

### 5.3 更新/日志/保存（337-369 行）

与 GRPO 相同：累积更新 + grad_clip、七个监控量（多了 GrpStd 组内奖励标准差）、保存 + `rollout_engine.update_policy`。删张量两行。

---

## 6. main（372-490 行）

与 GRPO main 的差异：

| 行 | 内容 | 讲解 |
|----|------|------|
| 374-411 | argparse | 新增：`max_total_len=2500`（多轮打包上限）、`num_generations=4`、`thinking_ratio=0.1`（agent 任务 90% 关思考——工具调用对思考依赖低，省 token） |
| 418-419 | lm_config.max_seq_len = max_seq_len + max_gen_len | RoPE 表要覆盖「prompt+生成长度」 |
| 433-450 | 初始化 | policy/ref/RM/rollout_engine 同 GRPO |
| 451 | AgentRLDataset | 原始 messages+tools+gt |
| 454-455 | **自定义 collate_fn** | 数据是变长结构（消息列表），默认 collate 会崩；简单把 batch 内各字段收成 list（逐条 rollout 处理） |
| 457-458 | 计数 loader + CosineAnnealingLR | 同 GRPO 的 iters 技巧 |
| 471-474 | DDP（broadcast_buffers=False）+ update_policy | 同 GRPO |

## 7. 本篇小结

1. **多轮轨迹 = prompt + Σ(每轮生成 + 工具观察)**；生成区 mask=1、观察区 mask=0——「模型只对自己说过的话负责」。
2. **标记法（marker trick）**把模板注入的结构 token 精确切出：唯一 marker 渲染 → partition → 编码 → 入账本（mask 0/logp 0）。
3. 奖励双层：**无工具题**走 RM+规则；**工具题**走可验证奖励（工具数量对齐 0.5 + GT 命中率 2.5）+ 格式扣分 + 未完成扣分，±3 clip。
4. 变长轨迹打包：prompt 区 old_logp 补 0 对齐错位；超长左截；EOS 截断 mask 复用 GRPO 逻辑。
5. 数学工具的真值来自 safe_math_eval——**环境要确定性，奖励才可信**。

下一篇：[09-知识蒸馏-train_distillation](./09-知识蒸馏-train_distillation.md)
