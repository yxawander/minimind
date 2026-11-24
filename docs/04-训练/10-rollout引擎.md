# 10 - Rollout 引擎 rollout_engine.py 全文

> 覆盖代码：`trainer/rollout_engine.py` **全文 225 行**。
> 职责：**把「用当前策略采样」这件事从 RL 训练脚本里解耦出来**。GRPO/PPO/AgentRL 都通过统一接口 `rollout()` 拿生成结果、`update_policy()` 同步权重，底层可插拔：`torch`（进程内直接 generate）或 `sglang`（HTTP 调用独立推理服务器）。
> 文件头注释（1-4 行）给出 SGLang 启动命令：`python -m sglang.launch_server --model-path ./minimind-3 --attention-backend triton --host 0.0.0.0 --port 8998`

---

## 1. 为什么要解耦

训练时模型的 forward 是「教学用」的（HF 接口、动态图）；而**批量采样**讲究吞吐（continuous batching、前缀缓存）。把采样交给专用推理引擎后：

- 训练侧不用改：`rollout(prompt_ids, num_generations, ...) -> RolloutResult`；
- 引擎可换：调试用 torch（零依赖）、生产用 SGLang（快几倍）；
- 权重同步被封装成 `update_policy()` 一个钩子。

这就是 RLHF 系统里经典的 **trainer / rollout 分离**架构的极简版。

---

## 2. 逐行精读

### 2.1 compute_per_token_logps（24-36 行）

```python
# ===== 计算每个 token 的 logprob =====
def compute_per_token_logps(model, input_ids: Tensor, n_keep: int, attention_mask: Optional[Tensor] = None) -> Tensor:
    if n_keep <= 0:
        return input_ids.new_empty((input_ids.size(0), 0), dtype=torch.float32)
    unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
    input_ids = input_ids.detach().clone() if input_ids.is_inference() else input_ids
    logits = unwrapped(input_ids, attention_mask=attention_mask, logits_to_keep=n_keep + 1).logits[:, :-1, :]
    per_token_logps = []
    for logits_row, ids_row in zip(logits, input_ids[:, -n_keep:]):
        ids_row = ids_row.detach().clone() if ids_row.is_inference() else ids_row
        per_token_logps.append(
            torch.gather(logits_row.log_softmax(dim=-1), 1, ids_row.unsqueeze(1)).squeeze(1)
        )
    return torch.stack(per_token_logps)
```

| 行 | 讲解 |
|----|------|
| 25-26 | n_keep≤0：返回空张量（省一次前向） |
| 27 | 剥 DDP 壳——推理用裸模型（DDP 包装的 forward 会在不需要时浪费通信） |
| 28 | `is_inference()`：inference-mode 张量（generate 产出的）不能直接进训练图，clone 成普通张量 |
| 29 | **`logits_to_keep=n_keep+1`**：只对最后 n_keep+1 个位置算 lm_head（模型 forward 的切片参数，01 篇讲过）。**为什么 +1**：要预测「最后 n_keep 个 token」的 logp，需要倒数第 n_keep+1 个位置的 logits（错位一步）→ 切 `[:, :-1]` 后正好剩 n_keep 行 |
| 31-35 | 逐样本（batch 内 prompt 长度不同，直接 gather 整批会错位，所以循环）：对每行 `log_softmax` 后 gather 出对应 token 的 logp |
| 36 | stack 成 `[B, n_keep]` |

> train_agent.py 里 ref model 的 logp 就用这个函数（而策略模型 logp 用全量 gather——因为策略需要梯度走全图，ref 只要数值）。

### 2.2 RolloutResult（40-47 行）

```python
@dataclass
class RolloutResult:
    output_ids: Tensor          # [B*num_gen, P+R] 完整序列（含 prompt）
    completion_ids: Tensor      # [B*num_gen, R] 生成的部分
    per_token_logps: Tensor     # [B*num_gen, R] π_old 的逐 token logp
    completions: List[str]      # 解码出的文本
    prompt_lens: Tensor         # [B*num_gen] 各样本 prompt 长度
    completion_mask: Tensor     # [B*num_gen, R] 生成区有效掩码
```

**六个字段是三个 RL 训练脚本约定的接口**：训练侧拿 output_ids 前向、拿 completion_mask 造损失掩码、拿 per_token_logps 当 ratio 分母、拿 prompt_lens 算错位索引（GRPO/PPO 的 logp_pos）、拿 completions 打分。

### 2.3 RolloutEngine 抽象基类（50-60 行）

```python
class RolloutEngine(ABC):
    tokenizer = None
    
    @abstractmethod
    def rollout(self, prompt_ids, attention_mask, num_generations, max_new_tokens, temperature=0.8) -> RolloutResult:
        pass
    
    @abstractmethod
    def update_policy(self, model: torch.nn.Module):
        pass
```

两个抽象方法即全部契约：**采样**与**权重同步**。ABC+abstractmethod 强制子类实现——缺一个方法实例化时直接报错。

### 2.4 TorchRolloutEngine（64-96 行）

```python
class TorchRolloutEngine(RolloutEngine):
    def __init__(self, policy_model, tokenizer, device="cuda", autocast_ctx=None):
        self.policy_model = policy_model
        self.tokenizer = tokenizer
        self.device = device
        self.autocast_ctx = autocast_ctx
    
    def rollout(self, prompt_ids, attention_mask, num_generations, max_new_tokens, temperature=0.8) -> RolloutResult:
        model = self.policy_model.module if isinstance(self.policy_model, DistributedDataParallel) else self.policy_model
        ctx = self.autocast_ctx if self.autocast_ctx else nullcontext()
        with torch.no_grad(), ctx:
            output_ids = model.generate(
                input_ids=prompt_ids.repeat_interleave(num_generations, dim=0),
                attention_mask=attention_mask.repeat_interleave(num_generations, dim=0),
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            ).clone()  # [B*num_gen, P+R]
```

| 行 | 讲解 |
|----|------|
| 72 | 剥 DDP 壳——**采样不需要各卡同步**，直接用本 rank 的模型 |
| 73-74 | 沿用训练脚本的 autocast 上下文（bf16 采样省显存）；no_grad 关图 |
| 76-84 | `repeat_interleave(num_generations, dim=0)`：**每个 prompt 连续复制 G 份**（[p1,p1,p1,p2,p2,p2,...]——注意与 `repeat` 的交错复制不同，这个顺序与奖励函数的 `idx//num_gen` 反解样本号对齐）；调 **模型自带的手写 generate**（01 篇/08 篇）采样。`.clone()`：generate 返回的张量可能共享内部存储，detach 后克隆成独立副本 |

```python
            prompt_len = prompt_ids.size(1)
            completion_ids = output_ids[:, prompt_len:]  # [B*num_gen, R]
            full_mask = torch.cat([attention_mask.repeat_interleave(num_generations, dim=0), attention_mask.new_ones(output_ids.size(0), completion_ids.size(1))], dim=1)
            per_token_logps = compute_per_token_logps(self.policy_model, output_ids, completion_ids.size(1), attention_mask=full_mask)
        completions = self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        return RolloutResult(output_ids, completion_ids, per_token_logps, completions,
                             prompt_ids.new_full((output_ids.size(0),), prompt_len),
                             attention_mask.new_ones(output_ids.size(0), completion_ids.size(1)))
    
    def update_policy(self, model: torch.nn.Module):
        self.policy_model = model
```

| 行 | 讲解 |
|----|------|
| 85-86 | 切出生成区 |
| 87 | **logp 计算的 mask**：prompt 区沿用输入 mask（含左 pad 的 0），生成区全 1——给 compute_per_token_logps 用 |
| 88 | 立刻算 π_old 的 logp（**在 no_grad 里**——RL 用它当常数分母） |
| 90 | 文本解码（skip_special_tokens：奖励函数想看到纯文本；注意 `<tool_call>` 不是 special 所以保留） |
| 91-92 | prompt_lens 全部等于输入长度（左 pad 下 prompt 区等长）；completion_mask 全 1（torch 引擎生成到 EOS 就停、pad 由 EOS 截断逻辑处理） |
| 94-95 | **update_policy**：换引用即可——训练脚本每 save_interval 调用，把新模型对象塞进来 |

### 2.5 SGLangRolloutEngine（99-205 行）

```python
class SGLangRolloutEngine(RolloutEngine):
    def __init__(self, base_url: str, model_path: str, shared_ckpt_path: str = "./sglang_ckpt", timeout: int = 120):
        self.base_url = base_url.rstrip('/')
        self.shared_ckpt_path = shared_ckpt_path
        self.timeout = timeout
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.http = requests
```

- 不持有模型——**模型在远端 SGLang 服务器进程里**，本类只是 HTTP 客户端。tokenizer 本地加载用于解码/停止符。

```python
    def rollout(self, prompt_ids, attention_mask, num_generations, max_new_tokens, temperature=0.8) -> RolloutResult:
        # 去除左侧 padding tokens，只保留有效 token
        input_ids_list = []
        for ids, mask in zip(prompt_ids, attention_mask):
            valid_ids = ids[mask.bool()].tolist()
            input_ids_list.append(valid_ids)
        all_input_ids = [ids for ids in input_ids_list for _ in range(num_generations)]
```

| 行 | 讲解 |
|----|------|
| 109-113 | **SGLang API 是变长请求**——把左 pad 剥掉（mask 取真 token），再每个 prompt 复制 G 份。列表推导 `[ids for ids in list for _ in range(G)]` 实现 repeat_interleave 的列表版 |

```python
        payload = {
            "input_ids": all_input_ids,
            "sampling_params": {
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
                "stop_token_ids": [self.tokenizer.eos_token_id] if self.tokenizer.eos_token_id else [],
            },
            "return_logprob": True,
        }
        
        resp = self.http.post(f"{self.base_url}/generate", json=payload, timeout=self.timeout)
        resp.raise_for_status()
        
        results = resp.json()
        if not isinstance(results, list):
            results = [results]
```

| 行 | 讲解 |
|----|------|
| 115-123 | 请求体：直接传 **input_ids**（跳过文本解析，省 CPU）；`return_logprob=True` 让服务端返回每 token logp（π_old 从服务端拿，省一次本地前向）；stop_token_ids 停止符 |
| 125-130 | POST `/generate`（SGLang 原生批量接口，自动 continuous batching）；单条返回时包成列表 |

```python
        for i, result in enumerate(results):
            meta = result.get("meta_info", {})
            completion_ids = meta.get("output_ids", result.get("output_ids", []))
            raw_logprobs = meta.get("output_token_logprobs", [])
            
            logprobs = []
            for item in raw_logprobs:
                if isinstance(item, (list, tuple)) and len(item) >= 1:
                    logprobs.append(item[0])
                elif isinstance(item, (int, float)):
                    logprobs.append(item)
            
            if len(logprobs) < len(completion_ids):
                logprobs = [0.0] * (len(completion_ids) - len(logprobs)) + logprobs
            elif len(logprobs) > len(completion_ids):
                logprobs = logprobs[-len(completion_ids):] if completion_ids else []
            prompt = all_input_ids[i]
            full_output = prompt + completion_ids
            all_output_ids.append(full_output)
            all_completion_ids.append(completion_ids)
            all_logprobs.append(logprobs)
            completions.append(self.tokenizer.decode(completion_ids, skip_special_tokens=True))
```

| 行 | 讲解 |
|----|------|
| 136-138 | **兼容不同 SGLang 版本的返回结构**（meta_info 里/顶层；output_token_logprobs 的元素可能是 (logp, id, ...) 元组或纯数）——第三方 API 版本漂移的防御 |
| 147-150 | 长度对齐兜底：logp 少了前补 0、多了取尾部——宁可占位也不让训练崩 |
| 151-156 | 拼 prompt+completion；解码文本 |

```python
        device = prompt_ids.device
        max_comp_len = max(1, max(len(ids) for ids in all_completion_ids))
        max_out_len = max(len(ids) for ids in all_input_ids) + max_comp_len
        
        def pad_to_tensor(seqs, max_len, pad_val=0):
            return torch.tensor([s + [pad_val] * (max_len - len(s)) for s in seqs], device=device)
        
        pad_id = self.tokenizer.pad_token_id
        return RolloutResult(
            output_ids=pad_to_tensor(all_output_ids, max_out_len, pad_val=pad_id),
            completion_ids=pad_to_tensor(all_completion_ids, max_comp_len, pad_val=pad_id),
            per_token_logps=pad_to_tensor(all_logprobs, max_comp_len, pad_val=0.0),
            completions=completions,
            prompt_lens=torch.tensor([len(ids) for ids in all_input_ids], device=device),
            completion_mask=torch.tensor([[1] * len(ids) + [0] * (max_comp_len - len(ids)) for ids in all_completion_ids], device=device),
        )
```

| 行 | 讲解 |
|----|------|
| 158-173 | **变长结果重整成矩形张量**（与 torch 引擎返回形状一致，训练侧无感知）：各样本长度可能不同 → pad 到 batch 最大长度；completion_mask 标出真实生成区（**这是 SGLang 引擎返回变长 mask、torch 引擎返回全 1 的原因**——训练侧的 EOS 截断/散点修正逻辑兼容两种） |

### 2.6 update_policy：SGLang 的权重热更新（175-194 行）

```python
    def update_policy(self, model: torch.nn.Module):
        ok = True
        if not dist.is_initialized() or dist.get_rank() == 0:
            try:
                unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
                unwrapped = getattr(unwrapped, '_orig_mod', unwrapped)
                abs_path = os.path.abspath(self.shared_ckpt_path)
                state_dict = {k: v.detach().half().cpu() for k, v in unwrapped.state_dict().items()}
                unwrapped.save_pretrained(abs_path, state_dict=state_dict, safe_serialization=False)
                self.tokenizer.save_pretrained(abs_path)
                resp = self.http.post(f"{self.base_url}/update_weights_from_disk", json={"model_path": abs_path}, timeout=self.timeout)
                if resp.status_code != 200: print(f"[SGLANG WARNING] update_weights 失败: ...")
                ok = resp.status_code == 200
            except Exception as e:
                print(f"[SGLANG WARNING] update_weights 异常: {e}"); ok = False
        if dist.is_initialized():
            ok_t = torch.tensor(int(ok), device=next(model.parameters()).device)
            dist.broadcast(ok_t, src=0); dist.barrier(); ok = bool(ok_t.item())
        if not ok: raise RuntimeError("SGLang update_policy failed")
        return ok
```

| 行 | 讲解 |
|----|------|
| 177 | **只有 rank 0 做更新**（写盘+HTTP 各一次，其他 rank 静等） |
| 179-183 | 剥 DDP/compile 壳 → state_dict 转 fp16 → `save_pretrained` 存成 **transformers 完整格式**（config+权重，SGLang 直接可加载）到**共享目录** |
| 185-188 | 调 SGLang 的 `/update_weights_from_disk`：**服务器原地热加载新权重**（不重启、不重新分配 KV cache 池）——这是在线 RL 的关键设施 |
| 190-193 | **多卡一致性**：rank 0 的成功/失败广播给全体（`broadcast+barrier`），失败则全体抛错——否则 rank 0 失败、其他 rank 继续训练，采样引擎还停留在旧权重，ratio 会静默漂移 |
| 196-205 | 辅助方法：flush_cache（清服务端 KV 缓存）、health（探活）——调试用 |

### 2.7 工厂（209-225 行）

```python
def create_rollout_engine(engine_type="torch", policy_model=None, tokenizer=None, device="cuda", autocast_ctx=None,
                          sglang_base_url=None, sglang_model_path=None, sglang_shared_path=None) -> RolloutEngine:
    if engine_type == "torch":
        return TorchRolloutEngine(policy_model, tokenizer, device, autocast_ctx)
    elif engine_type == "sglang":
        return SGLangRolloutEngine(sglang_base_url, sglang_model_path, sglang_shared_path)
    else:
        raise ValueError(f"不支持的引擎类型: {engine_type}")
```

- 简单工厂：字符串 → 引擎实例。训练脚本一行切换。

## 3. 两种引擎对比

| | TorchRolloutEngine | SGLangRolloutEngine |
|--|---------------------|---------------------|
| 模型位置 | 训练进程内 | 独立服务器 |
| 吞吐 | 低（无 continuous batching） | 高数倍（大批量时） |
| 额外显存 | 0 | 服务器一份模型 |
| 部署 | 开箱即用 | 要先 launch_server + 共享目录 |
| 权重同步 | 换引用（零成本） | 写盘 + HTTP 热加载 |
| 适用 | 学习/小规模验证 | 正经 RL 训练 |

## 4. 本篇小结

1. 接口只有两个：`rollout()`（采样 → 六字段 RolloutResult）与 `update_policy()`（权重同步）——三个 RL 脚本全部依赖它。
2. `compute_per_token_logps` 的 `logits_to_keep=n+1` + 逐行 gather 是「尾部 logp 高效抽取」的通用件。
3. torch 引擎细节：repeat_interleave 复制、剥 DDP、no_grad 下立刻固化 π_old。
4. SGLang 引擎细节：变长 HTTP 请求/响应重整、**多版本 API 兼容垫片**、rank0 写盘热更新 + **broadcast 保证全体一致**。
5. 这套抽象的意义：**RL 算法代码与推理引擎彻底解耦**，换引擎零改动训练代码。

---

训练系列完。下一篇进入推理部署：[05-推理与部署/01-命令行对话-eval_llm](../05-推理与部署/01-命令行对话-eval_llm.md)
