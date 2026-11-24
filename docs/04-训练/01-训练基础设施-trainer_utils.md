# 01 - 训练基础设施 trainer_utils.py 全文

> 覆盖代码：`trainer/trainer_utils.py` **全文 209 行**。这是所有训练脚本共享的工具箱：余弦学习率、DDP 初始化、随机种子、**断点续训**（lm_checkpoint）、模型初始化、跳批采样器、奖励模型封装、安全的数学表达式求值。
> 后续训练篇会频繁引用这里的函数，建议先读透本篇。

---

## 1. 文件头与导入（1-18 行）

```python
"""训练工具函数集合"""
import os, sys
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import ast, operator, random, math
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Sampler
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from model.model_minimind import MiniMindForCausalLM
```

| 行 | 讲解 |
|----|------|
| 6-7 | `__package__ = "trainer"`：把本模块标记为 `trainer` 包的成员（即使被直接 `python trainer_utils.py` 运行），这样相对导入语境正确。`sys.path.append` 把仓库根目录加进模块搜索路径——训练脚本用 `from model.model_minimind import ...` 这样的「根目录相对」导入，靠它才能找到 |
| 8-11 | `ast`/`operator`：给 safe_math_eval 用的 AST 求值；`random/np/torch`：给 setup_seed 用的三种随机源 |
| 15-17 | `Sampler` 基类（写 SkipBatchSampler 用）；transformers 的三个 Auto 类（奖励模型加载） |
| 18 | 引入模型类（init_model 构造用） |

---

## 2. get_model_params：参数量统计（20-30 行）

```python
def get_model_params(model, config):
    total = sum(p.numel() for p in model.parameters()) / 1e6
    n_routed = getattr(config, 'n_routed_experts', getattr(config, 'num_experts', 0))
    n_active = getattr(config, 'num_experts_per_tok', 0)
    n_shared = getattr(config, 'n_shared_experts', 0)
    expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.experts.0.' in n) / 1e6
    shared_expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.shared_experts.0.' in n) / 1e6
    base = total - (expert * n_routed) - (shared_expert * n_shared)
    active = base + (expert * n_active) + (shared_expert * n_shared)
    if active < total: Logger(f'Model Params: {total:.2f}M-A{active:.2f}M')
    else: Logger(f'Model Params: {total:.2f}M')
```

| 行 | 讲解 |
|----|------|
| 21 | 全部参数求和（单位 M）。注意：**tied 权重只计一次**（同一 Parameter 对象） |
| 22-24 | 路由专家数/每 token 激活数/共享专家数。`getattr` 链：config 里可能没有这些字段（dense 模型），兜底 0 |
| 25-26 | 单个专家的参数量：用「键名含 `mlp.experts.0.`」筛出第 0 个专家的参数求和（每层都有，所以这是**全部层第 0 专家**之和——单专家总量）。shared_expert 同理（本版已移除 shared expert，筛不到就是 0） |
| 27-28 | `base` = 非专家参数；`active` = base + 每层实际激活的专家（`expert × n_active` 是**每层激活的专家参数**——注意这里乘的是 num_experts_per_tok，隐含假设每层激活 n_active 个专家）。MoE 显示 `198.00M-A64.00M` 这种「总量-激活量」格式 |
| 29-30 | dense 时 active==total，只显示总量 |

---

## 3. is_main_process / Logger（33-39 行）

```python
def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0

def Logger(content):
    if is_main_process():
        print(content)
```

| 行 | 讲解 |
|----|------|
| 33-34 | **主进程判定**：没初始化分布式（单卡）→ 恒真；DDP 下只有 rank 0 是「主进程」。全仓库所有打印/保存都套这个判断——8 张卡各打一遍日志会淹死人 |
| 37-39 | `Logger`：带主进程过滤的 print。**注意它是小写 l 开头的函数**，命名模仿 wandb 的 Logger 风格 |

---

## 4. get_lr：余弦退火学习率（42-43 行）

```python
def get_lr(current_step, total_steps, lr):
    return lr*(0.1 + 0.45*(1 + math.cos(math.pi * current_step / total_steps)))
```

| 行 | 讲解 |
|----|------|
| 42-43 | **无 warmup 的余弦退火**，逐 step 手动调（不用 scheduler）：<br>• step=0：cos(0)=1 → `0.1+0.45×2 = 1.0` → **起始 lr 全额**<br>• step=total/2：cos(π/2)=0 → `0.1+0.45 = 0.55` → 一半行程时 0.55×lr<br>• step=total：cos(π)=−1 → `0.1` → **结尾 0.1×lr**<br>形状：从 lr 平滑降到 0.1·lr。没有 warmup 是小模型/短训练的取舍（大模型必须 warmup 是因为初始梯度爆炸风险，这里 lr 本身小）。调用处（train_pretrain.py:29）每步算一次并写回 optimizer.param_groups。GRPO/PPO 则用 torch 的 CosineAnnealingLR（原理相同，走 PyTorch 实现） |

---

## 5. init_distributed_mode：DDP 启动（46-53 行）

```python
def init_distributed_mode():
    if int(os.environ.get("RANK", -1)) == -1:
        return 0  # 非DDP模式
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank
```

| 行 | 讲解 |
|----|------|
| 47-48 | 判定是否 DDP：`torchrun`/`torch.distributed.launch` 启动时会在环境变量里注入 `RANK/WORLD_SIZE/LOCAL_RANK`；普通 `python train.py` 没有 RANK → 返回 0，单卡路径 |
| 50 | 初始化进程组（NCCL = NVIDIA 的 GPU 集合通信后端）。**各卡的 world 信息来自环境变量**，不需要命令行传参 |
| 51-52 | `LOCAL_RANK`：本机内的卡号。把当前进程绑定到对应 GPU（否则 8 个进程全挤在 0 号卡） |
| 53 | 返回 local_rank 给调用方（DataParallel 包装时用） |

**DDP 使用方式**：`torchrun --nproc_per_node=8 trainer/train_pretrain.py` → 每个进程跑同一份脚本、各自 init_distributed_mode → DistributedDataParallel 包装模型 → 每步自动做梯度 all-reduce（各卡梯度求平均）。

---

## 6. setup_seed：全源随机种子（56-63 行）

```python
def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

| 行 | 讲解 |
|----|------|
| 57-61 | 四个随机源全部播种：Python random（数据增强/采样）、numpy、torch CPU、torch GPU（单卡 + 所有卡）。任何一个漏掉都会造成「不可复现」 |
| 62-63 | cuDNN 两个开关：`deterministic=True` 强制确定性卷积/算子实现（可复现但略慢）；`benchmark=False` 关闭「自动挑最快 kernel」（它挑的结果可能因运行而异）。调用处都用 `seed + rank`（各卡不同数据顺序）与 `seed + epoch`（每轮重新洗牌），见训练篇 |

---

## 7. lm_checkpoint：断点续训的核心（65-117 行）

这个函数是**双面函数**：`model=None` 时是「读」，否则是「写」。

```python
def lm_checkpoint(lm_config, weight='full_sft', model=None, optimizer=None, epoch=0, step=0, wandb=None, save_dir='../checkpoints', **kwargs):
    os.makedirs(save_dir, exist_ok=True)
    moe_path = '_moe' if lm_config.use_moe else ''
    ckp_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}.pth'
    resume_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}_resume.pth'
```

| 行 | 讲解 |
|----|------|
| 66-69 | 两个文件名：`{weight}_{hidden}[_moe].pth` = **纯模型权重**（与 out/ 下推理用的同名格式）；`..._resume.pth` = **完整训练状态**（模型 fp32 + 优化器 + 进度 + kwargs）。MoE 后缀保证 Dense/MoE 互不覆盖 |

### 7.1 保存分支（71-107 行）

```python
    if model is not None:
        raw_model = model.module if isinstance(model, DistributedDataParallel) else model
        raw_model = getattr(raw_model, '_orig_mod', raw_model)
        state_dict = {k: v.cpu() for k, v in raw_model.state_dict().items()}
        ckp_tmp = ckp_path + '.tmp'
        torch.save({k: v.half() for k, v in state_dict.items()}, ckp_tmp)
        os.replace(ckp_tmp, ckp_path)
```

| 行 | 讲解 |
|----|------|
| 72-73 | **剥两层包装**：DDP 的 `model.module` → torch.compile 的 `_orig_mod`。保存的永远是裸模型 |
| 74 | state_dict 搬 CPU（fp32，给 resume 用） |
| 75-77 | **原子写入**：先存 `.tmp` 再 `os.replace` 改名。为什么？训练中突然断电/杀进程时，直接写目标文件可能写了一半——损坏且不可恢复。先写临时文件再原子改名（同一文件系统内 replace 是原子操作），保证磁盘上的 ckp **要么完整旧版要么完整新版**。工程细节但救命 |
| 78-84 | 提取 wandb run id（swanlab/wandb 两代 API：有 `get_run` 用新式，否则直接读 `wandb.id`），为的是续训时**接回同一条训练曲线**（wandb 的 `id+resume='must'`） |

```python
        resume_data = {
            'model': state_dict,
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'step': step,
            'world_size': dist.get_world_size() if dist.is_initialized() else 1,
            'wandb_id': wandb_id
        }
        for key, value in kwargs.items():
            if value is not None:
                if hasattr(value, 'state_dict'):
                    raw_value = value.module if isinstance(value, DistributedDataParallel) else value
                    raw_value = getattr(raw_value, '_orig_mod', raw_value)
                    resume_data[key] = raw_value.state_dict()
                else:
                    resume_data[key] = value

        resume_tmp = resume_path + '.tmp'
        torch.save(resume_data, resume_tmp)
        os.replace(resume_tmp, resume_path)
        del state_dict, resume_data
        torch.cuda.empty_cache()
```

| 行 | 讲解 |
|----|------|
| 86-92 | resume 数据包：fp32 模型（**注意**：与上面 half 版不同，恢复训练必须满精度，fp16 存 optimizer 状态会崩）、优化器状态（AdamW 的动量/二阶矩，恢复不恢复直接决定续训前几步 loss 是否跳变）、进度（epoch/step）、**world_size**（几卡训练的）、wandb id |
| 94-101 | **kwargs 机制**：调用方可塞任意额外状态（PPO 传 `critic_model/critic_optimizer/critic_scheduler`，GRPO 传 `scheduler`）。有 `state_dict` 属性的（模块/优化器/scheduler）同样剥包装存 state_dict；标量直接存。这让 lm_checkpoint 对所有训练脚本通用 |
| 103-107 | 同样的原子写；清缓存（显存里的大字典及时释放） |

### 7.2 加载分支（108-117 行）

```python
    else:  # 加载模式
        if os.path.exists(resume_path):
            ckp_data = torch.load(resume_path, map_location='cpu')
            saved_ws = ckp_data.get('world_size', 1)
            current_ws = dist.get_world_size() if dist.is_initialized() else 1
            if saved_ws != current_ws:
                ckp_data['step'] = ckp_data['step'] * saved_ws // current_ws
                Logger(f'GPU数量变化({saved_ws}→{current_ws})，step已自动转换为{ckp_data["step"]}')
            return ckp_data
        return None
```

| 行 | 讲解 |
|----|------|
| 109-110 | 有 resume 文件就整体加载（fp32 状态） |
| 111-115 | **跨卡数续训的 step 换算**：`step` 是「batch 迭代步数」，4 卡训练时每 step 消耗 4 倍数据；换成 8 卡后同样的数据对应 step/2。`step × saved_ws // current_ws` 做线性换算（整除截断）。没有它，跨卡数恢复会错位跳批 |
| 116-117 | 没有 resume 文件返回 None（从头训） |

---

## 8. init_model：统一的模型构建入口（120-132 行）

```python
def init_model(lm_config, from_weight='pretrain', tokenizer_path='../model', save_dir='../out', device='cuda'):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = MiniMindForCausalLM(lm_config)

    if from_weight!= 'none':
        moe_suffix = '_moe' if lm_config.use_moe else ''
        weight_path = f'{save_dir}/{from_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
        weights = torch.load(weight_path, map_location=device)
        model.load_state_dict(weights, strict=False)

    get_model_params(model, lm_config)
    Logger(f'Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f}M')
    return model.to(device), tokenizer
```

| 行 | 讲解 |
|----|------|
| 121 | AutoTokenizer 从 `model/` 目录加载（与训练词表强绑定） |
| 122 | 按配置空构建模型（随机权重） |
| 124-128 | `from_weight='none'` 才是纯随机初始化（预训练第一步）；否则从 `out/` 加载指定阶段的 fp16 权重。`strict=False`：允许 checkpoint 缺少部分键（例如模型新增 buffer）；**权重是 fp16 存储、载入 fp32 模型时 PyTorch 自动升精度** |
| 130-131 | 打印参数量（total 与 trainable——LoRA 训练时 trainable 一行会显著变小） |
| 132 | 搬设备 + 返回 (model, tokenizer) |

---

## 9. SkipBatchSampler：断点续训的跳批器（135-158 行）

```python
class SkipBatchSampler(Sampler):
    def __init__(self, sampler, batch_size, skip_batches=0):
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches

    def __iter__(self):
        batch = []
        skipped = 0
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:
                    skipped += 1
                    batch = []
                    continue
                yield batch
                batch = []
        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch

    def __len__(self):
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size
        return max(0, total_batches - self.skip_batches)
```

**解决的问题**：续训时恢复到了「第 3 epoch 第 500 step」，但 DataLoader 的 shuffle 顺序取决于随机种子——种子固定后顺序可复现，于是**重建同样的顺序、跳过前 500 个 batch、从 501 继续**，数据不重不漏。

| 行 | 讲解 |
|----|------|
| 136-139 | 包住一个底层 sampler（DDP 的 DistributedSampler 或纯索引列表 `indices`）+ batch_size + 要跳过的 batch 数 |
| 141-153 | 迭代逻辑：攒 idx 成批 → 若还在「待跳过」区间就丢弃该批继续；否则 `yield` 该批。尾部不足一 batch 的余数：只在没跳过时才 yield（跳批后余数批会与换算后的 step 对齐问题，宁缺勿重） |
| 156-158 | `__len__`：总 batch 数 − 跳过数（DataLoader 用它算进度） |

**顺序可复现的保证**在训练脚本里：`setup_seed(seed+epoch); indices = torch.randperm(len(train_ds)).tolist()`（train_pretrain.py:151）——同样的种子产生同样的洗牌顺序，SkipBatchSampler 才能精确跳到位。

---

## 10. LMForRewardModel：奖励模型封装（161-178 行）

```python
class LMForRewardModel:
    def __init__(self, model_path, device="cuda", dtype=torch.float16):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, torch_dtype=dtype, trust_remote_code=True)
        self.model = self.model.to(device).eval()
        self.device = device

    @torch.no_grad()
    def get_score(self, messages, response):
        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages[:-1]])
        last_query = messages[-1]['content'] if messages else ""
        message_context = f"{history_text}\n以上是对话历史。我的新问题是：\n{last_query}" if history_text else last_query
        eval_messages = [
            {"role": "user", "content": message_context},
            {"role": "assistant", "content": response}
        ]
        score = self.model.get_score(self.tokenizer, eval_messages)
        return max(min(score, 3.0), -3.0)
```

| 行 | 讲解 |
|----|------|
| 161 | **不是 nn.Module**，纯推理封装类。加载的是外部奖励模型（默认 `internlm2-1_8b-reward`，书生 1.8B 奖励模型）：`trust_remote_code=True` 允许执行模型仓库自带的自定义代码（get_score 方法就是它定义的） |
| 165 | eval 模式 + fp16（推理专用） |
| 168 | `@no_grad`：打分永远不反传 |
| 170-172 | **对话压缩**：奖励模型的输入格式只有一对 (user, assistant)，多轮历史被折叠成一段文本 + 「我的新问题是：」+ 最后一条 query——把多轮对话塞进单轮接口的变通 |
| 177-178 | 调用奖励模型自家 API 打分，然后 **clip 到 ±3**：极端分数会淹没 RL 的规则奖励（长度/格式分只有 ±0.5~1），裁剪保证信号平衡。reward 函数详解见 GRPO/PPO 篇 |

---

## 11. safe_math_eval：安全表达式求值（181-209 行）

**用途**：Agentic RL 里模型会调用 `calculate_math` 工具，参数是模型生成的**字符串表达式**（如 `"2**10"`）。绝不能用 `eval()` 直接执行——模型可能生成 `__import__('os').system(...)` 之类的攻击串。本函数用 **AST 白名单**方式安全求值。

```python
def safe_math_eval(expression):
    """对模型生成的数学表达式求值：支持 + - * / // % **、math 白名单函数与 pi/e/tau 常量。"""
    def pow_guard(base, exp):  # 幂运算统一走这里，拦截 9**9**9 / 10**99999 这类把进程算死的输入
        if abs(exp) > 1e4 or (abs(base) > 1 and abs(exp) * math.log10(abs(base)) > 100): raise ValueError('幂运算结果过大')
        return base ** exp
    def resolve(node):  # 同时兼容 sqrt(4) 与 math.sqrt(4) 两种写法
        if isinstance(node, ast.Name): return node.id
        if isinstance(node, ast.Attribute) and getattr(node.value, 'id', '') == 'math': return node.attr
    def walk(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float): return node.value
        name = resolve(node)
        if name in consts: return consts[name]
        if isinstance(node, ast.UnaryOp): return unary_ops[type(node.op)](walk(node.operand))
        if isinstance(node, ast.BinOp): return bin_ops[type(node.op)](walk(node.left), walk(node.right))
        if isinstance(node, ast.Call) and not node.keywords: return funcs[resolve(node.func)](*map(walk, node.args))
        raise ValueError(f'不支持的表达式语法: {type(node).__name__}')
    consts = {'pi': math.pi, 'e': math.e, 'tau': math.tau}
    funcs = {'pow': pow_guard, **{n: getattr(math, n) for n in 'sqrt exp log log2 log10 sin cos tan asin acos atan atan2 sinh cosh tanh floor ceil trunc fabs fmod hypot gcd degrees radians'.split()}}
    bin_ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: pow_guard}
    unary_ops = {ast.UAdd: operator.pos, ast.USub: operator.neg}
    chars = str.maketrans({'^': '**', '×': '*', '÷': '/', '−': '-', '²': '**2', '³': '**3', '（': '(', '）': ')'})
    expr = str(expression).translate(chars).strip()
    if not expr or len(expr) > 512: raise ValueError('表达式为空或过长')
    try:
        return walk(ast.parse(expr, mode='eval').body)
    except (KeyError, SyntaxError, TypeError):
        raise ValueError('不支持的表达式语法') from None
```

逐块讲解：

| 位置 | 讲解 |
|------|------|
| `pow_guard` | 幂运算守卫：指数绝对值 >10⁴，或「位数爆炸」（|base|>1 时 `|exp|×log₁₀|base|` 即结果十进制位数 >100）→ 拒绝。没有它，`9**9**9` 会尝试计算 3.7 亿位大数，进程卡死——**LLM 工具调用的输入是不可信的** |
| `resolve` | 把 AST 名字节点解析成字符串：`ast.Name`（裸名 `sqrt`）直接取 id；`ast.Attribute`（`math.sqrt`）且宿主是 `math` 时取属性名。这样两种写法统一处理 |
| `walk` | **递归求值器**，只认五种节点：<br>1. 常量（且必须是 int/float——字符串常量直接抛错，杜绝注入）；<br>2. 常量名（pi/e/tau）；<br>3. 一元 ±；<br>4. 二元运算（查 bin_ops 表——**白名单**：不在表里的运算符类型直接抛 ValueError）；<br>5. 函数调用（查 funcs 白名单；`not node.keywords` 禁止关键字参数；参数递归 walk）。<br>任何其他 AST 节点类型（Attribute 链、Lambda、字符串……）→ ValueError。**这就是「不是 eval 而是解释自己认识的最小语法」** |
| `consts/funcs/bin_ops/unary_ops` | 四张白名单表。math 函数 27 个；乘方统一走 pow_guard |
| `chars = str.maketrans({...})` | **符号归一化**：模型常输出 `^`（Python 里是异或！）`×`、`÷`、全角减号、上标 ²³、全角括号——translate 一次性换成 Python 语法。`^→**` 尤其重要：否则 `2^3` 会被算成 2 XOR 3 = 1，奖励信号全错 |
| 203-204 | 空串/超 512 字符拒绝（防超长输入消耗解析器） |
| 206-208 | `ast.parse(mode='eval')` 语法解析 → walk 求值。三类异常统一包装成 ValueError（调用方 execute_tool 捕获后返回 error JSON 给模型） |

---

## 12. 本篇小结

1. `get_lr`：无 warmup 余弦退火（lr → 0.1·lr），每步手动写回。
2. DDP 判定靠**环境变量 RANK**；主进程过滤（Logger/is_main_process）保证日志与保存只发生一次。
3. `lm_checkpoint`：**双文件设计**（fp16 模型给推理、fp32 resume 给续训）+ 原子写入（.tmp+replace）+ 跨卡数 step 换算 + kwargs 任意状态扩展。
4. `SkipBatchSampler` 依赖「种子固定的洗牌顺序」实现精确跳批。
5. `safe_math_eval` 是**AST 白名单求值**的教科书实现：只放行白名单语法/函数、幂运算防爆、符号归一化——LLM 工具安全的标准姿势。

下一篇：[02-预训练-train_pretrain](./02-预训练-train_pretrain.md)
