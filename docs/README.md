# MiniMind 源码精读文档（docs）

> 本目录是对 MiniMind 仓库（minimind-3，2026-04 版本）的**全量源码讲解**。
> 原则：**仓库内每一行代码都有对应讲解**，每个知识点一个独立文档，先讲原理、再逐行读代码。
> 你可以完全按照本目录的文档学习，不需要再对照其他教程。

## 这个仓库是什么

MiniMind 用 **纯 PyTorch 从 0 实现**（不依赖 trl/peft 等高层封装）跑通了一个现代 LLM 的完整生命周期：

```
分词器训练 → 预训练(Pretrain) → 监督微调(SFT) → 偏好对齐(DPO) → 强化学习(PPO/GRPO/CISPO)
                ↓                                                        ↓
         知识蒸馏(可选)                                    Agentic RL(多轮工具调用)
                ↓
推理部署（OpenAI 兼容 API / WebUI / 命令行）→ 评测
```

主线模型结构对齐 **Qwen3 / Qwen3-MoE**：RMSNorm + RoPE(含 YaRN) + GQA + QK-Norm + SwiGLU + (MoE) + 权重绑定，总代码量约 4300 行。

## 推荐学习路线

| 阶段 | 文档 | 你将学会 |
|------|------|----------|
| ① 先建立全局感 | [00-项目总览](./00-项目总览.md) | 仓库结构、训练流水线、每个文件的职责 |
| ② 精读模型 | [01-模型架构](./01-模型架构/01-整体架构与Config.md) 开始，按编号顺序 | Transformer Decoder 全部组件的原理与实现 |
| ③ 理解数据 | [02-分词器](./02-分词器/01-BPE分词原理.md) → [03-数据集](./03-数据集/01-数据集总览.md) | BPE、chat template、各阶段数据格式 |
| ④ 精读训练 | [04-训练](./04-训练/01-训练基础设施-trainer_utils.md) 开始，按编号顺序 | 每个训练阶段的算法原理与工程细节 |
| ⑤ 推理部署 | [05-推理与部署](./05-推理与部署/01-命令行对话-eval_llm.md) 开始 | 解码、流式输出、API 协议、工具调用 |
| ⑥ 动手跑通 | 回到 00-项目总览 的「实操指南」 | 下载数据、训练、对话、评测 |

---

## 文档目录

### [00-项目总览.md](./00-项目总览.md)
仓库目录结构逐项说明、训练/推理全流程图、环境依赖（requirements.txt 逐项）、模型规格、实操指南。

### 01-模型架构/（核心：`model/model_minimind.py` 292 行 + `model/model_lora.py` 71 行）

| 文档 | 覆盖代码 | 知识点 |
|------|----------|--------|
| [01-整体架构与Config](./01-模型架构/01-整体架构与Config.md) | MiniMindConfig、MiniMindBlock、MiniMindModel、MiniMindForCausalLM（含 loss 计算、权重绑定、KV Cache 管理） | Decoder-only 架构总装、PreTrainedModel/GenerationMixin、causal LM 的 shift loss |
| [02-RMSNorm](./01-模型架构/02-RMSNorm.md) | RMSNorm 类（50-60 行） | LayerNorm vs RMSNorm、为什么去掉均值中心化 |
| [03-RoPE旋转位置编码](./01-模型架构/03-RoPE旋转位置编码.md) | precompute_freqs_cis、apply_rotary_pos_emb（62-84 行） | 绝对位置→相对位置的数学推导、复数视角、缓存表 |
| [04-注意力机制与GQA](./01-模型架构/04-注意力机制与GQA.md) | repeat_kv、Attention 类（86-134 行） | MHA/MQA/GQA、QK-Norm、Flash Attention 分支、causal mask 的 -seq_len 技巧 |
| [05-FFN与SwiGLU](./01-模型架构/05-FFN与SwiGLU.md) | FeedForward 类（136-146 行） | 为什么用 SwiGLU、intermediate_size 的 π 公式 |
| [06-MoE混合专家](./01-模型架构/06-MoE混合专家.md) | MOEFeedForward 类（148-181 行） | 路由、top-k 加权、负载均衡 aux loss、top-1 直通估计器 |
| [07-YaRN长文本外推](./01-模型架构/07-YaRN长文本外推.md) | rope_scaling 分支（64-73 行）、inference_rope_scaling 配置 | 位置编码外推问题、YaRN 插值公式 |
| [08-生成解码策略](./01-模型架构/08-生成解码策略.md) | generate() 方法（262-293 行） | KV Cache 增量解码、temperature/top-k/top-p/repetition penalty 的数学、流式 streamer |
| [09-LoRA原理与实现](./01-模型架构/09-LoRA原理与实现.md) | model_lora.py 全文 | 低秩分解、monkey-patch forward、保存/合并权重 |

### 02-分词器/

| 文档 | 覆盖代码 | 知识点 |
|------|----------|--------|
| [01-BPE分词原理](./02-分词器/01-BPE分词原理.md) | — | Byte-Pair Encoding、ByteLevel、词表为什么是 6400、special token 设计 |
| [02-训练分词器-逐行讲解](./02-分词器/02-训练分词器-逐行讲解.md) | trainer/train_tokenizer.py 全文 | BpeTrainer、chat_template 逐块精读、编码/解码一致性测试、流式字节缓冲 |

### 03-数据集/

| 文档 | 覆盖代码 | 知识点 |
|------|----------|--------|
| [01-数据集总览](./03-数据集/01-数据集总览.md) | dataset/dataset.md、各 jsonl 格式 | 每个阶段的训练数据长什么样、loss mask 图示 |
| [02-lm_dataset-逐行讲解](./03-数据集/02-lm_dataset-逐行讲解.md) | dataset/lm_dataset.py 全文 | 5 种 Dataset 类、system prompt 随机化、空思考标签处理、loss mask 生成算法 |

### 04-训练/

| 文档 | 覆盖代码 | 知识点 |
|------|----------|--------|
| [01-训练基础设施-trainer_utils](./04-训练/01-训练基础设施-trainer_utils.md) | trainer/trainer_utils.py 全文 | 余弦退火 LR、DDP 初始化、断点续训设计、SkipBatchSampler、reward 封装、safe_math_eval |
| [02-预训练-train_pretrain](./04-训练/02-预训练-train_pretrain.md) | trainer/train_pretrain.py 全文 | 预训练目标、混合精度 AMP/GradScaler、梯度累积、梯度裁剪、DDP |
| [03-监督微调-train_full_sft](./04-训练/03-监督微调-train_full_sft.md) | trainer/train_full_sft.py 全文 | SFT 只学 assistant 回复（loss mask）、与预训练的差异、超参选择 |
| [04-LoRA训练-train_lora](./04-训练/04-LoRA训练-train_lora.md) | trainer/train_lora.py 全文 | 冻结主干只训 LoRA、与 full SFT 的工程差异 |
| [05-DPO-直接偏好优化](./04-训练/05-DPO-直接偏好优化.md) | trainer/train_dpo.py 全文 | RLHF 背景、DPO 损失完整推导、参考模型 |
| [06-GRPO-组相对策略优化](./04-训练/06-GRPO-组相对策略优化.md) | trainer/train_grpo.py 全文 | GRPO 组归一化优势、k3 KL 估计、CISPO 变体、奖励塑形 |
| [07-PPO-近端策略优化](./04-训练/07-PPO-近端策略优化.md) | trainer/train_ppo.py 全文 | Actor-Critic、GAE、PPO clip、value clip、早停防 DDP 死锁 |
| [08-AgenticRL-智能体强化学习](./04-训练/08-AgenticRL-智能体强化学习.md) | trainer/train_agent.py 全文 | 多轮工具调用 rollout、observation token 屏蔽技巧、可验证奖励 |
| [09-知识蒸馏-train_distillation](./04-训练/09-知识蒸馏-train_distillation.md) | trainer/train_distillation.py 全文 | 温度蒸馏、KL 散度、alpha 混合损失 |
| [10-rollout引擎](./04-训练/10-rollout引擎.md) | trainer/rollout_engine.py 全文 | 训练/推理解耦、Torch 与 SGLang 双引擎、热更新权重 |

### 05-推理与部署/

| 文档 | 覆盖代码 | 知识点 |
|------|----------|--------|
| [01-命令行对话-eval_llm](./05-推理与部署/01-命令行对话-eval_llm.md) | eval_llm.py 全文 | 本地权重加载、chat template 拼接、流式打印、测速 |
| [02-OpenAI兼容API服务-serve_openai_api](./05-推理与部署/02-OpenAI兼容API服务-serve_openai_api.md) | scripts/serve_openai_api.py 全文 | FastAPI、SSE 流式协议、边生成边解析 think/tool_call 的状态机 |
| [03-OpenAI客户端-chat_api](./05-推理与部署/03-OpenAI客户端-chat_api.md) | scripts/chat_api.py 全文 | OpenAI SDK 用法、多轮历史、reasoning_content 流式 |
| [04-WebUI-web_demo](./05-推理与部署/04-WebUI-web_demo.md) | scripts/web_demo.py 全文 | Streamlit 运行模型、多轮工具调用、思考折叠 UI |
| [05-模型格式转换-convert_model](./05-推理与部署/05-模型格式转换-convert_model.md) | scripts/convert_model.py 全文 | .pth ↔ transformers ↔ Qwen3 格式、LoRA 合并、MoE 权重重排 |
| [06-工具调用评测-eval_toolcall](./05-推理与部署/06-工具调用评测-eval_toolcall.md) | scripts/eval_toolcall.py 全文 | 工具调用全链路、local/api 双后端、流式 tool_calls 累积 |

---

## 约定

- 所有代码引用都带**真实行号**（如 `model_minimind.py:126`），可直接在编辑器跳转。
- 「原理」部分用最少数学讲清为什么这样设计；「逐行精读」部分按代码顺序逐行/逐小段解释。
- 训练脚本之间有大量重复代码（pretrain/sft/lora 骨架相同），每篇文档仍会完整讲解自己的每一行，但共同背景（如混合精度）只在首见文档详细展开，后续文档标注「详见」链接。
- 代码块中用 `# ←` 标注讲解重点，正文再逐行展开。

## 环境准备一句话

```bash
pip install -r requirements.txt   # 关键版本：transformers==4.57.6、torch(自装)≥2.x、numpy==1.26.4
```

数据集不在 git 里，需按 [00-项目总览](./00-项目总览.md#数据集下载) 自行下载放入 `dataset/`。
