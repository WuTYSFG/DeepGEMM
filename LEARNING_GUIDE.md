# DeepGEMM 学习文档

> 适用版本: DeepGEMM v2.6.1（基于本地仓库 `d:\code\DeepGEMM`）
> 读者: 初学者
> 目标: 帮助理解 DeepGEMM 提供了哪些能力、每个算子的具体功能,以及如何阅读这份代码仓库

---

## 目录

1. [项目一句话简介](#1-项目一句话简介)
2. [你需要先了解的关键背景](#2-你需要先了解的关键背景)
3. [仓库目录结构总览](#3-仓库目录结构总览)
4. [DeepGEMM 提供的核心能力清单](#4-deepgemm-提供的核心能力清单)
5. [每个算子的详细说明](#5-每个算子的详细说明)
   - [5.1 Dense GEMMs（稠密矩阵乘）](#51-dense-gemms稠密矩阵乘)
   - [5.2 Grouped GEMMs（分组矩阵乘，专为 MoE 设计）](#52-grouped-gemms分组矩阵乘专为-moe-设计)
   - [5.3 Attention 算子（Lightning Indexer 的 MQA logits）](#53-attention-算子lightning-indexer-的-mqa-logits)
   - [5.4 Einsum 算子（特定形状的批量矩阵乘）](#54-einsum-算子特定形状的批量矩阵乘)
   - [5.5 HyperConnection（HC）算子](#55-hyperconnectionhc-算子)
   - [5.6 Mega MoE（融合通信+计算的 MoE mega-kernel）](#56-mega-moe融合通信计算的-moe-mega-kernel)
   - [5.7 Layout / Scaling Factor 变换工具](#57-layout--scaling-factor-变换工具)
   - [5.8 Runtime / 编译期配置](#58-runtime--编译期配置)
   - [5.9 cuBLASLt GEMM（兜底实现）](#59-cublaslt-gemm兜底实现)
   - [5.10 Legacy（面向 A100 的 Triton 旧实现）](#510-legacy面向-a100-的-triton-旧实现)
6. [工具函数与辅助能力](#6-工具函数与辅助能力)
7. [建议的学习路径](#7-建议的学习路径)
8. [关键术语速查表](#8-关键术语速查表)
9. [按通信/计算特性分类](#9-按通信计算特性分类)

---

## 1. 项目一句话简介

**DeepGEMM 是 DeepSeek 开源的高性能 CUDA Tensor Core 内核库**，专注为大语言模型中**最热点的几种算子**（矩阵乘、MoE、Attention 评分、HyperConnection 等）提供极致的 FP8/FP4/BF16 实现。

它有两个显著特点：

1. **运行时 JIT 编译** — 不需要安装时编译 CUDA，第一次调用某个 kernel 时才编译并缓存到 `~/.deep_gemm/`。
2. **极简但高效** — 借鉴了 NVIDIA CUTLASS / CuTe 的概念，但去掉了厚重模板，写法更易读。

代码本身几乎全部用 **CUDA + C++** 编写（不是 Triton），只在 `deep_gemm/legacy/` 下保留了一份面向老 A100 的 **Triton** 实现。

---

## 2. 你需要先了解的关键背景

> 如果这些概念第一次见，先收藏这个文档，回头再回头看会清晰很多。

### 2.1 GEMM（矩阵乘）

DeepGEMM 的核心就是各种形态的 GEMM：`D = C + A @ B`（或它的转置版本）。

- `A` 形状 `[M, K]`，`B` 形状 `[N, K]`，`D` 形状 `[M, N]`。
- `K` 是归约维度（`sum` 掉的那一维）。
- "NT" 命名指：`A` 不转置（`N`）、`B` 转置（`T`），即 `D = A @ B.T`。

### 2.2 FP8 / FP4 低精度

| 精度 | 类型 | DeepGEMM 支持 |
|------|------|--------------|
| FP32 | 全精度 | 仅作累加器 |
| BF16 | 16 位浮点 | 是 |
| FP8 (e4m3) | 8 位浮点 | 是 |
| FP4 (e2m1, packed) | 4 位浮点（两个打包在一个 int8） | 是（SM100） |

低精度意味着计算更快、显存更省，但**必须配缩放因子（Scaling Factor / SF）**才能控制精度损失。

### 2.3 Scaling Factor (SF)

为了让 FP8 / FP4 不丢精度，DeepGEMM 把 K 维分成很多小段，每段一个 SF：

- **recipe**: 三元组 `(gran_mn, gran_n, gran_k)`，例如 `(1, 1, 128)` 表示每 128 个 K 元素共享一个 SF。
- **SF 的存储格式**:
  - **SM90 (Hopper)**: FP32 标量。
  - **SM100 (Blackwell)**: **packed UE8M0**，每 4 个 UE8M0 打包进 1 个 `int32`。
- SF 必须是 **TMA 对齐** + **MN-major** 布局（接口会帮你做转换）。

### 2.4 NVIDIA 架构支持

| 架构 | 计算能力 | 支持情况 |
|------|---------|---------|
| SM90 (Hopper, H100/H200) | 9.0 | 完全支持（FP8、B组 GEMM、B组、BF16） |
| SM100 (Blackwell, B100/B200) | 10.0 | 完全支持（增加 FP4 / packed UE8M0 / tcgen05） |
| 其他 (如 A100 SM80) | - | 仅 legacy 中的 Triton 实现可用 |

### 2.5 几个关键缩写

| 缩写 | 含义 |
|------|------|
| MMA | Matrix Multiply Accumulate，Tensor Core 指令 |
| TMA | Tensor Memory Accelerator，Hopper 引入的异步加载硬件 |
| WGMMA | Warp-Group MMA，Hopper 上的大块矩阵乘指令 |
| tcgen05 | Blackwell 上的新一代张量核心指令 |
| EP | Expert Parallel，MoE 中的专家并行 |
| MQA | Multi-Query Attention，多查询注意力 |
| SwiGLU | 一种激活函数（SiLU + 门控） |
| PDL | Programmatic Dependent Launch，下一个 kernel 自动接上一个 |
| PSUM | Prefix SUM layout，按组前缀和布局 |

---

## 3. 仓库目录结构总览

```
DeepGEMM/
├── csrc/                       # C++/CUDA 源代码
│   ├── apis/                   # 对 Python 的 API 入口（按算子类别）
│   │   ├── attention.hpp       # 注意力（MQA logits）
│   │   ├── einsum.hpp          # 批量矩阵乘
│   │   ├── gemm.hpp            # 所有 GEMM
│   │   ├── hyperconnection.hpp # HC
│   │   ├── layout.hpp          # SF / 布局变换
│   │   ├── mega.hpp            # Mega MoE
│   │   └── runtime.hpp         # 运行时配置
│   ├── jit/                    # JIT 编译器基础设施
│   ├── jit_kernels/
│   │   ├── heuristics/         # kernel 启动参数搜索
│   │   └── impls/              # 真正的 CUDA 实现（每个 kernel 一个文件）
│   ├── utils/                  # 通用工具（异常、layout、math 等）
│   └── python_api.cpp          # pybind11 总入口
├── deep_gemm/                  # Python 包
│   ├── __init__.py             # 顶层导出
│   ├── include/deep_gemm/      # CUDA 头文件（.cuh）
│   ├── mega/                   # Mega MoE 的 Python 接口
│   ├── utils/                  # 量化 / 布局等 PyTorch 工具
│   ├── testing/                # 测试 / benchmark 工具
│   └── legacy/                 # A100 上的 Triton 旧实现
├── tests/                      # 各算子的测试 + 生成器
├── third-party/                # 第三方依赖（tilelang 等）
├── build.sh / develop.sh / install.sh
└── CMakeLists.txt / setup.py
```

> **理解思路**：Python 调用 → `csrc/apis/*.hpp` 注册的 pybind 接口 → `csrc/jit_kernels/impls/*.hpp` 的具体 kernel 实现 → `deep_gemm/include/deep_gemm/*.cuh` 的 CUDA 头文件。

---

## 4. DeepGEMM 提供的核心能力清单

| # | 能力 | 适用场景 | API 入口（Python 名） |
|---|------|----------|----------------------|
| 1 | **FP8/FP4 Dense GEMM** | 稠密 LLM 的线性层 | `fp8_gemm_nt` / `fp8_fp4_gemm_nt` 等 4 个变体 |
| 2 | **FP8/FP4 M-axis Grouped GEMM (contiguous)** | MoE 前向（token 数已知） | `m_grouped_fp8_gemm_nt_contiguous` |
| 3 | **FP8/FP4 M-axis Grouped GEMM (masked)** | MoE 推理（CUDA Graph 下 token 数不定） | `m_grouped_fp8_gemm_nt_masked` |
| 4 | **FP8/FP4 K-axis Grouped GEMM** | MoE 反向 / 权重量化 | `k_grouped_fp8_gemm_tn_contiguous` |
| 5 | **BF16 Dense GEMM** | 不需要量化的训练/推理 | `bf16_gemm_nt` 等 4 个变体 |
| 6 | **BF16 Grouped GEMM** | MoE 的 BF16 路径 | `m_grouped_bf16_gemm_*` |
| 7 | **FP8 MQA Logits (非分页)** | Prefill 阶段的 Lightning Indexer 打分 | `fp8_mqa_logits` / `fp8_fp4_mqa_logits` |
| 8 | **FP8 MQA Logits (分页)** | Decode 阶段（KV Cache 是分页的） | `fp8_paged_mqa_logits` / `fp8_fp4_paged_mqa_logits` |
| 9 | **GEMM + Skip/Mid Head 拼接** | 输出需要按 head 分段拼接的特殊 GEMM | `fp8_gemm_nt_skip_head_mid` |
| 10 | **Einsum: `bmk,bnk->mn`** | 跨 batch 归约的批量 GEMM | `einsum('bmk,bnk->mn', ...)` |
| 11 | **Einsum: `bhr,hdr->bhd`** | 头维度归约（attention out proj 等） | `einsum('bhr,hdr->bhd', ...)` |
| 12 | **Einsum: `bhd,hdr->bhr`** | 反向（attention 权重对 V） | `einsum('bhd,hdr->bhr', ...)` |
| 13 | **FP8 Batch MatMul (bmm)** | 批量 FP8 GEMM，可喂 FP8 einsum | `fp8_einsum('...')` |
| 14 | **TF32 HC Prenorm GEMM** | HyperConnection 层的 prenorm + GEMM 融合 | `tf32_hc_prenorm_gemm` |
| 15 | **Mega MoE（FP8×FP4）** | **MoE 全流程融合：dispatch + linear1 + SwiGLU + linear2 + combine** | `deep_gemm.fp8_fp4_mega_moe` |
| 16 | **Mega MoE（BF16）** | 不量化版本的 Mega MoE | `deep_gemm.bf16_mega_moe` |
| 17 | **Layout / SF 变换** | 把 SF 转成 kernel 想要的布局 | `transform_sf_into_required_layout` 等 |
| 18 | **运行时配置** | 调整 SM 数、TC 利用率、PDL 等 | `set_num_sms` / `set_pdl` 等 |
| 19 | **cuBLASLt GEMM（兜底）** | 当 DeepGEMM 不支持时用 NVIDIA 官方库 | `cublaslt_gemm_*` |
| 20 | **Legacy Triton GEMMs** | A100 上的旧实现（仅前向） | `from deep_gemm.legacy import ...` |

---

## 5. 每个算子的详细说明

下面按类别逐一展开。**重点关注：
- 它计算什么
- 输入/输出形状是什么
- 何时该用它
- 关键参数的意义**

### 5.1 Dense GEMMs（稠密矩阵乘）

代码入口: [gemm.hpp](file:///d:/code/DeepGEMM/csrc/apis/gemm.hpp)
实现:
- SM90: [sm90_fp8_gemm_1d1d.hpp](file:///d:/code/DeepGEMM/csrc/jit_kernels/impls/sm90_fp8_gemm_1d1d.hpp), [sm90_fp8_gemm_1d2d.hpp](file:///d:/code/DeepGEMM/csrc/jit_kernels/impls/sm90_fp8_gemm_1d2d.hpp), [sm90_bf16_gemm.hpp](file:///d:/code/DeepGEMM/csrc/jit_kernels/impls/sm90_bf16_gemm.hpp)
- SM100: [sm100_fp8_fp4_gemm_1d1d.hpp](file:///d:/code/DeepGEMM/csrc/jit_kernels/impls/sm100_fp8_fp4_gemm_1d1d.hpp), [sm100_bf16_gemm.hpp](file:///d:/code/DeepGEMM/csrc/jit_kernels/impls/sm100_bf16_gemm.hpp)

#### 5.1.1 `fp8_gemm_*` / `fp8_fp4_gemm_*`

```python
deep_gemm.fp8_gemm_nt((a_fp8, a_sf), (b_fp8, b_sf), d, c=None, recipe=None, ...)
```

- **做什么**: 计算 `D = C + A @ B.T`，其中 A 是 FP8 或 FP4，B 是 FP8 或 FP4，C/D 是 BF16 或 FP32。
- **API 后缀说明**:
  - `nt` = A 不转置、B 转置
  - `nn` / `tn` / `tt` 是另外 3 个转置组合（接口内部自动转）
- **适用场景**: 任何稠密线性层（Q/K/V projection、output projection、MLP 等）。
- **关键参数**:
  - `a_sf` / `b_sf`: 缩放因子（per-token/per-block/per-channel）。
  - `recipe`: `(gran_mn, gran_n, gran_k)` 三元组，控制 SF 粒度。
  - `compiled_dims`: 哪些维度作为编译参数（默认 `nk`，即 M 维度变化不重新编译）。
  - `disable_ue8m0_cast`: 强制把 SF 当作 FP32 处理（SM100 上方便调试）。

> 💡 **初学者注意**: `fp8_gemm_nt` 和 `fp8_fp4_gemm_nt` 是同一个函数（fp8_gemm_* 是 fp8_fp4_gemm_* 的别名，源代码 [gemm.hpp:711](file:///d:/code/DeepGEMM/csrc/apis/gemm.hpp#L711-L717)）。

#### 5.1.2 `bf16_gemm_*`

```python
deep_gemm.bf16_gemm_nt(a, b, d)
```

- **做什么**: 完全相同的接口语义，但 a/b 都是 BF16，没有 SF。
- **适用场景**: 不需要量化的训练（如 BF16 baseline）。
- SM90 与 SM100 都支持 NT/TN/NN/TT 4 种布局。

### 5.2 Grouped GEMMs（分组矩阵乘，专为 MoE 设计）

> 这是 DeepGEMM **最核心、最独特**的能力。
> 区别于 CUTLASS 的标准 grouped GEMM，DeepGEMM **只对 M 轴分组**（N、K 保持相同），这正好契合 MoE 模型 "每个专家形状相同、只是 token 数量不同" 的特点。

#### 5.2.1 `m_grouped_fp8_gemm_nt_contiguous`

代码: [gemm.hpp:166](file:///d:/code/DeepGEMM/csrc/apis/gemm.hpp#L166-L232)

```python
deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
    (a_fp8, a_sf),        # [M, K] 所有专家的 token 拼接在一起
    (b_fp8, b_sf),        # [G, N, K] G 个专家的权重
    d,                    # [M, N] 输出
    grouped_layout,       # [M] 每个 token 属于哪个专家，或 [G] PSUM 偏移
    recipe=None,
    use_psum_layout=False,
    ensure_zero_padding=True,
    expected_m_for_psum_layout=None,
)
```

- **做什么**: `D = A @ B[g].T for each token g`，把 G 组结果写回对应位置。
- **何时用**:
  - **训练前向 / 推理 prefill**: 调度器已经知道每个专家分到多少 token，用 **contiguous（拼接）布局**最简单、最高效。
- **关键参数**:
  - `grouped_layout`: 两种模式
    - **Masked 模式（默认）**: `[M]` 的 int32，每个位置写它属于哪个专家；`-1` 表示 padding（必须 zero）。
    - **PSUM 模式**: `[G]` 的 int32，每组 token 的前缀和终止偏移（更省 padding 写入）。
  - `use_psum_layout`: 切到 PSUM 模式（性能更好但需要预先知道每组大小）。
  - `ensure_zero_padding`: padding 行是否保证被清零（影响 SFA 的打包）。

#### 5.2.2 `m_grouped_fp8_gemm_nt_masked`

代码: [gemm.hpp:250](file:///d:/code/DeepGEMM/csrc/apis/gemm.hpp#L250-L297)

```python
deep_gemm.m_grouped_fp8_gemm_nt_masked(
    (a_fp8, a_sf),     # [G, max_M, K] 每个专家有自己的 token 块
    (b_fp8, b_sf),     # [G, N, K]
    d,                 # [G, max_M, N]
    masked_m,          # [G] 每个专家的实际 token 数
    expected_m,        # 预分配的最大 token 数（编译时常量）
    recipe=None,
)
```

- **何时用**:
  - **推理 decode 阶段 + CUDA Graph**: CPU 不知道每个专家实际收到的 token 数。每行都有一个 `masked_m[g]`，kernel 只算前 `masked_m[g]` 行。
  - 典型输入源是 [DeepEP](https://github.com/deepseek-ai/DeepEP) 的 low-latency kernel 输出。
- **关键参数**:
  - `expected_m`: 必须传入，作为**编译时常量**。改一次就要重新编译 kernel。
  - `masked_m`: 每组的真实 token 数。

#### 5.2.3 `k_grouped_fp8_gemm_tn_contiguous` / `k_grouped_fp8_gemm_nt_contiguous`

代码: [gemm.hpp:299](file:///d:/code/DeepGEMM/csrc/apis/gemm.hpp#L299-L400)

```python
deep_gemm.k_grouped_fp8_gemm_tn_contiguous(
    (a_fp8, a_sf),         # [sum_k, M] 沿 K 拼接所有专家的输入
    (b_fp8, b_sf),         # [sum_k, N]
    d,                     # [G, M, N]  输出每个专家一份
    ks_cpu,                # [G] 每组 K 大小（host 上）
    grouped_layout,        # [G] 前缀和 或 各组的实际 K
    c=None,
    recipe=(1, 1, 128),
    use_psum_layout=False,
)
```

- **做什么**: 把不同专家按 K 维度拼接，输出拆回每个专家一份。即 `d[g] = c[g] + a[start:end].T @ b[start:end]`。
- **何时用**: **MoE 反向传播的 weight gradient**（每个专家的 K 对应不同 token 数）。
- **变体**:
  - `_tn_contiguous`: SM100 上使用，MN-major 布局 + 支持 PSUM。
  - `_nt_contiguous`: SM90 上使用，K-major 布局（仅支持 contiguous，不支持 psum）。

#### 5.2.4 BF16 版本

`m_grouped_bf16_gemm_*` / `k_grouped_bf16_gemm_tn_contiguous` 与上面 FP8 版完全对应，但输入是 BF16（无 SF），由 [gemm.hpp:464](file:///d:/code/DeepGEMM/csrc/apis/gemm.hpp#L464-L608) 实现。

### 5.3 Attention 算子（Lightning Indexer 的 MQA logits）

代码入口: [attention.hpp](file:///d:/code/DeepGEMM/csrc/apis/attention.hpp)
测试: [test_attention.py](file:///d:/code/DeepGEMM/tests/test_attention.py)

这是 DeepSeek V3.2 引入的 **Lightning Indexer** 算子：
给定 query `q`（多个 head）、共享 KV（单 head）、每个 head 的 `weights`，
输出每个 query 对每个 KV 位置的"重要性分数"，公式：

```
kv_j = kv[j, :] * kv_sf[j]                # [head_dim]
out_ij = q[i, :, :] @ kv_j                # [num_heads]
out_ij = relu(out_ij) * weights[i, :]     # [num_heads]
out_ij = sum(out_ij)                      # scalar
```

#### 5.3.1 `fp8_mqa_logits` / `fp8_fp4_mqa_logits`（非分页）

代码: [attention.hpp:76](file:///d:/code/DeepGEMM/csrc/apis/attention.hpp#L76-L182)

```python
deep_gemm.fp8_fp4_mqa_logits(
    q=(q_fp8, q_sf),                   # [seq_len, num_heads, head_dim]
    kv=(kv_fp8, kv_sf),                # [seq_len_kv, head_dim]
    weights=weights,                   # [seq_len, num_heads]
    cu_seq_len_k_start=cu_start,       # [seq_len]  每个 q 看 KV 的起点
    cu_seq_len_k_end=cu_end,           # [seq_len]  每个 q 看 KV 的终点
    clean_logits=True,
    logits_dtype=torch.float32,
)
# 返回 [seq_len, seq_len_kv] 的 logits
```

- **何时用**: **Prefill 阶段**，整个序列已知，KV 连续存储。
- `cu_seq_len_k_*` 描述 prefix caching / chunked prefill 等场景下每个 q 实际看到的 KV 区间。
- `clean_logits=True`: 把区间外的位置填成 `-inf`（便于后面 softmax）。

#### 5.3.2 `fp8_paged_mqa_logits` / `fp8_fp4_paged_mqa_logits`（分页）

代码: [attention.hpp:219](file:///d:/code/DeepGEMM/csrc/apis/attention.hpp#L219-L361)

```python
deep_gemm.fp8_fp4_paged_mqa_logits(
    q=(q_fp8, q_sf),             # [batch, next_n, num_heads, head_dim]
    kv_cache=fused_kv_cache,     # [num_kv_blocks, block_kv, 1, head_dim+sf]
    weights=weights,             # [batch*next_n, num_heads]
    context_lens=context_lens,   # [batch, next_n]  实际 KV 长度
    block_table=block_table,     # [batch, max_blocks]  KV 物理块索引
    schedule_meta=schedule_meta, # 预计算的 SM 调度信息
    max_context_len=...,
)
```

- **何时用**: **Decode 阶段**，KV 缓存在分页 KV cache 里（每个请求长度不同，且物理上不连续）。
- 需要先用 `get_paged_mqa_logits_metadata()` 预算一份 `schedule_meta`，告诉 kernel 怎么按 SM 切分。

#### 5.3.3 `fp8_gemm_nt_skip_head_mid`

代码: [attention.hpp:19](file:///d:/code/DeepGEMM/csrc/apis/attention.hpp#L19-L74)

```python
deep_gemm.fp8_gemm_nt_skip_head_mid(
    (a_fp8, a_sf), (b_fp8, b_sf), d,
    head_splits=(left, mid, right),   # 例 (128, 64, 128)
    recipe=(...),
)
```

- **做什么**: 标准 FP8 GEMM，但 epilogue 时把输出按 head 分组：在 head 序列中**每 `left+right` 个 head 之间插入 `mid` 个零 head**。
- **何时用**: DeepSeek V3.2 Indexer 的特殊结构（head 维度要拼出中段 head 用作 placeholder）。
- 输出 N 实际是 `(left+right) * num_heads + mid * num_heads = num_heads * (left+right+mid)`。

### 5.4 Einsum 算子（特定形状的批量矩阵乘）

代码入口: [einsum.hpp](file:///d:/code/DeepGEMM/csrc/apis/einsum.hpp)

这些是**针对 LLM 中反复出现的高频 einsum 模式**专门写的硬编码 kernel（不是通用 einsum 编译器）。

#### 5.4.1 `einsum('bmk,bnk->mn', a, b, d)`

代码: [einsum.hpp:23](file:///d:/code/DeepGEMM/csrc/apis/einsum.hpp#L23-L59)

```python
deep_gemm.einsum('bmk,bnk->mn', a, b, d)  # a, b: BF16; d: BF16 或 FP32
```

- **做什么**: `d = sum_b (a[b] @ b[b].T)`，跨 batch 维度求和。
- **何时用**: Attention 输出投影前的"先聚合多头"，或任何"先按 batch 算 GEMM 再求和"的模式。
- 实现在 [sm90_bmk_bnk_mn.hpp](file:///d:/code/DeepGEMM/csrc/jit_kernels/impls/sm90_bmk_bnk_mn.hpp) / [sm100_bmk_bnk_mn.hpp](file:///d:/code/DeepGEMM/csrc/jit_kernels/impls/sm100_bmk_bnk_mn.hpp)。

#### 5.4.2 `einsum('bhr,hdr->bhd', ...)`

代码: [einsum.hpp:61](file:///d:/code/DeepGEMM/csrc/apis/einsum.hpp#L61-L82)

```python
deep_gemm.einsum('bhr,hdr->bhd', x, y, z)  # 通用: d[b,h,d] = sum_r x[b,h,r] * y[h,d,r]
```

- **做什么**: 头维共享的批量 GEMM——`y` 在 head 维只有一个（broadcast），多个 query head 共用一套 K。
- **何时用**: Attention 的 **value projection**（V 矩阵被多个 head 共享时的反向）。
- 也可走 `use_cublaslt=True` 路径用 cuBLASLt 实现。

#### 5.4.3 `einsum('bhd,hdr->bhr', ...)`

代码: [einsum.hpp:84](file:///d:/code/DeepGEMM/csrc/apis/einsum.hpp#L84-L105)

- 类似上一个，是它的转置版本。

#### 5.4.4 `fp8_einsum(...)` — FP8 版本的批量 GEMM

代码: [einsum.hpp:179](file:///d:/code/DeepGEMM/csrc/apis/einsum.hpp#L179-L214)

底层都走 `fp8_bmm`：
```python
deep_gemm.fp8_einsum('bhr,hdr->bhd', (a, a_sf), (b, b_sf), d, recipe=(1, 128, 128))
deep_gemm.fp8_einsum('bhd,hdr->bhr', ...)  # SM100 only
deep_gemm.fp8_einsum('bhd,bhr->hdr', ...)  # SM100 only
```

- 实现在 [sm100_fp8_fp4_gemm_1d1d.hpp](file:///d:/code/DeepGEMM/csrc/jit_kernels/impls/sm100_fp8_fp4_gemm_1d1d.hpp) + [sm90_fp8_gemm_1d1d.hpp](file:///d:/code/DeepGEMM/csrc/jit_kernels/impls/sm90_fp8_gemm_1d1d.hpp)。

### 5.5 HyperConnection（HC）算子

代码入口: [hyperconnection.hpp](file:///d:/code/DeepGEMM/csrc/apis/hyperconnection.hpp)
实现:
- [sm90_tf32_hc_prenorm_gemm.hpp](file:///d:/code/DeepGEMM/csrc/jit_kernels/impls/sm90_tf32_hc_prenorm_gemm.hpp)
- [sm100_tf32_hc_prenorm_gemm.hpp](file:///d:/code/DeepGEMM/csrc/jit_kernels/impls/sm100_tf32_hc_prenorm_gemm.hpp)

#### 5.5.1 `tf32_hc_prenorm_gemm`

```python
deep_gemm.tf32_hc_prenorm_gemm(
    a,                # [M, K] BF16
    b,                # [N, K] FP32
    d,                # [M, N] FP32 (或 [num_splits, M, N])
    sqr_sum,          # [M]    (或 [num_splits, M])  每行 x 的平方和
    num_splits=None,
)
```

- **做什么**: 一次调用同时算出：
  - `d = a @ b.T`（TF32 精度 GEMM）
  - `sqr_sum[m] = sum_k a[m, k] ** 2`
- **何时用**: HyperConnection 层里，**GEMM 和 RMSNorm / prenorm 的方差**可以融合计算（共享一遍 a 的读）。
- `num_splits`: K 维切多份并行算（适合超大 K），结果累加。

### 5.6 Mega MoE（融合通信+计算的 MoE mega-kernel）

> **DeepGEMM 2026.04 版本的旗舰特性**。这是仓库里最复杂的一个算子。

代码入口: [mega.hpp](file:///d:/code/DeepGEMM/csrc/apis/mega.hpp)
Python 包装: [mega/__init__.py](file:///d:/code/DeepGEMM/deep_gemm/mega/__init__.py)
测试: [test_mega_moe.py](file:///d:/code/DeepGEMM/tests/test_mega_moe.py)
CUDA 实现:
- [sm100_fp8_fp4_mega_moe.cuh](file:///d:/code/DeepGEMM/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh)
- [sm100_bf16_mega_moe.cuh](file:///d:/code/DeepGEMM/deep_gemm/include/deep_gemm/impls/sm100_bf16_mega_moe.cuh)
- [scheduler/mega_moe.cuh](file:///d:/code/DeepGEMM/deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh)
- [layout/mega_moe.cuh](file:///d:/code/DeepGEMM/deep_gemm/include/deep_gemm/layout/mega_moe.cuh)
- [comm/barrier.cuh](file:///d:/code/DeepGEMM/deep_gemm/include/deep_gemm/comm/barrier.cuh)

#### 5.6.1 `fp8_fp4_mega_moe`

```python
# 1. 分配"对称内存"（PyTorch >= 2.9）
buffer = deep_gemm.get_symm_buffer_for_mega_moe(
    group, num_experts,
    num_max_tokens_per_rank, num_topk,
    hidden, intermediate_hidden,
    num_shared_experts=0,      # 可选：DeepSeek-V3 风格的 shared expert
    mma_type='fp8xfp4',        # 或 'bf16xbf16'
    activation='swiglu',
)

# 2. 转换权重（仅一次，离线做）
l1, l2 = deep_gemm.transform_weights_for_mega_moe(
    (l1_weights_fp4, l1_sf),   # gate + up 拼接的 FP4
    (l2_weights_fp4, l2_sf),
)

# 3. 把输入拷进对称 buffer
buffer.x[:num_tokens].copy_(x_fp8)
buffer.x_sf[:num_tokens].copy_(x_sf)
buffer.topk_idx[:num_tokens].copy_(topk_idx)
buffer.topk_weights[:num_tokens].copy_(topk_weights)

# 4. 调用融合 mega-kernel
y = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
deep_gemm.fp8_fp4_mega_moe(
    y, l1, l2, buffer,
    recipe=(1, 1, 32),
    activation='swiglu',
    activation_clamp=None,
    fast_math=True,
)
```

- **做了什么**（全部塞进 **一个 kernel**）：
  1. **EP dispatch** — 跨 rank 把 token 按 topk 派发到对应专家所在的 GPU（NVLink）。
  2. **Linear 1 (FP8 × FP4)** — 每个专家的 gate + up projection 一起算（输出 `2 * intermediate`）。
  3. **SwiGLU 激活** — 在 MMA 之后立即应用 `silu(gate) * up`。
  4. **Linear 2 (FP8 × FP4)** — 每个专家的 down projection。
  5. **EP combine** — 把每个 topk 的结果按权重累加回原始 token 位置（NVLink）。
- **关键设计**:
  - 通信和计算**完全重叠**：dispatch 时就在算其他 rank 的 GEMM；combine 时一边收一边累加。
  - 使用 **PyTorch 2.9 对称内存** + **NVSHMEM 风格的 barrier**，所有 rank 同步触发。
  - 使用 **FP8 × FP4**：激活是 FP8，权重是 FP4（最激进的量化），SF 用 packed UE8M0。
  - 依赖 **SM100**（tcgen05 指令）。SM90 不支持。
- **何时用**: 生产级 MoE 推理的最快路径之一（DeepSeek-V3 风格）。

#### 5.6.2 `bf16_mega_moe`

- 与上面相同路径，但权重和激活都是 BF16。
- 调试 / 校验数值正确性时用。

#### 5.6.3 与 DeepEP 的关系

- **DeepEP**: 单独的项目，专门做 MoE 的 dispatch / combine 通信原语（intra-node + inter-node）。
- **Mega MoE**: 把通信**与 GEMM 融合**成一个 kernel，不需要单独调 DeepEP。
- 两者是同一思路下的两种实现粒度选择。

### 5.7 Layout / Scaling Factor 变换工具

代码入口: [layout.hpp](file:///d:/code/DeepGEMM/csrc/apis/layout.hpp)

这一组工具把用户传入的 SF / 矩阵布局**自动转成 kernel 需要的格式**，省去调用方手写转换。

| 函数 | 功能 |
|------|------|
| `transform_sf_into_required_layout(sf, mn, k, recipe, ...)` | 把 SF 转成 TMA 对齐 + MN-major |
| `get_mn_major_tma_aligned_tensor(sf)` | 仅做 MN-major + TMA 对齐（SM90 FP32 路径） |
| `get_mn_major_tma_aligned_packed_ue8m0_tensor(sf)` | FP32 SF → packed UE8M0 (SM100) |
| `get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor(...)` | K-grouped 版本的 SF packing |
| `get_tma_aligned_size(size)` | 计算 TMA 对齐后的字节数 |
| `set_mk_alignment_for_contiguous_layout` / `get_...` | 控制 contiguous 布局的对齐粒度 |
| `get_theoretical_mk_alignment_for_contiguous_layout` | 理论最小对齐（用于性能估计） |

> 💡 一般用户**不需要直接调用**这些函数——GEMM / Mega MoE 的入口会自动帮你做。  
> 但如果你自己写更底层的 kernel，就要用它们。

### 5.8 Runtime / 编译期配置

代码入口: [runtime.hpp](file:///d:/code/DeepGEMM/csrc/apis/runtime.hpp)

| 函数 | 功能 |
|------|------|
| `set_num_sms(n)` / `get_num_sms()` | 限制 kernel 使用的 SM 数量（用于多 GPU 共享、抢占调度） |
| `set_tc_util(ratio)` / `get_tc_util()` | Tensor Core 利用率（小于 1 让部分 SM 空闲，给通信留资源） |
| `set_pdl(bool)` / `get_pdl()` | 开关 Programmatic Dependent Launch（节省调度开销） |
| `set_ignore_compile_dims(bool)` | 编译时忽略某些维度 → 减少 kernel 数量 |
| `set_block_size_multiple_of(...)` | 强制 block 大小取某个倍数 |
| `init(library_root, cuda_home)` | 初始化 JIT 编译器（安装时由 `__init__.py` 自动调用） |

### 5.9 cuBLASLt GEMM（兜底实现）

代码: [smxx_cublaslt.hpp](file:///d:/code/DeepGEMM/csrc/jit_kernels/impls/smxx_cublaslt.hpp)

```python
deep_gemm.cublaslt_gemm_nt(a, b, d)
deep_gemm.cublaslt_gemm_nn(a, b, d)
deep_gemm.cublaslt_gemm_tn(a, b, d)
deep_gemm.cublaslt_gemm_tt(a, b, d)
```

- 直接调用 NVIDIA 的 **cuBLASLt**。
- **何时用**: DeepGEMM 自己的 kernel 不支持的特殊 shape / dtype 时，作为兜底。
- 性能一般不如 DeepGEMM 自家实现，但稳定性有保障。

### 5.10 Legacy（面向 A100 的 Triton 旧实现）

代码: [legacy/](file:///d:/code/DeepGEMM/deep_gemm/legacy/)

```python
from deep_gemm import legacy
legacy.a_fused_k_grouped_gemm(...)
legacy.a_fused_m_grouped_gemm(...)
legacy.b_fused_k_grouped_gemm(...)
legacy.m_grouped_gemm(...)
```

- **仅在 A100（SM80）上加载**——H100/H200/B100+ 不需要它。
- 用 **Triton** 编写（前 DeepGEMM 时代的主路径），H100 上性能不如 CUDA 实现。
- 现已退居"legacy"，但保留了向后兼容接口。
- 测试: [test_legacy.py](file:///d:/code/DeepGEMM/tests/test_legacy.py)

---

## 6. 工具函数与辅助能力

### 6.1 PyTorch 侧的量化工具

代码: [deep_gemm/utils/layout.py](file:///d:/code/DeepGEMM/deep_gemm/utils/layout.py) + [utils/math.py](file:///d:/code/DeepGEMM/deep_gemm/utils/math.py)

| 函数 | 用途 |
|------|------|
| `per_token_cast_to_fp8(x, use_ue8m0, gran_k)` | 把 BF16 转 FP8（per-token，按 K 段 gran_k 分组 SF） |
| `per_channel_cast_to_fp8(x, ...)` | Per-channel 量化 |
| `per_block_cast_to_fp8(x, ...)` | Block-wise 量化（DeepSeek-V3 风格） |
| `per_token_cast_to_fp4(x, ...)` | 转 FP4（packed int8） |
| `transpose_packed_fp4(...)` | FP4 packed 数据的转置 |
| `per_custom_dims_cast_to_fp8(x, ...)` | 自定义维度的量化 |
| `cast_back_from_fp8 / fp4` | 反量化回来对比数值 |
| `align(x, n)` / `ceil_div(x, n)` | 对齐 / 向上取整 |

### 6.2 分布式 / 多进程工具

代码: [utils/dist.py](file:///d:/code/DeepGEMM/deep_gemm/utils/dist.py)

- `init_dist()`: 封装 `torch.distributed.init_process_group`。
- `uneven_all_gather(...)`: 不均匀张量的 all-gather（用于 MoE 的 dispatch）。
- `dist_print(...)`: 分布式环境下只在 rank 0 打印。

### 6.3 测试与 Benchmark 工具

代码: [testing/](file:///d:/code/DeepGEMM/deep_gemm/testing/__init__.py)

- `bench_kineto(...)`: 用 NVIDIA Kineto 打点测时。
- `calc_diff(a, b)`: 数值误差检查。
- `assert_bitwise_equal(...)`: 严格逐 bit 对比。
- `count_bytes(...)`: 算带宽用。
- `get_arch_major()`: 当前 GPU 是 SM 几。
- `test_filter(...)`: 装饰器，根据 GPU 架构跳过不兼容的测试。

### 6.4 重要环境变量

| 变量 | 作用 |
|------|------|
| `DG_JIT_DEBUG=1` | 打印 JIT 调试信息 |
| `DG_PRINT_CONFIGS=1` | 打印每次 kernel 选用的 block size 配置 |
| `DG_JIT_CACHE_DIR` | 编译产物缓存目录（默认 `~/.deep_gemm`） |
| `DG_JIT_USE_NVRTC=1` | 用 NVRTC 代替 NVCC（更快编译，性能可能略差） |
| `DG_JIT_NVCC_COMPILER` | 指定 nvcc 路径 |
| `DG_JIT_PRINT_COMPILER_COMMAND=1` | 打印完整编译命令 |
| `DG_JIT_PTXAS_VERBOSE=1` | 看 PTXAS 输出（分析寄存器使用） |
| `DG_JIT_DUMP_ASM=1` | 同时 dump PTX 和 SASS |
| `DG_JIT_WITH_LINEINFO=1` | 保留行号（便于 ncu / nsys 分析） |
| `DG_COMM_KERNEL_DEBUG=1` | Mega MoE: 每次调用前清零对称 buffer（调试用） |
| `DG_USE_NVIDIA_TOOLS=1` | 在 nsys / ncu 下运行，跳过内部 profiling |

---

## 7. 建议的学习路径

按下面这个顺序读，会事半功倍：

### 第 1 阶段：理解项目定位（半小时）
1. 通读 [README.md](file:///d:/code/DeepGEMM/README.md)。
2. 看 [deep_gemm/__init__.py](file:///d:/code/DeepGEMM/deep_gemm/__init__.py) —— 看所有对外导出的 API 名字。

### 第 2 阶段：从最简单的 Dense GEMM 开始（一小时）
1. 看 [gemm.hpp](file:///d:/code/DeepGEMM/csrc/apis/gemm.hpp) 的 `fp8_gemm_nt` 函数签名。
2. 看 [tests/test_fp8_fp4.py](file:///d:/code/DeepGEMM/tests/test_fp8_fp4.py)（如果存在），跑一次 `python tests/test_fp8_fp4.py`。
3. 看 [csrc/jit_kernels/heuristics/](file:///d:/code/DeepGEMM/csrc/jit_kernels/heuristics/) —— 理解 DeepGEMM 如何选 block size。

### 第 3 阶段：理解 Grouped GEMM（一小时）
1. 读 [m_grouped_gemm.py](file:///d:/code/DeepGEMM/deep_gemm/legacy/m_grouped_gemm.py) —— legacy 版本最容易理解。
2. 对比 `m_grouped_fp8_gemm_nt_contiguous` 和 `m_grouped_fp8_gemm_nt_masked` 的区别。
3. 跑 `tests/test_*.py` 看具体 demo。

### 第 4 阶段：理解 JIT 编译机制（半小时）
1. 看 [csrc/jit/](file:///d:/code/DeepGEMM/csrc/jit/) 下的 `compiler.hpp`、`kernel_runtime.hpp`、`cache.hpp`。
2. 设置 `DG_JIT_DEBUG=1` + `DG_PRINT_CONFIGS=1` 跑一次，看实际选了哪些配置。

### 第 5 阶段：深入 Mega MoE（最复杂，先跳过也行）
1. 读 README 中的 "Mega MoE" 一节 + Python [mega/__init__.py](file:///d:/code/DeepGEMM/deep_gemm/mega/__init__.py)。
2. 跑 [tests/test_mega_moe.py](file:///d:/code/DeepGEMM/tests/test_mega_moe.py)（需要多 GPU）。
3. 跟着 [scheduler/mega_moe.cuh](file:///d:/code/DeepGEMM/deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh) 看调度逻辑。

### 第 6 阶段：精读最优 kernel
1. 看 [sm90_fp8_gemm_1d1d.hpp](file:///d:/code/DeepGEMM/csrc/jit_kernels/impls/sm90_fp8_gemm_1d1d.hpp) —— 主体结构清晰。
2. 学 PTX 指令（`wgmma`、`cp.async.bulk` = TMA）。
3. 看 [sm100_fp8_fp4_gemm_1d1d.hpp](file:///d:/code/DeepGEMM/csrc/jit_kernels/impls/sm100_fp8_fp4_gemm_1d1d.hpp) —— Blackwell 怎么用 `tcgen05`。

---

## 8. 关键术语速查表

| 术语 | 含义 |
|------|------|
| **GEMM** | 通用矩阵乘（General Matrix Multiply） |
| **NT / TN / NN / TT** | A、B 是否转置（A 不转=N，转置=T） |
| **FP8 / FP4** | 8-bit / 4-bit 浮点 |
| **e4m3 / e5m2** | FP8 的两种编码格式（4 位指数+3 位尾数 / 5+2） |
| **e2m1** | FP4 的编码（2 位指数+1 位尾数） |
| **Scaling Factor (SF)** | 缩放因子，用于恢复低精度损失的精度 |
| **recipe** | `(gran_mn, gran_n, gran_k)` 元组，描述 SF 分组粒度 |
| **UE8M0** | 8-bit 无符号指数（用作 SF 的最简格式） |
| **packed UE8M0** | 4 个 UE8M0 打包进 1 个 int32（SM100 的存储格式） |
| **TMA** | Tensor Memory Accelerator，Hopper 引入的硬件加速器 |
| **WGMMA** | Warp-Group MMA，Hopper 上的大块矩阵乘指令 |
| **tcgen05** | Blackwell 上的新一代 MMA 指令 |
| **SM90 / SM100** | NVIDIA 计算能力 9.0（Hopper）/ 10.0（Blackwell） |
| **JIT** | Just-In-Time，运行时编译 |
| **NVRTC** | NVIDIA Runtime Compilation，C++ API 的运行时编译入口 |
| **NVCC** | NVIDIA 官方 CUDA 编译器 |
| **PTX / SASS** | NVIDIA 虚拟 ISA / 真实机器码 |
| **MMA / MGrouped / KGrouped** | M 轴分组 / K 轴分组的 GEMM |
| **Contiguous / Masked / PSUM** | Grouped GEMM 的三种 token 布局 |
| **Top-K** | MoE 中每个 token 选 K 个专家 |
| **EP** | Expert Parallel，专家并行（不同专家可分布在不同 GPU） |
| **MQA** | Multi-Query Attention，多 query head 共享 KV |
| **Lightning Indexer** | DeepSeek V3.2 的"轻量索引器"，用 MQA 方式筛 KV |
| **HyperConnection (HC)** | 一种残差/连接机制（见 DeepSeek HC 论文） |
| **Prenorm** | 在 norm 之前做 GEMM（与 norm 融合） |
| **Mega MoE** | DeepGEMM 中"把 MoE 全部融合成一个 kernel"的优化 |
| **SwiGLU** | Swish + GLU 组合，`silu(gate) * up` 激活函数 |
| **PDL** | Programmatic Dependent Launch，让连续 kernel 调度开销消失 |
| **Symmetric Memory** | PyTorch 2.9+ 的对称内存，多 GPU 共享一段地址空间 |
| **NVLink** | NVIDIA GPU 之间的高速直连总线 |
| **cuBLASLt** | NVIDIA 官方的优化 BLAS 库 |

---

## 9. 按通信/计算特性分类

把 DeepGEMM 的所有算子按"**是否涉及跨 rank 通信**"重新组织，结论很清晰：

> **19 个 kernel 是纯计算，2 个 kernel 是通信+计算融合。**

### 🟢 9.1 纯计算类（绝大多数算子）

这些 kernel **完全不碰 NVLink / NCCL / 对称内存**，只读 HBM / shared memory / register：

| 类别 | 算子 | 备注 |
|------|------|------|
| Dense GEMM | `fp8_gemm_*` / `fp8_fp4_gemm_*`<br>`bf16_gemm_*` | 稠密线性层 |
| MoE Grouped GEMM | `m_grouped_fp8_gemm_*_contiguous`<br>`m_grouped_fp8_gemm_*_masked`<br>`k_grouped_fp8_gemm_*_contiguous` | 假设 token **已被外部调度器派发好**，本 kernel 只做本地计算 |
| BF16 Grouped GEMM | `m_grouped_bf16_gemm_*`<br>`k_grouped_bf16_gemm_*_contiguous` | 同上 |
| 注意力 | `fp8_mqa_logits`<br>`fp8_paged_mqa_logits`<br>`fp8_fp4_mqa_logits` / `fp8_fp4_paged_mqa_logits` | 单卡内的 Indexer 打分 |
| 注意力变体 | `fp8_gemm_nt_skip_head_mid` | epilogue 加零段的特殊 GEMM |
| Einsum | `einsum('bmk,bnk->mn')`<br>`einsum('bhr,hdr->bhd')`<br>`einsum('bhd,hdr->bhr')`<br>`fp8_einsum(...)` | 硬编码的高频批量 GEMM |
| HyperConnection | `tf32_hc_prenorm_gemm` | GEMM + 平方和融合 |
| 兜底 | `cublaslt_gemm_*` | 调 cuBLASLt |
| Legacy | `deep_gemm.legacy.*`（A100 上的 Triton） | 仅前向，纯计算 |

> 💡 **关于 MoE 的"派发"语义**：DeepGEMM 的 grouped GEMM 本身**不负责把 token 送到对的专家所在 GPU**。它只假设输入已经按专家分组好了（由 [DeepEP](https://github.com/deepseek-ai/DeepEP)、NCCL 等外部工具完成）。所以从 kernel 视角看，它们都是纯计算。

### 🟡 9.2 通信+计算融合类（仅 2 个 kernel）

这两个 kernel **显式使用 NVLink 跨 rank 通信**，依赖 PyTorch ≥ 2.9 的对称内存：

| 算子 | 通信内容 |
|------|----------|
| `fp8_fp4_mega_moe` | EP **dispatch**（NVLink）+ 计算 + EP **combine**（NVLink），与计算**完全重叠** |
| `bf16_mega_moe` | 同上，但权重和激活是 BF16 |

它们把"派发 → 专家 GEMM1 → SwiGLU → 专家 GEMM2 → 收回"全部塞进**一个 mega-kernel**，通信和 Tensor Core 计算通过 SM 资源**分时复用**实现 overlap。

### 🟣 9.3 配置 / 工具类（非计算也非通信）

`set_num_sms`、`set_pdl`、`set_tc_util`、`init`、`transform_sf_into_required_layout` 等只是改全局配置或重排数据，不算计算 kernel，也不算通信。

### 9.4 一张全景图

```
                      DeepGEMM 算子全景
        ┌─────────────────┬─────────────────────┐
        │                 │                     │
     纯计算 (19 个)      通信+计算 (2 个)       配置/工具
        │                 │                     │
        ▼                 ▼                     ▼
  ┌─────────────┐    ┌─────────────┐     ┌──────────────┐
  │ Dense GEMM  │    │ fp8_fp4_    │     │ Layout 变换   │
  │ Grouped GEMM│    │   mega_moe  │     │ Runtime 配置 │
  │ Attention   │    │ bf16_mega_  │     │ 量化/Bench 工具│
  │ Einsum      │    │   moe       │     └──────────────┘
  │ HC Prenorm  │    └─────────────┘
  │ Legacy      │            │
  └─────────────┘            ▼
                       NVLink + Symmetric
                      Memory (PyTorch≥2.9)
```

### 9.5 实践含义

1. **如果只做单卡推理 / 训练** → 用纯计算类算子即可，**完全不需要考虑通信**。
2. **如果做 MoE 多卡推理** →
   - **传统方案**：[DeepEP](https://github.com/deepseek-ai/DeepEP) 做 dispatch/combine + DeepGEMM grouped GEMM 做计算（两个独立 kernel，自由组合）。
   - **极致方案**：直接用 `fp8_fp4_mega_moe`，通信计算融合（**仅 SM100 支持**，因为需要 tcgen05 指令 + 较新的 NVLink 特性）。
3. **如果做 MQA Indexer** → 用 `fp8_fp4_mqa_logits`（prefill）/ `fp8_fp4_paged_mqa_logits`（decode），**纯单卡计算**。
4. **通信和计算的重叠**：DeepGEMM 的通信能力目前**只通过 Mega MoE 体现**。其他算子保持纯计算，**让你自由组合通信库**（DeepEP、NCCL、自研均可）。

---

## 附录 A：快速定位算子源文件

| 算子 | C++ API 头 | CUDA 实现头 |
|------|-----------|------------|
| Dense FP8 GEMM (SM90) | [gemm.hpp](file:///d:/code/DeepGEMM/csrc/apis/gemm.hpp) | [sm90_fp8_gemm_1d1d.hpp](file:///d:/code/DeepGEMM/csrc/jit_kernels/impls/sm90_fp8_gemm_1d1d.hpp) / [sm90_fp8_gemm_1d2d.hpp](file:///d:/code/DeepGEMM/csrc/jit_kernels/impls/sm90_fp8_gemm_1d2d.hpp) |
| Dense FP8/FP4 GEMM (SM100) | [gemm.hpp](file:///d:/code/DeepGEMM/csrc/apis/gemm.hpp) | [sm100_fp8_fp4_gemm_1d1d.hpp](file:///d:/code/DeepGEMM/csrc/jit_kernels/impls/sm100_fp8_fp4_gemm_1d1d.hpp) |
| BF16 GEMM | [gemm.hpp](file:///d:/code/DeepGEMM/csrc/apis/gemm.hpp) | [sm90_bf16_gemm.hpp](file:///d:/code/DeepGEMM/csrc/jit_kernels/impls/sm90_bf16_gemm.hpp) / [sm100_bf16_gemm.hpp](file:///d:/code/DeepGEMM/csrc/jit_kernels/impls/sm100_bf16_gemm.hpp) |
| Grouped GEMM | [gemm.hpp](file:///d:/code/DeepGEMM/csrc/apis/gemm.hpp) | 同上 |
| MQA logits | [attention.hpp](file:///d:/code/DeepGEMM/csrc/apis/attention.hpp) | [sm90_fp8_mqa_logits.hpp](file:///d:/code/DeepGEMM/csrc/jit_kernels/impls/sm90_fp8_mqa_logits.hpp) / [sm100_mqa_logits.hpp](file:///d:/code/DeepGEMM/csrc/jit_kernels/impls/sm100_mqa_logits.hpp) |
| Paged MQA logits | [attention.hpp](file:///d:/code/DeepGEMM/csrc/apis/attention.hpp) | [sm90_fp8_mqa_logits.hpp](file:///d:/code/DeepGEMM/csrc/jit_kernels/impls/sm90_fp8_mqa_logits.hpp) |
| Skip-Head-Mid GEMM | [attention.hpp](file:///d:/code/DeepGEMM/csrc/apis/attention.hpp) | [sm90_fp8_gemm_1d2d.hpp](file:///d:/code/DeepGEMM/csrc/jit_kernels/impls/sm90_fp8_gemm_1d2d.hpp) |
| Einsum | [einsum.hpp](file:///d:/code/DeepGEMM/csrc/apis/einsum.hpp) | [sm90_bmk_bnk_mn.hpp](file:///d:/code/DeepGEMM/csrc/jit_kernels/impls/sm90_bmk_bnk_mn.hpp) / [sm100_bmk_bnk_mn.hpp](file:///d:/code/DeepGEMM/csrc/jit_kernels/impls/sm100_bmk_bnk_mn.hpp) |
| HC Prenorm | [hyperconnection.hpp](file:///d:/code/DeepGEMM/csrc/apis/hyperconnection.hpp) | [sm90_tf32_hc_prenorm_gemm.hpp](file:///d:/code/DeepGEMM/csrc/jit_kernels/impls/sm90_tf32_hc_prenorm_gemm.hpp) |
| Mega MoE | [mega.hpp](file:///d:/code/DeepGEMM/csrc/apis/mega.hpp) | [sm100_fp8_fp4_mega_moe.hpp](file:///d:/code/DeepGEMM/csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp) / [sm100_bf16_mega_moe.hpp](file:///d:/code/DeepGEMM/csrc/jit_kernels/impls/sm100_bf16_mega_moe.hpp) |
| Layout / SF 转换 | [layout.hpp](file:///d:/code/DeepGEMM/csrc/apis/layout.hpp) | [smxx_layout.hpp](file:///d:/code/DeepGEMM/csrc/jit_kernels/impls/smxx_layout.hpp) |
| cuBLASLt 兜底 | [gemm.hpp](file:///d:/code/DeepGEMM/csrc/apis/gemm.hpp) | [smxx_cublaslt.hpp](file:///d:/code/DeepGEMM/csrc/jit_kernels/impls/smxx_cublaslt.hpp) |

---

## 附录 B：一句话总结每个算子（速查卡片）

```
fp8_gemm_nt                D = C + A @ B.T, A/B 是 FP8/FP4
fp8_gemm_nt_skip_head_mid  同上，但输出在 head 维插入零段
fp8_fp4_mqa_logits         算 q @ kv.T * relu * weights，给 Indexer 打分
fp8_fp4_paged_mqa_logits   同上，但 KV 在 paged cache 里
m_grouped_fp8_gemm_nt_contiguous    MoE 前向，按 M 拼接所有专家的 token
m_grouped_fp8_gemm_nt_masked        MoE 解码，每组一份 buffer + mask
k_grouped_fp8_gemm_tn_contiguous    MoE 反向（权重梯度），按 K 拼接
bf16_gemm_*                同 fp8，但用 BF16
einsum('bmk,bnk->mn')      跨 batch 求和的 GEMM（多头聚合等）
einsum('bhr,hdr->bhd')     头维共享的 bmm（V projection 反向等）
fp8_einsum(...)            FP8 版本的 bmm
tf32_hc_prenorm_gemm       HyperConnection 的"GEMM + 平方和"融合
fp8_fp4_mega_moe           Mega-kernel：dispatch + GEMM1 + SwiGLU + GEMM2 + combine
bf16_mega_moe              同上，BF16 版本
cublaslt_gemm_*            兜底：用 NVIDIA cuBLASLt
transform_sf_into_required_layout  把 SF 转换成 kernel 需要的格式
```

祝你学习顺利！