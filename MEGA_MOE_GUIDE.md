# DeepGEMM 通信+计算算子深度学习文档

> 本文档专门讲解 DeepGEMM 中**仅有的 2 个**涉及跨 rank 通信的 kernel：  
> `fp8_fp4_mega_moe` 和 `bf16_mega_moe`
>
> 阅读建议：先看 LEARNING_GUIDE.md 第 9 章了解整体定位，再阅读本文档。

---

## 目录

1. [为什么只有 2 个？——再次确认结论](#1-为什么只有-2-个再次确认结论)
2. [Mega MoE 解决的痛点](#2-mega-moe-解决的痛点)
3. [整体架构：一次调用完成 MoE 全流程](#3-整体架构一次调用完成-moe-全流程)
4. [核心数据结构：对称内存与 Workspace](#4-核心数据结构对称内存与-workspace)
5. [通信机制：NVLink barrier 原理](#5-通信机制nvlink-barrier-原理)
6. [调度器：MegaMoEScheduler 如何编排任务](#6-调度器megamoescheduler-如何编排任务)
7. [Tensor Core 计算：FP8×FP4 与 BF16 路径](#7-tensor-core-计算fp8fp4-与-bf16-路径)
8. [Block Config 启发式（heuristics）](#8-block-config-启发式heuristics)
9. [Python 端完整调用流程](#9-python-端完整调用流程)
10. [源码定位速查表](#10-源码定位速查表)
11. [Mega MoE vs DeepEP 深度对比](#11-mega-moe-vs-deepep-深度对比)

---

## 1. 搜索跨 rank 通信引用

通过对仓库 `grep` 的全面搜索，所有与**跨 rank 通信**相关的引用（symmetric_memory / symm_mem / buffer_ptrs / NVLink / nvls / ProcessGroup / world_size / rank_idx / num_ranks / rendezvous / dist.barrier）**全部**集中在以下文件：

```
deep_gemm/mega/__init__.py                    # Python 包装
deep_gemm/utils/dist.py                       # dist 工具（init_dist / uneven_all_gather / dist_print）
deep_gemm/include/deep_gemm/comm/barrier.cuh  # NVLink barrier 实现
deep_gemm/include/deep_gemm/layout/sym_buffer.cuh
deep_gemm/include/deep_gemm/layout/mega_moe.cuh
deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh
deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh
deep_gemm/include/deep_gemm/impls/sm100_bf16_mega_moe.cuh
csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp
csrc/jit_kernels/impls/sm100_bf16_mega_moe.hpp
csrc/jit_kernels/heuristics/mega_moe.hpp
csrc/apis/mega.hpp
tests/test_mega_moe.py
scripts/quick_plot_pm.py
```

其他 19 个 kernel 完全不引用以上任何符号。

**结论**：DeepGEMM 中只有 `fp8_fp4_mega_moe` 和 `bf16_mega_moe` 涉及跨 rank 通信。

---

## 2. Mega MoE 解决的痛点

传统 MoE 前向（DeepSeek-V3 / Mixtral 等）需要 4 段独立操作：

```
┌─────────────┐    NVLink     ┌──────────────┐    Tensor Core   ┌──────────────┐
│ EP Dispatch │ ─────────────►│  GEMM1+SwiGLU│ ───────────────► │  GEMM2       │
│ (token→expert)              │  (per expert)│                  │  (per expert)│
└─────────────┘               └──────────────┘                  └──────────────┘
                                                                       │
                                                                       │ NVLink
                                                                       ▼
                                                                 ┌─────────────┐
                                                                 │ EP Combine  │
                                                                 │ (weight sum)│
                                                                 └─────────────┘
```

**问题**：
1. 通信和计算串行：dispatch 完成 → 等 → GEMM1 → 等 → GEMM2 → 等 → combine，**通信延迟完全暴露**
2. 多段 kernel 调度：每次 launch 都有 ~5μs 的 CPU 调度开销
3. dispatch / combine 需要单独工程（DeepEP 项目）

**Mega MoE 的解法**：把 4 段**全部塞进一个 mega-kernel**：

- **dispatch 与 L1 GEMM 重叠**：一边把 token 拉过来，一边在已经准备好的 token 上算 GEMM
- **combine 与 L2 GEMM 重叠**：一边把结果发回去，一边继续算下一批 token
- **SwiGLU 激活插在 L1 和 L2 之间**：MMA 后立即 epilogue，不需要单独的 kernel
- **无外部通信库**：内核内置 NVLink atomic 操作，依赖 PyTorch ≥ 2.9 的对称内存

性能：在 H800/B100 上对 EP=8 / 256 experts 的 MoE 推理，相对"DeepEP + 两次 grouped GEMM + SwiGLU + combine"的传统管线，端到端可提速 1.5x~2x（具体数字见 README 和 PR #316）。

---

## 3. 整体架构：一次调用完成 MoE 全流程

### 3.1 调用入口（Python 侧）

[`fp8_fp4_mega_moe`](file:///d:/code/DeepGEMM/deep_gemm/mega/__init__.py#L153-L176) 的签名：

```python
deep_gemm.fp8_fp4_mega_moe(
    y,                          # [num_tokens, hidden]          BF16 输出
    l1_weights,                 # (gate+up packed FP4, sf)      每个专家
    l2_weights,                 # (down FP4, sf)                每个专家
    sym_buffer,                 # SymmBuffer 对象
    shared_l1_weights=None,     # (可选) shared expert 的 L1
    shared_l2_weights=None,     # (可选) shared expert 的 L2
    cumulative_local_expert_recv_stats=None,  # 可选：专家接收统计计数器
    recipe=(1, 1, 32),          # SF 粒度
    activation='swiglu',
    activation_clamp=None,
    fast_math=True,
)
```

### 3.2 一个 mega-kernel 的内部阶段

```
┌─────────────────────────────────────────────────────────────────┐
│                    sm100_fp8_fp4_mega_moe                       │
│                                                                 │
│  阶段 1 (DMA warp):   NVLink PULL  ── 把 token 从其他 rank 拉过来 │
│  阶段 2 (TMA + MMA):  L1 GEMM (FP8 × FP4) → SwiGLU             │
│  阶段 3 (TMA + MMA):  L2 GEMM (FP8 × FP4) → 输出 BF16           │
│  阶段 4 (DMA warp):   NVLink PUSH  ── 把结果原子加回源 token     │
│                                                                 │
│  并行发生的:  shared expert L1/L2（可选）                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
        ▲                                                          │
        │ grid_sync + nvlink_barrier                               │
        └──────────────────────────────────────────────────────────┘
```

阶段间用 **grid_sync（SM 内 barrier）** 和 **nvlink_barrier（跨 rank barrier）** 串接。详见第 5 节。

---

## 4. 核心数据结构：对称内存与 Workspace

### 4.1 SymmBuffer：跨 rank 地址映射

代码：[`sym_buffer.cuh`](file:///d:/code/DeepGEMM/deep_gemm/include/deep_gemm/layout/sym_buffer.cuh)

```cpp
template <uint32_t kNumRanks = kNumMaxRanks>   // kNumMaxRanks = 72
struct SymBuffer {
    int64_t base;                              // 本 rank 对称 buffer 基址
    int64_t offsets[kNumMaxRanks];             // 其他 rank 的偏移
    uint32_t rank_idx;
};
```

**工作原理**：
- 所有 rank 通过 `torch.distributed._symmetric_memory.rendezvous()` 拿到**同一逻辑地址、但物理上各自显存**的 buffer。
- `offsets[i]` 存的是 rank i 的物理地址相对 rank_idx 的偏移。
- 在 kernel 内调用 `sym_buffer.map(ptr, dst_rank_idx)` 就能把本地指针映射到目标 rank 的物理指针——无需显式 NVLink send/recv。

**约束**：最多 72 个 rank（足够覆盖 DeepSeek-V3 的 EP=64 + 调度余量）。

### 4.2 Workspace：kernel 内部的"控制平面"

代码：[`layout/mega_moe.cuh`](file:///d:/code/DeepGEMM/deep_gemm/include/deep_gemm/layout/mega_moe.cuh#L46-L240) 中 `struct Workspace`

`Workspace` 是从 buffer 头部切出的一小段，存储**各种原子计数器**：

| 偏移 | 名称 | 用途 |
|------|------|------|
| 0..15 | 4× `uint32_t` grid_sync 计数器 | SM 内同步（不同阶段用不同 idx） |
| 16..19 | NVLink barrier counter | 跨 rank barrier 状态 |
| 20..27 | 2× `int` NVLink barrier signals | 跨 rank 信号（双 phase 翻转） |
| 28..31 | `uint32_t` L1 task counter | 调度器派发 L1 任务用 |
| 32..35 | `uint32_t` L2 task counter | 调度器派发 L2 任务用 |
| 36..39 | `uint32_t` shared L1 task counter | shared expert L1 任务 |
| 40..43 | `uint32_t` shared L2 task counter | shared expert L2 任务 |
| 44..127 | padding | 隔离"热"原子区，减少 L2 cache 行冲突 |
| 128+ | expert send/recv 计数器 | 每个专家的 token 收发数（64 位计数器） |
| ... | L1/L2 ring full/empty 计数器 | ring buffer 同步 |
| ... | src_token_topk_idx | dispatch 拉取源信息 |
| ... | token_src_metadata | combine 写回源信息 |

**核心思想**：所有跨 rank / 跨 SM 的同步都通过**这些共享计数器 + atomic** 完成，**没有显式的 MPI/NCCL 调用**。

### 4.3 MegaMoEBuffer：完整的 buffer 布局

`struct MegaMoEBuffer` 把整个对称 buffer 划分成多个子 buffer：

```
┌─────────────────────────────────────────────────────────────────┐
│                       对称内存区（所有 rank 共享）              │
├─────────────────────────────────────────────────────────────────┤
│ Workspace               │ 内核控制平面（counter 等）            │
├─────────────────────────┼───────────────────────────────────────┤
│ input_token_buffer      │ [num_max_tokens, hidden]   输入 x     │
│ input_sf_buffer         │ [num_max_tokens, hidden/32] FP8 SF   │
│ input_topk_idx_buffer   │ [num_max_tokens, num_topk] 专家 idx  │
│ input_topk_weights_buffer│ [num_max_tokens, num_topk] 专家权重  │
├─────────────────────────┤ （shared expert 才有以下）            │
│ shared_l1_sf_buffer     │                                       │
│ shared_l2_token_buffer  │                                       │
│ shared_l2_sf_buffer     │                                       │
├─────────────────────────┤ （routed expert ring）                │
│ l1_token_buffer         │ [num_ring_tokens, hidden]  ring      │
│ l1_sf_buffer            │ [num_sf_ring_tokens, ...]            │
│ l1_topk_weights_buffer  │                                       │
│ l2_token_buffer         │ [num_ring_tokens, intermediate]      │
│ l2_sf_buffer            │ [num_sf_ring_tokens, ...]            │
│ combine_token_buffer    │ [num_topk, num_max_tokens, hidden]   │
└─────────────────────────────────────────────────────────────────┘
```

> **Ring buffer** 设计：dispatch 持续填入 token，L1 持续消费，L2 持续消费形成 pipeline。

---

## 5. 通信机制：NVLink barrier 原理

代码：[`comm/barrier.cuh`](file:///d:/code/DeepGEMM/deep_gemm/include/deep_gemm/comm/barrier.cuh)

### 5.1 为什么需要 barrier？

Mega MoE 中**通信和计算都在同一个 kernel 内**，但计算阶段需要等待：
1. **dispatch 完成**：L1 GEMM 才能开始
2. **所有 rank 的 L1 完成**：combine 阶段才能开始（避免 deadlock）

传统做法是用 NCCL；但 NCCL 假设独占进程 / stream，且 launch 成本高。这里用**更轻量的 in-kernel barrier**。

### 5.2 `nvlink_barrier` 工作流

```cpp
template <uint32_t kNumRanks, uint32_t kNumSMs, ...>
void nvlink_barrier(workspace, sym_buffer, sm_idx, thread_idx, sync_scope, ...) {
    // 阶段 A: SM 内 grid sync（确保所有 SM 都到了这一步）
    grid_sync<kNumSMs, kGridSyncIndex>(workspace, sm_idx, thread_idx, sync_scope);

    // 阶段 B: 仅 SM 0 参与跨 rank 通信
    if (sm_idx == 0) {
        // 1. 读出当前 barrier 状态（2 bit: phase + sign）
        auto status = (*counter_ptr) & 3;
        auto signal_phase = status & 1;
        auto signal_sign = status >> 1;
        auto* signal_ptr = workspace.get_nvl_barrier_signal_ptr(signal_phase);

        // 2. 每个 thread 给一个远端 rank 发信号（+1 或 -1）
        if (thread_idx < kNumRanks)
            ptx::red_add_rel_sys(sym_buffer.map(signal_ptr, thread_idx), signal_sign ? -1 : 1);
        sync_scope();

        // 3. 翻转 phase + 自旋等待 signal 累加到目标值
        if (thread_idx == 0) {
            ptx::red_add(counter_ptr, 1);  // 翻转
            int target = signal_sign ? 0 : kNumRanks;
            while (ptx::ld_acq_sys(signal_ptr) != target) {
                if (clock64() - start_clock >= kNumTimeoutCycles)
                    DG_DEVICE_ASSERT(false and "NVLink barrier timeout");
            }
        }
    }

    // 阶段 C: SM 内 grid sync（确保所有 SM 都看到 barrier 完成）
    grid_sync<kNumSMs, kGridSyncIndex>(workspace, sm_idx, thread_idx, sync_scope);
}
```

### 5.3 关键技术点

| 技术 | 作用 |
|------|------|
| **`red_add_rel_sys` / `ld_acq_sys`** | NVLink 上的 release/acquire 原子操作（Hopper 起硬件支持） |
| **双 phase 翻转** | 避免 reuse 同一 buffer 时的数据竞争（每次 barrier phase +1） |
| **60s 超时** | 防止死锁时 GPU hang 整个集群 |
| **`grid_sync` 包裹** | 确保所有 SM 都到达 barrier，避免部分 SM 跳过 |

### 5.4 与传统通信库的对比

| 方案 | dispatch 延迟 | 调度开销 | 多 stream 兼容 |
|------|---------------|----------|----------------|
| **DeepEP + 两段 GEMM** | ~20-50μs/次 | 3-4 次 launch | 需 stream 编排 |
| **NCCL all-to-all + GEMM** | ~30-100μs/次 | 2 次 launch | 需 stream 编排 |
| **Mega MoE（in-kernel）** | **<5μs**（NVLink 原子延迟） | **1 次 launch** | 内置 overlap |

---

## 6. 调度器：MegaMoEScheduler 如何编排任务

代码：[`scheduler/mega_moe.cuh`](file:///d:/code/DeepGEMM/deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh)

### 6.1 任务模型

```cpp
enum class BlockPhase : uint32_t {
    None = 0,
    Linear1 = 1,        // 专家 GEMM1
    Linear2 = 2,        // 专家 GEMM2
    SharedLinear1 = 3,  // shared expert GEMM1
    SharedLinear2 = 4   // shared expert GEMM2
};

struct TaskInfo {
    BlockPhase block_phase;
    uint32_t local_expert_idx;  // 当前 rank 内的专家编号
    uint32_t m_block_idx;       // token 块编号
    uint32_t n_cluster_idx;     // N 维 cluster 编号（2 个 CTA 共一个 cluster）
    uint32_t pool_block_idx;    // ring buffer 中的目标槽位
    uint32_t valid_m;           // 实际有效 token 数（≤ BLOCK_M）
    uint32_t shape_n, shape_k;  // 该任务的 N/K 形状
};
```

### 6.2 调度流程

调度器运行在**专门的 producer warp**（不参与 MMA）：

```
           ┌─────────────────────────────────────┐
           │        MegaMoEScheduler              │
fetch ──►  │ fetch_expert_recv_count()            │  ◄── 等待所有 rank
           │   从 workspace 读各专家 token 数    │      dispatch 完成
           └──────────────┬──────────────────────┘
                          ▼
            ┌─────────────────────────────┐
   L1 task │ get_next_task()  ──► 计数 +1  │  ──► publish 到 ring buffer
            │   （轮流 L1 / L2）           │
   L2 task │   L1 任务发完后再发 L2        │  ──► consumer warp 取走执行
            └─────────────────────────────┘
```

**关键点**：
1. **先发完所有 L1，再发 L2**：保证 combine 之前所有专家都算完 L1
2. **Warming waves**：`get_num_l1_warmup_waves()` 保证有足够的 L1 任务在飞，避免 L2 等待 L1 时 deadlock
3. **TaskInfo 在 SMEM 通过 barrier 传递**：`task_info_full_barriers` / `task_info_empty_barriers` 实现 producer-consumer 同步（双 buffer 流水线）

### 6.3 Cluster 内 2 个 CTA 协作

- `kNumCTAsPerCluster = 2`：相邻 2 个 CTA 组成 cluster
- `cluster_arrive_relaxed() + cluster_wait()`：cluster 内轻量同步
- N 维切成 2 份分别由 2 个 CTA 处理
- **通过 DSMEM（Distributed Shared Memory）交换中间结果**

### 6.4 任务依赖图

```
                ┌─────────┐
                │ dispatch │  NVLink PULL  (DMA warp)
                └────┬─────┘
                     │ token 写入 ring
                     ▼
              ┌──────────────┐
       ┌──────│   L1 GEMM     │  (MMA warp, FP8×FP4)
       │      │ + SwiGLU epilogue│
       │      └──────┬────────┘
       │             │ 写回 ring
       │             ▼
       │       ┌──────────────┐
       │       │   L2 GEMM     │  (MMA warp, FP8×FP4)
       │       └──────┬────────┘
       │              │
       └──────────────┤ (loop 同一 cluster 处理不同 token)
                      ▼
              ┌──────────────┐
              │ NVLink PUSH  │  (DMA warp)
              │ + atomic add │
              └──────┬───────┘
                     │
                     ▼
                 输出 y
```

---

## 7. Tensor Core 计算：FP8×FP4 与 BF16 路径

### 7.1 FP8×FP4 路径（`fp8_fp4_mega_moe`）

- **激活**：FP8 (e4m3)，per-32-element SF，使用 packed UE8M0
- **权重**：FP4 (e2m1)，per-32-element SF，使用 packed UE8M0
- **指令**：SM100 `tcgen05.mma`（Blackwell 第五代张量核）
- **精度恢复**：`accumulate in FP32`（kernel 输出 BF16，但内部累加是 FP32）
- **TMA 描述符**：所有数据通过 TMA 加载（Hopper 起硬件异步）

### 7.2 BF16 路径（`bf16_mega_moe`）

- **激活 + 权重**：BF16，无 SF
- **指令**：`tcgen05.mma` BF16 variant
- **适用场景**：调试、数值正确性验证、对精度敏感的训练

### 7.3 SwiGLU 激活的位置

```
L1 GEMM 输出 [num_tokens, 2*intermediate]
       │
       ▼ epilogue（无需额外 kernel）
gate, up = split(...)
y = silu(gate) * up         # SwiGLU
       │
       ▼ 写回 ring buffer（FP8 重量化）
[FP8, SF=UE8M0]
       │
       ▼ L2 GEMM 输入
```

> **优势**：SwiGLU 和 FP8 重量化都在 L1 的 epilogue 完成，**无需 launch 额外 kernel**。

### 7.4 Combine 写回

- 每个 token 的 `num_topk` 个专家结果**分别**写回 combine_token_buffer（位于源 rank）
- 源 rank 端用 `atomic_add` 把权重加到 y 输出（避免多 rank 写冲突）
- Combine 在最后一次 nvlink_barrier 之后启动

---

## 8. Block Config 启发式（heuristics）

代码：[`csrc/jit_kernels/heuristics/mega_moe.hpp`](file:///d:/code/DeepGEMM/csrc/jit_kernels/heuristics/mega_moe.hpp)

### 8.1 Block 大小选择

按"**每个专家预期接收的 token 数**"动态选择 `block_m` / `block_k` / `cluster_size`：

| `expected_tokens_per_expert` | `block_m` | `block_k` | 典型场景 |
|------------------------------|-----------|-----------|----------|
| ≤ 8.5 | 16 | 256 | RL 长尾 rollout |
| ≤ 16.5 | 32 | 128 | 小 batch + 小 EP + decode |
| ≤ 32.5 | 64 | 128 | 中 batch + 小 EP + decode |
| ≤ 64.5 | 96 | 128 | 大 batch + 小 EP + decode |
| ≤ 96.5 | 128 | 128 | 中 batch + 中 EP |
| > 96.5 | 192 | 128 | Prefill 或大 EP decode |

### 8.2 Ring buffer 容量计算

```cpp
// 在 csrc/apis/mega.hpp:37-65 中
int num_ring_tokens = 0;
for (block_m in kCandidateBlockM = {8, 16, 32, 64, 96, 128, 192}) {
    num_live_pool_blocks = sched::get_num_max_live_pool_blocks(num_pool_blocks, num_sms, hidden, intermediate_hidden);
    num_ring_tokens = max(num_ring_tokens, num_live_pool_blocks * block_m);
}
num_ring_tokens = align(num_ring_tokens, kLCMCandidateBlockM=384);
```

**为什么对所有候选 block_m 取 max**：因为 JIT 时不知道实际 token 分布，要预留最大可能空间。

### 8.3 Shared Memory 分配

heuristics 计算并分配以下 smem 区域：

| 区域 | 用途 |
|------|------|
| `dispatch` | DMA warp 用的 expert counter + send buffer |
| `cd` | L1/L2 输出 staging（BF16 或 FP8） |
| `amax_reduction` | SwiGLU 跨 warp amax 归约 |
| `task_info` | 调度器 TaskInfo 双 buffer |
| `barriers` | 各类同步 barrier（dispatch + epilogue + schedule） |
| `pipeline stages × stage_size` | A/B/SFA/SFB tile + 同步 |

总大小保证 `num_stages >= 2`（双缓冲）。

---

## 9. Python 端完整调用流程

### 9.1 标准用法（多 GPU）

```python
import torch
import torch.distributed as dist
import deep_gemm
from deep_gemm.utils import per_token_cast_to_fp8, per_token_cast_to_fp4

# ── 1. 初始化分布式 ──
rank, world_size, group = deep_gemm.utils.init_dist(local_rank, num_local_ranks)
torch.cuda.set_device(local_rank)

# ── 2. 分配对称内存 ──
buffer = deep_gemm.get_symm_buffer_for_mega_moe(
    group,
    num_experts=256,                  # 总专家数
    num_max_tokens_per_rank=1024,     # 每个 rank 最大 token 数
    num_topk=8,                       # 每个 token 选 8 个专家
    hidden=7168,
    intermediate_hidden=2048,
    num_shared_experts=2,             # 可选：shared expert
    mma_type='fp8xfp4',               # 或 'bf16xbf16'
    activation='swiglu',
)

# ── 3. 准备权重（离线做一次即可）──
l1_weights_fp4, l1_sf = per_token_cast_to_fp4(l1_weights_bf16, use_ue8m0=True, gran_k=32)
l2_weights_fp4, l2_sf = per_token_cast_to_fp4(l2_weights_bf16, use_ue8m0=True, gran_k=32)
transformed_l1, transformed_l2 = deep_gemm.transform_weights_for_mega_moe(
    (l1_weights_fp4, l1_sf),
    (l2_weights_fp4, l2_sf),
)

# ── 4. 把输入拷进对称 buffer ──
buffer.x[:num_tokens].copy_(x_fp8)
buffer.x_sf[:num_tokens].copy_(x_sf)
buffer.topk_idx[:num_tokens].copy_(topk_idx)
buffer.topk_weights[:num_tokens].copy_(topk_weights)

# ── 5. 调用 mega-kernel ──
y = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
deep_gemm.fp8_fp4_mega_moe(
    y,
    transformed_l1, transformed_l2,
    buffer,
    recipe=(1, 1, 32),
    activation='swiglu',
    fast_math=True,
)
```

### 9.2 关键参数解读

| 参数 | 含义 | 调优建议 |
|------|------|----------|
| `num_max_tokens_per_rank` | 每个 rank 的最大 token 数 | **越大越能 overlap 通信**，但占更多显存 |
| `num_experts` | 总专家数（所有 rank 之和） | 模型配置决定 |
| `num_topk` | 每个 token 选几个专家 | DeepSeek-V3 = 8 |
| `num_shared_experts` | DeepSeek-V3 风格的 shared expert 数 | 0 表示无 |
| `recipe=(1, 1, 32)` | SF 粒度（gran_mn=1, gran_n=1, gran_k=32） | FP8×FP4 路径**必须**是这个 |
| `fast_math=True` | 是否用近似指令（更快的 SwiGLU） | 推荐 True |
| `activation_clamp` | SwiGLU 输入上限（防止溢出） | `None` = 不限制 |

### 9.3 单卡调试模式

如果 `group.size() == 1`，`get_symm_buffer_for_mega_moe` 会自动用 `torch` 分配（而非 symmetric memory）。这样可以在单卡上**只跑计算部分**，跳过通信——非常便于调试。

---

## 10. 源码定位速查表

| 模块 | 文件 |
|------|------|
| Python 入口 | [deep_gemm/mega/__init__.py](file:///d:/code/DeepGEMM/deep_gemm/mega/__init__.py) |
| C++ API 注册 | [csrc/apis/mega.hpp](file:///d:/code/DeepGEMM/csrc/apis/mega.hpp) |
| Python 端 dist 工具 | [deep_gemm/utils/dist.py](file:///d:/code/DeepGEMM/deep_gemm/utils/dist.py) |
| 对称内存结构 | [deep_gemm/include/deep_gemm/layout/sym_buffer.cuh](file:///d:/code/DeepGEMM/deep_gemm/include/deep_gemm/layout/sym_buffer.cuh) |
| Mega MoE buffer 布局 | [deep_gemm/include/deep_gemm/layout/mega_moe.cuh](file:///d:/code/DeepGEMM/deep_gemm/include/deep_gemm/layout/mega_moe.cuh) |
| 跨 rank barrier | [deep_gemm/include/deep_gemm/comm/barrier.cuh](file:///d:/code/DeepGEMM/deep_gemm/include/deep_gemm/comm/barrier.cuh) |
| 调度器（task 编排） | [deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh](file:///d:/code/DeepGEMM/deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh) |
| FP8×FP4 mega kernel | [deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh](file:///d:/code/DeepGEMM/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh) |
| BF16 mega kernel | [deep_gemm/include/deep_gemm/impls/sm100_bf16_mega_moe.cuh](file:///d:/code/DeepGEMM/deep_gemm/include/deep_gemm/impls/sm100_bf16_mega_moe.cuh) |
| FP8×FP4 启动逻辑 | [csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp](file:///d:/code/DeepGEMM/csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp) |
| BF16 启动逻辑 | [csrc/jit_kernels/impls/sm100_bf16_mega_moe.hpp](file:///d:/code/DeepGEMM/csrc/jit_kernels/impls/sm100_bf16_mega_moe.hpp) |
| Block config 启发式 | [csrc/jit_kernels/heuristics/mega_moe.hpp](file:///d:/code/DeepGEMM/csrc/jit_kernels/heuristics/mega_moe.hpp) |
| 集成测试 | [tests/test_mega_moe.py](file:///d:/code/DeepGEMM/tests/test_mega_moe.py) |
| NCU profiling 脚本 | [scripts/run_ncu_mega_moe.sh](file:///d:/code/DeepGEMM/scripts/run_ncu_mega_moe.sh) |

---

## 11. Mega MoE vs DeepEP 深度对比

> DeepGEMM 的 Mega MoE 和 [DeepEP](https://github.com/deepseek-ai/DeepEP) 都是 DeepSeek 团队为 MoE 专家并行（EP）打造的工具，但**定位和能力截然不同**。本章详细对比两者。

### 11.1 一句话定位

| | DeepEP | Mega MoE |
|--|--------|----------|
| **定位** | 独立的 MoE 通信原语库 | DeepGEMM 内置的 MoE 融合 mega-kernel |
| **能做什么** | EP dispatch + EP combine（**只管通信**） | dispatch + GEMM1 + SwiGLU + GEMM2 + combine（**通信+计算全包**） |
| **类比** | "快递公司"——只送包裹 | "工厂+流水线"——从原料到成品一条龙 |

### 11.2 能力对比表

| 维度 | DeepEP | Mega MoE |
|------|--------|----------|
| **通信原语** | dispatch / combine | 同样包含 dispatch / combine |
| **GEMM 计算** | ❌ 不包含 | ✅ 内置 GEMM1（gate+up）、SwiGLU、GEMM2（down） |
| **SwiGLU 激活** | ❌ 不包含 | ✅ 在 L1 epilogue 中完成 |
| **精度支持** | FP8 / BF16 dispatch | FP8×FP4（默认）/ BF16×BF16 |
| **GPU 架构** | SM90 (Hopper) 完整支持 | **仅 SM100 (Blackwell)** |
| **拓扑** | Intra-node + Inter-node（NVSHMEM + RDMA） | 仅 Intra-node（NVLink 原子） |
| **软件依赖** | NVSHMEM、RCCL | PyTorch ≥ 2.9 对称内存 |
| **通信实现** | NVSHMEM 远程读写 + RDMA | 直接 NVLink `red_add` 原子 |
| **CPU 调度** | 多次 kernel launch | **单次 kernel launch** |
| **训练支持** | ✅（normal / low-latency 两套） | ❌ 仅前向 |
| **推理支持** | ✅ | ✅（极致性能） |
| **Group-Limited 专家** | ✅（DeepSeek-V3 风格） | ✅（`num_shared_experts` 参数） |
| **可独立使用** | ✅（可单独调 dispatch/combine） | ❌（必须经 DeepGEMM 入口） |

### 11.3 两者的重叠（共同能做）

下列能力**两者都提供**，可以二选一：

| 能力 | DeepEP 的实现 | Mega MoE 的实现 |
|------|---------------|-----------------|
| **EP dispatch** | `ep_buffer.dispatch(x, topk_idx, topk_weights, ...)` | 内置在 mega-kernel 内（自动） |
| **EP combine** | `ep_buffer.combine(...)` | 内置在 mega-kernel 内（自动） |
| **FP8 量化 dispatch** | `use_fp8_dispatch=True` | 原生（FP8×FP4 路径） |
| **不均匀 token 派发** | ✅（每个 rank token 数可以不同） | ✅（`expected_m_for_psum_layout` 等机制） |
| **Top-K 路由** | ✅ | ✅ |
| **共享专家支持** | ✅ | ✅ |
| **多进程分布式启动** | ✅ | ✅ |

### 11.4 两者的差异（核心区别）

#### ① 设计目标不同

- **DeepEP**：通用通信库——为**任何需要 EP 通信**的场景服务（无论训练 / 推理 / 不同的 GEMM 库 / 不同的精度）。需要**自己配 GEMM**。
- **Mega MoE**：极致的"全栈 MoE 前向"——把通信和计算**彻底焊死**在一个 kernel 里，只为追求**最低延迟**。

#### ② 与 GEMM 的关系

```
DeepEP 使用方式（需要 5 步）:
  ① deep_ep.dispatch(x, topk_idx, ...)         # 通信
  ② deep_gemm.m_grouped_fp8_gemm_*_contiguous  # GEMM1
  ③ tilelang_ops.swiglu_apply_weight_to_fp8    # SwiGLU（第三方）
  ④ deep_gemm.m_grouped_fp8_gemm_*_contiguous  # GEMM2
  ⑤ deep_ep.combine(...)                       # 通信
  → 5 次 kernel launch

Mega MoE 使用方式（1 步）:
  ① deep_gemm.fp8_fp4_mega_moe(...)           # 全包
  → 1 次 kernel launch
```

> **Mega MoE 的极致优化**：通信和计算在 SM 资源上**分时复用**——dispatch 时其他 SM 已经在算 GEMM；combine 时一边收一边算下一批。

#### ③ 硬件约束

- **DeepEP**：可在 **H100/H200/H800**（SM90）上运行，是目前 H800 集群上生产部署的事实标准。
- **Mega MoE**：**仅 B100/B200/GB200**（SM100）可用，因为依赖 `tcgen05.mma` 指令 + 新一代 NVLink atomic。

#### ④ 使用门槛

| | DeepEP | Mega MoE |
|--|--------|----------|
| 安装复杂度 | 中（要装 NVSHMEM） | 低（pip 装 DeepGEMM 即可） |
| 调用代码量 | 30-50 行 | 5-10 行 |
| 调参难度 | 中（要调 num_warp_groups / expert_alignment） | 低（heuristics 已自动选 block size） |
| 调试友好度 | 好（每段独立） | 一般（一个 kernel 不易拆解） |

#### ⑤ 性能取舍

| 指标 | DeepEP + 2 个 grouped GEMM | Mega MoE |
|------|---------------------------|----------|
| **延迟（decode, 小 batch）** | 基线 | **显著领先**（dispatch 与 GEMM 完全重叠） |
| **吞吐（prefill, 大 batch）** | 接近 TMA 带宽上限 | 略快（少几次 launch） |
| **显存占用** | 多份中间 buffer | 单一对称 buffer |
| **代码灵活性** | 高（可换 GEMM 库、可换精度） | 低（绑定 FP8×FP4） |

### 11.5 怎么选？

| 场景 | 推荐 |
|------|------|
| **H100/H200/H800 集群** | DeepEP + DeepGEMM grouped GEMM（**唯一选项**，Mega MoE 不支持） |
| **B100/B200/GB200 + 极致延迟** | ✅ Mega MoE |
| **B100/B200 + 训练/反向** | DeepEP + DeepGEMM grouped GEMM |
| **需要 FP8/BF16 都用** | DeepEP（灵活）|
| **只有 SM100 + 极简代码** | Mega MoE |
| **需要 inter-node 通信** | DeepEP（支持 RDMA，Mega MoE 只走 NVLink） |

### 11.6 共同点总结

如果只能记住一句话：**DeepEP 和 Mega MoE 都是 DeepSeek 团队在不同代 GPU 上对"MoE 通信"这一问题的不同回答**：
- DeepEP = **通用解**（库，可组配）
- Mega MoE = **专用解**（mega-kernel，最快）

两者在 `tests/test_mega_moe.py` 中**作为对照基准被同时使用**：Mega MoE 的正确性测试就是用 DeepEP + DeepGEMM 的传统管线作为 baseline 对比的。

### 11.7 参考资料

- [DeepEP GitHub 仓库](https://github.com/deepseek-ai/DeepEP) — MoE 通信原语库
- [DeepGEMM GitHub 仓库](https://github.com/DeepSeek-AI/DeepGEMM) — 本仓库
- [DeepGEMM README 第 116 行](file:///d:/code/DeepGEMM/README.md#L116-L131) — Mega MoE 官方说明
- [DeepGEMM README 第 88 行](file:///d:/code/DeepGEMM/README.md#L88-L90) — DeepEP 作为 masked GEMM 的输入源
- [`tests/test_mega_moe.py`](file:///d:/code/DeepGEMM/tests/test_mega_moe.py) — 用 DeepEP 作为 baseline 的正确性测试

---

## 附录 A：Mega MoE 涉及的关键硬件特性

| 硬件特性 | 用途 | 必需 |
|----------|------|------|
| **TMA** (cp.async.bulk) | 异步加载 tile 到 SMEM | SM90+ |
| **WGMMA / tcgen05** | Tensor Core 矩阵乘 | SM90 / SM100 |
| **Distributed Shared Memory** | Cluster 内 CTA 共享 SMEM | SM90+ |
| **`bar.cluster`** | Cluster 同步 | SM90+ |
| **`red_add` 系统级原子** | NVLink 远程原子操作 | SM90+（最强 SM100） |
| **Symmetric Memory** | 多 GPU 共享地址空间 | PyTorch ≥ 2.9 |
| **NVLink** | GPU 互联 | NVSwitch 拓扑 |

> ⚠️ **仅 SM100 支持 Mega MoE**，因为需要最新的 `tcgen05` 指令 + 较新的 NVLink atomic 性能。

---

## 附录 B：常见问题

### Q1: Mega MoE 和 DeepEP 怎么选？

| 场景 | 推荐方案 |
|------|----------|
| 单卡或不需要极致 MoE 性能 | grouped GEMM（任何方案都行） |
| 多卡推理 + 想要最简代码 | **DeepEP + grouped GEMM**（DeepGEMM 自带） |
| 多卡推理 + 极致性能（仅 SM100） | **Mega MoE** |
| 训练 / 反向 | 不建议 Mega MoE，grouped GEMM 更通用 |

### Q2: 能否在 PyTorch < 2.9 上跑？

不能。`torch.distributed._symmetric_memory` 是 PyTorch 2.9 引入的 Mega MoE 必需依赖。

### Q3: num_topk 必须和模型一致吗？

**是的**。这是编译时常量（在 JIT 中作为模板参数），改了就重新编译。

### Q4: 如何调试？

1. 单卡测试：`group.size() == 1` 时自动跳过通信
2. 设置 `DG_COMM_KERNEL_DEBUG=1`：每次调用前清零 buffer（可帮助定位竞态）
3. 用 `DG_JIT_DEBUG=1` + `DG_PRINT_CONFIGS=1` 看每次 kernel 选择的配置
4. 用 `nsys` / `ncu` profile，看 timeline 中的通信 vs 计算 overlap

### Q5: SM90 能用吗？

**不能**。Mega MoE 依赖 `tcgen05` 指令（仅 Blackwell）。SM90 上应使用 [DeepEP](https://github.com/deepseek-ai/DeepEP) + `m_grouped_fp8_gemm_*`。

---

## 附录 C：阅读顺序建议

1. **先看本文档第 3 节**，建立整体流程的"心智模型"
2. **再读 [deep_gemm/mega/__init__.py](file:///d:/code/DeepGEMM/deep_gemm/mega/__init__.py)** (~200 行) —— 全 Python 端最薄的一层
3. **看 [csrc/apis/mega.hpp](file:///d:/code/DeepGEMM/csrc/apis/mega.hpp) (~400 行)** —— C++ 端的"接线"
4. **重点读 [comm/barrier.cuh](file:///d:/code/DeepGEMM/deep_gemm/include/deep_gemm/comm/barrier.cuh) (~90 行)** —— 这是通信的核心
5. **浏览 [layout/mega_moe.cuh](file:///d:/code/DeepGEMM/deep_gemm/include/deep_gemm/layout/mega_moe.cuh) (~440 行)** —— buffer 布局
6. **挑战 [scheduler/mega_moe.cuh](file:///d:/code/DeepGEMM/deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh) (~420 行)** —— 调度逻辑
7. **最后看 [sm100_fp8_fp4_mega_moe.cuh](file:///d:/code/DeepGEMM/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh)** —— 主 kernel（可能上千行，需要耐心）

---

祝你深入理解 DeepGEMM 最核心的优化！