# IntentIR

**Language:** [English](README.md) | [中文](README.zh-CN.md)

IntentIR 是 TianChen Stack 的一个子项目，用来把优化过的机器学习
kernel 整理成可审计、可验证、可移植的算子资产。

现代机器学习系统越来越依赖面向特定硬件优化的 kernel。这些 kernel
可能来自 Triton、TileLang、CUDA，也可能来自某个算子库或模型系统的专用
实现。它们快，是因为源码里同时写进了两类信息：一类是“这个算子到底在算
什么”，另一类是“它如何被调度到某个机器上”。这两类信息纠缠在一起之后，
kernel 就很难检查、验证、复用和迁移。

IntentIR 要解决的就是这个问题。它通过 certified semantic lifting，把不同
前端里的 kernel 提升到统一的 intent-level IR；再用前端证据和源程序执行结
果验证这个 IR；最后让下游后端消费同一个带 contract 的表示，而不是为每一组
前端和后端都写一套专门翻译器。

![IntentIR frontend/backend interface](docs/assets/fig1_cropped.png)

上图展示的是 IntentIR 的核心接口：如果没有统一 IR，`N` 个 kernel 前端和
`M` 个目标后端会形成 `N x M` 的翻译和重调优问题。IntentIR 把这个问题拆成
`N` 个前端 lifter 加 `M` 个后端 consumer，中间通过同一个语义资产连接。

## 在 TianChen Stack 中的位置

在 TianChen Stack 里，IntentIR 是前端和算子资产的统一 IR 层。它不是一个
单独的 benchmark harness，也不是一次性原型；它负责把异构算子资产变成
TianChen 编译栈可以稳定消费的编译输入。

- 它接收 Triton、TileLang、CUDA，以及 FlagGems 这类 Triton-backed 算子库
  中的优化 kernel 和 provider 资产。
- 它把这些资产规范化到统一的 IntentIR 表示中，记录算子计算了什么、哪些
  正确性约束必须成立、哪些 schedule 信息只是提示。
- 它给后端编译提供稳定 contract：CUDA、RVV 以及未来 TianChen target 都
  应该消费同一个 recovered intent，而不是各自重新理解一遍前端语义。
- 它保留源 kernel 里的优化知识，但把这些知识作为 non-binding guidance，
  让后端可以针对自己的硬件重新调优，而不是把源 schedule 冻结成语义。

一句话说，IntentIR 是 TianChen Stack 中连接前端/算子资产和后端编译器的
语义桥梁。

## 核心思想

从概念上说，IntentIR 做的是普通 lowering 的反方向。普通编译从干净的张量
程序出发，把它 lowering 成特化 kernel；IntentIR 则从已经 schedule-specialized
的实现出发，恢复出一个可执行、可检查、可迁移的 intent artifact。

恢复出来的表示分三层：

- **Layer A: algorithmic intent。** 记录张量级计算，包括算子结构、数据类
  型、符号 shape、轴角色，以及会影响输出解释的语义 layout。
- **Layer B: portable execution structure。** 记录正确性相关的执行结构，
  包括 index expression、mask、bounds、shape/layout 约束、依赖关系、同步
  和受控 atomics。
- **Layer C: non-binding schedule hints。** 记录 tile size、thread mapping、
  vector width、pipeline depth 等源实现里的优化选择。这些信息可以指导调优，
  但不定义语义。

这个三层拆分是 IntentIR 的关键。Layer A 和 Layer B 是 binding 的，需要被
验证；Layer C 很有用，但后端可以重新选择。

## 验证边界

IntentIR 不把 LLM 或启发式提取器当成真值来源。lifter 可以使用前端证据，
也可以让 LLM 作为 constrained proposer 提出候选 IR；但候选 artifact 必须
通过验证之后才会被接受。

验证过程包括：

- 对 IntentIR artifact 做 schema 和 type 检查；
- 从前端提取 canonical evidence，例如计算 anchor、访存、predicate、shape、
  layout、同步和 atomic 摘要；
- 检查 obligation，包括 anchor coverage、structured access、guarded bounds、
  interface consistency、address independence、synchronization 和 atomics；
- 在源路径可执行时，对比源 kernel 执行结果和恢复出的 IntentIR artifact；
- 生成 certificate，记录已经检查过的 obligation、witness、assumption，以及
  FULL/PARTIAL/OOS contract 状态。

所以这个仓库是按编译 pipeline 组织的，而不是把生成物堆在一起。

## 仓库结构

- `intent_ir/`：核心 IR 数据结构、parser、验证辅助、macro expansion、
  canonicalization 和 MLIR bridge。
- `frontends/`：Triton、TileLang、CUDA 的前端 adapter 和 evidence extraction。
- `pipeline/`：连接前端 lifting、验证、provider integration、MLIR contract
  emission 和后端 handoff 的编排代码。
- `backends/`：CUDA 和 RVV 后端 contract/runtime 路径。
- `kernels/`：精简后的源 kernel 资产和 smoke case。
- `verify/`：interpreter、diff runner、metamorphic check、mutation utility、
  tolerance 和 case generation。
- `compiler/intentir_mlir_plugin/`：可选 C++ MLIR plugin 源码。
- `scripts/`：精选过的运行入口，包括环境检查、前端验证、MLIR 工具、后端
  smoke check 和 provider matrix。
- `tests/`：快速可复现测试，覆盖 semantic core、MLIR/backend lowering、
  provider boundary 和 pipeline 行为。
- `docs/assets/`：README 中引用的设计图资产。

这个清理后的仓库刻意不包含历史 archive、manuscript source、workflow state、
实验输出、本地 toolchain、Triton dump 或 Python bytecode cache。

## 快速开始

创建环境：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements/dev.txt
```

运行快速可复现 gate：

```bash
pytest -q
```

检查本地可选依赖和加速器可见性：

```bash
python scripts/intentir.py env
```

运行纯 IntentIR/MLIR sanity check：

```bash
python scripts/intentir.py mlir check
```

可选 GPU/frontend 流程需要 CUDA-compatible PyTorch 和 GPU extras：

```bash
pip install -r requirements/gpu.txt
python scripts/intentir.py suite --suite triton-native-coverage --kernel add2d --cases-limit 1
```

长周期 FlagGems matrix 和远程 RVV run 会在 `artifacts/` 下重新生成 evidence。
这些内容应由脚本生成，不提交到源码仓库。

## 可复现边界

这个仓库的可复现性分两层：

- **快速本地 gate：** `pytest -q` 覆盖 IntentIR schema/parser、interpreter 行为、
  MLIR conversion 和 lowering contract、provider boundary，以及部分 pipeline
  utility。这个 gate 不依赖历史 artifact tree。
- **系统级 evidence：** 完整前端/provider/后端 run 会在 `artifacts/` 下重新生成
  输出。CUDA、Triton、TileLang、FlagGems 和 RVV 是否可用取决于宿主环境，所以
  这些路径是可选的，并通过脚本参数化。

核心原则是：提交进仓库的源码是 source of truth。输出、workflow snapshot、
manuscript table、编译后的 kernel 和 cache 都不是理解或重建项目的必要条件。

## 开发说明

扩展系统时，尽量沿用现有 frontend/backend 拆分：

- 新前端相关的提取逻辑放到 `frontends/<name>/`；
- 可复用编排逻辑放到 `pipeline/`；
- 目标相关 lowering/runtime 代码放到 `backends/<target>/`；
- 语义算子放到 `intent_ir/ops/`，interpreter 覆盖放到 `verify/`；
- 生成出来的 evidence 放到 `artifacts/`，不要放进源码目录。

IntentIR 应该始终作为 TianChen Stack 的共享语义接口：足够贴近前端以保留
证据，又足够后端中立以支持 retargeting。
