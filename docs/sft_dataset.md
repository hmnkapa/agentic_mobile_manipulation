# Codex 轨迹 → Qwen3.5-4B SFT

转换器从本项目保存的 Codex RGB 轨迹生成 ms-swift `messages/images/tools` JSONL。
它只需要 Python 3.11 标准库，导出不调用模型或模拟器。数据筛选使用独立评估结果；
finish、客户端退出和证据验证分别记录。转换后的数据与当前 Codex 运行接口相互独立。

## 使用

先按项目安装说明安装本包，或设置 `PYTHONPATH=src`。安装后的入口是 `manip-sft`；
仓库脚本与 `python -m manipulation_agent.sft.cli` 调用同一个入口。

单条轨迹检查，默认不生成文件：

```bash
python scripts/build_manipulation_sft_dataset.py \
  --run-dir /data/run/episode \
  --controller-dir /data/run/controller
```

指定新的输出目录后，复制图片并生成可独立使用的数据包：

```bash
python scripts/build_manipulation_sft_dataset.py \
  --run-dir /data/run/episode \
  --controller-dir /data/run/controller \
  --out-dir /data/sft/radio_v1
```

批量输入是显式配对的 JSONL 清单。相对路径从清单所在目录解析，不猜测 episode/controller 配对：

```jsonl
{"run_dir":"scene_a/run_1/episode","controller_dir":"scene_a/run_1/controller"}
{"run_dir":"scene_b/run_2/episode","controller_dir":"scene_b/run_2/controller"}
```

```bash
python scripts/build_manipulation_sft_dataset.py \
  --manifest /data/cohort/sources.jsonl \
  --out-dir /data/sft/cohort_v1 \
  --k 2 \
  --max-len 24576
```

批量按 evaluator 记录的场景整体划分，seed=42。在所有完整场景组合中，
选择最接近 10% 样本数的组合作为验证集，至少留一个场景训练。
多个组合距离相同时，按 `SHA256("42:" + scene)` 排序后的场景索引元组确定取舍，确保可复现。
统计记录实际比例，不保证场景大小差异很大时仍能接近 10%。相同轨迹的重复输入会报告错误。
只有一个合格场景时批量模式报错，请用单条模式导出，不构造伪验证集。

退出码 0 表示检查或导出完成，2 表示输入/导出错误。批量中无法转换的来源隔离到
`errors.jsonl`，其余来源继续处理；标准输出同时报告错误数量和原因。
评估失败或结果未知是正常筛选项，写入 `skipped.jsonl`。单条格式错误直接失败。
批量剩余场景不足时，错误详情通过标准输出返回，输出目录不创建。

## 样本与历史

- 一个成功的公开工具调用生成一个样本，包括 initialize、技能读取和 task_notes。
  工具结果必须有 `ok=true`，没有 MCP/传输错误；finish 还要求 `closed=true`。
  独立评估成功即可满足轨迹成功条件，不要求模型声明 achieved 或进程正常退出。
- 输入止于目标调用的 dispatch 边界。目标结果只参与筛选，目标调用后的图片不进入该样本。
  initialize 样本因此可以没有图片。末尾独立的 assistant 总结不单独生成样本。
- 调用开始前的 assistant 公开文字与随后的调用合并。调用期间收到的公开文字
  只进入后续样本的历史，按“调用 → 交错文字 → 回执”保留为独立 assistant 消息。
  所有历史 assistant 都设置 `loss=false`，
  最后一个 assistant 使用空 `<think>\n\n</think>` 加公开文字和工具调用。
  reasoning/summary/encrypted reasoning 都不进入教师思考内容。
- 失败调用及真实错误保留为历史；不会把缺少的回执从模拟器记录补成模型输入。
  原生 MCP 的 `status="failed"` 也按失败处理，即使 `error=null` 且回执没有 `isError`。
  原始错误文字和失败状态保留在历史中，该调用不作为训练目标，后续有效调用继续转换。
  失败回执可以没有 `evidence_id`；若提供了该字段，仍须通过原有的证据匹配校验。
  原始事件、技能快照和最终 task_notes 文件保持原状。私有执行器与评估信息只用于验证/来源记录。

基础文字压缩使用 `--k`，必须是正整数，默认 2，按**返回四张 RGB 的工具回执**计数；重复返回同一拍摄也算一个回执。
保留最近 K 次回执及其之后的历史，较早调用转换成确定性的动作/结果文本。
历史动作保留参数、结果状态、错误、control、revision 和图片引用，所有图片与视角标签保留。
已经读到的技能/技能列表和最新已返回的笔记内容及版本完整保留；旧笔记正文从较旧摘要中移除。
未结构化的传输错误保持完整。
压缩按 `tool_call_id` 找到调用，允许中间出现公开文字；这些文字的位置、内容和 mask 保持不变。

### 超长上下文

基础文字压缩后，按 `rgb_window_budget_v1` 策略处理超长样本。
`--max-len` 默认为 **24,576**；`0` 关闭预算裁剪，负值报错。
`approx_tokens` 使用 `文本字符数/4 + 图片数×64` 的粗估，包含工具定义和目标格式，
**不是 Qwen processor 编码长度**。仅严格超过预算时触发：

1. 窗口外每个较早 RGB 回执仅保留按原始交付顺序出现的第一张图片，同时删除其他图片的视角标签。
   其余动作/结果文字及历史引用保留；首图不一定是 front。
2. 若仍超限，从最早历史开始逐个删除完整调用组，包括调用 assistant、调用期间的公开文字和 tool 回执，
   每次删除后重新估算，达到预算即停止。窗口外技能、技能列表、最新 task_notes 和失败历史均可随组删除。
3. 始终保护 system/user、目标回答和最近 K 个 RGB 回执的完整调用组及其后的历史；不足 K 个时保留全部历史。
   无可删除历史时停止，保留样本并记录 `over_budget=true`。

每条样本从原始前缀独立计算，原始轨迹与目标监督不变。长度不用于筛除样本，也没有轨迹回合数门槛。
即使裁剪后粗估低于预算，训练侧仍须检查真实编码长度。

## 学生协议与来源

本项目原生 `act.target.point=[x,y]` 范围为 0–1。学生使用：

```json
{"primitive":"toggle_on","revision":0,"target":{"image_ref":"rgb-00001-left","point_2d":[571,487]},"placement_yaw_degrees":null,"wait_seconds":null}
```

转换按 Python `round(1000*v)` 量化，逆转换除以 1000；有效原坐标的量化误差不超过 0.0005。
image_ref、revision、空 target 和其他动作参数保持不变。目标参数必须符合记录时的 schema，
不会通过补零、截断范围或补参数修复无效成功目标。失败历史中的越界坐标仍保持越界；
无法表示的失败参数保留原文，并在统计中标记。

共享接口位于 `manipulation_agent.sft`：

```python
from manipulation_agent.sft import (
    build_student_context, compress_history, build_sample_messages,
    to_student_arguments, from_student_arguments,
)

# frozen_system 来自该 run 的 developer_instructions，catalog 是已校验的 MCP 工具目录。
system, tools = build_student_context(frozen_system, catalog)
student_args = to_student_arguments("act", native_args)
native_args = from_student_arguments("act", student_args)
context = compress_history(raw_oai_history, k=2)
# target_assistant 是当前目标调用，不包含它的回执或调用后的公开文字。
messages, length_stats = build_sample_messages(
    raw_oai_history, target_assistant, tools, k=2, max_len=24576,
)
```

`compress_history` 的输入是尚未压缩的完整前缀，工具消息使用 text/image blocks，
调用与结果通过 `tool_call_id` 配对。每次决策都从该原始前缀计算，不把上次压缩结果再次压缩。
`compress_history` 继续只做基础文字压缩，保留全部图片；`build_sample_messages` 在此基础上
追加目标并应用预算，返回消息与长度/裁剪统计。`episode_samples`、`prepare_dataset` 和
`Dataset.max_len` 使用同一默认预算；`max_len=0` 恢复原有消息与图片输出行为。
`build_student_context` 同步转换 schema 和提示词中的明确坐标说明；
`protocol.rewrite_interface_text` 用于学生侧技能回执中的明确 API 示例。
未来推理还应保留和验证工具 schema，坐标逆转换本身不是完整的工具参数验证器。
本次仅提供这些共享函数，不接入学生运行器或修改原生 MCP。

提示词取自 controller 保存的实际 `developer_instructions`。
工具目录通过该 controller 的 `source_snapshot` 在独立 CPU 子进程加载，
先验证源码哈希，再与 `mcp_preflight.schema_sha256/tool_names` 对照。
恢复时按归档 `tool_specs` 签名传入 `run.config.executor_profile`（或 `executor`）和
`nonofficial_actions`；两种 executor 字段同时存在但不一致时拒绝。
缺失参数只使用该归档函数自身的默认值，仅接受 profile 的历史函数继续兼容。
`sources.jsonl` 的 `catalog_restore` 记录实际参数、取值来源及归档签名不支持的已记录选项；
不尝试其他配置凑哈希。原生 schema 校验通过后才转换学生坐标接口。
子进程禁用字节码写入和模拟器导入。此操作执行归档中的 Python 工具定义，输入应为可信的运行归档。
历史版本缺失、不匹配或无法导入时报告错误；不会套用当前仓库的提示词或工具目录。

支持当前串行 Codex `item.started/item.completed` 轨迹。额外客户端 turn/resume 缺少对应用户提示词、
重叠工具调用、非 MCP 执行、损坏 JSONL 等情况会明确拒绝，
不会静默删掉上下文。只有末尾未完成的调用时，保留之前可验证的成功前缀并报告警告。
完全相同的重复 completion 不重复产生样本；冲突 completion 报错。

## 可选的旧成功回执摘要

`old_success_observation_summary_v1` 是独立版本的文字压缩策略。默认 `legacy` 行为保留；
原始轨迹转换可通过 `manip-sft --history-compression old_success_observation_summary_v1` 启用。
训练及学生推理可复用同一 CPU 函数：

```python
from manipulation_agent.sft import COMPACT_HISTORY, compress_history, compact_history

# 从原始历史构建：先执行既有历史规则，再精简旧成功观察回执。
context = compress_history(raw_oai_history, k=2, history_compression=COMPACT_HISTORY)
# 对已有的模型历史单独执行摘要；支持 text/image blocks 和已导出的字符串格式。
compact_context, stats = compact_history(existing_history, k=2)
```

保护区从倒数第二条带图片的工具回执开始，包含其后的全部消息；不足两条时不压缩。
保护区外仅处理可解析为 JSON、顶层 `ok=true` 且没有有效 `error/isError` 标记的
`act/look` 回执。删除其顶层 `observation/evidence_id/execution_budget/actions_used`，
保留实际图片、顺序、正文中的 view/ref 标签、技能、task notes、initialize、失败和
无法解析的回执，以及全部 assistant 文字、调用参数、监督目标和 loss 标记。

`effect` 的语义字段和未知字段保留。已知结构的 `local_adjustment` 在
`operation_committed=true` 且 `recovery_required=false` 时，仅保留这两个标记，删除
`repositions/primitive_attempts/departure/message`；出现未知字段或条件不符时整块保留。
`effect.message` 仅删除代码列出的两条固定执行说明。`verification=executor_operation_only`
保持原样，执行器完成不改写为任务成功。

## 数据包与验证

单条输出 `dataset.jsonl`；批量输出 `train.jsonl` 和 `val.jsonl`。
转换版本为 `manipulation_sft_v1.3.0`，数据 `schema_version` 仍为 1。附带：

| 文件 | 内容 |
|---|---|
| `images/`、`images.jsonl` | 按内容哈希去重的图片、源路径、MIME 和哈希 |
| `sample_sources.jsonl` | 每个训练行的来源、调用行号、目标工具、切分及长度估算 |
| `sources.jsonl` | 轨迹文件哈希、模型/effort、任务/场景、配置、主机和独立状态 |
| `config.json`、`stats.json` | 转换版本/源码、压缩配置、分布与验证级别 |
| `errors.jsonl`、`skipped.jsonl` | 无法转换与正常筛选的来源 |
| `build.json` | 完成或失败状态 |

`config.json` 记录 `context_policy=rgb_window_budget_v1`、`max_len` 和 `keep_all_images`。
预算启用时 `keep_all_images=false` 表示允许减图，实际是否裁剪以逐样本统计为准；关闭时为 true。
`length_limit=null` 表示没有拒绝样本的硬长度上限。
`sample_sources.jsonl` 中 `approx_tokens_before`、`image_count_before` 是基础文字压缩后、预算裁剪前的数值；
`approx_tokens`、`image_count` 是最终数值。另记录 `slim_triggered`、`slimmed_images`、
`dropped_history_pairs`、`images_removed` 和 `over_budget`。图片数量按出现次数计算，
`slimmed_images` 仅统计第一阶段减图，`images_removed` 包含随后删除调用组带走的图片。
`stats.json` 汇总前后长度分布和 `length_control` 中的触发、裁剪、剩余超限样本数及删除数量。

图片顺序来自模型实际收到的 blocks；逐张校验其 Base64 字节、观测哈希、capture 清单和存档文件。
图片路径相对于数据包根目录。训练从数据包根目录启动，或在读取时以 JSONL 所在目录解析图片；
可移动整份数据包，不需要保留原始轨迹路径。输出目录必须不存在且不能与源目录重叠。
导出失败时保留 `build.json` 与 `.incomplete` 文件作为证据；训练只使用 `status=complete` 的数据包。

CPU 检查：

```bash
python -m pytest -q tests/test_sft*.py
```

参考：`Cytosi/habitat-gs` 的 `zimeng/point_nav` 分支，commit
`c7322f4944ff182508083f3a73282e09964696d2`，
`bench/tool/nav_sft/scripts/build_nav_sft_dataset.py` 与 `bench/tool/harness/nav_window.py`。
两阶段裁剪沿用该版本先减图再删历史的顺序；本项目按 RGB 回执计 K，并支持调用期间交错的公开文字。
轨迹读取和 manipulation 规则独立实现，不依赖 Habitat 运行环境。
