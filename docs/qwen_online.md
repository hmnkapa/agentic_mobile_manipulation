# Qwen3.5-4B 在线执行

本地 Qwen 是现有通用控制器的一个客户端。任务指令、
`vision_policy.system_prompt`、MCP catalog、技能、task_notes、四路 RGB、
执行器及独立任务评分沿用 Codex 路径。模型自己选择 initialize、动作和 finish。

模型侧与 SFT 使用同一套坐标协议：`act.target.point_2d=[x,y]`，两项均为 0–1000
的整数，x 从左到右、y 从上到下。系统提示、工具 schema 和技能文档的接口示例
复用 SFT 转换；模型历史也保留 `point_2d`。调用 MCP 前才将坐标除以 1000，
改为原生 `act.target.point=[x,y]`、范围 `[0,1]`；例如 `[250,750]` → `[0.25,0.75]`。
`image_ref`、revision 和其他参数保持原值，release/wait 继续使用 `target=null`。
不继承训练图片预算或历史压缩。
模型的 tokenizer、原生 chat template、视觉 processor 和上下文容量属于模型属性；
Codex CLI 内部提示和上下文管理不宣称与本地模型逐 token 相同。

## 模型环境与配置

在独立 Python 3.11 环境安装项目及 `requirements/inference.in`。
普通核心导入不加载 torch、transformers 或模拟器。
模型文件、环境和轨迹放在 Git 外。配置只接受以下位置与设备信息：

```toml
python = "/data/qwen-online/env/bin/python"
model = "/data/models/Qwen3.5-4B"
device = "cuda:0"
gpu_uuid = "GPU-your-physical-device-uuid"
# LoRA 轨迹才增加此行，必须明确选择 checkpoint：
# adapter = "/data/training/swift/checkpoint-1122"
```

路径相对于配置文件解析，允许命名环境变量。CUDA 必须指定物理 GPU UUID，
子进程通过 `CUDA_VISIBLE_DEVICES` 只看见该设备。CPU 诊断用 `device="cpu"`
并省略 gpu_uuid。GPU 使用 BF16，CPU 使用 FP32。
仓库内 `configs/inference/` 提供基座和 LoRA 配置示例。

LoRA 使用指定基座的 tokenizer、processor 和 generation config，
仅通过 PEFT 增加 adapter 权重，不从 checkpoint 读取在线行为设置。
加载不要求 optimizer、RNG、原训练机器路径或训练契约可用。
权重、配置和可用训练 marker 归档；基座与 LoRA 均使用上述 SFT 坐标协议，
checkpoint 的训练 marker 不覆盖在线配置。

## Bash 单任务及批量入口

使用 `scripts/run_qwen_behavior.sh` 可以直接复用原 BEHAVIOR 监督器，
不必手工分别启动 episode 和模型。`--runner-template` 指向已有站点的 runner JSON；
默认从其中 `batch` 目录读取 `manifest.json`，也可用 `--manifest FILE` 明确指定。
模板中的 SSH、资产、模拟器、准入、初始化/清理 watchdog 和录像选项继续使用。
任务指令、instruction hash、实例、seed、技能 profile 和 goal-skill 设置均来自原清单。
脚本只选择任务，不生成目标点或动作序列。

2026-11-07 已在脚本开头的“本机配置区”填写当前机器的实际路径，每组第一项启用，
备选项注释保留。切换时先注释当前赋值，再取消所选备选项的注释：

- 基座为本机固定 `851bf6e...` 快照；`lora=""` 默认仅使用基座。
- LoRA 备选为已有 `checkpoint-1122`、`checkpoint-1120`、`checkpoint-720`，仅显式选择一个。
- 控制器使用项目 `.venv`；推理使用已有 `/tmp/mas-qwen-native-20261107-7_fq5k15/env/bin/python`。
  该临时环境可能被清理；原训练环境缺 MCP，不能直接替代它。
- runner 默认 `configs/inference/behavior_local.json`：本机 GPU 0 推理、本机 GPU 1 仿真，
  worker 为 `local-gpu1`，默认仅 radio；两张卡分别通过物理 UUID 校验。
- 本机传输通过 `local_worker_transport.py --env ... local-v02` 直接执行命令，
  预检、MCP 和 systemd 仿真均使用本机模拟器环境，不经过 SSH。
  `behavior_local_env.sh` 沿用已有 runtime、资产及只读上游包，不修改原环境。
- 原 100 任务清单已复制到忽略 Git 的 `operations/qwen_local_runtime/manifest.json`；
  来源和哈希记录在同目录的 `local_setup.json`。静态资产检查所需的
  `operations/runtime_local/src/BEHAVIOR-1K` 指向已有只读源码。
- 输出默认在 `/home/liqi/agentic_mobile_manipulation_runs/qwen_behavior/` 下创建独立时间戳目录。
  仿真输出和缓存使用本机 NVMe 上的 `operations/runtime_local/`，
  共享锁使用 `operations/resource_leases_local/`。保留 14 GiB 仿真显存门槛和原有资源等待。

```bash
# 先只生成配置；不加载权重、不启动任务。
scripts/run_qwen_behavior.sh --prepare-only

# 正式执行当前注释选定的配置；沿用原有干净冻结源码检查和资源准入。
scripts/run_qwen_behavior.sh

# CLI 可覆盖配置区的任务和 worker；首次 --worker 会替换默认 worker，而不是追加。
scripts/run_qwen_behavior.sh --all --limit 10 --worker local-gpu1
```

远程 runner 作为注释备选保留；切换时须同时选择该模板内的 worker。
新运行不继承另一控制器的 Codex API 环境要求、控制器 PATH 或本地 spool 路径，
使用当前控制器环境及监督器原有的本地默认 spool。下面保留显式 CLI 覆盖方式供换站点时使用。

将下面的路径和 GPU UUID 替换为实际值。`--worker` 是原模板中的仿真 worker ID，
可以重复指定；省略时使用脚本配置区的 worker。每个选中 worker 都必须提供原有的
`expected_gpu_uuid`，且与 Qwen 推理 GPU 不同。模型环境须安装本页所列推理依赖。

```bash
common=(
  --runner-template /data/behavior/site/runner.json
  --controller-python /data/controller-env/bin/python
  --qwen-python /data/qwen-online/env/bin/python
  --model /data/models/Qwen3.5-4B
  --gpu-uuid GPU-your-inference-device-uuid
  --worker gpu0
)

# 基座：默认仅 turning_on_radio，实例和 seed 保留原清单值。
scripts/run_qwen_behavior.sh "${common[@]}" \
  --output /data/online/radio-base

# LoRA：明确选定 PEFT checkpoint；其余在线接口与基座相同。
scripts/run_qwen_behavior.sh "${common[@]}" \
  --lora /data/training/swift/checkpoint-1122 \
  --output /data/online/radio-lora-1122

# 批量：原清单顺序中的前十项。--limit 是任务数，不是 episode 动作预算。
scripts/run_qwen_behavior.sh "${common[@]}" \
  --all --limit 10 --output /data/online/qwen-first10

# 批量选定任务；使用 --all 且省略 --limit 才选择完整清单。
scripts/run_qwen_behavior.sh "${common[@]}" \
  --task turning_on_radio --task clean_a_keyboard \
  --output /data/online/qwen-selected
```

这些命令分别创建新运行，输出目录必须不存在且位于 Git 和权重目录之外。
新目录包含 `qwen.toml`、`manifest.json`、`runner.json` 和 `launch.json`。
任务保留原输入定义，使用新 run ID 和 systemd batch tag；旧执行结果、进程归属和
接管配置不会复制为新成绩。报告写入新目录的 `reports/`，保留原共享资源锁目录；
supervisor spool 使用原监督器按当前 UID 和新 batch 路径生成的本地默认位置。
原清单和模板文件不会改写，已有推理配置中的 adapter 不会被自动沿用。

只准备配置、暂不加载模型或启动模拟器时，添加 `--prepare-only`：

```bash
scripts/run_qwen_behavior.sh "${common[@]}" \
  --output /data/online/radio-prepared --prepare-only

# 之后由原监督器恢复同一运行，保留已完成、失败及进行中的记录。
scripts/run_qwen_behavior.sh \
  --controller-python /data/controller-env/bin/python \
  --resume /data/online/radio-prepared
```

执行与恢复都要求干净、冻结的源码；`--prepare-only` 允许未提交修改，且不运行测试。
脚本将自身 checkout 设为运行源码，所有仿真 worker 需能访问该 checkout 的相同绝对路径，
沿用原调度器的部署约定。准备时源码若已干净，会锁定 commit；存在本地修改时先记录
该状态，正式执行仍由监督器核对并记录实际冻结源码。
`--resume` 只接受原运行目录和控制器 Python，不重新选择权重、任务或重试已失败任务。
中断后的接管和收尾仍由原监督器按进程归属执行。

`simulator_gpu_memory_admission` 控制仿真显存容量准入，未配置时默认为 `true`。
设为 `false` 会跳过调度预检、GPU 候选筛选和进程内显存容量判断；
GPU 存在性、UUID、资源锁及其他资源检查继续生效，原显存门槛保留供恢复。

执行和 `--resume` 默认在终端用英文实时显示批次进度、当前任务序号/名称、instance、seed、
任务指令、worker、初始化阶段、模型加载状态、推理轮次和 token 数，以及每次 MCP
工具调用的动作参数、返回状态、错误和 act/look 请求计数。每秒读取一次已有的本地
模型事件；等待初始化、资源或推理时每 30 秒输出一次状态。任务结束后分别显示独立
评估结果与 Q 分、finish 确认、控制器状态、录像验证及记录路径，工具成功或 finish
确认不会代替官方任务成功。完整事件和图像仍保存在原日志中；终端只显示摘要。
直接调用 `scripts/run_behavior100.py` 时可用 `--console-progress` 启用相同显示。

## 启动原有 episode 和本地控制器

先用已有模拟器环境准备 episode：

```bash
PYTHONPATH=src /path/to/simulator/python -m manipulation_agent.vision_cli \
  --backend omnigibson --policy serve --agent-profile skills \
  --task turning_on_radio --instance 301 --seed 0 --unbounded-episode \
  --controller qwen --record-video --port 29440 \
  --output /data/online/radio-base/episode
```

再使用原通用入口：

```bash
PYTHONPATH=src /path/to/controller/python scripts/run_model_controller.py \
  --client qwen --qwen-config /data/online/base.toml \
  --instruction "Turn on the radio receiver that's on the table in the living room." \
  --agent-profile skills --goal-skill hierarchical --timeout 0 \
  --mcp-command /path/to/controller/python \
  --mcp-args-json '["-m","manipulation_agent.mcp_server","--bridge","http://127.0.0.1:29440"]' \
  --output /data/online/radio-base/controller
```

控制器解释器需要项目和 MCP；配置内解释器还需模型依赖。
Qwen 默认 `--model Qwen/Qwen3.5-4B`，实际身份以基座、adapter 字节哈希为准。
现有 Codex 命令、模型与 low 设置不变。Qwen 不接受 GPT 的 reasoning-effort；
thinking 使用基座模板默认，基座与 LoRA 相同。
生成在基座模型的 EOS 或 tokenizer 声明的回合结束 token 处停止，避免把下一角色的
虚构文本混入工具调用；两类权重使用相同的停止规则。

每条轨迹只加载一次权重，使用真实官方 MCP SDK。调用结果进入后续历史时，
仅 `read_skill` 回执的 `text` 字段按 SFT 规则转换接口描述；task_notes 和其他反馈保留原文。
图片逐张验证 SHA-256 并保存原始字节、视角与 image_ref。没有额外观察或轮询动作。
客户端严格校验 `point_2d` 的整数类型、长度和 0–1000 范围，不接受旧 `point` 字段、
混合字段、布尔值或小数，不夹紧或猜测单位。转换后的原生参数仍由 harness 校验。
MCP 派发后发生传输异常记录为 execution_uncertain，不自动重发动作。

原生 XML 参数兼容模板的标量写法：非字符串参数同时接受 `None/True/False`
和 JSON 的 `null/true/false`；nullable 字符串参数中的 `None/null` 表示空值。
普通字符串及其空白原样保留，对象和数组内部仍须使用严格 JSON。
兼容解码不修正参数类型；坐标通过上述 SFT 转换后，其他语义错误继续交给 harness 校验；
离线评估解析器仍使用原有严格规则。

每次响应的全部工具调用解析和坐标转换通过后才派发。缺少 `</think>`、损坏的工具调用结构、
未知工具、非法参数编码、非法学生坐标或结束时没有工具调用，均作为可恢复的协议错误：
客户端在同一会话中追加来源为 `qwen_native_client_protocol` 的 user 反馈，
说明错误码、可定位的工具/参数、当前响应未执行任何工具，并请求重新生成完整调用。
原有有效历史和四路图片继续保留；格式重试不会自动 initialize、look 或增加 act/look 计数。
连续格式失败没有次数上限，直到生成有效调用，或遇到上下文耗尽等运行故障。
该恢复仅适用于派发前的解析和坐标校验，OOM、上下文耗尽和 execution_uncertain 仍按故障处理。

所有公共历史与图片都保留。没有累计动作、轮数、token 或墙钟截止。
单次输出在 EOS 或原生剩余上下文处结束；没有继承离线 2048 输出上限或 17408 输入预算。
上下文耗尽、OOM 等终止故障交现有监督器收尾；缺少正式 finish 仍单独记录。
模型加载有独立的 1800 秒初始化 watchdog；它不限制任务执行时间。

## 批量、证据及验证

`prepare_behavior100.py --client qwen --qwen-config FILE` 接受原有 catalog、BDDL、
capabilities 等参数，准备相同任务列表及指令。所有新清单统一使用 unbounded，
历史与已执行 attempt 的原政策不改写。基座和 LoRA 使用不同 cohort 和输出目录。

已有批处理监督器增加推理 GPU 租约，复用物理 GPU 锁命名空间；worker 可覆盖
qwen_config，并须声明模拟器 expected_gpu_uuid。模型与模拟器使用不同物理 GPU，
资源不足时保持等待，不停止其他任务。监督器重启会核对原控制器及模型 PID 的归属。

`controller.json`、`model_events.jsonl`、canonical/compat 事件继续用于原 replay/audit。
附加的 `qwen_request.json`、`qwen_inputs.json`、`qwen_runtime.json`、
`generations.jsonl` 和 `received_images.jsonl` 保存公共输入、配置、权重、原始生成、
token 指纹和实际图片证据。原始 think 内容不冒充公开 reasoning summary。
`qwen_inputs.json` 保存实际传给模型的提示与 OpenAI 格式工具 schema，同时保留
`native_inputs` 和两侧输入指纹；`qwen_request.json` 保留原始控制器输入。
接口标记为 `sft_point_2d_integer_0_1000`，MCP 接口仍为 `native_mcp_point_0_1`。
工具事件的 `arguments` 记录实际 MCP 原生参数，`model_arguments` 记录对应模型参数；
原始 MCP 回执保留在事件中，便于和模拟器调用记录对齐。历史运行记录不改写。
失败生成仍完整保存在 `generations.jsonl`；`model_events.jsonl` 中的
`generation.rejected` 是非终止诊断，记录轮次、连续失败次数、结构化错误、
实际反馈消息及零派发标记。每次格式失败也记录 `turn.completed` 的 token 用量，
其 status 为 `retrying`，不会生成 `turn.failed` 或虚构 MCP 工具回执。
成功仍读取 episode 的独立官方评估记录，finish、退出及录像质量分别报告。

```bash
PYTHONPATH=src:tests python -m pytest -q \
  tests/test_qwen_native.py tests/test_qwen_native_mcp.py
```

这里的 CPU fixture 是接口验证，不代表视觉自主任务成功。
真实验收需在同一冻结源码及资产下分别运行 Codex、Qwen 基座和明确指定的 LoRA，
使用 turning_on_radio / instance 301 / seed 0，逐条记录官方评分与验证级别。
本次操作、真实验证状态及失败证据见 `operations/qwen_native_20261107/`；
未完成的模型或模拟器检查不得标记通过。

2026-11-07：真实 processor 的零图、四张 512×512 图片、74,642 token 完整历史检查已通过。
真实 SDK→bridge→CPU fixture 已覆盖七个工具、超过 50 次动作、非法参数及丢失回执。
真实 CPU 基座生成暴露了本地快照 EOS 与 tokenizer 回合结束 token 不一致，已修正在线停止规则；
修正后的真权重检查和 Codex/基座/LoRA 的 OmniGibson 单任务验收按用户要求暂缓。
已有生成诊断和失败记录保留；不将这些组件记录计为自主任务成功。

2026-11-08：原生标量兼容和格式反馈恢复完成组件验收。CPU MCP fixture 覆盖连续
六次格式失败后继续执行、整轮零部分派发、失败用量记录及终止故障边界；真实 processor
验证 `None/True/False` 历史调用往返解析，以及反馈消息加入后四路 RGB 编码不变。
三个历史故障样本已重放，原始记录保留。验证记录位于
`operations/qwen_native_20261107/format_recovery_20261108/`；本次未加载模型权重或启动新 BEHAVIOR 批次。
