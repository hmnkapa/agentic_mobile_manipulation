# Qwen3.5 4B SFT 工具链

`manip-train` 消费本项目的完整 SFT 数据包，提供结构检查、全量真实编码、LoRA
训练、checkpoint 恢复和权重合并。数据转换继续由 `manip-sft` 负责；转换格式及历史
语义见 [数据说明](sft_dataset.md)，模块边界见 [架构说明](sft_training_architecture.md)。

CPU 测试通过不代表真实编码、GPU 训练或任务成功。每项运行的验证级别和状态写在对应
JSON 报告中。四步冒烟用于验证训练基础设施，不用于报告模型任务成功率。

## 启动脚本

已有独立训练环境时，可直接使用仓库脚本。默认直接启动完整 LoRA 训练，
不执行启动前的 CPU 全量编码预检或第二轮 loader 编码核验；模型与数据默认使用本机已准备的固定版本：

```bash
cd /home/liqi/agentic_mobile_manipulation

# 只检查启动计划，不编码、不训练、不创建输出目录。
bash scripts/train_qwen35_4b.sh --gpu 0 --dry-run

# 直接启动正式训练；W&B 沿用 recipe 和环境变量。
bash scripts/train_qwen35_4b.sh --gpu 0

# 验证训练基础设施：16 条 train、8 条 val，第 2 步保存后停止。
bash scripts/train_qwen35_4b.sh --gpu 0 --smoke --wandb-mode offline
```

默认工作根目录是 `/mnt/data/home/liqi/manipulation_sft`，解释器为其下的
`env/bin/python`，模型是 `models/Qwen3.5-4B/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`。
当前数据包为 `/mnt/nas/home--liqi/latest94_scene_split_real_tokens17408_20261104`，
含 2,956 条训练样本和 334 条验证样本，均已通过 17,408 真实 token 上限的编码检查。
K=2 保持不变；另外 33 条无法达标的训练样本保存在包内 `over_budget/train.jsonl`，
不进入主训练/验证文件。脚本显式使用 17,408 的预算。启动时仍检查数据包结构、来源与
文件哈希、模型文件和 GPU 资源，然后准备训练输入；不逐条调用 processor 编码。

训练结果保存在工作根目录的 `runs/` 下；只有显式 `--preflight-only` 才在 `preflights/` 下创建新预检。
每次使用唯一目录，不删除失败记录。`--dry-run` 只检查 CPU 启动计划，
不证明编码、样本覆盖、CUDA 资源或 W&B 登录已验证。冒烟的恢复步骤见下文。

只做 CPU 真实预检，或明确复用一份已有报告：

```bash
bash scripts/train_qwen35_4b.sh --preflight-only
bash scripts/train_qwen35_4b.sh --gpu 0 \
  --preflight /data/sft_checks/encode_001 --output /data/sft_runs/full_001
```

`--preflight` 是可选参数。提供时由 `manip-train` 验证已有报告的数据、模型、模板、依赖与源码指纹，
并使用报告中的长度信息，不重新编码。省略时，`run.json` 和 `training_contract.json` 的
`startup_validation.encoding_preflight=not_run`，`loader_validation.json.status=not_run`；
不将跳过检查记录为通过。编码和超长错误由原生训练的数据加载过程报告，`strict=true` 保持启用。
可用
`--work-root`、`--python`、`--data`、`--model`、`--recipe`、`--launch`、`--max-length`、
`--encoding-workers` 覆盖设置，完整参数见 `bash scripts/train_qwen35_4b.sh --help`。
`SFT_WORK_ROOT`、`SFT_PYTHON`、`SFT_DATA`、`SFT_MODEL_DIR` 也可配置默认路径。
路径相对于调用目录解析，脚本从任意工作目录使用当前 checkout 的源码。

## 每图 256 token 的独立配置

`configs/training/qwen35_4b_lora_img256.toml` 是低显存实验 recipe：每张图片最多
256 个视觉 token，真实上下文预算 17,408，K=2。LoRA、BF16、视觉冻结及其他训练
参数沿用原 recipe。原 `qwen35_4b_lora.toml`、默认启动数据和既有数据包保持原样。
图片 token 上限降低会减少输入细节，训练显存峰值仍需另做 GPU 实测。

真实裁剪使用同一 recipe 的 Qwen processor/template：超预算时先精简窗口外旧图片，
再从最早的完整调用组开始删除，每次以真实 token 长度判断是否停止。system/user、
工具定义、目标监督和最近 K 个 RGB 回执及其后历史受到保护；仍无法达标的行单独
保存在新包的 `over_budget/`，保留原 split 和目标，不混入主训练文件。
裁剪搜索使用原生文本 token 与真实 processor 图片网格展开，输出每行还会进行完整
processor 编码和反事实 loss-mask 检查，再执行独立的 `manip-train preflight`。

生成与裁剪时均须明确使用新 recipe；`sft.retrim --recipe` 必须与其基线预检一致：

```bash
"$SFT_PYTHON" -m manipulation_agent.training.cli preflight \
  --data /data/source_package --model "$SFT_MODEL_DIR" \
  --recipe configs/training/qwen35_4b_lora_img256.toml \
  --max-length 17408 --output /data/checks/img256_before_trim

"$SFT_PYTHON" -m manipulation_agent.sft.retrim \
  --data /data/source_package --model "$SFT_MODEL_DIR" \
  --recipe configs/training/qwen35_4b_lora_img256.toml \
  --baseline-preflight /data/checks/img256_before_trim \
  --max-length 17408 --out-dir /data/img256_trimmed
```

`retrim` 保留无法裁剪的行并在 `trimming_report.json` 中标记；如有这类行，必须先
单独归档，随后对主数据包重新预检。不能把 `retrim` 完成直接当作全量可训练。
新配置的编码结果不能复用每图 1,024 token 的旧预检。

本轮低显存版本使用相同的 94 条轨迹和原始场景划分，主数据包含 2,985 条 train、
334 条 val；4 条受保护上下文仍超长的样本完整保存在新包的 `over_budget/`。
新数据目录为 `/mnt/nas/home--liqi/latest94_scene_split_img256_real_tokens17408_20261105`，
对应预检为 `/mnt/data/home/liqi/manipulation_sft/preflights/lowmem_img256_real17408_20261105`。
使用独立配置与数据检查启动计划：

```bash
bash scripts/train_qwen35_4b.sh --gpu 0 \
  --recipe configs/training/qwen35_4b_lora_img256.toml \
  --data /mnt/nas/home--liqi/latest94_scene_split_img256_real_tokens17408_20261105 \
  --preflight /mnt/data/home/liqi/manipulation_sft/preflights/lowmem_img256_real17408_20261105 \
  --dry-run
```

实际训练移除 `--dry-run`。这组数据与配置的 CPU 编码检查不代表 GPU 显存峰值或训练
效果已经验证。运行、恢复及故障证据保存在 `operations/sft_lowmem256_20261105/`。

### 每图 64 token 配置

`configs/training/qwen35_4b_lora_img64.toml` 将每张图片的 token 上限降为 **64**，
其余训练参数与 img256 recipe 相同：上下文上限 17,408、K=2、LoRA r16/alpha32、
学习率 1e-4、batch=1、梯度累积 8、3 epoch、梯度检查点和 W&B 记录。
W&B 使用独立的 `img64` 标签。方形截图经处理器缩为 256×256，对应 64 个视觉 token。

可复用现有 img256 数据包，样本和历史文本保持不变，图片在编码时按新上限处理：

```bash
scripts/train_qwen35_4b.sh --gpu 0 \
  --recipe configs/training/qwen35_4b_lora_img64.toml \
  --data /mnt/nas/home--liqi/latest94_scene_split_img256_real_tokens17408_20261105 \
  --dry-run
```

移除 `--dry-run` 即启动训练。需要新配置的全量 CPU 编码预检时，将 `--dry-run`
替换为 `--preflight-only`，得到新的预检目录；随后可通过 `--preflight DIR` 用于训练。
图片参数属于编码契约，img256 的预检报告不能作为 img64 的报告复用。
原始、img256 配置、默认启动配置和既有数据包均保留。

### 每图 64 token 与旧成功回执摘要

`configs/training/qwen35_4b_lora_img64_compact.toml` 在 img64 参数基础上设置
`data.history_compression="old_success_observation_summary_v1"`，并增加 `compact-history`
W&B 标签；原有三个 recipe 和启动脚本默认值保留。压缩范围和共享推理接口见
[数据说明](sft_dataset.md#可选的旧成功回执摘要)。

重生成的数据保留当前 3,319 条样本及场景划分（2,985 train / 334 val），
不删除图片或监督目标。原包中的 4 条超长归档仍保存在原包 `over_budget/`。

```bash
bash scripts/train_qwen35_4b.sh --gpu 0 \
  --recipe configs/training/qwen35_4b_lora_img64_compact.toml \
  --data /mnt/nas/home--liqi/latest94_scene_split_img64_compact_real_tokens17408_20261105 \
  --preflight /mnt/data/home/liqi/manipulation_sft/preflights/lowmem_img64_compact_real17408_20261105 \
  --dry-run
```

移除 `--dry-run` 启动训练。全量真实 CPU 编码由新配置独立验证，不复用 img256 或
未压缩 img64 的编码报告。运行证据在 `operations/sft_compact64_20261105/`，
完整原生编码报告在 `/mnt/data/home/liqi/manipulation_sft/compact64_20261105/preflight`。
发布目录的预检仅复用这一轮相同数据字节、模型、配置和代码下的已验证编码缓存。

## 安装独立环境

使用 Python 3.11，环境、模型、缓存和输出放在 Git 外部。模型是
`Qwen/Qwen3.5-4B` 的完整 Hugging Face 权重。记录 snapshot revision；如果使用本地
下载目录，可附上 `model_manifest.json`，其中包含 `model_id` 和 `revision`。
工具还会计算配置、processor、tokenizer 和权重文件的哈希。

```bash
export SFT_WORK_ROOT=/mnt/data/home/liqi/manipulation_sft
mkdir -p "$SFT_WORK_ROOT/tmp" "$SFT_WORK_ROOT/cache/pip"
export TMPDIR="$SFT_WORK_ROOT/tmp"
export PIP_CACHE_DIR="$SFT_WORK_ROOT/cache/pip"
export HF_HOME="$SFT_WORK_ROOT/cache/huggingface"

python3.11 -m venv "$SFT_WORK_ROOT/env"
export SFT_PYTHON="$SFT_WORK_ROOT/env/bin/python"
```

先按目标 GPU 驱动安装官方 CUDA PyTorch 和对应 torchvision，再安装本仓库依赖：

```bash
"$SFT_PYTHON" -m pip install -r requirements/training.in
"$SFT_PYTHON" -m pip install -e .
```

ms-swift 适配版本固定为 **4.6.0**。依赖锁必须来自实际环境的验证结果，不要将候选
依赖解析结果当作训练通过。将真实验证后的 `pip freeze` 保存为
`requirements/training.lock.txt`，在注释中注明编码、训练、恢复及导出的验证状态。

`MAX_PIXELS`、`MIN_PIXELS` 等旧环境参数会与 recipe 的图片参数冲突，执行前应去除。
图片参数在导入训练库之前设置；在同一 Python 进程内切换已初始化的图片配置会报错。

## 配置

基础 recipe 是 `configs/training/qwen35_4b_lora.toml`。`--smoke` 加载同目录下的
`smoke.toml`，再应用 `--launch`。launch 只允许包含进程和资源设置。

默认 LoRA 为 r16/alpha32/dropout0.05、BF16、冻结视觉编码器和 aligner、学习率
1e-4。`loss_scale=default`、`disable_ignore_empty_think=true` 保证空 think 仍计入监督。
历史 assistant 的 loss=false，只有最后目标计 loss。框架实际可训练参数由回调检查。

ms-swift 4.6.0 的原生 default 对最终 EOS 不继承 message.loss。适配层注册命名空间
扩展 `default+manipulation_message_mask`，仅在最终 response 全部被屏蔽时，将其
SUFFIX 按 OTHER 处理。recipe 仍表达 default 监督语义，真实 argv 和编码契约记录扩展名。
这项修正不改变任何输入 token；预检与训练使用相同扩展，并由反事实编码验证。

recipe 默认长度预算为 16,384，图片 token 上限为每图 1,024。它们不是数据转换筛选条件。
本轮全量检查发现验证集最短 finish 为 16,841 tokens，经用户明确确认将冒烟预算改为
**17,408**。下面的真实编码和冒烟示例用 `--max-length 17408` 显式覆盖；基础 recipe
仍保留默认值。命令行覆盖发生在 recipe、smoke 和 launch 合成之后，恢复时禁止覆盖预算。
转换器 v1.3.0 的 `rgb_window_budget_v1` 策略使用 `K=2` 和默认粗估预算 24,576，
超限时先减旧图，再删完整旧调用组，保护最近窗口和目标；详见[转换指南](sft_dataset.md#超长上下文)。
转换预算是软上限，不能替代这里的真实 token 长度检查。训练读取器同时支持 v1.3.0 策略包和
既有 v1.2.1 全图包，拒绝未知策略；recipe 的 `data.keep_all_images=true` 表示保留数据包内全部图片，
原始轨迹是否减图由数据包的上下文策略记录。读取器保持包内图片和消息原样。
策略变更需要生成新的数据包并重新预检；包内配置参与缓存指纹。未来学生推理需采用相同的上下文策略。

## 结构检查和真实编码

设置实际的模型目录与每次新建的输出目录：

```bash
export SFT_DATA=/mnt/nas/home--liqi/latest94_scene_split_real_tokens17408_20261104
export SFT_MODEL_DIR=/path/to/complete/Qwen3.5-4B

manip-train preflight --data "$SFT_DATA" \
  --structure-only --output /data/sft_checks/structure_001

manip-train preflight --data "$SFT_DATA" --model "$SFT_MODEL_DIR" \
  --recipe configs/training/qwen35_4b_lora.toml \
  --max-length 17408 \
  --output /data/sft_checks/encode_001
```

`--structure-only` 只依赖标准库。它检查所有来源、split、行配对、图片文件和哈希，
写出可从任意工作目录读取的 `prepared/train.jsonl`、`prepared/val.jsonl`。
它的 `encoding_passed` 始终为 false，不能作为 `train --preflight` 的编码证据；训练可省略该参数。

真实预检不加载模型权重，使用实际 ms-swift 数据加载与模板编码，逐条测量完整长度。
输出包括每条样本的 token、图片及监督统计、输入/标签摘要、失败历史标记和来源。
反事实编码只关闭最终 assistant 的 loss，要求输入 token 不变、所有标签均变为 -100。

默认逐条编码。CPU 内存充足时可使用 `--encoding-workers 2` 启动两个独立 processor，
每个进程只处理当前样本，最多保留两倍进程数的待处理记录，并按原始顺序写报告。
该选项不改变 ms-swift 数据读取的 `dataset_num_proc=1` 或编码契约；结构检查不使用并行编码。

已结束的真实编码报告会生成 `encoding_cache.json`，中断时已完成的样本也可以封存。
用 `--encoding-cache /data/sft_checks/encode_001` 在新输出目录继续检查；数据、模型、
模板、图片设置、依赖与编码代码的指纹必须完全相同，缓存结果文件本身也会校验哈希。
只复用已通过的真实编码结果，失败样本重新检查；CPU 模拟结果不能用作缓存。
汇总的 `cache_hits` 和逐条 `encoding_cache_source` 保留来源。训练显式复用报告时验证报告与输入契约，
不再对所选样本进行第二轮 CPU 编码。

`summary.json` 的字段含义：

| 字段 | 含义 |
|---|---|
| `scan_complete` | 是否检查了全部样本 |
| `encoding_passed` | 是否所有图片、模板和标签检查通过 |
| `eligible_samples` | 当前预算及模型容量以内的 train/val 数量 |
| `full_dataset_trainable` | 是否完整数据都满足当前训练预算 |

编码成功但存在超长样本时，预检可以完成；完整训练显式提供该报告时会根据报告拒绝启动。
未提供报告时不提前判断全量编码长度，超限错误在训练实际读取样本时报告。
结构、图片或标签错误会让预检失败。`samples.jsonl` 和 `errors.jsonl` 保留每项结果，
报告哈希保存在 summary 中。修改数据、模型、编码代码、模板、依赖或图片处理配置后，
不能继续复用旧预检。

## 训练、冒烟和恢复

先做只读 dry-run。该模式不要求已安装训练框架；训练和 dry-run 均可省略预检，它不会声称样本选择或环境
已经验证，也不创建输出目录。

```bash
manip-train train --data "$SFT_DATA" --model "$SFT_MODEL_DIR" \
  --recipe configs/training/qwen35_4b_lora.toml \
  --launch configs/training/launch/single_gpu.toml \
  --smoke --max-length 17408 --dry-run --output /data/sft_runs/smoke_001
```

确认所选 GPU 有足够空闲资源，显式设置 `CUDA_VISIBLE_DEVICES`，然后执行：

```bash
manip-train train --data "$SFT_DATA" --model "$SFT_MODEL_DIR" \
  --recipe configs/training/qwen35_4b_lora.toml \
  --launch configs/training/launch/single_gpu.toml \
  --smoke --max-length 17408 --output /data/sft_runs/smoke_001
```

冒烟从 train 选择 16 条、从原 val 选择 8 条。覆盖七种目标工具、无图、多组图片和失败历史。
无预检报告时按图片数优先选择结构覆盖，再加入图片最多的样本及确定性补齐样本，
`selection.json.lengths_verified=false`，不生成推测的 token 数。
显式提供 `--preflight DIR` 时沿用报告中的真实长度筛选，并包含预算内最长样本。
选择算法、sample_id 和原因写在 `selection.json`。
覆盖不足直接报错，不复制样本、不自动扩大长度预算。

第一段始终按四步总学习率计划创建训练器，在第 2 步保存、验证并停止。
`run.json.status=awaiting_resume` 表示第一段完成。取 `run.json.checkpoint.path` 作为恢复输入：

```bash
manip-train train --resume-from /actual/path/to/checkpoint-2 \
  --output /data/sft_runs/smoke_001_resumed
```

恢复只能指定新输出目录，参数从 checkpoint 绑定的不可变契约中读取。优化器、调度器、
RNG、样本和数据跳过行为交给原生 trainer 恢复并审计，最终应达到第 4 步及完整一轮的
16 条训练数据。仅加载 adapter 后从零训练不属于恢复。

去掉 `--smoke` 是完整训练接口。本轮实现验证不要求启动完整微调。
启动时不自动验证全部样本的编码长度；框架继续使用 lazy_tokenize=true、strict=true，
在实际读取样本时报告编码错误，禁止静默替换错误行。
恢复同样不执行全量编码核验；如果原契约绑定了预检报告，则继续验证该报告，否则保存 `preflight=null`。

## W&B 实验管理

新训练默认上报到
[qili0502-zhejiang-university / agentic_distillation](https://wandb.ai/qili0502-zhejiang-university/agentic_distillation)。
`configs/training/qwen35_4b_lora.toml` 的 `training.report_to="wandb"` 和 `[wandb]`
保存默认目标；在线认证仍由训练机器上的 W&B 登录或 `WANDB_API_KEY` 环境变量提供。
API key 不写入 recipe、命令行、Git 或运行记录。
W&B 0.31 使用 `https://api.forge.coreweave.com` 作为默认云端 API；接入代码与它保持一致，
因此可读取新版 `wandb login` 保存的凭据，也兼容旧 `api.wandb.ai` 的登录记录。
云端 run 链接可能显示为 `forge.coreweave.com/wandb/...`，entity 和 project 保持一致。
见 [W&B 0.31 发布说明](https://github.com/wandb/wandb/releases/tag/v0.31.0)。

```bash
"$SFT_PYTHON" -m pip install -r requirements/training-wandb.in
"$SFT_WORK_ROOT/env/bin/wandb" login
```

W&B SDK 固定为 0.31.0。`training.in` 已包含它；`training.lock.txt` 保留此前真实编码
环境的历史快照，不包含这次新增依赖。`training-wandb.lock.txt` 在历史锁上固定实际安装的
新增依赖，通过 CPU 离线 SDK 验证；它不代表新环境已通过 Qwen 编码或 GPU 训练。
先安装 SDK。需要真实编码证据时可在新目录单独执行预检；
代码和依赖变化会使旧预检／checkpoint 的环境指纹不匹配，显式复用和恢复仍核对这些指纹，不改写历史证据。

普通训练沿用上面的 `manip-train train` 命令。可加以下参数命名和分组：

```bash
--wandb-name qwen35-4b-lora-baseline --wandb-group latest94
```

`--wandb-mode offline` 将记录写到本次输出目录的 `wandb/`，无需联网或登录；
`--wandb-mode disabled` 完全关闭 W&B，保持原生训练入口。
新训练的优先级是 CLI 的 `--wandb-{entity,project,mode,name,group}`，再到
`WANDB_ENTITY`／`WANDB_PROJECT`／`WANDB_MODE`／`WANDB_NAME`／`WANDB_RUN_GROUP`，最后是 recipe。
dry-run 不导入 W&B、不创建 run，也不要求登录。

W&B 通过原生 Transformers 回调记录 `train/loss`、`eval/loss`、学习率、梯度范数、
global step、训练吞吐及 SDK 的系统资源指标。额外记录解析配置、数据和模型指纹、模型 revision、
样本数量、源码摘要、依赖、主机、解释器、GPU UUID、worker PID 和验证范围。
仅全局 rank 0 创建 run。`wandb.json` 保存 run ID、在线 URL 或离线 `sync_dir`；
控制器结束时也将它放入 `run.json.wandb`。

checkpoint 在本地完成哈希验收后，把路径、步数、指纹和 best checkpoint 写入 W&B summary。
默认关闭模型权重上传、梯度直方图、控制台捕获和代码上传；数据图片和原始轨迹不作为附件上传。
离线同步使用 `wandb.json` 中的 **sync_dir**，而不是整个训练输出目录：

```bash
"$SFT_WORK_ROOT/env/bin/wandb" sync \
  --entity qili0502-zhejiang-university --project agentic_distillation \
  /actual/run/wandb/offline-run-TIMESTAMP-ID
```

`--resume-from` 始终使用 checkpoint 绑定的不可变配置，不接受 W&B 参数覆盖。
在线恢复使用原 run ID 和 `resume="must"`；环境中的身份设置不会改写原目标。
本项目采用离线分段记录策略：恢复段获得新 ID，并保留原 group 和 parent ID，
每段独立同步后可在同组比较。W&B run 恢复不能代替 optimizer／scheduler／RNG 恢复检查。

worker 在模型加载前初始化 W&B，异常和中断分别以失败退出码、130 结束 SDK run。
SDK 收尾超时为 15 秒，超时保留本地记录并报告错误。
W&B 的 `lifecycle/status=trainer_completed` 只表示训练子进程完成；
`validation/level=gpu_training_pending_controller_validation` 表示尚待父进程核对产物与输入。
最终训练验收以本地 `run.json` 为准；这两者均不是独立官方任务评估器的任务成功结论。

真实 SDK 的 CPU 离线验证可单独运行，使用明确标注的合成指标，不加载模型、不训练 GPU：

```bash
MANIP_TEST_WANDB=1 CUDA_VISIBLE_DEVICES="" PYTHONPATH=src \
  "$SFT_PYTHON" -m pytest -q tests/integration/test_wandb_tracking.py
```

该检查读取真实 `.wandb` 事件文件，验证原生回调的 loss、学习率、梯度、step、checkpoint
摘要和任务评估标记。在线登录、云端权限、真实 Qwen 训练及在线恢复需各自的运行证据。
参考：[Transformers 的 W&B 集成](https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/integration_utils.py)、
[W&B 离线记录和同步](https://docs.wandb.ai/support/models/articles/can-i-run-wandb-offline)。

## 导出

```bash
manip-train export --run /data/sft_runs/smoke_001_resumed \
  --checkpoint /actual/path/to/checkpoint-4 \
  --output /data/sft_exports/smoke_001
```

`--checkpoint best` 使用 trainer 记录的最佳 checkpoint，不按最大步数猜测。
导出先固定完整 checkpoint 快照，再用 CPU merge。产物模型目录为输出下的 **`model/`**；
导出清单、日志和生成验证结果位于其父目录。

合并后使用无图和多图验证输入重新加载、前向并生成。生成文本和工具调用解析结果保留，
四步训练的生成质量不作为自主任务成功标准。任何失败都保留目录及错误记录。

## 状态和测试

退出码 0 表示请求操作成功，2 表示输入／配置／依赖／资源前置条件失败，1 表示训练或
导出进程及产物验收失败，130 表示用户中断。子进程原始退出码另行记录。

```bash
PYTHONPATH=src python -m pytest -q tests/test_sft_*.py tests/test_training_*.py
```

真实集成测试检查实际运行证据，默认缺少路径时跳过；显式请求对应标记却未配置证据时失败：

```bash
MANIP_TEST_PREFLIGHT=/data/sft_checks/encode_001 \
  python -m pytest -q -m qwen_encode tests/integration

MANIP_TEST_SMOKE_FIRST=/data/sft_runs/smoke_001 \
MANIP_TEST_SMOKE_RESUMED=/data/sft_runs/smoke_001_resumed \
MANIP_TEST_EXPORT=/data/sft_exports/smoke_001 \
  python -m pytest -q -m gpu tests/integration
```

运行记录包括完整解析配置、不可变 training contract、输入与环境指纹、可训练参数列表、
真实数据量、checkpoint 状态和日志。认证信息不写入记录，失败目录不会自动删除。

## 常见错误

| 错误 | 处理方式 |
|---|---|
| `package_provenance_mismatch` / `package_count_mismatch` | 检查完整包及 sidecar；保留两个 split 的独立读取顺序 |
| `image_hash_mismatch` / `source_changed` | 检查数据包副本和源文件；数据变化后重新预检 |
| `preflight_contract_mismatch` | 显式报告与当前输入不匹配；使用匹配报告、另做独立预检，或在新训练时省略 `--preflight` |
| `overlength_samples` | 显式提供的报告包含超限行；检查报告中的真实长度和当前预算 |
| `smoke_coverage_gap` | 查缺失的工具或上下文类别；不复制样本、不自动增大预算 |
| `encoding_worker_failed` | 保留失败证据，检查内存和进程退出码；可在新目录使用默认单进程编码 |
| `gpu_unavailable` / `insufficient_gpu_memory` | 检查选定 GPU、CUDA 环境和空闲资源；保留已有任务 |
| `incomplete_checkpoint` | 找回完整 optimizer、scheduler、RNG 和 trainer state；adapter 本身不足以恢复 |
| `export_processor_mismatch` / `incomplete_export` | 检查原生导出产物、基础模型与精确依赖；加载一致性未通过时不报告导出成功 |

训练 OOM 会保留解析配置、原始退出码和日志，不自动降低图片质量、删除历史或更换模型。

## 本轮验证记录（2026-11-04）

用户已取消后续 GPU 测试，本轮没有完成真实冒烟闭环。代码实现与下列已验证项目保留。
完整机器记录见 [implementation.json](/home/liqi/agentic_mobile_manipulation/operations/sft_training_20261104/implementation.json)。

| 验收项 | 当前状态及证据 |
|---|---|
| 转换器整理 | 14 条冻结 golden 一致；[重构与数据完整性记录](/tmp/manipulation_sft_implementation_20261103/refactor_validation.json) |
| 数据读取 | 2,989 train、334 val、94 来源、9,584 图片完整性检查通过；原包 JSON/JSONL 哈希与基线一致 |
| CLI、配置与 CPU 边界 | 81 项测试、34 项子测试通过；真实数据 dry-run 不创建输出目录 |
| 全量真实编码 | 3,323 条全部通过，零图片／标签／工具渲染错误；[完整测量报告](/mnt/data/home/liqi/manipulation_sft/preflight_v6/summary.json) |
| 17,408 预算资格 | train 927、val 109；复用相同指纹的完整真实编码结果，固定 16/8 覆盖检查通过；[报告](/mnt/data/home/liqi/manipulation_sft/preflight_v7_17408/summary.json) |
| GPU LoRA 冒烟 | 三次尝试因 CUDA OOM 失败，未得到通过验收的 checkpoint-2；[第一次失败记录](/mnt/data/home/liqi/manipulation_sft/smoke_first_v1/run.json) |
| checkpoint 恢复 | 未执行；缺少验收通过的 checkpoint-2 |
| checkpoint-4 合并导出 | 未执行；基础模型的 GPU0 加载、有限前向、无图／四图生成另行通过，属于未微调模型探测 |
| 可复现性与文档 | 实际依赖锁、数据／模型／代码／配置指纹、运行日志和取消记录已保存；GPU 闭环未验证 |

真实长度中位数 28,725，P95 为 404,577，最大 828,233；338 条超过模型声明的 262,144。
因此 `encoding_passed=true` 与 `full_dataset_trainable=false` 同时成立。全部历史图片保留，
没有通过裁剪制造可训练样本。

本机账户内存限额为 32GB，长样本曾使单进程峰值接近该限额并大量换页；最终以单进程和
已封存缓存完成 CPU 编码。GPU0 总显存约 24GB，训练进程曾使用约 20.1GiB，同卡常驻
服务约占 2.6GiB，期间另有任务新增约 8.3GB 占用，实际前向出现 CUDA OOM。

该 PyTorch 2.9.1 构建的实际分配器快照显示：`PYTORCH_ALLOC_CONF` 被忽略，
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 才生效，尽管会出现弃用提示。
原始诊断与失败日志保留。资源探测已改为短生命周期子进程，以消除控制进程额外的 CUDA
上下文；该改动通过 CPU 测试，因用户取消而没有继续 GPU 复测。
