# Agentic Mobile Manipulation

An independent research system for **RGB observation -> LLM action and target selection -> idealized execution -> fresh observation** in OmniGibson.

Maintained at [echo636/agentic_mobile_manipulation](https://github.com/echo636/agentic_mobile_manipulation). Reference repositories and existing navigation environments remain separate from this project.

## Observation: four fixed cameras

- Four 512 x 512 RGB views: **front, back, left, and right**, relative to the robot heading.
- Cameras follow the robot rigidly. All four render products are read after a shared render barrier and carry the same `capture_id`, `captured_at`, and `sim_step`. Observation does not rotate the robot or advance physics.
- The research rig replaces stock head and wrist sensors. The four cameras are mounted 35 cm outward in their respective viewing directions; their fixed base-relative height is fitted to the robot's initial visual geometry, with 5 cm clearance. They use a 90-degree horizontal field of view and 35-degree downward pitch. This is a recorded simulation configuration, not a real-hardware calibration claim.
- The model receives RGB and capture metadata. Depth, camera extrinsics, simulator object identity, and task ground truth remain inside the executor or offline evaluator.

## Observation and action tools

```text
initialize({}) -> current revision and the prepared episode's first four RGB views
act({...})     -> execution feedback, next revision and four RGB views
look({...})    -> turn feedback, next revision and four RGB views
finish({...})  -> formal episode closure
```

Call `initialize` first and inspect the returned images. It exposes the prepared episode's current snapshot without resetting the scene or rendering an extra frame. Each subsequent `act` or `look` completes on the simulator's owning thread and returns its resulting four-camera snapshot in the same response. The model receives those pixels before choosing its next action.

Use the latest returned revision and image references for the next action. Older captures remain historical evidence and cannot target a new action after they are superseded. Any current camera view can supply an action target without rotating the robot first. `look` explicitly turns the base; it is not surround capture. If the world needs time to change, use `act` with `primitive="wait"`, `target=null` and an appropriate `wait_seconds`, then inspect the returned RGB. The latest action images can support final verification; no extra observation request is required before `finish`.

The default `skills` profile exposes six MCP tools: `initialize`, `look`, `act`, `finish`, `list_skills`, and `read_skill`. `minimal` exposes the first four. The legacy `workflow` profile retains `observe` and asynchronous observation jobs for compatibility; they are absent from the default tool catalog.

Five readable workflow skills are available: `visual-manipulation`, `goal-decomposition`, `visual-exploration`, `pick-and-place`, and `failure-recovery`. The goal-decomposition skill guides the model to form observable subgoals, order prerequisites, and revise its working plan in conversation. Separate planning stacks and persistent memory are currently deferred. Skills provide guidance; the LLM selects actions through tools.

## Local manipulation control

The default RGB executor uses `target_locked_local_v1`: the brain chooses the task, general navigation approach and current-image target; the motor controller makes bounded local position/heading adjustments and then executes the operation on that same target. `initialize.control` declares the protocol. Set `MAS_LOCAL_MANIPULATION=0` for the previous explicit-navigation behavior. See [coordination, limits and validation](docs/local_manipulation.md).

## Installation and execution

Python >=3.11 is required. The CPU core does not require simulator packages:

```bash
python -m pip install -e '.[mcp,navigation]'
PYTHONPATH=src python -m unittest discover -s tests -v
```

The recorded simulator environment uses OmniGibson 3.9.2, Isaac Sim 5.1.0, torch 2.7.0+cu128, and BDDL 3.7.0. BEHAVIOR assets must be installed under their official license and are not included in this repository. Default navigation uses private GT floor support and current scene collision geometry and does not require Cartographer or `MAS_CARTOGRAPHER_RUNTIME`. The retained [online mapping experiment](docs/online_navigation.md) has a separate native runtime dependency.

```bash
PYTHONPATH=src python -m manipulation_agent.vision_cli \
  --backend omnigibson --policy serve --agent-profile skills \
  --task turning_on_radio --instance 301 --output runs/radio --port 29440
PYTHONPATH=src python -m manipulation_agent.mcp_server --bridge http://127.0.0.1:29440
```

The [asynchronous MCP observation probe](scripts/probe_async_observation.py) covers the legacy `workflow` observation interface. It is a scripted compatibility test, not a test of the default tool catalog or an autonomous task result.

## Harness and model clients

All MCP transports use the official SDK's `FastMCP`: stdio by default, plus SSE and
Streamable HTTP on loopback. A shared tool registry and episode `ToolContext` serve
both MCP clients and the native RGB loop. Each server represents one episode.

The default Codex controller is preserved. A common `prepare_project -> run -> parse`
interface also provides OpenCode and Kimi adapters; their live RGB integration is
not yet validated. The native Responses loop has configurable turn/token budgets,
verified image attachments and optional historical-image retention, with replay
records for each tool call. Explicit planning and memory remain deferred.

See [harness architecture and transport usage](docs/architecture.md) and
[client capabilities and validation limits](src/manipulation_agent/clients/README.md).

The 2026-10-04 acceptance used frozen `0760f64`: Astra through Codex and FastMCP
completed radio instance 301 with Q=1 in 73.942 seconds of agent execution, excluding
initialization. [Recorded replay](http://10.76.5.241:8765/manipulation_runs/mas_harness_phase1_original_000_turning_on_radio_i301_s0_r1/replay.html).
This validates one Original episode; native Responses, OpenCode and Kimi do not
inherit this real-client acceptance. CPU/interface validation passed 224 tests,
with nine tests skipped for unavailable simulator dependencies.

## SFT dataset export

Local Qwen3.5-4B base and PEFT LoRA weights can use the existing MCP task loop via
`scripts/run_model_controller.py --client qwen --qwen-config FILE`. The client
uses the shared SFT `point_2d` integer `[0,1000]` protocol in model prompts, tool
schemas and history, then converts to native MCP `point` coordinates in `[0,1]`
before dispatch. It preserves four-camera image bytes and public feedback, with
skill interface examples adapted to the same coordinates. It does not inherit
SFT image budgets or history compression. For supervised single-task or batch
runs, use [`scripts/run_qwen_behavior.sh`](scripts/run_qwen_behavior.sh) with an
existing site runner template; add `--lora CHECKPOINT` for explicit PEFT weights.
See [online execution and validation limits](docs/qwen_online.md).

The optional `manip-train` toolchain adds full encoding preflight, LoRA training,
checkpoint resume and model export. See [Qwen3.5-4B SFT training](docs/sft_training.md)
for the separate environment, CPU checks and real-data smoke procedure.
New SFT runs track metrics and provenance in W&B project
`qili0502-zhejiang-university/agentic_distillation`; see the training guide for
login, offline logging and checkpoint resume behavior.

`manip-sft` converts recorded Codex RGB trajectories into per-decision Qwen3.5-4B
ms-swift data on CPU. It verifies archived prompts/tools and image bytes, masks
historical assistant turns, and provides shared point_2d and RGB-window adapters.
See [SFT dataset conversion](docs/sft_dataset.md) for single/batch export, provenance,
scene splits, and optional processor-only encoding checks.

## Execution and evidence

The model selects a pixel in a current RGB image. The private executor directly backprojects that pixel with its captured depth to obtain the target position, without a depth-to-mesh consistency check. Navigation uses that point without an object lookup. Manipulation additionally identifies the first visual object on the selected ray, solely to obtain its simulator handle; the mesh intersection never replaces the backprojected position. The executor then performs navigation, controlled carrying without a robot/object fixed joint, official state operations, or placement sampling. Base motion uses incremental idealized pose control; grasp and placement can be discontinuous. This is not a physical-control leaderboard submission. Final BDDL predicates and TaskMetric are evaluated independently after termination and are not returned to the active model. See the [RGB protocol](docs/rgb_protocol.md) and [executor v7 history](docs/executor_v7.md).

For `place_on_top`, the selected surface and shelf determine the landing region. Candidate poses use the carried object's collision geometry and requested relative yaw, without expanding its footprint into a diagonal-sized square. The executor checks physical contacts, lets candidates settle under gravity, and verifies support near the selected surface after release. Rejected candidates restore the scene; failed actions also restore the carry relationship. Collision geometry and diagnostics remain private to the executor. `place_inside` uses official volume sampling with a geometric fallback that covers distinct poses and supports on existing contents. Ideal grasps transfer the assembly directly to a checked carry destination; pickup trajectories are not required. Loaded base motion still checks continuous collision sweeps. Paired scripted placement probes are component evidence, not autonomous task success or a benchmark success-rate estimate.

Default navigation backprojects the model's selected RGB pixel with private calibrated depth. A floor click requests the nearest reachable point, without a camera-distance standoff; an elevated click requests a nearby approach with an unobstructed view of the target. Target visibility is checked again at arrival. The private GT grid combines floor support with current PhysX obstacle geometry and updates when doors or objects move. Normal travel uses checked turns and straight forward or reverse driving. Backing out is allowed without first sweeping the body or payload around a doorway. If the original corners are unsafe, a bounded search over position and heading looks for turning space. Endpoints now require checked local arrival and departure paths; arrival revalidates the departure. Ground routes compare a bounded set of nearby endpoints and final headings by movement plus turning time, with checked shortcuts and no independent pre-turn. This is not a globally shortest continuous planner. Robot and carried collision hulls are checked throughout each turn and translation; the explicit contact-shedding recovery retains its fixed-heading retreat. Motion still uses bounded incremental ideal pose updates and actual feedback, not physical wheel control. Maps, depth and world coordinates stay private. See [GT-grid navigation](docs/gt_navigation.md). The `initialize` → action → four-RGB protocol remains in use. [Online Cartographer navigation](docs/online_navigation.md) is retained as an experiment and is not called by this executor. Frozen evaluations retain their original source and results.

## Batch evaluation

Latest Original harness development is on `feat/harness-fastmcp`; Official is on `explore/official-symbolic`. Both methods use the same task list, while archived attempts retain their actual frozen source versions. Motor exploration is paused. The [current comparison](http://10.76.5.241:8765/compare100_20261002/) is separate from the historical 32-task replay portal.

The current queue protocol gives each task 30 minutes of actual agent execution, starting immediately before model launch after simulator initialization and MCP setup. Model inference, observations and actions consume this budget; resource waiting and initialization do not. Initialization has a separate watchdog and an initialization failure is recorded separately from a task execution timeout. A task has one terminal outcome: `success`, `failure` or `timeout`. Independent evaluator scores, controller completion and recording quality remain separate fields. A native crash or timeout may prevent a final Q score; the supervisor still records an explicit terminal outcome without inventing a score.

Both methods can use multiple workers in separately assigned resource pools. Within each method, host-memory and GPU leases prevent duplicate resource claims. Workers claim tasks in list order after resource admission; concurrently executing tasks may finish out of order. Cleanup must confirm the owned simulator has exited before its resource lease is released. A faulty worker waits for cleanup while other workers continue. Media validation and replay publication run in a bounded CPU queue after simulator cleanup.

Launch a frozen checkout with `scripts/start_behavior100_service.py --config <config.json> --unit mas-b100-<name>.service`. The supervisor restarts after an unexpected exit and retains task identities, deadlines and original records. For disjoint method resource pools, configure a separate `shared_lease_dir` per method and respect each host's actual GPU and memory capacity. A supervisor may finish the cohort only when every manifest task has a terminal outcome; unexpected worker exit cannot silently drop a task. Source versions and protocol changes remain attached to individual attempts; earlier completed attempts are not silently reclassified as new-code experiments.

Each run retains video, images, available model messages and returned reasoning summaries, tool traces, skill snapshots, versions, host/interpreter/GPU/PID provenance, and failure records. Replays do not invent hidden reasoning or unrecorded arm motion. Raw experiment data, credentials, and assets are excluded from Git.

`scripts/prepare_behavior100.py --instruction-mode mixed-coarse` preserves the 50 official gallery instructions and gives the other 50 tasks short task-level goals instead of detailed BDDL translations. The official evaluator is unchanged; omitted goal information can affect scores, so this condition must be reported separately from detailed-instruction runs. Batch configuration `export_spectator_video: true` exports every third-person frame to `spectator.mp4` and produces `spectator_walltime.mp4` from actual frame timestamps and the controller interval. The latter preserves model/network waiting by holding recorded frames and excludes simulator initialization. Both editions validate their full duration and remain separate from the original recording; neither invents motion. The replay links directly to the full wall-time third-person video when available.

- [Observation and tool protocol](docs/rgb_protocol.md)
- [Legacy asynchronous four-camera compatibility](docs/async_observation.md)
- [2026-09-30 validation](docs/four_camera_validation_20260930.md): 60 CPU checks, a real four-camera MCP probe, and a successful radio instance 301 loop; [report and replay](http://10.76.5.241:8765/surround_observation.html) require the lab network or VPN.
- [Earlier three-camera experiments](docs/history_before_four_camera.md): historical results do not validate a different camera configuration.

## Current executor validation

The v8 candidate enforces the selected placement surface, carries rigid supported contents such as food on plates, anchors the base during sampling physics, and uses consistent navigation segment checks. Camera calibration annotators initialize eagerly; RGB, private depth, and intrinsics pass a bounded render-only readiness check. No synthetic intrinsics or robot movement bypass initialization failures.

The recorded v8 validation includes 134 CPU tests and a real four-camera startup retest on `preparing_lunch_box`, instance 301. Component probes and complete model episodes have separate records. Passing code tests or initialization does not establish task success or a perfect executor. The [repair report](http://10.76.5.241:8765/executor_v8_20261001/index.html) preserves failed attempts and recordings. The [comparison with lvzhang and wenbo](http://10.76.5.241:8765/executor_v8_20261001/comparison.html) distinguishes program policies, navigation harnesses, and this project's RGB manipulation loop using pinned source snapshots.

## Results and replay

The [current system guide](http://10.76.5.241:8765/manipulation_system.html) explains the complete RGB loop and the Original / Official implementations with pinned source references, interactive tool and navigation explanations, a real four-camera grasp example, and a presentation mode. Its maintained source is [web/system_guide.html](web/system_guide.html); publish it with `python scripts/publish_system_guide.py --output-dir <report-root>`.

The [current comparison](http://10.76.5.241:8765/compare100_20261002/index.html) shows the Original and Official 100-task queues. The [historical 32-task replay portal](http://10.76.5.241:8765/retest32_v7_20261001/replays.html) retains only that completed cohort, synchronized camera views, and the model timeline. The complete task table is collapsed by default. Refreshes preserve playback. Retried tasks remain in the original batch without increasing its task count; previous attempts and bookmarked URLs retain their original execution versions. Bookmarked replays load independently of live progress metadata.

The portal source is [web/replays.html](web/replays.html). It reads `behavior100/progress.json` from the report directory and embeds existing replays without changing recordings, model outputs, or scores. `scripts/publish_results_page.py --output-dir <report-directory>` publishes the replay portal and analysis page.

The [historical 32-task comparison](http://10.76.5.241:8765/retest32_v7_20261001/index.html) retains that cohort and the earlier 2/32 baseline. [Scripted component regressions](http://10.76.5.241:8765/executor_retest_20261001/components.html) are displayed separately and do not count as model task successes.

The [earlier results and failure analysis](http://10.76.5.241:8765/retest32_gt_20260930/results.html) contains task status, old/new Q scores, failure evidence, timing, four-view observations, and model output. The detailed replay edition retains 1x simulation actions and reading intervals for available model text. The front view is largest; the other three views are smaller, and the spectator view is replay-only. Wall-time, earlier compact, and original video editions remain selectable. Returned reasoning summaries and assistant messages are displayed separately; missing history is not reconstructed. See the [replay review](http://10.76.5.241:8765/replay_readable_20261001/index.html).

[web/results.html](web/results.html) requires no build step or external CDN. It reads the batch's `behavior100/progress.json`, `behavior100/walltime_index.json`, `failure_review/review.json`, and per-task `replay.json`. Progress refreshes every 30 seconds. Manual failure analyses use explicitly frozen snapshots; their conclusions do not automatically extend to new runs. Audit scripts, input hashes, publication records, and browser checks remain in the corresponding operations batch.

- [Replay and model trace recording](docs/replay_trace.md) and a [recorded model episode](http://10.76.5.241:8765/replay_trace_20261001/manipulation_runs/mas_trace_000_turning_on_radio_i301_s0_r1/replay.html#step=9)
- [Startup and placement repair scope](docs/executor_repairs_20261001.md) and [repair attempts](http://10.76.5.241:8765/executor_fixes_20261001/index.html)
- [Measured host memory admission and shared startup coordination](docs/memory_admission.md)
