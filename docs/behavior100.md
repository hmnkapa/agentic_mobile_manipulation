# Current experiment defaults (2026-10-13)

New batches use `gpt-6.1-sol` with `low` reasoning effort, up to 50 unique
`act` / `look` requests, and no episode wall-time deadline. Failed requests count;
request-ID replays, initialization, skill reads and `finish` do not. Execution
attempts (`actions`) and agent steps (`agent_steps`) are recorded separately.
The final admitted action returns its RGB before a 120-second closure grace
starts. `finish` always permits independent evaluation. Startup, transport and
cleanup monitoring are separate from the task budget; legacy episode action,
call and physics-step caps are inactive in this mode. Historical manifests
retain their original models and budgets.

# BEHAVIOR 100-task research evaluation

The September 30, 2026 batch fixes one public instance (301), seed 0, for each of the 100 task types. This is 100 episodes, not all public instances and not a leaderboard submission. Existing fixed front/back/left/right RGB cameras and the ideal motor executor are unchanged. The model is gpt-6-astra with the nine-tool skills profile and four readable skills; no plan or memory tools are enabled.

`prepare_behavior100.py` verifies the official catalog and static BDDL hashes, preserving 50 official instructions and labeling the other 50 as project translations of static BDDL goals. Runtime state, task object IDs and scoring information are not supplied to the model. The offline manifest contains predicate/capability metadata for reviewers only; the policy receives the task instruction alone.

`run_behavior100.py` allows local uncommitted changes and records the source commit, dirty state and source hash. An explicitly configured `source_commit` must still match. A site JSON configuration supplies the batch/report directories, SSH argument list, simulator interpreter/environment file, physical GPU indices and bridge ports. `behavior100_remote.py` checks GPU ownership pressure, cgroup headroom, disk, quota and port availability before each simulator launch. Each episode gets a dedicated systemd unit and new output directory; the scheduler never overwrites an attempt or stops a unit without matching ownership. Existing running records require explicit reconciliation before a supervisor restart.

Run limits: 80 motor actions, 20,000 control steps, 30-minute model timeout, eight-minute startup timeout, 40-minute simulator-unit limit. Unsupported actions, model failures, native crashes and resource blocks remain in the denominator. No automatic task retries or policy changes are made within the batch. Raw records preserve source, dependencies, assets, task hashes, host, interpreter, GPU UUID, unit/PID and output paths.

Each completed run is archived with original images, simulator logs, exact tool calls/results, original controller stream, checksums, independent goal evaluation when available, five-view continuous video and an HTML Replay. The fifth spectator view is offline only. Web exports include original public assistant messages, not private reasoning. Videos contain actual control frames and observation boundaries, omitting model waiting and volume-sampler search ticks. Fragmented MP4 and checkpoints retain useful partial video after some failures; partial video never passes complete-episode validation.

The dashboard `behavior100.html` reports the fixed 100-task denominator, finalized episodes, task successes, available scores and complete videos separately. A row passes only when task success, controller finish, evidence alignment and video validation pass. Final task score alone does not prove a valid autonomous trajectory. A supervisor-requested abort is explicitly recorded and may fail strict model/simulator alignment. Missing evaluation is unknown, never invented as success. Initialization failures may have no images or video.

Validation includes CPU contract tests, an encoder decode check, all-100 static instance availability, per-episode transport/camera/video audits and browser inspection of real episode playback. Validation artifacts and the append-only operations journal live outside Git.

The initial service environment lacked the model executable on PATH. Its zero-action failure remains archived. The runner now checks the explicit controller PATH before allocating a simulator. A separately declared pre-policy infrastructure retry is allowed only when the original stream has no generated model item (client error items are allowed) and no actions occurred. It uses a new attempt ID/unit/output; the dashboard preserves the prior failure and reports first-attempt success separately from latest-attempt success. This is not an automatic policy retry.

The second service startup revealed missing proxy variables: routing discovery timed out before any policy output. Preserve this separate r2 failure. `start_behavior100_service.py` inherits only the explicitly named network variables without logging values; the runner verifies their presence. An authenticated text-only connectivity probe is a separate environment check, never a benchmark result. `--bootstrap-first` runs the first pending episode, checks transport alignment and complete video, then automatically continues the remaining queue. A failed bootstrap leaves unattempted tasks planned.

## Actual elapsed-time viewing edition

`walltime_video.py` uses controller start/end, simulator tool timestamps and recorded RGB capture timestamps to preserve real elapsed duration at 1×. It separates synchronous tool time from outside-tool waiting, which includes model inference, transport, client overhead and asynchronous work. It is not a measurement of pure model thinking time. Simulator initialization is excluded.

The original recorder did not timestamp individual control frames. Their display times are estimated uniformly within each measured tool interval; no motion interpolation is generated. Waiting repeats real frames. Exact public-message emission times were not recorded, so original public messages remain aligned by call order. All limitations are explicit in the MP4/Replay and timing sidecar. The first radio run spans 94.46 seconds of real control time versus 5.3 seconds in the compact simulation-time clip.

`publish_walltime_batch.py --watch` runs independently on CPU alongside the frozen evaluation, generates this edition as episodes finish, and publishes `behavior100_realtime.html`. It does not alter raw policy/evaluator evidence or interrupt active simulator units. Prior Replay exports are retained with `_before_walltime` names; new artifact hashes are separate from the original archive manifest. Real-time video is the default Replay edition when available; the compact original remains selectable.

## October 1 update (old batch remains frozen)

Newly prepared manifests use a separate 1200-second startup limit and 3300-second unit limit, retaining the 1800-second model limit. Original batches keep their original budgets. The replay defaults to an additive compact viewing edition when available; original and wall-time videos remain selectable. Returned provider reasoning summaries are recorded separately from assistant output. See [replay trace](replay_trace.md) and [repair validation](executor_repairs_20261001.md).
