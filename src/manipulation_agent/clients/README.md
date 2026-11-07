# CLI policy adapters

`get_adapter(name)` exposes `probe()`, `prepare_project(config, instructions, tool_names)`,
`run(prepared, output, timeout, on_start)`, and `parse(raw_path)`. Contracts are in
`types.py`. The controller owns simulator/MCP preflight and the execution clock;
client setup is excluded when `--execution-clock-command-json` is used.

The controller passes `vision_policy.system_prompt(profile)` unchanged to each
adapter. Codex receives it as developer instructions, OpenCode as the configured
agent prompt, and Kimi through its project-local system.md. Native Responses uses
the same function. The current minimal catalog is initialize/look/act/finish;
skills adds list_skills/read_skill/task_notes. Goal-skill mode is supplied by the controller CLI to the common prompt builder. Initialize supplies the prepared episode's
current four-camera RGB, and act/look supply the next images in their results.
Clients need no separate capture request or polling loop. Legacy observe and
observation-job calls remain parseable in historical traces and are exposed only
by the workflow profile.

The existing `scripts/run_codex_controller.py` interface and Codex argv remain the
batch default. `scripts/run_model_controller.py --client codex|opencode|kimi` uses
the same lifecycle and arguments. `--client kimi --probe` requires no model or MCP
arguments and reports `available`, `unavailable`, or `unsupported` with exit 0 or 3.
A probe establishes executable/version/required CLI flags only, not authentication,
image delivery, model execution, or task success.

`--client qwen --qwen-config FILE` adds a local Qwen3.5-4B base/PEFT client with
the same project prompt, native MCP schemas, original four-view RGB and full
public history. It does not apply SFT coordinates, image budgets or compression.
See [native online execution](../../../docs/qwen_online.md). Its registration
probe is not a dependency/model/GPU validation; preparation checks the configured
interpreter. The existing Codex argv and low default are unchanged.

Codex keeps its read-only sandbox, disabled unrelated tools, profile allowlist and
optional bubblewrap session storage. OpenCode prepares a project-local primary
agent and deny-by-default permissions allowing only selected manipulation tools.
Kimi prepares an agent with no built-in tools, an explicit MCP config, and an empty
native skills directory; our task skills remain accessible through MCP. Existing
client authentication is inherited. No credentials are copied and HOME is unchanged.
OpenCode/Kimi storage isolation is explicitly unsupported. Their configuration and
parsers are implemented, but neither CLI is installed on the main machine as of
2026-10-04; live integration and RGB delivery have **not** been accepted. Inherited
OpenCode global provider/plugin configuration may still influence client behavior;
its real tool catalog must be checked when an installed client is first validated.

Files written:

- `model_events.jsonl`: original stdout, unchanged even when the final row is partial.
- `canonical_events.jsonl`: public events with `at`, `kind`, `client`, `turn`, call ID,
  tool, arguments, content/result/usage and original `source_line`. Missing times stay
  null. Explicit `reasoning_summary` remains distinct from public assistant text;
  token counts or opaque reasoning never become invented summaries.
- `model_events.compat.jsonl`: derived Codex-shaped envelopes for non-Codex replay,
  with adapter provenance. Images remain as actually recorded. This does not prove
  that a provider received them; the RGB evidence audit checks that separately.
- `controller.json`: capability, process, budget, finish and record paths. A normal
  CLI exit without a successful `finish` response is not harness completion. Harness
  completion still does not assert benchmark success; join the simulator score.

Canonical/compat files are finalized after client exit. Raw stdout is written while
the client runs. Replay and audit prefer compat when present; native-loop replay can
read a compat stream in the run directory without an external controller directory.

Format references: the local Jinkai snapshot's `bench/tool/harness/{opencode,kimi}_agent.py`,
[OpenCode configuration](https://opencode.ai/docs/config),
[OpenCode agent permissions](https://opencode.ai/docs/agents),
[Kimi custom agents](https://moonshotai.github.io/kimi-cli/en/customization/agents.html),
and [Kimi stream JSON](https://moonshotai.github.io/kimi-cli/en/customization/print-mode.html).
The adapters reuse the interaction patterns; they do not import navigation code,
copy authentication, or alter navigation environments.
