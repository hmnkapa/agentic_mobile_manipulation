# Navigation SFT design reference

The independent `manipulation_agent.sft` converter follows the per-decision loss
masking and history-window design in `Cytosi/habitat-gs`, branch `zimeng/point_nav`,
commit `c7322f4944ff182508083f3a73282e09964696d2`, specifically
`bench/tool/nav_sft/scripts/build_nav_sft_dataset.py` and
`bench/tool/harness/nav_window.py`. It implements manipulation-specific evidence,
coordinates and compression without importing or bundling the Habitat runtime.
Its `rgb_window_budget_v1` policy also follows that commit's two-stage soft
budget handling: first retain one image per older observation, then remove old
assistant/tool groups. This project counts RGB-bearing replies and handles
interleaved public assistant text as part of the original call group.

The optional training toolchain also refers to that commit's
`sft_qwen35_nav_lora_last1_gate.sh`, `sft_qwen35_4b_lora_smoke.sh`, and merge
scripts. Its configuration, provenance, validation and launch code is independently
implemented. Training uses ms-swift as a separately installed Apache-2.0 dependency;
no ms-swift trainer or upstream simulator source is bundled.

# Jinkai visual-point GT navigation strategy

The candidate sampling and scoring in `executors/gt_navigation.py` are adapted
from https://github.com/dadwadw233/habitat-gs, branch `jinkai/harness`, commit
`0815cf234ee591bacd8017e9b1def4fac13e649b`, specifically
`tools/habitat_agent/oracle_local_nav/visual_point.py`.

The bounded projection, reachable-candidate selection, and feedback-navigation
design also refer to `oracle_local_nav/standoff.py` and
`src_python/habitat_sim/habitat_adapter_internal/mixins_navigation.py`.
Habitat's C++ navmesh and native greedy follower are not bundled. The project
provides an OmniGibson grid and kinematic follower adapter. See
`docs/gt_navigation.md` for the exact adaptation and validation boundary.

MIT License

Copyright (c) Meta Platforms, Inc. and its affiliates.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
