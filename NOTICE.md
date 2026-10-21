# Attribution and release provenance

This project builds on [RoboHarness](https://github.com/LZY-1021/RoboHarness),
commit `716defc19c2062e613e81dd19592595ef289b3b6`, by Zhiyang Li and contributors.
The selected, unmodified runtime Python modules from that release are in `vendor/src/`; their
original `soma_*` names are retained. Its MIT notice is preserved in
`vendor/LICENSE` and the root `LICENSE`. This project is an independent research
extension, not an official RoboHarness release.

`repair_release.py` creates a separate runtime copy and records the original
and repaired hashes. Shared release repairs apply to both comparison arms;
the request-gated geometric fallback is enabled only for the improved arm.

The simulator adapter uses [RLinf/RPent](https://github.com/RLinf/RPent) at
`945682a55c4e17d6bb686a80b162ff801e768157` and its LIBERO/LIBERO-PRO integration.
RPent, the simulator, LeRobot, OpenPI, SAM3, and model weights are separate
dependencies with their own licenses. Their source and weights are not
redistributed here.

Publication changes replace absolute machine paths, the local username check,
and GPU IDs with portable configuration, pin the OpenPI patch download, and add
setup/documentation/results. No policy, recovery, grounding, or horizon logic
was changed after the pilot. `results/publication-changes.json` records hashes
of files copied and adapted from the evaluated working directory;
`results/pilot-20.json` retains the evaluated seven core source hashes.

The published data contains metrics and provenance only. Raw images, model
responses, memory files, machine logs, credentials, and checkpoints are excluded.
