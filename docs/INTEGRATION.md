# Integration and arm definitions

The selected controller is `experiments/libero_pro_120/protocol.py`. The evaluated native loop in the same directory uses the same LeRobot policy service for both arms. `baseline` means the authors' harness plus the disclosed common release repairs; `improved` adds my harness. It does not mean a bare policy.

Start with the existing [SETUP.md](../SETUP.md). Before using the archived simulator integration, set `RPENT_SOURCE` to an installed, pinned RPent checkout supplying `robots.libero.env_server`. Prepare the pinned upstream RoboHarness source under the experiment's expected `upstream/` or `vendor/` directory, real LIBERO-PRO states, policy/perception services, and memory. See `pilot_common.py`, `repair_release.py`, and `run_episode.py`; paths in the preserved scripts describe the historical runtime and are not a universal installer. `RPENT_SOURCE` is the one publication-only path binding; the controller is unchanged.

The original campaign used two A6000 GPUs for BF16 Qwen3-VL-32B and another shared GPU for policy/SAM/simulation. No minimum-VRAM or fresh-machine performance claim is made. Do not copy historical GPU indices into a new allocation without checking availability.
