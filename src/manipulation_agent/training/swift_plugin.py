"""Optional native trainer callbacks, imported only by the training subprocess."""
from pathlib import Path
import math
import os

from manipulation_agent.sft.common import read_json
from manipulation_agent.training.artifacts import fingerprint, journal, write_json
from manipulation_agent.training.run import checkpoint_manifest
from manipulation_agent.training.swift_loss import register_loss

def adapter_digest(model):
    import hashlib
    import torch
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            if not bool(torch.isfinite(parameter.detach()).all().item()):
                raise RuntimeError("Non-finite adapter parameters")
            digest.update(name.encode())
            # numpy cannot directly represent BF16.
            digest.update(parameter.detach().float().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class _ManipulationSftAudit:
    def on_train_begin(self, args, state, control, model=None, optimizer=None, lr_scheduler=None, train_dataloader=None, **kwargs):
        if not state.is_world_process_zero:
            return
        self.root = Path(os.environ["MANIP_SFT_RUN_ROOT"])
        self.contract = read_json(self.root / "training_contract.json")
        self.config = self.contract["config"]
        parameters = [{"name": name, "numel": p.numel(), "dtype": str(p.dtype)}
                      for name, p in model.named_parameters() if p.requires_grad]
        if not parameters or any(any(part in p["name"].split(".") for part in ("visual", "vision_tower", "aligner")) for p in parameters):
            raise RuntimeError("Visual/aligner freeze or LoRA parameter contract failed")
        if any("lora_" not in p["name"] for p in parameters):
            raise RuntimeError("Only language LoRA adapter parameters may be trained")
        count = len(train_dataloader.dataset)
        if count != self.contract["expected_splits"]["train"]:
            raise RuntimeError("Actual training dataset count changed")
        if state.global_step != self.contract["expected_start_step"]:
            raise RuntimeError("Trainer did not restore the expected global step")
        restored = bool(optimizer is not None and len(optimizer.state) > 0)
        if state.global_step and not restored:
            raise RuntimeError("Resume did not restore optimizer state")
        scheduler_step = getattr(getattr(lr_scheduler, "scheduler", lr_scheduler), "last_epoch", None)
        if state.global_step and scheduler_step != state.global_step:
            raise RuntimeError("Resume did not restore scheduler step")
        self.initial_digest = adapter_digest(model)
        if state.global_step:
            checkpoint = Path(read_json(self.root / "run.json")["resume_checkpoint"]["path"])
            if self.initial_digest != read_json(checkpoint / "manipulation_training.json")["adapter_sha256"]:
                raise RuntimeError("Resume adapter does not match the saved training segment")
            if self.config["smoke"]["enabled"] and abs(float(state.epoch) - state.global_step / 4) > 1e-6:
                raise RuntimeError("Resume sampler position differs from the saved smoke segment")
        self.audit = {"status": "running", "start_step": state.global_step, "train_samples": count,
                      "trainable_parameters": parameters, "initial_adapter_sha256": self.initial_digest,
                      "optimizer_restored": restored, "scheduler_start_step": scheduler_step,
                      "eval_counts": [], "finite_logs": True, "finite_gradient_steps": [],
                      "rng_restored": False, "ignore_data_skip": args.ignore_data_skip, "start_epoch": state.epoch}
        import torch
        settings = torch.cuda.memory._snapshot().get("allocator_settings", {})
        self.audit["allocator_settings"] = {k: settings.get(k) for k in
            ("expandable_segments", "PYTORCH_CUDA_ALLOC_CONF")}
        if state.global_step:
            if args.ignore_data_skip:
                raise RuntimeError("Resume must preserve the native sampler skip position")
            self._audit_rng_restore()
        write_json(self.root / "audit.json", self.audit)
        journal(self.root, "trainer_started", step=state.global_step, optimizer_restored=restored)

    def _audit_rng_restore(self):
        # Observe the native restore call, without replacing the trainer's RNG
        # or sampler behavior. It occurs after on_train_begin in transformers.
        native_restore = self.trainer._load_rng_state
        expected = Path(read_json(self.root / "run.json")["resume_checkpoint"]["path"]).resolve()

        def restore(checkpoint):
            import torch
            native_restore(checkpoint)
            if Path(checkpoint).resolve() != expected:
                raise RuntimeError("Trainer restored RNG from an unexpected checkpoint")
            name = "rng_state.pth" if self.config["launch"]["nproc_per_node"] == 1 else "rng_state_0.pth"
            saved = torch.load(expected / name, map_location="cpu", weights_only=False)
            if not torch.equal(saved["cpu"], torch.random.get_rng_state()):
                raise RuntimeError("Native CPU RNG restoration did not match the checkpoint")
            if torch.cuda.is_available():
                actual = torch.cuda.get_rng_state_all()
                wanted = saved.get("cuda")
                wanted = wanted if isinstance(wanted, list) else [wanted]
                if len(actual) != len(wanted) or any(not torch.equal(a.cpu(), b.cpu()) for a, b in zip(actual, wanted)):
                    raise RuntimeError("Native CUDA RNG restoration did not match the checkpoint")
            self.audit["rng_restored"] = True
            journal(self.root, "rng_restore_verified", checkpoint=str(expected))

        self.trainer._load_rng_state = restore

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        if not state.is_world_process_zero:
            return
        import torch
        gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        if not gradients or any(not bool(torch.isfinite(g).all().item()) for g in gradients):
            raise RuntimeError("Missing or non-finite adapter gradients")
        self.audit["finite_gradient_steps"].append(state.global_step + 1)

    def on_step_end(self, args, state, control, **kwargs):
        # All ranks must take the same stop/save/evaluate branches.
        stop = read_json(Path(os.environ["MANIP_SFT_RUN_ROOT"]) / "training_contract.json")["config"]["smoke"]["stop_after_step"]
        if stop and state.global_step >= stop:
            control.should_save = control.should_evaluate = control.should_training_stop = True
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero:
            return
        if any(isinstance(v, (int, float)) and not math.isfinite(v) for v in (logs or {}).values()):
            raise RuntimeError("Non-finite training loss/gradient log")

    def on_evaluate(self, args, state, control, eval_dataloader=None, **kwargs):
        if not state.is_world_process_zero:
            return
        if eval_dataloader is None or len(eval_dataloader.dataset) != self.contract["expected_splits"]["val"]:
            raise RuntimeError("Actual validation dataset count changed")
        self.audit["eval_counts"].append({"step": state.global_step, "samples": len(eval_dataloader.dataset)})

    def on_save(self, args, state, control, model=None, **kwargs):
        if args.world_size > 1:
            import torch.distributed
            torch.distributed.barrier()
        if not state.is_world_process_zero:
            return
        checkpoint = Path(args.output_dir) / ("checkpoint-" + str(state.global_step))
        write_json(checkpoint / "manipulation_training.json", {
            "run_root": str(self.root), "contract_fingerprint": fingerprint(self.contract),
            "adapter_sha256": adapter_digest(model)})
        item = checkpoint_manifest(checkpoint)
        index_path = self.root / "checkpoints.json"
        entries = read_json(index_path)["checkpoints"] if index_path.exists() else []
        entries = [e for e in entries if e["global_step"] != state.global_step]
        entries.append(item)
        write_json(index_path, {"checkpoints": entries})
        journal(self.root, "checkpoint_verified", step=state.global_step, path=str(checkpoint))
        from manipulation_agent.training.tracking import log_checkpoint
        log_checkpoint(item, state.best_model_checkpoint)

    def on_train_end(self, args, state, control, model=None, **kwargs):
        if not state.is_world_process_zero:
            return
        final_digest = adapter_digest(model)
        self.audit.update(status="passed", end_step=state.global_step, final_adapter_sha256=final_digest,
                          adapter_updated=final_digest != self.initial_digest,
                          best_model_checkpoint=state.best_model_checkpoint, epoch=state.epoch)
        if not self.audit["eval_counts"] or not self.audit["adapter_updated"]:
            raise RuntimeError("Training did not evaluate or update adapter parameters")
        if self.audit["start_step"] and not self.audit["rng_restored"]:
            raise RuntimeError("Trainer did not perform the checkpoint RNG restore")
        if self.audit["finite_gradient_steps"] != list(range(self.audit["start_step"] + 1, state.global_step + 1)):
            raise RuntimeError("Training lacks a finite-gradient check for every optimizer update")
        if self.config["smoke"]["enabled"] and abs(float(state.epoch) - state.global_step / 4) > 1e-6:
            raise RuntimeError("Smoke sampler did not consume the expected portion of the 16-sample epoch")
        write_json(self.root / "audit.json", self.audit)


def register_callbacks():
    from swift.callbacks import TrainerCallback, callbacks_map
    register_loss()

    class ManipulationSftAudit(_ManipulationSftAudit, TrainerCallback):
        pass

    callbacks_map["manipulation_sft_audit"] = ManipulationSftAudit


# Native external_plugins import this file in a managed training subprocess.
# A regular CPU import remains independent of the training libraries.
if os.environ.get("MANIP_SFT_RUN_ROOT"):
    register_callbacks()
