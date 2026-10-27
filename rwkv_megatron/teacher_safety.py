"""Prepare complete teacher steps before pipeline work; checkpoint safe failures."""

import json
import math
import os
import time
from collections import deque
from pathlib import Path

import torch
import torch.distributed as dist

from megatron.training import get_args
from megatron.training.callbacks import Callback

from .online_teacher import TeacherPrefetchIterator, teacher_request_deadline


class TeacherDataUnavailable(RuntimeError):
    """All ranks agreed to stop before starting the next pipeline schedule."""


class TeacherStepBuffer:
    """Replay only CPU input/target pairs staged for the next complete step."""

    teacher_prepared = True

    def __init__(self, source):
        self.source = source
        self.rows = deque()

    def __iter__(self):
        return self

    def __next__(self):
        if not self.rows:
            raise RuntimeError("Pipeline consumed more microbatches than prepared")
        return self.rows.popleft()

    def prepare(self, count, client, deadline, need_teacher):
        if self.rows:
            raise RuntimeError("Previous pipeline step left unconsumed teacher data")
        with teacher_request_deadline(client, deadline):
            for _ in range(count):
                if time.monotonic() >= deadline:
                    raise TimeoutError("Teacher step preparation deadline exhausted")
                row = next(self.source)
                if need_teacher and "teacher_token_ids" not in row:
                    future = row.get(TeacherPrefetchIterator.future_key)
                    if future is not None:
                        targets = future.result(timeout=max(0.0, deadline - time.monotonic()))
                    else:
                        prompts = row.get("raw_prompts")
                        kwargs = {"prompts": prompts}
                        if getattr(client, "prompt_mode", "text") == "token_ids" and prompts is not None:
                            lengths = (row["loss_mask"].sum(-1).long() + 1).clamp_max(row["tokens"].shape[-1])
                            kwargs = {"attention_mask": torch.arange(row["tokens"].shape[-1])[None] < lengths[:, None]}
                        targets = client.get_logits(row["tokens"], **kwargs)
                    row = dict(row, teacher_token_ids=targets.token_ids.cpu(),
                               teacher_logprobs=targets.logprobs.cpu())
                if time.monotonic() >= deadline:
                    raise TimeoutError("Teacher step preparation deadline exhausted")
                self.rows.append(row)

    def get_stats(self):
        return getattr(self.source, "get_stats", lambda: {})()


def validate_teacher_safety(args):
    if not args.rwkv_distill:
        return
    timeout = args.rwkv_teacher_step_timeout
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("rwkv_teacher_step_timeout must be finite and positive")
    collective_timeout = (getattr(args, "distributed_timeout_seconds_after_init", None)
                          or args.distributed_timeout_minutes * 60)
    if timeout + 60 >= collective_timeout:
        raise ValueError("distributed_timeout_minutes must exceed teacher step timeout by over 60 seconds")
    if args.dataloader_type != "single" or args.virtual_pipeline_model_parallel_size is not None:
        raise ValueError("Teacher step preparation requires single dataloader and non-interleaved PP")


def prepare_teacher_step(data_iterator, num_microbatches, training, teacher_factory, prefetchers):
    args = get_args()
    if not args.rwkv_distill:
        return
    from megatron.core import parallel_state as ps
    from megatron.core.rerun_state_machine import RerunMode, get_rerun_state_machine

    validate_teacher_safety(args)
    if get_rerun_state_machine().get_mode() != RerunMode.DISABLED:
        raise ValueError("Teacher step preparation requires rerun_mode=disabled")
    deadline = time.monotonic() + args.rwkv_teacher_step_timeout
    error, buffer = None, None
    reader = ps.get_tensor_model_parallel_rank() == 0 and (
        ps.is_pipeline_first_stage() or ps.is_pipeline_last_stage())
    if reader:
        try:
            client = teacher_factory()
            if not isinstance(data_iterator.iterable, TeacherStepBuffer):
                source = data_iterator.iterable
                if (training and args.rwkv_teacher_async_prefetch
                        and not args.rwkv_teacher_replace_bad_samples
                        and ps.is_pipeline_last_stage() and ps.get_context_parallel_rank() == 0
                        and not isinstance(source, TeacherPrefetchIterator)):
                    from megatron.core.num_microbatches_calculator import get_num_microbatches

                    end_iteration = args.train_iters
                    if args.exit_interval:
                        end_iteration = min(end_iteration, (args.iteration // args.exit_interval + 1) * args.exit_interval)
                    source = TeacherPrefetchIterator(
                        source, client, num_workers=args.rwkv_teacher_prefetch_workers,
                        lookahead=args.rwkv_teacher_prefetch_lookahead,
                        min_workers=args.rwkv_teacher_prefetch_min_workers,
                        latency_target_seconds=args.rwkv_teacher_prefetch_latency_target,
                        max_batches=None if args.rampup_batch_size else
                        max(0, end_iteration - args.iteration) * get_num_microbatches())
                    prefetchers.append(source)
                data_iterator.iterable = TeacherStepBuffer(source)
            buffer = data_iterator.iterable
            buffer.prepare(num_microbatches, client, deadline,
                           ps.is_pipeline_last_stage() and ps.get_context_parallel_rank() == 0)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"[:2048]
    device = torch.device("cuda", torch.cuda.current_device())
    failed = torch.tensor([int(error is not None)], device=device, dtype=torch.int32)
    dist.all_reduce(failed, op=dist.ReduceOp.MAX)
    if failed.item():
        if buffer is not None:
            buffer.rows.clear()
        messages = [None] * dist.get_world_size()
        dist.all_gather_object(messages, error)
        detail = "; ".join(f"rank {rank}: {message}" for rank, message in enumerate(messages) if message)
        raise TeacherDataUnavailable("Teacher data unavailable before pipeline step; " + detail)


class TeacherFailureCheckpoint(Callback):
    def __init__(self):
        self.context = None
        self.iteration = None
        self.flops = 0

    def on_train_start(self, context):
        args = get_args()
        if not args.rwkv_distill:
            return
        self.context, self.iteration = context, args.iteration
        self.flops = args.num_floating_point_operations_so_far

    def on_train_step_end(self, context):
        if self.iteration is not None:
            from megatron.core.num_microbatches_calculator import get_current_running_global_batch_size
            from megatron.training.training import num_floating_point_operations

            self.iteration += 1
            self.flops += num_floating_point_operations(get_args(), get_current_running_global_batch_size())

    def save(self, error):
        args = get_args()
        if not args.rwkv_teacher_save_on_failure:
            return None
        if self.context is None or self.iteration is None:
            raise RuntimeError("Cannot save teacher emergency checkpoint before training state is initialized") from error
        from megatron.training.async_utils import maybe_finalize_async_save
        from megatron.training.checkpointing import save_checkpoint
        from megatron.training.training import force_param_sync, should_disable_forward_pre_hook

        root = args.save or str(Path("training-smoke") / ("teacher-checkpoints-" + os.environ.get("MASTER_PORT", "single")))
        destination = str(Path(root) / "teacher-emergency")
        maybe_finalize_async_save(blocking=True)
        if should_disable_forward_pre_hook(args):
            force_param_sync(self.context.model, optimizer=self.context.optimizer)
        fields = dict(save=destination, async_save=False, no_save_optim=False, no_save_rng=False)
        previous = {key: getattr(args, key) for key in fields}
        try:
            for key, value in fields.items():
                setattr(args, key, value)
            save_checkpoint(self.iteration, self.context.model, self.context.optimizer,
                            self.context.scheduler, self.flops, checkpointing_context={})
            maybe_finalize_async_save(blocking=True)
            dist.barrier()
            if dist.get_rank() == 0:
                report = dict(reason=str(error), iteration=self.iteration,
                              consumed_train_samples=args.consumed_train_samples,
                              checkpoint_directory=destination, exit_code=75,
                              safe_before_pipeline=True)
                path = Path(destination) / "teacher-failure.json"
                temporary = path.with_suffix(".tmp")
                temporary.write_text(json.dumps(report, indent=2) + "\n")
                temporary.replace(path)
                print(f"Teacher emergency checkpoint saved at iteration {self.iteration}: {destination}; "
                      f"resume with --load {destination}", flush=True)
            dist.barrier()
        finally:
            for key, value in previous.items():
                setattr(args, key, value)
        return destination
