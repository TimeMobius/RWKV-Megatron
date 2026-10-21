"""Derive student perplexity only after Megatron has aggregated token CE."""

from megatron.training import get_args, get_tensorboard_writer, get_wandb_writer
from megatron.training.callbacks import Callback


class StudentPerplexityCallback(Callback):
    """PPL = exp(sum NLL / number of valid tokens), never mean(exp(batch CE))."""

    def __init__(self):
        self._iteration = None

    def on_train_start(self, context):
        self._iteration = get_args().iteration

    def on_train_step_end(self, context):
        if self._iteration is not None:
            self._iteration += 1
        metrics = context.loss_dict
        if metrics is not None and 'distill/student_ce_loss' in metrics:
            metrics['distill/student_ppl'] = metrics['distill/student_ce_loss'].detach().double().exp()

    def on_eval_end(self, context):
        self._log_quality(context, 'validation')

    def on_test_end(self, context):
        self._log_quality(context, 'test')

    def _log_quality(self, context, split):
        metrics = context.total_loss_dict
        if metrics is None or 'distill/student_ce_loss' not in metrics:
            return
        ppl = metrics['distill/student_ce_loss'].detach().double().exp()
        metrics['distill/student_ppl'] = ppl
        args = get_args()
        values = {f'distill/{split}_student_ce_loss': metrics['distill/student_ce_loss'].item(),
                  f'distill/{split}_student_ppl': ppl.item()}
        writer, wandb_writer = get_tensorboard_writer(), get_wandb_writer()
        iteration = args.iteration if self._iteration is None else self._iteration
        if writer:
            for name, value in values.items():
                writer.add_scalar(name, value, iteration)
        if wandb_writer:
            wandb_writer.log(values, iteration)
