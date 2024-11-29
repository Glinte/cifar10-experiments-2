from torch.optim.lr_scheduler import SequentialLR, LinearLR, LRScheduler


def warmup(scheduler: LRScheduler, start_factor: float, warmup_iters: int):
    """
    Append a linear warmup scheduler to the given scheduler. Do not try to chain multiple schedulers with this function.

    Args:
        scheduler (LRScheduler): The scheduler to append the warmup scheduler to.
        start_factor (float): The factor to start the warmup with.
        warmup_iters (int): The number of iterations to warm up for.
    """
    return SequentialLR(
        scheduler.optimizer,
        schedulers=[
            LinearLR(scheduler.optimizer,start_factor=start_factor, end_factor=1, total_iters=warmup_iters),
            scheduler
        ],
        milestones=[warmup_iters]
    )
