from math import ceil

from optimizers.schedulers import *

SCHEDULERS = {
    'warmup': WarmUpScheduler,
    'step': StepDecayScheduler,
    'exponential': ExponentialDecayScheduler,
    'cosine': CosineAnnealingScheduler
}

class SchedulerFactory:
    def __init__(self, dataset_len: int, batch_size: int):
        self.scheduler_map = SCHEDULERS
        self.steps_per_epoch = ceil(dataset_len / batch_size)

    def create(self, scheduler_config):
        scheduler = scheduler_config['name']
        params = scheduler_config.get('params', {}).copy()
        if scheduler not in self.scheduler_map:
            raise ValueError(f'Unknown scheduler type: {scheduler}')
        if scheduler == 'warmup':
            base_scheduler = self.create(scheduler_config['params']['base_scheduler'])
            params['base_scheduler'] = base_scheduler
            return self.scheduler_map[scheduler](**params)
        else:
            return self.scheduler_map[scheduler](**params)