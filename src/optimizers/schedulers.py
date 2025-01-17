from abc import ABC, abstractmethod
import math
import matplotlib.pyplot as plt

class Scheduler:
    """Interface that defines a standardized structure for implementing learning rate scheduling strategies in training workflows."""
    def __init__(self):
        self.base_scheduler = None
        self.learning_rate = 0
        self.current_step = 0

    def step(self):
        self.current_step += 1

    def reset(self):
        self.learning_rate = 0
        self.current_step = 0

    @abstractmethod
    def get_lr(self):
        """Compute the learning rate for a given step."""
        pass

class WarmUpScheduler(Scheduler):
    """The WarmUp Scheduler linearly increases the learning rate during a warm-up period."""
    def __init__(self, base_scheduler: Scheduler, lr_start: float, lr_max: float, warmup_steps: float):
        """Initilize the WarmUp Scheduler.

        Args:
            base_scheduler (Scheduler): The main scheduler to use after warm-up.
            lr_start (float): Starting learning rate for the warm-up period.
            lr_max (float): Maximum learning rate to reach at the end of the warm-up period.
            warmup_steps (int): Number of steps for the warm-up phase. When 0, then no warm up. 
        """
        super(WarmUpScheduler, self).__init__()
        self.base_scheduler: Scheduler = base_scheduler
        self.lr_start = lr_start
        self.lr_max = lr_max
        self.warmup_steps = warmup_steps

    def get_lr(self):
        if self.current_step < self.warmup_steps:
            self.learning_rate = self.lr_start + (self.lr_max - self.lr_start) * (self.current_step / self.warmup_steps)
            return self.learning_rate
        else:
            self.learning_rate = self.base_scheduler.get_lr()
            self.base_scheduler.step()
            return self.learning_rate

class StepDecayScheduler(Scheduler):
    """The Step Decay Scheduler reduces the learning rate by a factor after a set number of steps."""
    def __init__(self, lr_start: float, step_size: int = 1, decay_factor: float = 0.9):
        """Initilize the Step Decay Scheduler. 

        Args:
            lr_start (float): Starting learning rate.
            step_size (int): Number of steps between learning rate adjustments
            decay_factor (float): Multiplicative factor by which to reduce the learning rate. (e.g., 0.1 for reducing the learning rate by 90%)
        """
        super(StepDecayScheduler, self).__init__()
        self.learning_rate = lr_start
        self.step_size = step_size
        self.decay_factor = decay_factor
    
    def get_lr(self):
        """Returns the learning rate at the current step based on step decay.

        Returns:
            float: Adjusted learning rate after applying step decay.
        """
        if self.current_step > 0 and self.current_step % self.step_size == 0:
            self.learning_rate *= self.decay_factor
        return self.learning_rate

class ExponentialDecayScheduler(Scheduler):
    """The Exponential Decay Scheduler smoothly decreases the learning rate exponentially."""
    def __init__(self, lr_start: float, decay_rate: float = 0.001):
        """Initilize the Exponential Decay Scheduler.

        Args:
            lr_start (float): Initial learning rate
            decay_rate (float): The rate at which the learning rate decays after each step.
        """
        super(ExponentialDecayScheduler, self).__init__()
        self.lr_start = lr_start
        self.decay_rate = decay_rate
    
    def get_lr(self):
        """Returns the learning rate at the current step based on exponential decay.

        Returns:
            float: Adjusted learning rate after applying exponential decay.
        """
        self.learning_rate = self.lr_start * math.exp(-self.decay_rate * self.current_step)
        return self.learning_rate

class CosineAnnealingScheduler(Scheduler):
    """The Cosine Annealing Scheduler decreases the learning rate following a cosine function pattern."""
    def __init__(self, lr_start: float, total_steps: int, min_lr: float = 0.01):
        """Initilize the Cosine Annealing Scheduler. 

        Args:
            lr_start (float): Initial learning rate
            total_steps (float): Total number of steps for the annealing schedule
            min_lr (float): Minimum learning rate to reach by the end
        """
        super(CosineAnnealingScheduler, self).__init__()
        self.lr_start = lr_start
        self.total_steps = total_steps
        self.min_lr = min_lr
    
    def get_lr(self):
        """Returns the learning rate based on cosine annealing.

        Returns:
            float: Adjusted learning rate after applying cosine function decay.
        """
        self.learning_rate = self.min_lr + 0.5 * (self.lr_start - self.min_lr) * (1 + math.cos(math.pi * self.current_step / self.total_steps))
        return self.learning_rate

def plot_schedule(scheduler: Scheduler, epochs: int, steps_per_epoch: int, filepath: str = None):
    learning_rates = []
    for epoch in range(epochs):
        for step in range(steps_per_epoch):
            learning_rates.append(scheduler.get_lr())
            scheduler.step()
    scheduler.reset()

    # Add markers for the start of each epoch
    epoch_starts = range(0, epochs*steps_per_epoch, steps_per_epoch)
    for start in epoch_starts:
        plt.axvline(x=start, color='red', linestyle='--', alpha=0.6, label="Epoch Start" if start == epoch_starts[0] else None)

    scheduler_name = type(scheduler.base_scheduler).__name__ if scheduler.base_scheduler else type(scheduler).__name__
    # Add labels and title
    plt.plot(range(epochs * steps_per_epoch), learning_rates)
    plt.xlabel("Training Steps")
    plt.ylabel("Learning Rate")
    full_scheduler_name = scheduler_name + f' with {type(scheduler).__name__}' if scheduler.base_scheduler else scheduler_name
    plt.suptitle(f"Learning Rate Schedule", fontsize=12)
    plt.title(f"{full_scheduler_name}", fontsize=10)
    plt.grid()
    # Save or show the plot
    plt.savefig(f'{filepath}/{scheduler_name}.png') if filepath else plt.show()
    plt.clf()
    
if __name__ == '__main__':

    from yacs.config import CfgNode
    from experiments.config import get_cfg_defaults
    from optimizers.scheduler_factory import SchedulerFactory, SCHEDULERS

    # Load default configuration
    config: CfgNode = get_cfg_defaults()
    config.merge_from_file("./src/optimizers/test_case.yaml")
    test_config = config['test']
    sch_config = config['scheduler']['params']
    bs_config = sch_config['base_scheduler']
    
    lr_start = sch_config['lr_start']
    lr_max = sch_config['lr_max']

    epochs = config['epochs']
    num_augmentations = test_config['dataset']['num_augmentations']
    total_data_length = test_config['dataset']['data_length'] * num_augmentations
    batch_size = config['dataset']['batch_size']
    num_batches = math.ceil(total_data_length / batch_size)

    warmup_steps = sch_config['warmup_steps']

    # Scheduler
    scheduler_factory = SchedulerFactory(
        dataset_len = total_data_length,
        batch_size = batch_size
    )
    scheduler = scheduler_factory.create(config['scheduler'])
    plot_schedule(scheduler, epochs, num_batches, config['scheduler']['plot_filepath'])