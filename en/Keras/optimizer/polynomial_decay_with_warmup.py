from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util.tf_export import keras_export
from tensorflow.compat.v1.keras.optimizers.schedules import (
    LearningRateSchedule,
)
from tensorflow.compat.v1 import logging
import tensorflow.compat.v1 as tf
import tensorflow.compat.v1.keras.backend as K


@keras_export("keras.optimizers.schedules.PolynomialDecayWarmup")
class PolynomialDecayWarmup(LearningRateSchedule):
    """A LearningRateSchedule that uses a polynomial decay schedule."""

    def __init__(
        self,
        initial_learning_rate: float,
        decay_steps: int,
        num_warmup_steps: int,
        end_learning_rate: float = 0.0,
        power: float = 1.0,
        start_warmup_step: int = 0,
        cycle: bool = False,
        name: str = "PolynomialDecayWarmup",
    ):
        """
        Reimplementation of PolynomialDecay learning rate scheduler with warm
        up similar to Albert implementation.

        Args:
            initial_learning_rate (float): initial_learning_rate
            decay_steps (int): decay_steps
            num_warmup_steps (int): num_warmup_steps
            end_learning_rate (float): end_learning_rate
            power (float): power
            start_warmup_step (int): start_warmup_step
            cycle (bool): cycle
            name (str): name
        """
        super(PolynomialDecayWarmup, self).__init__()

        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.warm_up_steps = num_warmup_steps
        self.start_warmup_step = start_warmup_step
        self.end_learning_rate = end_learning_rate
        self.power = power
        self.cycle = cycle
        self.name = name

    def __call__(self, step: int):
        """
        Call function from optimizer function.

        Args:
            step (int): step
        """
        with ops.name_scope_v2(
            self.name or "PolynomialDecayWithWarmup"
        ) as name:
            initial_learning_rate = ops.convert_to_tensor_v2(
                self.initial_learning_rate, name="initial_learning_rate"
            )
            dtype = initial_learning_rate.dtype
            end_learning_rate = math_ops.cast(self.end_learning_rate, dtype)
            power = math_ops.cast(self.power, dtype)
            warm_up_steps = math_ops.cast(self.warm_up_steps, dtype)
            start_warmup_step = math_ops.cast(self.start_warmup_step, dtype)

            global_step_recomp = math_ops.cast(step, dtype)
            decay_steps_recomp = math_ops.cast(self.decay_steps, dtype)
            if self.cycle:
                # Find the first multiple of decay_steps that is bigger than
                # global_step. If global_step is zero set the multiplier to 1
                multiplier = control_flow_ops.cond(
                    math_ops.equal(global_step_recomp, 0),
                    lambda: 1.0,
                    lambda: math_ops.ceil(
                        global_step_recomp / self.decay_steps
                    ),
                )
                decay_steps_recomp = math_ops.multiply(
                    decay_steps_recomp, multiplier
                )
            else:
                # Make sure that the global_step used is not bigger than decay_steps.
                global_step_recomp = math_ops.minimum(
                    global_step_recomp, decay_steps_recomp
                )

            p = math_ops.divide(global_step_recomp, decay_steps_recomp)
            decay_learning_rate = math_ops.multiply(
                initial_learning_rate - end_learning_rate,
                math_ops.pow(1 - p, power),
            )

            global_step_warmup = math_ops.sub(
                global_step_recomp, start_warmup_step
            )
            warmup_percent_done = math_ops.divide(
                global_step_warmup, warm_up_steps
            )
            warmup_learning_rate = math_ops.multiply(
                initial_learning_rate, warmup_percent_done,
            )
            learning_rate = control_flow_ops.cond(
                math_ops.greater(global_step_warmup, warm_up_steps),
                lambda: decay_learning_rate,
                lambda: warmup_learning_rate,
            )
            return learning_rate

    def get_config(self):
        """
        Returns config for restoration.
        """
        config = {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_steps": self.decay_steps,
            "num_warmup_steps": self.warm_up_steps,
            "end_learning_rate": self.end_learning_rate,
            "power": self.power,
            "start_warmup_step": self.start_warmup_step,
            "cycle": self.cycle,
            "name": self.name,
        }
        return config
