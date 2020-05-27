from tensorflow.python.keras.optimizer_v2.optimizer_v2 import OptimizerV2
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend_config
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.compat.v1 import logging
import re


class AdamWeightDecayOptimizer(OptimizerV2):
    """AdamWeightDecayOptimizer."""

    def __init__(
        self,
        learning_rate: float,
        weight_decay_rate: float = 0.0,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-6,
        exclude_from_weight_decay: list = ["LayerNorm", "layer_norm", "bias"],
        name: str = "AdamWeightDecayOptimizer",
    ):
        """
        A basic Adam optimizer that includes "correct L2 weight decay".  BERT
        Adamw implementation.

        Args:
            learning_rate (float): learning_rate
            weight_decay_rate (float): weight_decay_rate
            beta_1 (float): beta_1
            beta_2 (float): beta_2
            epsilon (float): epsilon
            exclude_from_weight_decay (list): exclude_from_weight_decay
            name (str): name
        """
        super(AdamWeightDecayOptimizer, self).__init__(name)
        self._set_hyper("learning_rate", learning_rate)
        self._set_hyper("beta_1", beta_1)
        self._set_hyper("beta_2", beta_2)
        self._set_hyper("weight_decay_rate", weight_decay_rate)
        self.epsilon = epsilon or backend_config.epsilon()
        self.exclude_from_weight_decay = exclude_from_weight_decay
        self._use_locking = False
        logging.debug(f"exclude layers: {self.exclude_from_weight_decay}")

    def _create_slots(self, var_list):
        # Create slots for the first and second moments.
        # Separate for-loops to respect the ordering of slot variables from v1.
        for var in var_list:
            self.add_slot(var, "m")
        for var in var_list:
            self.add_slot(var, "v")

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super(AdamWeightDecayOptimizer, self)._prepare_local(
            var_device, var_dtype, apply_state
        )

        local_step = math_ops.cast(self.iterations + 1, var_dtype)
        w_d = array_ops.identity(
            self._get_hyper("weight_decay_rate", var_dtype)
        )
        beta_1_t = array_ops.identity(self._get_hyper("beta_1", var_dtype))
        beta_2_t = array_ops.identity(self._get_hyper("beta_2", var_dtype))
        beta_1_power = math_ops.pow(beta_1_t, local_step)
        beta_2_power = math_ops.pow(beta_2_t, local_step)
        lr = apply_state[(var_device, var_dtype)]["lr_t"] * (
            math_ops.sqrt(1 - beta_2_power) / (1 - beta_1_power)
        )
        apply_state[(var_device, var_dtype)].update(
            dict(
                lr=lr,
                epsilon=ops.convert_to_tensor(self.epsilon, var_dtype),
                weight_decay_rate=w_d,
                beta_1_t=beta_1_t,
                beta_1_power=beta_1_power,
                one_minus_beta_1_t=1 - beta_1_t,
                beta_2_t=beta_2_t,
                beta_2_power=beta_2_power,
                one_minus_beta_2_t=1 - beta_2_t,
            )
        )

    def _do_use_weight_decay(self, var_name):
        """Whether to use L2 weight decay for `param_name`."""
        if not self.weight_decay_rate:
            return False
        if self.exclude_from_weight_decay:
            for r in self.exclude_from_weight_decay:
                if re.search(r, var_name) is not None:
                    return False
        return True

    def _resource_apply_dense(self, grad, var, apply_state):
        """
        Add ops to apply dense gradients to the variable `handle`.

        Args:
          grad: a `Tensor` representing the gradient.
          var: a `Tensor` of dtype `resource` which points to the variable to
            be updated.
          apply_state: A dict which is used across multiple apply calls.
        Returns:
          An `Operation` which updates the value of the variable.
        """

        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = (apply_state or {}).get(
            (var_device, var_dtype)
        ) or self._fallback_apply_state(var_device, var_dtype)
        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")
        w_d = coefficients["weight_decay_rate"]
        beta_1_t = coefficients["beta_1_t"]
        beta_2_t = coefficients["beta_2_t"]
        epsilon_t = coefficients["epsilon"]
        lr_t = coefficients["lr_t"]

        m_t = state_ops.assign(
            m,
            beta_1_t * m + (1.0 - beta_1_t) * grad,
            use_locking=self._use_locking,
        )
        v_t = state_ops.assign(
            v,
            beta_2_t * v + (1.0 - beta_2_t) * math_ops.square(grad),
            use_locking=self._use_locking,
        )
        var_delta = m_t / (math_ops.sqrt(v_t) + epsilon_t)

        # Weight decays
        logging.debug(f"Optimizer Dense layer: {var.name}")
        if self._do_use_weight_decay(var.name):
            var_delta += w_d * var
            logging.debug(f"Not applying decay on {var.name}")

        var_delta_with_lr = lr_t * var_delta
        var_t = var - var_delta_with_lr

        var_update = state_ops.assign(
            var, var_t, use_locking=self._use_locking
        )
        updates = [var_update, m_t, v_t]
        return control_flow_ops.group(*updates)

    def _resource_apply_sparse(self, grad, var, indices, apply_state):
        """
        Add ops to apply sparse gradients to the variable `handle`.

        Similar to `_apply_sparse`, the `indices` argument to this method has
        been de-duplicated. Optimizers which deal correctly with non-unique
        indices may instead override `_resource_apply_sparse_duplicate_indices`
        to avoid this overhead.

        Args:
          grad: a `Tensor` representing the gradient for the affected indices.
          var: a `Tensor` of dtype `resource` which points to the variable to
            be updated.
          indices: a `Tensor` of integral type representing the indices for
            which the gradient is nonzero. Indices are unique.
          apply_state: A dict which is used across multiple apply calls.
        Returns:
          An `Operation` which updates the value of the variable.
        """

        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = (apply_state or {}).get(
            (var_device, var_dtype)
        ) or self._fallback_apply_state(var_device, var_dtype)
        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")
        w_d = coefficients["w_d"]
        beta_1_t = coefficients["beta_1_t"]
        beta_2_t = coefficients["beta_2_t"]
        epsilon_t = coefficients["epsilon"]
        lr_t = coefficients["lr_t"]

        m_scaled_g_values = grad * (1 - beta_1_t)
        m_t = state_ops.assign(m, m * beta_1_t, use_locking=self._use_locking)
        with ops.control_dependencies([m_t]):
            m_t = self._resource_scatter_add(m, indices, m_scaled_g_values)

        v_scaled_g_values = (grad * grad) * (1 - beta_2_t)
        v_t = state_ops.assign(v, v * beta_2_t, use_locking=self._use_locking)
        with ops.control_dependencies([v_t]):
            v_t = self._resource_scatter_add(v, indices, v_scaled_g_values)

        var_delta = m_t / (math_ops.sqrt(v_t) + epsilon_t)

        var_t = math_ops.sub(var, self.eta_t * lr_t * var_delta)

        logging.debug(f"Optimizer Sparce layer: {var.name}")
        # Weight decays
        if self._do_use_weight_decay(var.name):
            var_delta += w_d * var
            logging.debug(f"Not applying decay on {var.name}")

        var_delta_with_lr = lr_t * var_delta
        var_t = var - var_delta_with_lr

        var_update = state_ops.assign(
            var, var_t, use_locking=self._use_locking
        )
        updates = [var_update, m_t, v_t]
        return control_flow_ops.group(*updates)

    def get_config(self):
        """
        Returns the config of the optimimizer.

        An optimizer config is a Python dictionary (serializable)
        containing the configuration of an optimizer.
        The same optimizer can be reinstantiated later
        (without any saved state) from this configuration.

        Returns:
            Python dictionary.
        """
        config = super(AdamWeightDecayOptimizer, self).get_config()
        config.update(
            {
                "learning_rate": self._serialize_hyperparameter(
                    "learning_rate"
                ),
                "beta_1": self._serialize_hyperparameter("beta_1"),
                "beta_2": self._serialize_hyperparameter("beta_2"),
                "weight_decay_rate": self._serialize_hyperparameter(
                    "weight_decay_rate"
                ),
                "epsilon": self.epsilon,
                "exclude_from_weight_decay": self.exclude_from_weight_decay,
            }
        )
        return config
