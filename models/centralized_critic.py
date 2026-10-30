from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork
from ray.rllib.utils.typing import TensorType


class CentralizedCriticModel(RecurrentNetwork, nn.Module):
    """Recurrent actor + centralized feed-forward critic (MAPPO).

    Observation layout (built by MultiAgentFireEnv with append_global_state):
        obs row = [ agent-local obs (agent_obs_dim) | global state (state_dim) ]

    - Actor (LSTM) sees ONLY obs[..., :agent_obs_dim].
    - Critic sees the FULL row (local obs + global state).

    Because the global state is part of the observation itself, it is
    sampled, stored and replayed by RLlib exactly like any other obs
    column: state t is always aligned with observation t, at rollout time
    (so VF_PREDS are computed with the real state) and at training time.
    No callback injection, no ViewRequirement, no zero substitution.

    custom_model_config:
        agent_obs_dim     width of the actor's input (defaults to the full
                          obs width, i.e. no appended state -> critic is
                          then local-obs-only).
        global_state_size optional; if given it is checked against
                          obs_dim - agent_obs_dim.
        lstm_cell_size    default 128.
        fc_size           default 256.
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        RecurrentNetwork.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        self.obs_dim = int(np.prod(obs_space.shape))  # full row
        custom_config = model_config.get("custom_model_config", {})
        self.agent_obs_dim = int(custom_config.get("agent_obs_dim", self.obs_dim))
        self.global_state_size = self.obs_dim - self.agent_obs_dim
        expected_state = custom_config.get("global_state_size")
        if expected_state is not None and int(expected_state) != self.global_state_size:
            raise ValueError(
                f"obs width {self.obs_dim} - agent_obs_dim {self.agent_obs_dim} = "
                f"{self.global_state_size}, but global_state_size={expected_state}"
            )
        if self.agent_obs_dim <= 0 or self.global_state_size < 0:
            raise ValueError(
                f"Bad dims: obs_dim={self.obs_dim}, agent_obs_dim={self.agent_obs_dim}"
            )
        self.lstm_cell_size = int(custom_config.get("lstm_cell_size", 128))
        fc_size = int(custom_config.get("fc_size", 256))

        # -------------------- Actor (recurrent, local obs only) ----------
        self.pre_fc = nn.Sequential(
            nn.Linear(self.agent_obs_dim, fc_size),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(fc_size, self.lstm_cell_size, batch_first=True)
        self.action_branch = nn.Linear(self.lstm_cell_size, num_outputs)

        # -------------------- Critic (feed-forward, full row) ------------
        self.critic = nn.Sequential(
            nn.Linear(self.obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        self._value: Optional[torch.Tensor] = None

    def get_initial_state(self) -> List[TensorType]:
        weight = self.pre_fc[0].weight
        h = weight.new_zeros(self.lstm_cell_size)
        c = weight.new_zeros(self.lstm_cell_size)
        return [h, c]

    def forward_rnn(self, inputs: torch.Tensor, state: List[TensorType], seq_lens: TensorType):
        # inputs: [B, T, obs_dim] (padded to seq_lens by RLlib)
        batch_size, seq_len, _ = inputs.shape
        inputs = inputs.float()

        actor_in = inputs[..., : self.agent_obs_dim]  # actor never sees global state
        x = self.pre_fc(actor_in)
        h_in = state[0].unsqueeze(0)
        c_in = state[1].unsqueeze(0)
        lstm_out, [h, c] = self.lstm(x, [h_in, c_in])
        logits = self.action_branch(lstm_out)

        # Critic on the full row, flattened in the same [B*T] order RLlib
        # reshapes the logits back from.
        obs_flat = inputs.reshape(batch_size * seq_len, self.obs_dim)
        self._value = self.critic(obs_flat).squeeze(-1)

        return logits, [h.squeeze(0), c.squeeze(0)]

    def value_function(self) -> TensorType:
        if self._value is None:
            return torch.zeros((1,), dtype=torch.float32)
        return torch.reshape(self._value, [-1])