"""S25 same-state twin-action intervention adapter: development-only, no fitting.

Exact S20 coefficients/rate and hidden x,z stepping are evaluator-private.
Each emit() supplies only *past* numeric history, current query actions and
the two contemporaneous noisy sensor observations. Both counterfactual probes
originate from the same x,z snapshot, using one identical dt. One randomly
selected probe becomes the realized next hidden state; never advance twice.

No frozen fit/validation/heldout seeds are accepted by this engineering API.
This is simulator-clone access, NOT real-world causal identification.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch

import g7_s20_hidden_multicycle as s20
from g7_s25_numeric_ttt import WorldStep

ENGINEERING_SEEDS = (3101, 3109, 3119)
PAIR_ACTIONS = (-0.8, 0.8)
MAX_STEPS = 32
SWITCH_STEP = 20
MEASUREMENT_SIGMA = 0.002
# The S25 frozen protocol fixes this range, not a post-result choice.
SIGNFLIP_B_LOWER = -1.10
SIGNFLIP_B_UPPER = -0.45
REGIMES = {
    "source_stable": ("source", False),
    "source_signflip": ("source", True),
    "sine_signflip": ("sine", True),
    "rational_signflip": ("rational", True),
}


class PairedS20World:
    """Private physical state; public step data have the frozen 3 tensor fields."""

    def __init__(self, seed: int, batch: int, regime: str):
        if seed not in ENGINEERING_SEEDS:
            raise ValueError("S25_ENGINEERING_SEEDS_ONLY")
        if regime not in REGIMES:
            raise ValueError("S25_UNREGISTERED_WORLD_REGIME")
        if not isinstance(batch, int) or batch < 1:
            raise ValueError("S25_INVALID_BATCH")
        self.seed = seed
        self.batch = batch
        self.regime = regime
        self._family, self._flipped = REGIMES[regime]
        # S20's original generator order is intentionally preserved: old/new
        # coefficient draws, x/z initialization, initial sensor measurement.
        self._gen = torch.Generator(device="cpu").manual_seed(seed)
        self._old = s20.coefficients(self._gen, batch, False)
        self._new = s20.coefficients(self._gen, batch, False)
        self._x = -1 + 2 * torch.rand((batch, 1), generator=self._gen)
        self._z = -.8 + 1.6 * torch.rand((batch, 1), generator=self._gen)
        self._obs = self._x + .25 * self._z + (
            MEASUREMENT_SIGMA * torch.randn((batch, 1), generator=self._gen))
        if self._flipped:
            # Independent negative-b RNG stream: pre-switch worlds with the
            # same seed have IDENTICAL actions, noise and physical prefixes.
            flip = torch.Generator(device="cpu").manual_seed(seed * 1009 + 727)
            magnitude = torch.rand((batch,), generator=flip)
            self._new[:, 1] = SIGNFLIP_B_LOWER + (
                SIGNFLIP_B_UPPER - SIGNFLIP_B_LOWER) * magnitude
        self._step = 0
        self._frames: list[torch.Tensor] = []
        self._last_primary_index: torch.Tensor | None = None

    def _counterfactual_physics(
        self, pair_actions: torch.Tensor, dt: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Equivalent S20 numeric step; two clones, zero observation updates."""
        coefficient = self._new if self._step >= SWITCH_STEP else self._old
        x = self._x.expand(-1, 2)
        z = self._z.expand(-1, 2)
        rate = s20.rate(x, pair_actions, coefficient, self._family)
        dx = rate + .35 * z
        dz = -.85 * z + .25 * pair_actions + .15 * x
        return x + dt * dx, z + dt * dz

    def emit(self) -> WorldStep:
        """Advances exactly one *realized* world step after cloning the probes."""
        if self._step >= MAX_STEPS:
            raise ValueError("S25_EPISODE_ALREADY_FINISHED")
        # Numeric sentinel carries the initial sensor, not a synthetic
        # observation of the current intervention or a hidden state.
        if self._frames:
            history = torch.stack(self._frames, dim=1).clone()
        else:
            history = torch.cat((
                self._obs, torch.zeros_like(self._obs),
                torch.zeros_like(self._obs), self._obs), dim=1
            ).unsqueeze(1)
        before = self._obs.clone()
        dt = .05 + .11 * torch.rand((self.batch, 1), generator=self._gen)
        pair = torch.tensor(PAIR_ACTIONS, dtype=before.dtype).view(1, 2)
        actions = pair.expand(self.batch, -1)
        next_x, next_z = self._counterfactual_physics(actions, dt)
        targets = next_x + .25 * next_z + (
            MEASUREMENT_SIGMA * torch.randn(
                (self.batch, 2), generator=self._gen))
        # This selector is evaluator-private and is drawn AFTER both
        # counterfactual outcomes. It is independent of action-effect signs.
        index = torch.randint(0, 2, (self.batch,), generator=self._gen)
        indices = index[:, None]
        chosen_x = next_x.gather(1, indices).clone()
        chosen_z = next_z.gather(1, indices).clone()
        chosen_obs = targets.gather(1, indices).clone()
        selected_action = actions.gather(1, indices).clone()
        queries = torch.stack((
            before.expand(-1, 2), actions, dt.expand(-1, 2)), dim=-1)
        if not bool(torch.isfinite(history).all()
                    and torch.isfinite(queries).all()
                    and torch.isfinite(targets).all()):
            raise RuntimeError("S25_NONFINITE_WORLD_DATA")
        # Only historical realized observations enter the next step's
        # history, never hypothetical counterfactuals.
        self._frames.append(torch.cat((
            before, selected_action, dt, chosen_obs), dim=1))
        self._x, self._z, self._obs = chosen_x, chosen_z, chosen_obs
        self._last_primary_index = index.clone()
        self._step += 1
        return WorldStep(history=history, query_pair=queries.clone(),
                         observed_pair=targets.clone(),
                         realized_action_index=index.clone())

    def close(self) -> dict[str, int | str]:
        if self._step != MAX_STEPS or len(self._frames) != MAX_STEPS:
            raise ValueError("S25_INCOMPLETE_WORLD")
        return {"worlds": self.batch, "steps": self._step,
                "paired_observations": 2 * self.batch * self._step,
                "realized_transitions": self.batch * self._step,
                "regime": self.regime}


def identical_arm_streams(
    seed: int, batch: int, regime: str
) -> tuple[PairedS20World, PairedS20World]:
    """Independent object identities with exactly the same numeric exposures."""
    return PairedS20World(seed, batch, regime), PairedS20World(
        seed, batch, regime)
