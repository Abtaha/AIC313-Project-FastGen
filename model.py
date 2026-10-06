"""Student-facing NFE-specific model definitions and checkpoint utilities."""

import torch
from torch import nn

from src.utils import count_parameters, parameter_summary


class Model(nn.Module):
    """DO NOT MODIFY: Common parent class for both NFE-specific models."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def count_parameters(self) -> int:
        """Return the total number of parameters, including frozen ones."""
        return count_parameters(self, trainable_only=False)

    @property
    def num_parameters(self) -> int:
        return self.count_parameters()

    def parameter_summary(self) -> dict[str, int]:
        return parameter_summary(self)

    def forward(self, x: torch.Tensor, timestep: torch.Tensor, **kwargs):
        return self.backbone(x, timestep, **kwargs)

    def sample(
        self,
        shape: tuple[int, ...],
        *,
        device: torch.device | str = "cuda",
        category: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError("Implement sample() in the selected model class.")

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint_path: str,
        evaluate_mode: str,
        device="cpu",
        **kwargs,
    ) -> "Model":
        """Instantiate the selected model class and load its checkpoint."""
        model_classes = {
            "one_nfe": ModelOneNFE,
            "few_nfe": ModelFewNFE,
        }
        try:
            model_class = model_classes[evaluate_mode]
        except KeyError as error:
            raise ValueError(
                f"Unsupported evaluate_mode={evaluate_mode!r}; "
                "choose 'one_nfe' or 'few_nfe'."
            ) from error

        model = model_class(device=device, **kwargs)
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        state_dict = (
            checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        )
        model.load_state_dict(state_dict)
        return model.to(device).eval()


def build_backbone(config=None):
    from src.models.dit import PixelDiT
    from src.models.hybrid import HybridFlowNet
    from src.models.hybrid_v2 import HybridV2FlowNet
    from src.models.hybrid_v3 import HybridV3FlowNet
    from src.models.edm_unet import EDMUNetFlowNet

    if config is None:
        return PixelDiT()

    if isinstance(config, dict):
        backbone = config.get("backbone", "dit")
    else:
        backbone = getattr(config, "backbone", "dit")

    if backbone == "dit":
        return PixelDiT(config)

    if backbone == "hybrid":
        return HybridFlowNet(config)

    if backbone == "hybrid_v2":
        return HybridV2FlowNet(config)

    if backbone == "hybrid_v3":
        return HybridV3FlowNet(config)

    if backbone == "edm_unet":
        return EDMUNetFlowNet(config)

    raise ValueError(f"Unknown backbone: {backbone!r}")


class _FlowModel(Model):
    """Shared construction and self-describing state, without changing Model."""

    default_objective = "flow_matching"

    def __init__(self, device="cpu", config=None, knots=(0.25, 0.5, 0.75), objective=None):
        super().__init__(build_backbone(config))
        self.objective = self._validate_objective(objective or self.default_objective)
        self.knots = self._validate_knots(knots)
        if self.num_parameters > 100_000_000:
            raise ValueError("Model must have no more than 100M parameters")
        self.to(device)

    def _validate_objective(self, objective):
        allowed = ("meanflow", "flow_matching") if self.mode == "one_nfe" else ("flow_matching",)
        if objective not in allowed:
            raise ValueError(f"Unsupported objective {objective!r} for {self.mode}")
        return objective

    @staticmethod
    def _validate_knots(knots):
        knots = tuple(float(t) for t in knots)
        if len(knots) != 3 or not 0 < knots[0] < knots[1] < knots[2] < 1:
            raise ValueError(
                "Require exactly three strictly increasing knots in (0, 1)"
            )
        return knots

    def get_extra_state(self):
        return {
            "version": 1,
            "config": self.backbone.architecture_config(),
            "knots": self.knots,
            "mode": self.mode,
            "objective": self.objective,
        }

    def set_extra_state(self, state):
        if state["version"] != 1 or state["mode"] != self.mode:
            raise ValueError("Incompatible checkpoint version or NFE mode")
        self.knots = self._validate_knots(state["knots"])
        # Existing checkpoints trained with the noise->data FM convention.
        self.objective = self._validate_objective(state.get("objective", "flow_matching"))

    def load_state_dict(self, state_dict, strict=True, assign=False):
        # The fixed loader constructs a default model BEFORE reading the file.
        # Reconstruct the backbone from state metadata, keeping that API intact.
        metadata = state_dict.get("_extra_state")
        if metadata is None:
            raise ValueError("Missing FastGen architecture metadata in checkpoint")
        self.set_extra_state(metadata)
        if metadata["config"] != self.backbone.architecture_config():
            parameter = next(self.parameters())
            self.backbone = build_backbone(metadata["config"]).to(
                device=parameter.device,
                dtype=parameter.dtype,
            )

        if self.num_parameters > 100_000_000:
            raise ValueError("Checkpoint exceeds parameter budget")
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def _noise(self, shape, device, category, generator):
        if len(shape) != 4 or tuple(shape[1:]) != (3, 64, 64) or shape[0] < 1:
            raise ValueError("shape must be [positive B, 3, 64, 64]")
        parameter = next(self.parameters())
        device = torch.device(device)
        if device.type != parameter.device.type or (
            device.index is not None and device.index != parameter.device.index
        ):
            raise ValueError("Move the model to the sampling device first")
        if (
            category is None
            or category.shape != (shape[0],)
            or category.dtype != torch.long
        ):
            raise ValueError("Provide category as an int64 tensor of shape [B]")
        category = category.to(device)
        if ((category < 0) | (category >= 151)).any():
            raise ValueError("category IDs must be in [0, 150]")
        return (
            torch.randn(
                shape, device=device, dtype=parameter.dtype, generator=generator
            ),
            category,
        )


class ModelOneNFE(_FlowModel):
    """One MeanFlow update across [r=0,t=1], with legacy FM checkpoint support."""

    mode = "one_nfe"
    default_objective = "meanflow"

    @torch.no_grad()
    def sample(self, shape, *, device="cuda", category=None, generator=None, **kwargs):
        z, category = self._noise(shape, device, category, generator)
        if self.objective == "meanflow":
            t = torch.ones(shape[0], device=z.device, dtype=z.dtype)
            return (z - self.backbone(z, t, category=category, interval=t)).clamp(-1, 1)
        t = torch.zeros(shape[0], device=z.device, dtype=z.dtype)
        return (z + self.backbone(z, t, category=category)).clamp(-1, 1)


class ModelFewNFE(_FlowModel):
    """Exactly four Euler evaluations; no auxiliary trained network or CFG."""

    mode = "few_nfe"

    @torch.no_grad()
    def sample(
        self,
        shape,
        *,
        device="cuda",
        category=None,
        generator=None,
        knots=None,
        **kwargs,
    ):
        x, category = self._noise(shape, device, category, generator)
        schedule = (
            0.0,
            *self._validate_knots(self.knots if knots is None else knots),
            1.0,
        )
        for start, end in zip(schedule[:-1], schedule[1:]):
            t = torch.full((shape[0],), start, device=x.device, dtype=x.dtype)
            x = x + (end - start) * self.backbone(x, t, category=category)
        return x.clamp(-1, 1)


# class ModelOneNFE(Model):
#     """Student model evaluated with exactly one model function evaluation."""
#
#     def __init__(self, device="cpu", **kwargs):
#         raise NotImplementedError("Implement the one-NFE model constructor.")
#
#     def sample(
#         self,
#         shape: tuple[int, ...],
#         *,
#         device: torch.device | str = "cuda",
#         category: torch.Tensor | None = None,
#         **kwargs,
#     ) -> torch.Tensor:
#         raise NotImplementedError("Implement sample() in ModelOneNFE.")
#
#
# class ModelFewNFE(Model):
#     """Student model evaluated with no more than four model evaluations."""
#
#     def __init__(self, device="cpu", **kwargs):
#         raise NotImplementedError("Implement the few-NFE model constructor.")
#     def sample(
#         self,
#         shape: tuple[int, ...],
#         *,
#         device: torch.device | str = "cuda",
#         category: torch.Tensor | None = None,
#         **kwargs,
#     ) -> torch.Tensor:
#         raise NotImplementedError("Implement sample() in ModelFewNFE.")
#
