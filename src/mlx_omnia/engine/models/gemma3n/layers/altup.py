import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.engine.models.gemma3n.config import Gemma3nTextConfig


def rescale(streams: mx.array, target: mx.array, floor: float) -> mx.array:
    """The extra AltUp streams carry the active stream's magnitude."""
    mags = mx.mean(streams**2, axis=-1, keepdims=True) ** 0.5
    return streams * (target / mx.maximum(mags, floor))


class AltUp(nn.Module):
    def __init__(self, config: Gemma3nTextConfig) -> None:
        super().__init__()
        streams = config.altup_num_inputs
        self.correct_output_scale = mx.zeros((config.hidden_size,))
        self.correction_coefs = nn.Linear(streams, streams, bias=False)
        self.prediction_coefs = nn.Linear(streams, streams * streams, bias=False)
        self.modality_router = nn.Linear(config.hidden_size, streams, bias=False)
        self.router_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.streams = streams
        self.active = config.altup_active_idx
        self.hidden = config.hidden_size

    def modalities(self, x: mx.array) -> mx.array:
        routed = self.modality_router(self.router_norm(x) * (self.hidden**-1.0))
        return mx.tanh(routed.astype(mx.float32))

    def predict(self, x: mx.array) -> mx.array:
        coefs = (
            self.prediction_coefs(self.modalities(x[self.active]))
            .reshape(1, -1, self.streams, self.streams)
            .transpose(0, 1, 3, 2)
        )
        lifted = x.astype(mx.float32)
        predicted = mx.matmul(lifted.transpose(1, 2, 3, 0), coefs).transpose(3, 0, 1, 2)
        return (predicted + lifted).astype(x.dtype)

    def correct(self, predictions: mx.array, activated: mx.array) -> mx.array:
        coefs = self.correction_coefs(self.modalities(activated)) + 1.0
        innovation = activated - predictions[self.active]
        corrected = innovation[None] * coefs.moveaxis(2, 0)[..., None]
        return (corrected + predictions).astype(activated.dtype)
