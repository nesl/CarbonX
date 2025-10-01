
# Sundial, TimeMoe
from transformers import AutoModelForCausalLM

# Moment
from momentfm import MOMENTPipeline
from momentfm.utils.utils import control_randomness
from momentfm.data.informer_dataset import InformerDataset
from momentfm.utils.forecasting_metrics import get_forecasting_metrics
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="momentfm.models.moment")

import torch
import numpy as np

# Caronos:
from chronos import ChronosPipeline
# TimesFM:
import timesfm


class Sundial:
    def __init__(self, repo, model_kwargs, other_kwargs):
        self.repo = repo
        self.model_kwargs = model_kwargs
        self.device = other_kwargs["device"]
        self.model = AutoModelForCausalLM.from_pretrained(repo, **model_kwargs)

    def _get_forecasts(self, input_data, horizon=96, num_samples=20):
        forecast = self.model.generate(input_data, max_new_tokens=horizon, num_samples=num_samples)
        meanForecast = forecast[0].mean(dim=0)
        meanForecast = meanForecast.cpu().numpy()
        return meanForecast
    
class Moment:
    def __init__(self, repo, model_kwargs, other_kwargs):
        self.repo = repo
        self.model_kwargs = model_kwargs
        self.device = other_kwargs["device"]
        model_kwargs["seq_len"] = other_kwargs["lookback"]
        model_kwargs["forecast_horizon"] = other_kwargs["forecast_horizon"]
        self.batch_size = 1
        self.channels = 1
        self.model = MOMENTPipeline.from_pretrained(repo, model_kwargs=model_kwargs)
        self.model.init()

    def _get_forecasts(self, input_data, horizon=96, num_samples=None):
        print("MOEMENT forecasting")
        self.model = self.model.to(self.device)
        self.model.eval()
        mean, std = input_data.mean(dim=-1, keepdim=True), input_data.std(dim=-1, keepdim=True)
        normed_seqs = (input_data - mean) / std # shape = (1, 512)
        normed_seqs = normed_seqs.reshape((self.batch_size, self.channels, normed_seqs.shape[1]))
        normed_seqs_t = normed_seqs.to(self.device)
        input_mask = torch.ones_like(input_data)
        input_mask_t = input_mask.to(self.device)

        with torch.no_grad():
            with torch.amp.autocast("cuda"):
                normed_forecast = self.model(x_enc=normed_seqs_t, input_mask=input_mask_t).forecast.cpu().numpy().reshape(horizon)
        mean = mean.cpu().numpy()
        std = std.cpu().numpy()
        forecast = normed_forecast * std + mean
        forecast = forecast.reshape(horizon)
        return forecast
    
class TimeMoe:
    def __init__(self, repo, model_kwargs, other_kwargs):
        self.repo = repo
        self.model_kwargs = model_kwargs
        self.device = other_kwargs["device"]
        model_kwargs["device_map"] = self.device
        self.model = AutoModelForCausalLM.from_pretrained(repo, **model_kwargs)

    def _get_forecasts(self, input_data, horizon=96, num_samples=None):
        # normalize seqs
        mean, std = input_data.mean(dim=-1, keepdim=True), input_data.std(dim=-1, keepdim=True)
        normed_seqs = (input_data - mean) / std
        normed_seqs = normed_seqs.to(self.device)
        # forecast
        output = self.model.generate(normed_seqs, max_new_tokens=horizon)  # shape is [batch_size, lookback + horizon]
        normed_predictions = output[:, -horizon:]  # shape is [batch_size, horizon]
        normed_predictions = normed_predictions.cpu().numpy()

        # inverse normalize
        mean = mean.cpu().numpy()
        std = std.cpu().numpy()
        predictions = normed_predictions * std + mean

        return predictions

class Chronos:
    """
    Args
    ----
    repo: str
        e.g., "amazon/chronos-t5-large" or "amazon/chronos-bolt-small"
    model_kwargs: dict
        Passed to ChronosPipeline.from_pretrained (e.g., {"torch_dtype": torch.float16})
    other_kwargs: dict
        - device: "cuda" or "cpu" (optional; auto-detected if missing)
        - lookback: optional (not required here)
    """

    def __init__(self, repo, model_kwargs=None, other_kwargs=None):
        other_kwargs = other_kwargs or {}
        model_kwargs = dict(model_kwargs or {})

        self.repo = repo
        self.device = other_kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")

        # Prefer fp16 on CUDA unless caller overrides
        if self.device.startswith("cuda") and "torch_dtype" not in model_kwargs:
            model_kwargs["torch_dtype"] = torch.float16

        # Let HF place weights unless caller specified
        model_kwargs.setdefault("device_map", "auto")

        # Load pipeline
        self.model = ChronosPipeline.from_pretrained(self.repo, **model_kwargs)

    @torch.no_grad()
    def _get_forecasts(self, input_data, horizon=96, num_samples=100):
        """
        input_data: torch.Tensor with shape (L,) or (B, L) — univariate only.
        Returns: np.ndarray (horizon,)  median over samples by default.
        """
        # ---- normalize input & shape handling ----
        ts = input_data if isinstance(input_data, torch.Tensor) else torch.as_tensor(input_data)
        if ts.ndim == 1:
            ts = ts.unsqueeze(0)  # (1, L)
        elif ts.ndim != 2:
            raise ValueError(f"Chronos expects 1D or 2D tensor, got shape {tuple(ts.shape)}")

        ts = ts.to(dtype=torch.float32)

        mean = ts.mean(dim=-1, keepdim=True)
        std = ts.std(dim=-1, keepdim=True)
        std = torch.where(std == 0, torch.full_like(std, 1e-6), std)
        ts_norm = (ts - mean) / std

        # ---- IMPORTANT: Chronos in your env requires a 1-D *CPU* torch.Tensor ----
        # Provide only the first series if (B, L) was passed.
        context = ts_norm[0].to("cpu", dtype=torch.float32).contiguous().view(-1)  # shape (L,)

        # ---- inference ----
        # predict(...) returns a tensor of shape (num_samples, horizon) for single series
        out = self.model.predict(
            context=context,                         # must be torch.Tensor (CPU)
            prediction_length=int(horizon),
            num_samples=int(num_samples) if num_samples is not None else 100,
            temperature=1.0,
            top_k=50,
            top_p=1.0,
            limit_prediction_length=False,
        )

        samples = out.detach().cpu().numpy().astype(np.float32) # (B, S, H)
        
        # ---- aggregate & inverse normalization ----
        # out may be Tensor or ndarray with shapes like:
        # (S, H), (1, S, H), (B, S, H), (S, B, H), or (H,)
        out_t = out if isinstance(out, torch.Tensor) else torch.as_tensor(out)

        # Normalize to (S, H) by picking the first series if a batch is present
        if out_t.ndim == 3:
            # assume last dim is horizon
            if out_t.shape[-1] != horizon:
                raise ValueError(f"Unexpected output shape {tuple(out_t.shape)}; expected horizon {horizon} as last dim.")
            # Prefer (B, S, H) -> first batch
            if out_t.shape[0] == 1 or (out_t.shape[0] > 1 and out_t.shape[1] == horizon):
                samples_2d = out_t[0]                       # (S, H)
            # Handle (S, 1, H) or (S, B, H) -> first series in second dim
            elif out_t.shape[1] == 1:
                samples_2d = out_t[:, 0, :]                 # (S, H)
            else:
                # default: treat first batch
                samples_2d = out_t[0]                       # (S, H)
        elif out_t.ndim == 2:
            samples_2d = out_t                              # (S, H)
        elif out_t.ndim == 1:
            # Single path -> fake samples dim
            samples_2d = out_t.unsqueeze(0)                 # (1, H)
        else:
            raise ValueError(f"Unsupported output ndim={out_t.ndim}, shape={tuple(out_t.shape)}")

        samples = samples_2d.detach().cpu().numpy().astype(np.float32)  # (S, H)
        agg = np.median(samples, axis=0)                                 # (H,)  or .mean(axis=0)

        mean_np = mean[0].detach().cpu().numpy().astype(np.float32)
        std_np  = std[0].detach().cpu().numpy().astype(np.float32)
        forecast = agg * std_np.squeeze() + mean_np.squeeze()

        return np.asarray(forecast, dtype=np.float32).reshape(horizon)


# ---------- TimesFM wrapper (legacy 2.0 API, PyTorch) ----------
# Requires: pip install timesfm==1.3.* (or your env's version that exposes TimesFmHparams/TimesFmCheckpoint/TimesFm)
# Checkpoint: google/timesfm-2.0-500m-pytorch

class TimesFM:
    """
    Wrapper for TimesFM 2.0 (PyTorch) with a stable interface:
        __init__(repo, model_kwargs, other_kwargs)
        _get_forecasts(input_data, horizon=96, num_samples=None) -> np.ndarray [B, H]

    Notes:
    - Matches the 500M checkpoint's fixed architecture hyperparameters exactly.
    - Uses per-series mean/std normalization and zero-pads the normalized series
      to a multiple of input_patch_len (32) to avoid reshape errors.
    """

    def __init__(self, repo, model_kwargs, other_kwargs):
        # --- required checkpoint (fixed architecture) ---
        self.repo = repo

        # --- caller hints ---
        model_kwargs = dict(model_kwargs or {})
        other_kwargs = dict(other_kwargs or {})
        self.device = other_kwargs.get("device", "cuda")
        self.lookback = int(other_kwargs.get("lookback", 512))          # your context length (can be < or > 32)
        self.default_horizon = int(other_kwargs.get("forecast_horizon", 96))

        backend = "gpu" if str(self.device).startswith("cuda") else "cpu"

        # --- CRITICAL: hparams MUST match the 500M checkpoint ---
        # You can change per_core_batch_size and context_len safely, but the following must stay fixed:
        # input_patch_len=32, output_patch_len=128, num_layers=50, model_dims=1280, use_positional_embedding=False, horizon_len=128
        hparams = timesfm.TimesFmHparams(
            backend=backend,
            per_core_batch_size=int(model_kwargs.get("per_core_batch_size", 1)),  # match how many series you pass per call
            horizon_len=128,                       # fixed by ckpt; we slice to requested horizon later
            input_patch_len=32,                    # fixed by ckpt
            output_patch_len=128,                  # fixed by ckpt
            num_layers=50,                         # fixed by ckpt
            model_dims=1280,                       # fixed by ckpt
            use_positional_embedding=False,        # fixed by ckpt
            context_len=max(32, int(self.lookback))# can be set freely (>= lookback is fine)
        )

        ckpt = timesfm.TimesFmCheckpoint(huggingface_repo_id=self.repo)
        self.model = timesfm.TimesFm(hparams=hparams, checkpoint=ckpt)

        # cache for padding
        self._patch_len = 32

    def _get_forecasts(self, input_data, horizon=96, num_samples=None):
        """
        Args:
            input_data: torch.Tensor or np.ndarray
                shape [L] or [B, L]; values are raw (unnormalized) time series.
            horizon: int, forecast horizon to return (<= 128).
            num_samples: unused (TimesFM returns point forecast + optional quantiles).

        Returns:
            preds: np.ndarray with shape [B, horizon] on the original scale.
        """
        import numpy as np
        import torch

        # ---- coerce to tensor [B, L] ----
        if isinstance(input_data, np.ndarray):
            input_data = torch.from_numpy(input_data)
        if input_data.dim() == 1:
            input_data = input_data.unsqueeze(0)  # [1, L]
        input_data = input_data.to(torch.float32)

        B, L = input_data.shape

        # ---- per-series mean/std normalization ----
        mean = input_data.mean(dim=-1, keepdim=True)                 # [B, 1]
        std = input_data.std(dim=-1, keepdim=True).clamp_min(1e-8)   # [B, 1]
        normed = (input_data - mean) / std                           # [B, L]

        # ---- pad normalized series to multiple of patch_len (=32) ----
        pad_needed = (-L) % self._patch_len
        if pad_needed > 0:
            # right-pad with zeros (which corresponds to mean after normalization)
            pad = torch.zeros((B, pad_needed), dtype=normed.dtype, device=normed.device)
            normed = torch.cat([normed, pad], dim=-1)  # [B, L_pad]
        L_pad = normed.shape[-1]

        # ---- convert to Python lists of 1D arrays as expected by legacy API ----
        series_list = [normed[i, :].detach().cpu().numpy().astype(np.float32) for i in range(B)]
        # legacy API requires a frequency list; choose 0 (high-frequency) by default
        freq_list = [0] * B

        # ---- run forecast ----
        point_list, _q = self.model.forecast(series_list, freq_list)  # list of arrays, each length >= 128
        H = int(horizon)
        if H > 128:
            raise ValueError(f"horizon={H} exceeds checkpoint horizon 128.")
        preds_norm = np.stack([arr[:H] for arr in point_list], axis=0)  # [B, H] in normalized space

        # ---- inverse normalization ----
        mean_np = mean.detach().cpu().numpy()  # [B, 1]
        std_np  = std.detach().cpu().numpy()   # [B, 1]
        preds = preds_norm * std_np + mean_np  # [B, H]

        return preds
# ---------- end TimesFM wrapper ----------
