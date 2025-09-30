
# Sundial, TimeMoe
from transformers import AutoModelForCausalLM

# Moment
from momentfm import MOMENTPipeline
from momentfm.utils.utils import control_randomness
from momentfm.data.informer_dataset import InformerDataset
from momentfm.utils.forecasting_metrics import get_forecasting_metrics
import torch

# Caronos:
import numpy as np
try:
    from chronos import ChronosPipeline
except Exception:
    from amazon_chronos import ChronosPipeline  # fallback if environment uses this name

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
        e.g., "amazon/chronos-t5-large"
    model_kwargs: dict
        Passed to ChronosPipeline.from_pretrained (e.g., {"torch_dtype": torch.float16})
    other_kwargs: dict
        - device: "cuda" or "cpu"
        - lookback: optional (for reference; not strictly required by pipeline)
    """
    def __init__(self, repo, model_kwargs, other_kwargs):
        self.repo = repo
        self.model_kwargs = dict(model_kwargs or {})
        self.device = other_kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")

        # Let the pipeline place itself; if user passes device_map/torch_dtype we honor it.
        # For CUDA, default to float16 to save memory unless the user overrode it.
        if self.device.startswith("cuda") and "torch_dtype" not in self.model_kwargs:
            self.model_kwargs["torch_dtype"] = torch.float16

        # Some envs prefer explicit device_map
        if "device_map" not in self.model_kwargs:
            self.model_kwargs["device_map"] = "auto"

        self.model = ChronosPipeline.from_pretrained(self.repo, **self.model_kwargs)

    @torch.no_grad()
    def _get_forecasts(self, input_data, horizon=96, num_samples=100):
        """
        input_data: torch.Tensor with shape (L,) or (B, L) — univariate only.
        horizon:    int prediction length
        num_samples:int number of sample paths for quantiles/mean

        Returns:
            np.ndarray of shape (horizon,), the median forecast by default.
            (flip to mean by changing the aggregation line below)
        """
        # ---- shape + device handling ----
        if isinstance(input_data, torch.Tensor):
            ts = input_data
        else:
            ts = torch.as_tensor(input_data)

        # Accept (L,) or (B, L). Chronos expects per-series; we handle B=1 here.
        if ts.ndim == 1:
            ts = ts.unsqueeze(0)  # (1, L)
        elif ts.ndim != 2:
            raise ValueError(f"Chronos expects 1D or 2D tensor, got shape {tuple(ts.shape)}")

        ts = ts.to(self.device, dtype=torch.float32)

        # ---- per-series z-norm (matches your other models’ behavior) ----
        mean = ts.mean(dim=-1, keepdim=True)
        std = ts.std(dim=-1, keepdim=True)
        std = torch.where(std == 0, torch.full_like(std, 1e-6), std)
        ts_norm = (ts - mean) / std

        # Convert to cpu numpy for pipeline if needed (pipeline handles numpy inputs well)
        past_values = ts_norm[0].detach().float().cpu().numpy()  # use first series

        # ---- inference ----
        # We request samples so we can aggregate (median/mean) and then inverse-scale.
        # quantiles is optional; we aggregate from samples for robustness.
        out = self.model.predict(
            past_values=past_values,
            prediction_length=horizon,
            num_samples=num_samples if num_samples is not None else 100,
            return_samples=True,         # get sample paths
            temperature=1.0,             # default sampling temperature
        )

        # out["samples"]: (num_samples, horizon)
        samples = out["samples"] if isinstance(out, dict) else out
        samples = np.asarray(samples, dtype=np.float32)

        # choose your aggregator:
        # agg = samples.mean(axis=0)     # mean forecast
        agg = np.median(samples, axis=0) # median forecast (robust)

        # ---- inverse normalization ----
        mean_np = mean[0].detach().cpu().numpy().astype(np.float32)
        std_np = std[0].detach().cpu().numpy().astype(np.float32)
        forecast = agg * std_np.squeeze() + mean_np.squeeze()

        # ensure 1-D shape (horizon,)
        return np.asarray(forecast, dtype=np.float32).reshape(horizon)


class TimesFM:
    """
    Direct wrapper over google/timesfm (no custom TimesFMModel, no scaling).

    Args
    ----
    repo: str
        Unused (kept for API symmetry), e.g. "google/timesfm-2.0-500m-pytorch".
    model_kwargs: dict
        May include:
          - "seq_len" (int)
          - "horizon_len" or "horizen_len" (int)
          - "patch_len" or "pacth_len" (int) -> mapped to per_core_batch_size
          - "num_layers" (int, default 50)
          - "use_positional_embedding" (bool, default False)
          - "freq_level" (int, default 0)
    other_kwargs: dict
        - "device": "cuda" or "cpu" (default auto)
        - "lookback": context length if not in model_kwargs
        - "forecast_horizon": horizon if not in model_kwargs
    """
    def __init__(self, repo, model_kwargs, other_kwargs):

        self.repo = repo
        mk = dict(model_kwargs or {})
        self.device = other_kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        backend = "gpu" if self.device == "cuda" else "cpu"

        # lengths
        seq_len = mk.get("seq_len", other_kwargs.get("lookback"))
        horizon = mk.get("horizon_len", mk.get("horizen_len", other_kwargs.get("forecast_horizon", 96)))
        patch_len = mk.get("patch_len", mk.get("pacth_len", 2))
        if seq_len is None:
            raise ValueError("TimesFM: seq_len / lookback must be provided.")
        if horizon is None:
            raise ValueError("TimesFM: forecast_horizon must be provided.")

        self.seq_len = int(seq_len)
        self.horizon = int(horizon)
        self.patch_len = int(patch_len)

        # extra hparams (optional)
        num_layers = int(mk.get("num_layers", 50))
        use_pos_emb = bool(mk.get("use_positional_embedding", False))

        # default frequency level (can be overridden per-call if you want)
        self.freq_level = int(mk.get("freq_level", 0))

        # instantiate TimesFm directly
        self.model = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                backend=backend,
                per_core_batch_size=self.patch_len,
                horizon_len=self.horizon,
                num_layers=num_layers,
                use_positional_embedding=use_pos_emb,
                context_len=self.seq_len,
            ),
            checkpoint=timesfm.TimesFmCheckpoint(
                huggingface_repo_id="google/timesfm-2.0-500m-pytorch"
            ),
        )

    @torch.no_grad()
    def _get_forecasts(self, input_data, horizon=96, num_samples=None, freq_level=None):
        """
        input_data: torch.Tensor or np.ndarray, shape (L,) or (1, L); ALREADY NORMALIZED upstream.
        horizon:    override forecast length (defaults to init horizon).
        num_samples: unused (kept for API parity).
        freq_level: optional override; 0 (daily/high freq), 1 (weekly/monthly), 2 (quarterly+).

        Returns:
            np.ndarray of shape (horizon,)
        """
        # normalize shape -> 1D numpy
        if isinstance(input_data, torch.Tensor):
            if input_data.ndim == 2:
                if input_data.size(0) != 1:
                    raise ValueError("TimesFM wrapper currently supports batch size 1.")
                series = input_data[0].detach().cpu().numpy()
            elif input_data.ndim == 1:
                series = input_data.detach().cpu().numpy()
            else:
                raise ValueError(f"TimesFM expects 1D or (1, L) input, got {tuple(input_data.shape)}")
        else:
            arr = np.asarray(input_data)
            if arr.ndim == 2:
                if arr.shape[0] != 1:
                    raise ValueError("TimesFM wrapper currently supports batch size 1.")
                series = arr[0]
            elif arr.ndim == 1:
                series = arr
            else:
                raise ValueError(f"TimesFM expects 1D or (1, L) input, got {arr.shape}")

        # choose horizon/freq
        pred_len = int(horizon) if horizon is not None else self.horizon
        freq = int(self.freq_level if freq_level is None else freq_level)

        # timesfm API expects list of series and list of freq
        forecast_input = [series.astype(np.float32)]
        freq_list = [freq]

        point_forecast, experimental_quantile_forecast = self.model.forecast(forecast_input, freq_list)
        yhat = np.asarray(point_forecast[0], dtype=np.float32)

        # enforce requested length
        if yhat.shape[0] != pred_len:
            yhat = yhat[:pred_len]
        return yhat.reshape(pred_len)

