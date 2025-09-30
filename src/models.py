from transformers import AutoModelForCausalLM
from momentfm import MOMENTPipeline
from momentfm.utils.utils import control_randomness
from momentfm.data.informer_dataset import InformerDataset
from momentfm.utils.forecasting_metrics import get_forecasting_metrics
import torch

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
    def __init__(self, repo, model_kwargs, other_kwargs):
        pass

    def _get_forecasts(self, input_data, horizon=96, num_samples=None):
        pass

class TimesFM:
    def __init__(self, repo, model_kwargs, other_kwargs):
        pass

    def _get_forecasts(self, input_data, horizon=96, num_samples=None):
        pass