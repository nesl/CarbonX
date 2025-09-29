from transformers import AutoModelForCausalLM

class Sundial:
    def __init__(self, repo, device, model_kwargs):
        self.repo = repo
        self.model_kwargs = model_kwargs
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(repo, **model_kwargs)

    def _get_forecasts(self, input_data, horizon=96, num_samples=20):
        forecast = self.model.generate(input_data, max_new_tokens=horizon, num_samples=num_samples)
        meanForecast = forecast[0].mean(dim=0)
        meanForecast = meanForecast.cpu().numpy()
        return meanForecast
    
class Moment:
    def __init__(self, repo, model_kwargs):
        pass

    def _get_forecasts(self, input_data, horizon=96, num_samples=None):
        pass

class TimeMoe:
    def __init__(self, repo, device, model_kwargs):
        self.repo = repo
        self.model_kwargs = model_kwargs
        self.device = device
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
    def __init__(self, repo, model_kwargs):
        pass

    def _get_forecasts(self, input_data, horizon=96, num_samples=None):
        pass

class TimesFM:
    def __init__(self, repo, model_kwargs):
        pass

    def _get_forecasts(self, input_data, horizon=96, num_samples=None):
        pass