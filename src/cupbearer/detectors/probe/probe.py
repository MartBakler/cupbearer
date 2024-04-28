import torch 
from cupbearer.detectors.anomaly_detector import AnomalyDetector




class Probe(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, out_dim: int):
        super().__init__()
        self.model = model 
        self.probe = torch.nn.Linear(out_dim, 1)
    
    def forward(self, x):
        with torch.no_grad():
            emb = self.model(x)
        out = self.probe(emb)
        return out


class ProbeAnomalyDetector(AnomalyDetector):
    def set_model(self, model, out_dim):
        self.model = Probe(model=model, out_dim=out_dim)
    
    def scores(self, batch) -> torch.Tensor:
        # return logits of probe as score
        scores = self.model(batch)
        assert scores.shape == (batch.shape[0], 1)
        return scores

