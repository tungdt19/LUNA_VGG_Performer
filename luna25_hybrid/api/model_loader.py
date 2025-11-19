import torch
from models.hybrid_model import HybridVGGPerformer

class ModelService:
    def __init__(self, checkpoint_path, device):
        self.device = device
        self.model = HybridVGGPerformer()
        state = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(state["model_state_dict"])
        self.model.to(device)
        self.model.eval()

    def predict_tensor(self, tensor):
        with torch.no_grad():
            logits = self.model(tensor)
            prob = torch.sigmoid(logits).item()
            label = 1 if prob >= 0.5 else 0
        return prob, label

model_service = None

def init_model(checkpoint_path):
    global model_service
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_service = ModelService(checkpoint_path, device)
    return model_service
