import numpy as np
import torch
import torch.nn.functional as F
import transformers
import gc
from tqdm.auto import tqdm

class EnsemblePipeline:
    def __init__(self, model1: torch.nn.Module, model2: torch.nn.Module, device: torch.device,
                 val_loader1: torch.utils.data.DataLoader, val_loader2: torch.utils.data.DataLoader):
        self.device = device
        self.model1 = model1.to(device)
        self.model2 = model2.to(device)
        self.val_loader1 = val_loader1
        self.val_loader2 = val_loader2
        self.predictions = []
        self.true_labels = []

    def load_model_weights(self, model1_weights_path, model2_weights_path):
        self.model1.load_state_dict(torch.load(model1_weights_path))
        self.model2.load_state_dict(torch.load(model2_weights_path))

    def validate(self):
        self.model1.eval()
        self.model2.eval()
        with torch.inference_mode():
            for batch1, batch2 in zip(self.val_loader1, self.val_loader2):
                inputs1 = batch1['input_ids'].to(self.device)
                masks1 = batch1['attention_mask'].to(self.device)
                labels1 = batch1['labels'].to(self.device)
    
                inputs2 = batch2['input_ids'].to(self.device)
                masks2 = batch2['attention_mask'].to(self.device)
                labels2 = batch2['labels'].to(self.device)
    
                outputs_model1 = self.model1(inputs1, attention_mask=masks1, labels=labels1)
                outputs_model2 = self.model2(inputs2, attention_mask=masks2, labels=labels2)
    
                prob_outputs = (F.softmax(outputs_model1.logits, dim=1) + F.softmax(outputs_model2.logits, dim=1)) / 2
                self.predictions.extend(torch.argmax(prob_outputs, dim=1).cpu().numpy())
                self.true_labels.extend(labels1.argmax(dim=1).cpu().numpy())
                
        ensemble_accuracy = (np.array(self.predictions) == np.array(self.true_labels)).astype(int).mean() * 100
        print(f"Ensemble Validation Accuracy: {ensemble_accuracy:.2f}%")
        del self.predictions, self.true_labels
        torch.cuda.empty_cache()
        gc.collect()