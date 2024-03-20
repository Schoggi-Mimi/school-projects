import numpy as np
import os
import torch
import torch.nn.functional as F
import transformers
import random
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from datetime import datetime
import gc
from tqdm.auto import tqdm
from captum.attr import visualization as viz
from captum.attr import LayerConductance, LayerIntegratedGradients, TokenReferenceBase


class TrainingPipeline:
    def __init__(self, model:transformers.PreTrainedModel, device:torch.device,
                 train_loader:torch.utils.data.DataLoader, val_loader:torch.utils.data.DataLoader, 
                 optimizer:torch.optim.Optimizer, scheduler:torch.optim.lr_scheduler, track:bool=False, num_epochs:int=10, lr:float=0.0001, model_checkpoint:bool=False):
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.track = track
        self.num_epochs = num_epochs
        self.lr = lr
        self.train_accuracy_list = []
        self.val_accuracy_list = []
        self.train_loss_list = []
        self.val_loss_list = []
        self.predictions = []
        self.true_labels = []
        self.model_predictions = []
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.choices = ['A', 'B', 'C', 'D']
        self.checkpoint = model_checkpoint

        if self.track:
            self._init_wandb_logger()

    def _init_wandb_logger(self):
        wandb.init(
            project='hslu-stableconfusion-nlp',
            config={
                "epochs": self.num_epochs,
                "learning_rate": self.lr,
            }
        )
    def _calculate_accuracy(self, predictions, true_labels):
        if not isinstance(predictions, np.ndarray):
            predictions = predictions.cpu().numpy() if torch.is_tensor(predictions) else np.array(predictions)
        if not isinstance(true_labels, np.ndarray):
            true_labels = true_labels.cpu().numpy() if torch.is_tensor(true_labels) else np.array(true_labels)
        correct = np.sum(predictions == true_labels)
        total = len(true_labels)
        accuracy = correct / total * 100.0
        return accuracy

    def train(self):
        #progress_bar = tqdm(range(self.num_epochs*len(self.train_loader)))
        progress_bar = tqdm(total=self.num_epochs*len(self.train_loader))
        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0.0
            correct_predictions = 0
            total_predictions = 0

            for batch in self.train_loader:
                self.optimizer.zero_grad()

                inputs = batch['input_ids'].to(self.device)
                masks = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(inputs, attention_mask=masks, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()
                indx_prob = torch.argmax(F.softmax(outputs.logits, dim=1), dim=1)
                
                indx_labels = torch.argmax(labels, dim=1)
                correct_predictions += torch.sum(indx_prob == indx_labels).item()
                total_predictions += len(indx_labels)
                
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                progress_bar.update(1)
                if self.track:
                    wandb.log({"train_loss": loss})
            average_train_loss = total_loss / len(self.train_loader)
            accuracy = (correct_predictions / total_predictions) * 100
            self.train_accuracy_list.append(accuracy)
            self.train_loss_list.append(average_train_loss)
            if self.track:
                wandb.log({"train_accuracy": accuracy})

            self.model.eval()
            total_val_loss = 0.0
            correct_val_predictions = 0
            total_val_predictions = 0

            for batch in self.val_loader:
                inputs = batch['input_ids'].to(self.device)
                masks = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                with torch.inference_mode():
                    outputs = self.model(inputs, attention_mask=masks, labels=labels)
                val_loss = outputs.loss
                total_val_loss += val_loss.item()

                indx_prob = torch.argmax(F.softmax(outputs.logits, dim=1), dim=1)
                indx_labels = torch.argmax(labels, dim=1)
                correct_val_predictions += torch.sum(indx_prob == indx_labels).item()
                total_val_predictions += len(indx_labels)
                self.predictions.extend(indx_prob.cpu().numpy())
                self.true_labels.extend(labels.argmax(dim=1).cpu().numpy())
                if self.track:
                    wandb.log({"val_loss": val_loss})
            average_val_loss = total_val_loss / len(self.val_loader)
            val_accuracy = (correct_val_predictions / total_val_predictions) * 100
            self.val_accuracy_list.append(val_accuracy)
            self.val_loss_list.append(average_val_loss)
            if self.track:
                wandb.log({"val_accuracy": val_accuracy})
                
            print(f"Epoch {epoch + 1}/{self.num_epochs} - Average Train Loss: {average_train_loss:.4f} - Train Accuracy: {accuracy:.2f}% - Validation Loss: {average_val_loss:.4f} - Validation Accuracy: {val_accuracy:.2f}%")
        if self.checkpoint:
            current_time = datetime.now().strftime("%d-%m_%H-%M")
            #model_path = f"./models/{current_time}_{self.model.config._name_or_path}_model.pth"
            model_path = f"./model_state_dicts/{current_time}.pth"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(self.model.state_dict(), model_path)
            if self.track:
                model_artifact = wandb.Artifact(name=f'{current_time}_bert_model',type="model", description=f"Best {self.model.config._name_or_path} Model pth file NLP")
                model_artifact.add_file(model_path)
                wandb.log_artifact(model_artifact)
        if self.track:
            wandb.finish()
        del total_loss, correct_predictions, total_predictions, total_val_loss, correct_val_predictions, total_val_predictions
        torch.cuda.empty_cache()
        gc.collect()
        return self.train_loss_list, self.train_accuracy_list, self.val_loss_list, self.val_accuracy_list
    
    def test(self, test_loader:torch.utils.data.DataLoader):
        self.model.eval()
        total_test_loss = 0.0
        correct_test_predictions = 0
        total_test_predictions = 0

        with torch.inference_mode():
            for batch in test_loader:
                inputs = batch['input_ids'].to(self.device)
                masks = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(inputs, attention_mask=masks, labels=labels)
                test_loss = outputs.loss
                total_test_loss += test_loss.item()

                indx_prob = torch.argmax(F.softmax(outputs.logits, dim=1), dim=1)
                indx_labels = torch.argmax(labels, dim=1)
                correct_test_predictions += torch.sum(indx_prob == indx_labels).item()
                total_test_predictions += len(indx_labels)

        average_test_loss = total_test_loss / len(test_loader)
        test_accuracy = (correct_test_predictions / total_test_predictions) * 100
        print(f"Final Test Loss: {average_test_loss:.4f} - Test Accuracy: {test_accuracy:.2f}%")
        return average_test_loss, test_accuracy

    def confusion_matrix(self, report=False):
        if report:
            report = classification_report(self.true_labels, self.predictions)
            print("Classification Report:")
            print(report, '\n')
        conf_matrix = confusion_matrix(self.true_labels, self.predictions)
        conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
        fig, ax = plt.subplots(figsize=(4, 4))
        cmd = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=self.choices)
        cmd.plot(cmap="Blues", ax=ax)
        ax.set_title('Normalized Confusion Matrix')
        plt.show()