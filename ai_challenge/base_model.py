import numpy as np
import torch
from tqdm import tqdm
from datetime import datetime
import gc
import random
import wandb
import optuna
#from torch.optim.lr_scheduler import CosineAnnealingLR

class BaseModel():
    """
    model: The model object
    model_path: The path to where the model should be stored
    """
    def __init__(self, optimizer, model, model_path='model.pth', scheduler=None, 
                 enable_wandb=True, pretrained=False, device=None):
        self.best_loss = np.inf
        self.best_model = None
        self.pretrained = pretrained
        
        # Set torch processor to GPU if available
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        self.model = model.to(self.device)
        self.model_path = model_path
        self.model_name = type(model).__name__
        if pretrained:
            self.model_config = model.config
        else:
            self.model_config = model.model_config
        
        self.criterion_mae = torch.nn.L1Loss()
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.enable_wandb = enable_wandb
        self._seed(42)
    
    def __del__(self):
        self.dispose()

    def criterion(self, y, y_pred, loss_weights, y_mask_reshaped):
        y_diff = y - y_pred
        y_diff = torch.abs(y_diff)

        loss_weights_reshaped = torch.flatten(loss_weights)
        loss_weights_reshaped = loss_weights_reshaped[y_mask_reshaped]
        y_diff = torch.multiply(y_diff, loss_weights_reshaped)

        y_diff = torch.sum(y_diff) / y.shape[0]
        return y_diff

    def l1_clipped(self, y, y_pred):
        y = torch.clip(y, min=0.0, max=1.0)
        y_pred = torch.clip(y_pred, min=0.0, max=1.0)

        y_diff = y - y_pred
        y_diff = torch.abs(y_diff)
        y_diff = torch.sum(y_diff) / y.shape[0]

        return y_diff

    def _seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
            
    def _train_model(self, trial, train_loader, verbose):
        """
        trial is only passed for Optuna Hyperparameter Search runs
        """
        if trial:
            #Define Hyperparameters for Optuna Run
            #Overwrite Optimizer and Scheduler in Optuna Context with hyperparameters
            lr = trial.suggest_float('lr', self.optuna_params['lr'][0], self.optuna_params['lr'][1], log=True)
            beta_0 = trial.suggest_float('beta_0', self.optuna_params['beta_0'][0], self.optuna_params['beta_0'][1], log=True)
            beta_1 = trial.suggest_float('beta_1', self.optuna_params['beta_1'][0], self.optuna_params['beta_1'][1], log=True)
            eps = trial.suggest_float('eps', self.optuna_params['eps'][0], self.optuna_params['eps'][1], log=True)
            weight_decay = trial.suggest_float('weight_decay', self.optuna_params['weight_decay'][0], self.optuna_params['weight_decay'][1], log=True)
            # (...)

            #TODO: Allow selection of optimizer, no static optimizer override

            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, betas=(beta_0,beta_1), eps=eps, weight_decay=weight_decay)

        self.model.train()

        total_mae_loss = 0.0
        total_clipped_loss = 0.0
        total_loss = 0.0
        num_iterations = 0.0

        if type(train_loader) != list:
            train_loader = [train_loader]
            
        for loader_i, train_loader_part in enumerate(train_loader):
            model_training_iter = tqdm(train_loader_part)
            for model_train_it in model_training_iter:
                if self.pretrained:
                    X, attention_mask, y, y_mask, loss_weights = model_train_it
                    y_pred = self.model(X, attention_mask=attention_mask, labels=y)
                    y_pred = y_pred.logits
                else:
                    X, y, y_mask, loss_weights = model_train_it
                    y_pred = self.model(X)
                
                y_mask_reshaped = torch.flatten(y_mask)
                y_pred_reshaped = torch.flatten(y_pred)
                y_reshaped = torch.flatten(y)
    
                # Apply sequence mask
                y_pred_reshaped = y_pred_reshaped[y_mask_reshaped]
                y_reshaped = y_reshaped[y_mask_reshaped]
                
                mae_loss = self.criterion_mae(y_pred_reshaped, y_reshaped).item()
                clipped_loss = self.l1_clipped(y_pred_reshaped, y_reshaped).item()
                total_mae_loss += mae_loss
                total_clipped_loss += clipped_loss
                loss = self.criterion(y_pred_reshaped, y_reshaped, loss_weights, y_mask_reshaped)
                
                self.optimizer.zero_grad()
                
                loss.backward()
    
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)

                total_loss += loss.item()
                num_iterations += 1
                
                self.optimizer.step()
                            
                model_training_iter.set_description(f'Fold {loader_i+1}, Step Train Loss Weighted: {loss:.7f}, Step Train Loss MAE: {mae_loss:.7f}')

        # Adjust the scheduler after each epoch
        if self.scheduler != None:
            self.scheduler.step()

        # Optuna: Report Epoch Loss
        if trial:
            #TODO: Fix so it correctly logs epochs https://optuna.readthedocs.io/en/v2.0.0/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial.report
            trial.report(total_loss / num_iterations, 1)
            
        # Optuna: Prune Criteria
        if trial:
            if trial.should_prune():
                raise optuna.TrialPruned()
            
        return (total_loss / num_iterations, total_mae_loss / num_iterations, total_clipped_loss / num_iterations)
            
    def _evaluate_model(self, test_loader, verbose):
        pred_max_epoch = -np.inf
        pred_min_epoch = np.inf
        true_max_epoch = -np.inf
        true_min_epoch = np.inf
        
        self.model.eval()
        
        loss = 0.0
        loss_mae = 0.0
        loss_clipped = 0.0
        num_iterations = 0.0
        
        with torch.no_grad():
            for model_test_it in tqdm(test_loader):
                if self.pretrained:
                    X, attention_mask, y, y_mask, loss_weights = model_test_it
                    y_pred = self.model(X, attention_mask=attention_mask, labels=y)
                    y_pred = y_pred.logits
                else:
                    X, y, y_mask, loss_weights = model_test_it
                    y_pred = self.model(X)

                y_mask_reshaped = torch.flatten(y_mask)
                y_pred_reshaped = torch.flatten(y_pred)
                y_reshaped = torch.flatten(y)
                
                # Apply sequence mask
                y_pred_reshaped = y_pred_reshaped[y_mask_reshaped]
                y_reshaped = y_reshaped[y_mask_reshaped]
                
                loss += self.criterion(y_pred_reshaped, y_reshaped, loss_weights, y_mask_reshaped).item()
                loss_mae += self.criterion_mae(y_pred_reshaped, y_reshaped).item()
                loss_clipped += self.l1_clipped(y_pred_reshaped, y_reshaped).item()
                num_iterations += 1
                # additional min/max pred/true to estimate trend over epochs
                if y_pred.max() > pred_max_epoch:
                    pred_max_epoch = y_pred.max()
                if y_pred.min() < pred_min_epoch:
                    pred_min_epoch = y_pred.min()            
                if y.max() > true_max_epoch:
                    true_max_epoch = y.max()
                if y.min() < true_min_epoch:
                    true_min_epoch = y.min()
        if self.enable_wandb:
            wandb.log({"pred_max":pred_max_epoch, "pred_min":pred_min_epoch}) 
        
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_model = self.model.state_dict()
        if verbose:
            print(f'(Pred: {pred_max_epoch:.3f} {pred_min_epoch:.3f} /  True: {true_max_epoch:.3f} {true_min_epoch:.3f})')
        
        return (loss / num_iterations, loss_mae / num_iterations, loss_clipped / num_iterations)
    
    def _init_wandb_logger(self, project_name, epochs, experiment_type, 
                           batch_size_train, batch_size_valid, data_len_train, 
                           data_len_valid, preprocessing_config, tags):
        """Initializes the wanbd logger with all essential run parameters"""
        """TBD: Add suitable run names"""

        wandb.init(
            project=project_name,
            tags=tags,
            config={
                "model_name": self.model_name,
                "model_config": self.model_config,
                "preprocessing_config": preprocessing_config,
                "epochs": epochs,
                "experiment_type": experiment_type,
                "batch_size_train": batch_size_train,
                "batch_size_valid": batch_size_valid,
                "data_len_train": data_len_train,
                "data_len_valid": data_len_valid,
                "learning_rate" : self.optimizer.param_groups[0]['lr'],
                "betas" : self.optimizer.param_groups[0]['betas'],
                "eps" : self.optimizer.param_groups[0]['eps'],
                "weight_decay" : self.optimizer.param_groups[0]['weight_decay'],
                "scheduler": f"{type(self.scheduler)}"
            }
        )
        
    def fit(self, train_loader, test_loader, experiment_type, epochs=1, 
            verbose=True, optuna_study=None, optuna_params=None, preprocessing_config=None,
            tags=[]):
        self.best_loss = np.inf
        self.best_model = None

        self.optuna_study = optuna_study
        self.optuna_params = optuna_params

        if self.optuna_study:
            print(f"INFO: Optuna Hyperparameter Searching Mode is on. Best Trial for {self.optuna_study.study_name} so far:")
            print(self.optuna_study.best_trial)
            print(50*'#')
            if not optuna_params:
                raise Exception("Must provide optuna_params as argument if Optuna tuning is enabled.")


        if self.enable_wandb:
            if type(train_loader) == list:
                num_folds = len(train_loader)
                batch_size_folds = train_loader[0].batch_size
                total_data_count = sum([len(t_loader.dataset) for t_loader in train_loader])
                single_data_count = len(train_loader[0].dataset)
                
                self._init_wandb_logger(project_name='hslu-stableconfusion',
                                        epochs=epochs,
                                        experiment_type=experiment_type,
                                        batch_size_train=batch_size_folds,
                                        batch_size_valid=batch_size_folds,
                                        data_len_train=total_data_count - single_data_count,
                                        data_len_valid=single_data_count,
                                        preprocessing_config=preprocessing_config,
                                        tags=tags)
            else:
                self._init_wandb_logger(project_name='hslu-stableconfusion',
                                        epochs=epochs,
                                        experiment_type=experiment_type,
                                        batch_size_train=train_loader.batch_size,
                                        batch_size_valid=test_loader.batch_size,
                                        data_len_train=len(train_loader.dataset),
                                        data_len_valid=len(test_loader.dataset),
                                        preprocessing_config=preprocessing_config,
                                        tags=tags)
        
        training_losses = []
        validation_losses = []

        use_cv = False
        current_cv = 0
        
        if test_loader is None and type(train_loader) == list:
            print('Using CV for training')
            use_cv = True
            train_loaders = train_loader
            num_folds = len(train_loaders)
            
        for i in range(epochs):
            if use_cv:
                test_loader = train_loaders[current_cv]
                train_loader = [x for i_x, x in enumerate(train_loaders) if i_x != current_cv]
                if current_cv + 1 == num_folds:
                    current_cv = 0
                else:
                    current_cv += 1
                
            # Model training
            if self.optuna_study:
                self.optuna_study.optimize(lambda trial: self._train_model(trial, train_loader, verbose))
            else:
                train_loss, train_loss_mae, train_clipped_loss = self._train_model(None, train_loader, verbose)
            training_losses.append(train_loss)
            
            # Model validation
            val_loss, val_loss_mae, val_clipped_loss = self._evaluate_model(test_loader, verbose)
            validation_losses.append(val_loss)
            
            if self.enable_wandb:
                wandb.log({
                    "train_loss_weighted": train_loss,
                    "val_loss_weighted": val_loss,
                    "train_loss": train_loss_mae, 
                    "val_loss": val_loss_mae,
                    "train_loss_clipped": train_clipped_loss, 
                    "val_loss_clipped": val_clipped_loss
                    })

            if verbose:
                print(f"""
                      Epoch: {i+1}, 
                      Train Loss MAE: {train_loss_mae:.7f}, Valid Loss MEA: {val_loss_mae:.7f},
                      Train Loss Weighted: {train_loss:.7f}, Valid Loss Weighted: {val_loss:.7f}, 
                      Train Loss Clipped: {train_clipped_loss:.7f}, Valid Loss Clipped: {val_clipped_loss:.7f}""", end=' ', sep='')
                print(f'Time: {datetime.now().strftime("%H:%M:%S")}', end=' ', sep='')
        
        torch.save(self.best_model, self.model_path)

        if self.enable_wandb:
            model_artifact = wandb.Artifact(name=f'{datetime.now().strftime("%d-%m-%Y_%H-%M-%S")}',type="model", description=f"Best Model pth file {self.model_name}")
            model_artifact.add_file(self.model_path)
            wandb.log_artifact(model_artifact)
        
        
        if verbose:
            print(f'Model saved to: {self.model_path}')

        if self.enable_wandb:
            wandb.finish()

        return training_losses, validation_losses

    def predict(self, x_data_loader, single_model_mode=False, new_init=True, finish_wandb=True):
        if self.enable_wandb and new_init:
            self.iteration = 0
            wandb.init(
                project='hslu-stableconfusion',
                tags=['Prediction'],
                config={
                    "model_name": self.model_name,
                    "model_config": self.model_config,
                    "learning_rate" : self.optimizer.param_groups[0]['lr'],
                    "betas" : self.optimizer.param_groups[0]['betas'],
                    "eps" : self.optimizer.param_groups[0]['eps'],
                    "weight_decay" : self.optimizer.param_groups[0]['weight_decay'],
                    "scheduler": f"{type(self.scheduler)}"
                }
            )

        self.model.eval()
        predictions = []
        with torch.no_grad():
            for data_loader_it in tqdm(x_data_loader):
                if self.enable_wandb:
                    wandb.log({
                        "iteration": self.iteration,
                        })
                    self.iteration += 1

                if self.pretrained:
                    X, attention_mask, sequence_length = data_loader_it
                    y_pred = self.model(X, attention_mask=attention_mask)
                    y_pred = y_pred.logits
                    y_pred = torch.unflatten(y_pred, dim=1, sizes=[457, 2])
                else:
                    X, sequence_length = data_loader_it
                    y_pred = self.model(X)

                y_pred = y_pred.round(decimals=4)
                # Add each row to the 
                for i in range(len(sequence_length)):
                    row = y_pred[i, :sequence_length[i]+1, :]
                    predictions.append(row)

        predictions = torch.cat(predictions, dim=0)
        
        if not single_model_mode:
            predictions = predictions.view(-1)
            
        gc.collect()
        
        if self.enable_wandb and finish_wandb:
            wandb.finish()

        return predictions

    def load_model(self, verbose=True):
        self.model.load_state_dict(torch.load(self.model_path, map_location=torch.device(self.device)))
        #self.best_model.load_state_dict(torch.load(self.model_path, map_location=torch.device(self.device)))

        if verbose:
            print(f'Model loaded from: {self.model_path}')

    def clear_gpu(self, verbose=True):
        if verbose:
            print("Before empty_cache:", torch.cuda.memory_allocated())
        torch.cuda.empty_cache()
        gc.collect()
        if verbose:
            print("After empty_cache:", torch.cuda.memory_allocated())

    def dispose(self, verbose=True):
        self.clear_gpu(verbose)
        del self.best_model
        del self.model
        gc.collect()