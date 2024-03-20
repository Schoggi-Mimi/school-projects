import noise_filter
import torch
import time
import numpy as np
from tqdm import tqdm
import gc
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torch.utils.data import DataLoader
from transformers.tokenization_utils_base import BatchEncoding

class Preprocessor():
    def __init__(self, max_sequence_length=457):
        self.max_sequence_length = max_sequence_length
        self.char_to_categorical = {
                'A': [1], 
                'C': [2], 
                'G': [3], 
                'U': [4]
        }
        self.seq_to_categorical = {
                'A': 1, 
                'C': 2, 
                'G': 3, 
                'U': 4
        }
        self.struc_to_categorical = {
                    '(': 1,
                    '.': 2,
                    ')': 3
        }
        self.loop_to_categorical = {
                    'E': 1,
                    'H': 2,
                    'I': 3,
                    'M': 4,
                    'S': 5
        }
        self.char_to_binary = {
                'A': [1, 0, 0, 0], 
                'C': [0, 1, 0, 0], 
                'G': [0, 0, 1, 0], 
                'U': [0, 0, 0, 1]
        }
    
    def _seq_struc_to_categorical(self,seq,struc,loop):
        return [self.seq_to_categorical.get(seq)*100 + self.struc_to_categorical.get(struc)*10 + self.loop_to_categorical.get(loop)]
        
    def _get_x_y_columns(self, train_data):
        X_cols = ['sequence']
        y_cols = []
        reactivity_errors = []
        
        for col in train_data.columns:
            if col.startswith('reactivity_error_'):
                reactivity_errors.append(col)
                continue
            
            if col.startswith('reactivity_'):
                y_cols.append(col)
                continue
        
        return X_cols, y_cols, reactivity_errors

    def _seperate_by_experiment_type(self, train_data, shuffle=True, verbose=True):
        if shuffle:
            if verbose:
                print('Shuffeling training data')
            np.random.seed(42)
            train_data = train_data.sample(frac=1)

        if verbose:
            print('Splitting DMS_MaP')
            
        train_dms = train_data[train_data['experiment_type'] == 'DMS_MaP']
        
        if verbose:
            print('Splitting 2A3_MaP')
            
        train_2A3 = train_data[train_data['experiment_type'] == '2A3_MaP']

        return [('DMS_MaP', train_dms), ('2A3_MaP', train_2A3)]

    def _group_by_sequence(self, train_data, duplicate_mode='high_signal', shuffle=True, verbose=True):
        """duplicate_mode defines how duplicates(multiple experiments for 1 sequence) are handled."""
        if shuffle:
            if verbose:
                print('Shuffeling training data')
            np.random.seed(42)
            train_data = train_data.sample(frac=1)

        if verbose:
            print('Combining DMS and 2A3')
        
        #Split into DMS and 2A3
        train_dms = train_data[train_data['experiment_type'] == 'DMS_MaP']
        train_2A3 = train_data[train_data['experiment_type'] == '2A3_MaP']

        #Duplicate handling
        if duplicate_mode=='keep_first':
            if verbose:
                print('Duplicate Handling: Keep first entry')
            train_dms = train_dms.drop_duplicates(subset=['sequence'], keep='first')
            train_2A3 = train_2A3.drop_duplicates(subset=['sequence'], keep='first')

        elif duplicate_mode=='high_signal':
            """only keep entries with highest stn (signal_to_noise)"""
            if verbose:
                print('Duplicate Handling: Keep entry with highest signal_to_noise')
            max_stn_indices_dms = train_dms.groupby(['sequence'])['signal_to_noise'].idxmax()
            train_dms = train_dms.loc[max_stn_indices_dms]

            max_stn_indices_2A3 = train_2A3.groupby(['sequence'])['signal_to_noise'].idxmax()
            train_2A3 = train_2A3.loc[max_stn_indices_2A3]

        elif duplicate_mode=='average':
            # TODO: Take average of all entries with same sequence
            raise NotImplementedError()

        else:
            raise ValueError("duplicate_mode not recognized")
        
        #Merge (Merge second df on sequence with suffix _2)
        train_merged = train_dms.merge(train_2A3, on='sequence', how='inner', suffixes=("", "_2"))

        return [('DMS_AND_2A3_MaP', train_merged)]

    def _sequence_to_encoding(self, train_data, column_name, categorical, structure, device):
        if not isinstance(categorical, (bool)):
            X_train_final = list(train_data[column_name])
            X_train_final = categorical(X_train_final)
            
            for key in X_train_final.keys():
                X_train_final[key] = X_train_final[key].to(device)

            return X_train_final
        
        if categorical:
            if structure:
                X_train_final = [
                    [
                        self._seq_struc_to_categorical(seq,struc,loop)
                        for seq,struc,loop in zip(row['sequence'],row['structure'],row['predicted_loop_type']) 
                    ] + ([[0]] * (self.max_sequence_length - len(row['sequence'])))
                    for _,row in tqdm(train_data[['sequence','structure','predicted_loop_type']].iterrows(), position=0, leave=True)
                ]
            else:
                X_train_final = [
                    [
                        self.char_to_categorical.get(char, [0])
                        for char in i
                    ] + ([[0]] * (self.max_sequence_length - len(i)))
                    for i in tqdm(train_data[column_name], position=0, leave=True)
                ]
        else:
            if structure:
                raise NotImplementedError("Encoding of structural features has not been implemented for binary representation.")
            X_train_final = [
                [
                    self.char_to_binary.get(char, [0, 0, 0, 0])
                    for char in i
                ] + ([[0, 0, 0, 0]] * (self.max_sequence_length - len(i)))
                for i in tqdm(train_data[column_name], position=0, leave=True)
            ]

        X_train_final = torch.tensor(X_train_final, dtype=torch.float32).to(device)

        return X_train_final

    def _pad_y(self, y, dual_model):
        
        if dual_model:
            new_y = np.zeros((y.shape[0], self.max_sequence_length))
            new_y[:, :y.shape[1]] = y
            y = new_y
        else:
            # Reshape from (datapoints, 412) to (datapoints, 206, 2) to pad DMS and 2A3 seperately
            # And fill to max seq lenght e.g. (datapoints, 457, 2)
            first_half = int(y.shape[1]/2)
            new_y = np.zeros(shape=(y.shape[0], self.max_sequence_length, 2))
            for row in tqdm(range(y.shape[0])):
                new_y[row, :first_half, 0] = y[row, :first_half]
                new_y[row, :first_half, 1] = y[row, first_half:]
            y = new_y
        del new_y
        gc.collect()
        
        return y
    
    def _y_mask(self, dataset, y_cols, dual_model):
        if dual_model:
            y_mask = dataset[y_cols].notnull().astype('bool')
        else:
            y_shape = dataset[y_cols].shape
            first_half = int(y_shape[1]/2)

            y_cols1 = [col for col in y_cols if not col.endswith('_2')]
            y_cols2 = [col for col in y_cols if col.endswith('_2')]

            nan_mask_values_1 = dataset[y_cols1].notnull().to_numpy(dtype='bool')
            nan_mask_values_2 = dataset[y_cols2].notnull().to_numpy(dtype='bool')

            y_mask = np.zeros(shape=(y_shape[0], self.max_sequence_length, 2)).astype('bool')

            for row in tqdm(range(y_shape[0])):
                y_mask[row, :first_half, 0] = nan_mask_values_1[row]
                y_mask[row, :first_half, 1] = nan_mask_values_2[row]

        y_mask = torch.from_numpy(y_mask)

        return y_mask
    
    def _weight_loss(self, dataset, reactivity_errors, min_weight, additive_weight, dual_model):
        if dual_model:
            print('Not supported yet')
            raise NotImplementedError()
        
        loss_weights = dataset[reactivity_errors].to_numpy()
        loss_weights = np.nan_to_num(1 - loss_weights, nan=min_weight)
        loss_weights = np.clip(loss_weights, a_min=min_weight, a_max=1)

        first_half = int(loss_weights.shape[1]/2)
        new_loss_weights = np.full(shape=(loss_weights.shape[0], self.max_sequence_length, 2), 
                                   fill_value=min_weight)
        for row in tqdm(range(loss_weights.shape[0])):
            new_loss_weights[row, :first_half, 0] = loss_weights[row, :first_half]
            new_loss_weights[row, :first_half, 1] = loss_weights[row, first_half:]
        loss_weights = new_loss_weights

        if additive_weight:
            loss_weights += 1

        del new_loss_weights
        gc.collect()

        #loss_weights = loss_weights[nan_mask_values]
        
        return loss_weights

    def _data_loaders(self, X, y, validation_split=0.1, batch_size=32, y_mask=None, k_fold=None, loss_weights=None):
        if validation_split is not None:
            if y_mask is None:
                print('y_mask required for data loader in training')
                return None

            train_idx, validation_idx = train_test_split(list(range(len(X))), test_size=validation_split)
            

            X_train = Subset(X, train_idx)
            X_val = Subset(X, validation_idx)

            y_train = Subset(y, train_idx)
            y_val = Subset(y, validation_idx)
            
            y_mask_train = Subset(y_mask, train_idx)
            y_mask_val = Subset(y_mask, validation_idx)

            training_subsets = [X_train, y_train, y_mask_train]
            validation_subsets = [X_val, y_val, y_mask_val]

            loss_weights_train = Subset(loss_weights, train_idx)
            training_subsets.append(loss_weights_train)
            
            loss_weights_val = Subset(loss_weights, validation_idx)
            validation_subsets.append(loss_weights_val)

            train_loader = DataLoader(list(zip(*training_subsets)), batch_size=batch_size, shuffle=False)
            validation_loader = DataLoader(list(zip(*validation_subsets)), batch_size=batch_size, shuffle=False)
        elif k_fold is not None:
            if y_mask is None:
                print('y_mask required for data loader in training')
                return None
            
            if isinstance(X, (BatchEncoding)):
                print('X is BatchEncoding')
                num_elements = len(X['input_ids'])

                y = torch.flatten(y, start_dim=1)
                y_mask = torch.flatten(y_mask, start_dim=1)
                loss_weights = torch.flatten(loss_weights, start_dim=1)

                print('Reshaped y to: ', y.shape)
            else:
                num_elements = len(X)

            per_fold_indecies = num_elements // k_fold
            print(f'k_fold is set to {k_fold} fold size {per_fold_indecies}')
            train_loader = []
            for i in range(k_fold):
                train_idx = list(range(per_fold_indecies * i, per_fold_indecies * (i+1) if i+1 != k_fold else num_elements))
                train_subsets = []
                if isinstance(X, (BatchEncoding)):
                    print('X is BatchEncoding')
                    X_train = Subset(X['input_ids'], train_idx)
                    train_subsets.append(X_train)
                    X_train_attention = Subset(X['attention_mask'], train_idx)
                    train_subsets.append(X_train_attention)
                else:
                    X_train = Subset(X, train_idx)
                    train_subsets.append(X_train)

                y_train = Subset(y, train_idx)

                y_mask_train = Subset(y_mask, train_idx)

                train_subsets.append(y_train)
                train_subsets.append(y_mask_train)
                
                loss_weights_train = Subset(loss_weights, train_idx)
                train_subsets.append(loss_weights_train)

                train_loader.append(
                    DataLoader(list(zip(*train_subsets)), batch_size=batch_size, shuffle=False)
                )
            
            validation_loader = None
        else:
            data_loader_list = []

            if isinstance(X, (BatchEncoding)):
                data_loader_list.append(X['input_ids'])
                data_loader_list.append(X['attention_mask'])
            else: 
                data_loader_list.append(X)
            data_loader_list.append(y)
            train_loader = DataLoader(list(zip(*data_loader_list)), batch_size=batch_size, shuffle=False)
            validation_loader = None
        
        return train_loader, validation_loader

    def prepare_prediction_dataset(self, prediction_data, batch_size=32, categorical=True, structure=False, verbose=True):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        X_cols, _, _ = self._get_x_y_columns(prediction_data)
        
        if verbose:
            print(f'Preparing dataset: with categorical encoding: {categorical}')
        X = self._sequence_to_encoding(prediction_data, X_cols[0], categorical, structure=structure, device=device)
        gc.collect()
        
        if verbose:
            print(f'Calculating sequence lengths')
        y = prediction_data['id_max'] - prediction_data['id_min']
        y = y.to_numpy()
        y = torch.tensor(y, dtype=torch.int16).to('cpu')
        
        if verbose:
            if isinstance(categorical, (bool)):
                print(f'Data shape X: {X.shape}')
            else:
                print(f'Data shape X: {X["input_ids"].shape}')

            print(f'batch_size: {batch_size}')
            print('Data loaders')
            
        
        train_loader, _ = self._data_loaders(X, y, validation_split=None, batch_size=batch_size)
        gc.collect()
        return train_loader

    def prepare_xy_split(self, train_data, categorical=True, 
                         shuffle=True, validation_split=0.1, batch_size=32, 
                         filter_noise=False, dual_model=True, k_fold=None, structure=False, 
                         clip=True, weighted_loss=0.1, additive_weight=False,
                         verbose=True):
        preprocessing_config = {
            'categorical': categorical,
            'shuffle': shuffle,
            'validation_split': validation_split,
            'batch_size': batch_size,
            'filter_noise': filter_noise,
            'dual_model': dual_model,
            'k_fold': k_fold,
            'structure': structure,
            'clip': clip,
            'weighted_loss': weighted_loss,
            'additive_weight': additive_weight
        }

        if k_fold is None and validation_split is None or k_fold is not None and validation_split is not None:
            print('Use k fold or validation split')
            
        if filter_noise:
            train_data = noise_filter.NoiseFilter().apply_filters(train_data)
            gc.collect()
            
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        X_cols, y_cols, reactivity_errors = self._get_x_y_columns(train_data)

        if dual_model:
            datasets = self._seperate_by_experiment_type(train_data, shuffle, verbose)
        else:
            y_cols = y_cols + [y_col + '_2' for y_col in y_cols]
            datasets = self._group_by_sequence(train_data, duplicate_mode='high_signal', shuffle=shuffle, verbose=verbose)

        split_datasets = []
        for experiment_type, dataset in datasets:
            if verbose:
                print(f'Preparing dataset: {experiment_type} with categorical encoding: {categorical}')
            
            if verbose:
                print(f'Calculating sequence lengths')
            
            if verbose:
                print('Creating y mask')
            y_mask = self._y_mask(dataset, y_cols, dual_model).to(device)

            if not weighted_loss is None:
                if verbose:
                    print('Creating weighted loss matrix')
                loss_weights = self._weight_loss(dataset, reactivity_errors, 
                                                 weighted_loss, additive_weight, dual_model)
                loss_weights = torch.tensor(loss_weights, dtype=torch.float32).to(device)
            else: 
                loss_weights = torch.ones(size=y_mask.shape).to(device)

            if verbose:
                print(f'Sequential encoding')
            
            if structure and verbose:
                print(f'Structure / Loop Encoding: Enabled')
            
            X = self._sequence_to_encoding(dataset, X_cols[0], categorical, structure=structure, device=device)
            gc.collect()
                           
            y = dataset[y_cols]
            if verbose:
                print('Replacing y nan values')
                
            y = np.nan_to_num(y)
            gc.collect()

            if clip:
                if verbose:
                    print('Clipping y values')
                y = np.clip(y, a_min=0, a_max=1)

            if verbose:
                print('Padding y values')
            y = self._pad_y(y, dual_model)
            
            y = torch.tensor(y, dtype=torch.float32).to(device)
            gc.collect()

            if verbose:
                if isinstance(categorical, (bool)):
                    print(f'Data shape X: {X.shape}, Data shape y: {y.shape}, y mask shape: {y_mask.shape}')
                else:
                    print(f'Data shape X: {X["input_ids"].shape}, Data shape y: {y.shape}, y mask shape: {y_mask.shape}')
                if weighted_loss:
                    print('Weighted loss shape: ', loss_weights.shape)
                print(f'batch_size: {batch_size}')
                print('Data loaders')

            train_loader, validation_loader = self._data_loaders(X, y, validation_split, batch_size, y_mask, 
                                                                 k_fold=k_fold, loss_weights=loss_weights)
            gc.collect()
            
            split_datasets.append((experiment_type, train_loader, validation_loader))
        del datasets
        del X_cols
        del y_cols
        
        gc.collect()
        
        if verbose:
            print('Output format: (Experiment type, train data loader, validation data loader)')
        
        return split_datasets, preprocessing_config
