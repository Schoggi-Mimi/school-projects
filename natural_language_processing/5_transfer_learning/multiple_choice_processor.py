import pandas as pd
import torch
import transformers
from torch.utils.data import DataLoader

class MultipleChoiceProcessor:
    def __init__(self, tokenizer:transformers.PreTrainedModel,
                 train:pd.DataFrame, dev:pd.DataFrame, test:pd.DataFrame, set='easy'):
        if set == 'easy':
            self.max_length=147 #144+3
            #self.max_length=512
        elif set == 'challenge':
            self.max_length=160 #157+3
        
        self.tokenizer = tokenizer
        self.train_tokenized = self.tokenize_dataframe(train)
        self.val_tokenized = self.tokenize_dataframe(dev)
        self.test_tokenized = self.tokenize_dataframe(test)

    def tokenize_dataframe(self, dataframe:pd.DataFrame):
        tokenized_examples = []
        answer_key_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, '1': 0, '2': 1, '3': 2, '4': 3}
        df = dataframe.copy()
        try:
            df['choices'] = df['choices'].apply(lambda x: x['text'])
        except:
            pass
        df = df[df['choices'].apply(len) == 4]
        for _, row in df.iterrows():
            labels = []
            for choice_index, _ in enumerate(row['choices']):
                try:
                    label = 1 if choice_index == answer_key_mapping[row['answerKey']] else 0
                except:
                    label = 1 if choice_index == row['answerKey'] else 0
                labels.append(label)

            question = [row['question']] * len(labels)
            answers = [f"{text}" for text in row['choices']]

            tokenized = self.tokenizer(
                question,
                answers,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt',
                return_attention_mask=True,
            )

            tokenized['labels'] = torch.tensor(labels)
            tokenized_examples.append(tokenized)

        return tokenized_examples

    def create_datasets(self, batch_size:int=16, train_batch_size:int=16):
        train_dataset = self.MultipleChoiceDataset(self.train_tokenized)
        val_dataset = self.MultipleChoiceDataset(self.val_tokenized)
        test_dataset = self.MultipleChoiceDataset(self.test_tokenized)
        if train_batch_size == None:
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        else:
            train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_dataloader, val_dataloader, test_dataloader

    class MultipleChoiceDataset(torch.utils.data.Dataset):
        def __init__(self, tokenized_data):
            self.tokenized_data = tokenized_data

        def __len__(self):
            return len(self.tokenized_data)

        def __getitem__(self, idx):
            return {
                'input_ids': self.tokenized_data[idx]['input_ids'],
                'attention_mask': self.tokenized_data[idx]['attention_mask'],
                'labels': self.tokenized_data[idx]['labels'].clone().detach().float()
            }
