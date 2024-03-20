import torch
import torch.nn as nn
import torch.nn.functional as F

class Word_embedding_classifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, num_layers=2, activation_function=F.relu):
        super(Word_embedding_classifier, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(5 * embedding_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.activation_function = activation_function

    def forward(self, x1, x2, x3, x4, x5):
        x = torch.cat((x1, x2, x3, x4, x5), dim=-1)
        for layer in self.layers:
            x = self.activation_function(layer(x))
        x = self.output_layer(x)
        x = F.softmax(x, dim=-1)
        return x