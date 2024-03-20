import torch

def calculate_accuracy_from_predicitions(predictions, truths):
    _, predicted_classes = torch.max(predictions, 1)
    _, true_classes = torch.max(truths, 1)

    correct = torch.eq(predicted_classes, true_classes).sum().item()
    total = truths.size(0)
    return correct / total