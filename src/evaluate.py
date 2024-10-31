import torch

def test(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total += labels.size(0)

            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            running_loss += loss.item()
    test_accuracy = 100 * correct / total
    test_loss = running_loss / total

    return test_loss, test_accuracy