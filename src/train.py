import torch

def train(model, dataloader, criterion, optimizer):
    total = 0.
    correct = 0.
    running_loss = 0.

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        _, predicted = outputs.max(1)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        running_loss += loss.item()

    train_accuracy = 100 * correct / total
    train_loss = running_loss / total

    return train_loss, train_accuracy