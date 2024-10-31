# Melasma Skin Disease Diagnosis Using Deep Neural Networks

This repository contains code for diagnosing Melasma skin disease using various deep neural network architectures. The model training is performed in two stages: pre-training on CIFAR-10, followed by fine-tuning on a custom Melasma dataset. The networks used in this project include VGGNet16, ResNet50, and AlexNet.

## Project Structure
- `src/`: Contains source code for loading datasets, defining model architectures, and training/evaluation functions.
- `data/`: Contains data folders for CIFAR-10 and the custom Melasma dataset.
- `history/`: Stores JSON files with training history logs for each model.
- `models/`: Saves trained model checkpoints.

## Workflow

### Part 1: Pre-train on CIFAR-10
1. **Dataset**: The CIFAR-10 dataset is used to initialize the models.
2. **Models**: The VGGNet16, ResNet50, and AlexNet architectures are trained on CIFAR-10.
3. **Training**: The models are trained for 100 epochs, and training history is logged in JSON files.
4. **Checkpoints**: Model checkpoints are saved to the `models/` folder after training.(For access to saved model feel free to contact me)

### Part 2: Fine-tune on Melasma Dataset
1. **Dataset**: The Melasma skin disease dataset, containing labeled images, is used for fine-tuning the pre-trained models.
2. **Models**: Checkpoints from Part 1 are loaded and further trained on the Melasma dataset.
3. **Training**: The models are fine-tuned for 40 epochs, and updated training history is logged in JSON files.
4. **Checkpoints**: Final fine-tuned models are saved for each architecture.


## Files and Functions

### Models
- **VGGNet16**: A VGG-style network, customized for CIFAR-10 and fine-tuned on Melasma.
- **ResNet50**: A 50-layer ResNet architecture for both CIFAR-10 and Melasma.
- **AlexNet**: A smaller network architecture suitable for initial testing.

### Training and Evaluation
- **train()**: Trains the model for each epoch and returns loss and accuracy.
- **test()**: Evaluates the model on the test set for each epoch.

## Results
Each model's training and testing performance metrics (loss and accuracy) are saved in the `history/` directory in JSON format.

## Requirements
- Python 3
- PyTorch
- Torchvision
- Pandas

## License
This project is open-source and is available for free use, modification, and distribution.