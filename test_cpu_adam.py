import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from cpu_adam import CPUAdam, construct_for_parameters

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def train(model, train_loader, optimizers, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # Zero all gradients
        for opt in optimizers:
            opt.zero_grad()
            
        # Forward pass
        output = model(data)
        loss = F.nll_loss(output, target)
        
        # Backward pass
        loss.backward()
        
        # Step each parameter with its optimizer
        for param, opt in zip(model.parameters(), optimizers):
            if param.grad is not None:
                param.grad = param.grad.cpu()  # Ensure grad is on CPU
                opt.step(param.cpu(), param.grad)
        
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

def main():
    # Training settings
    batch_size = 64
    epochs = 5
    
    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    
    # Initialize model
    model = SimpleNet()
    
    # Move model to CPU and create optimizers
    model = model.cpu()
    optimizers = construct_for_parameters(
        params=model.parameters(),
        lr=0.01,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Print SIMD level being used
    print(f"Using SIMD level: {CPUAdam.simd_level()}")
    
    # Train the model
    for epoch in range(1, epochs + 1):
        train(model, train_loader, optimizers, epoch)

if __name__ == '__main__':
    main()
