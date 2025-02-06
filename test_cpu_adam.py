import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from cpu_adam import CPUAdam
from torch.optim import Adam

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

def train_epoch(model_cpu, model_torch, train_loader, cpu_opt, torch_opt, epoch):
    model_cpu.train()
    model_torch.train()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        # Zero gradients
        cpu_opt.zero_grad()
        torch_opt.zero_grad()
        
        # Forward pass - CPU Adam
        output_cpu = model_cpu(data)
        loss_cpu = F.nll_loss(output_cpu, target)
        loss_cpu.backward()
        
        # Forward pass - Torch Adam
        output_torch = model_torch(data)
        loss_torch = F.nll_loss(output_torch, target)
        loss_torch.backward()
        
        # Step CPU Adam
        for param in model_cpu.parameters():
            if param.grad is not None:
                param.grad = param.grad.cpu()
                cpu_opt.step(param.cpu(), param.grad)
        
        # Step Torch Adam
        torch_opt.step()
        
        if batch_idx % 10 == 0:
            print(f'Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}]')
            print(f'CPU Adam Loss: {loss_cpu.item():.6f}')
            print(f'Torch Adam Loss: {loss_torch.item():.6f}')

def main():
    # Training settings
    batch_size = 32
    epochs = 5
    lr = 0.01
    betas = (0.9, 0.999)
    eps = 1e-8
    
    # Load digits dataset
    digits = load_digits()
    scaler = StandardScaler()
    images = scaler.fit_transform(digits.data).astype(np.float32)
    labels = digits.target.astype(np.int64)
    
    # Create DataLoader
    images = torch.from_numpy(images)
    labels = torch.from_numpy(labels)
    dataset = torch.utils.data.TensorDataset(images, labels)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize two identical models
    model_cpu = SimpleNet()
    model_torch = SimpleNet()
    
    # Copy initial weights to ensure same starting point
    model_torch.load_state_dict(model_cpu.state_dict())
    
    # Create optimizers
    cpu_opt = CPUAdam(lr=lr, betas=betas, eps=eps)
    torch_opt = Adam(model_torch.parameters(), lr=lr, betas=betas, eps=eps)
    
    print(f"Using SIMD level: {CPUAdam.simd_level()}")
    
    # Train both models
    for epoch in range(1, epochs + 1):
        train_epoch(model_cpu, model_torch, train_loader, cpu_opt, torch_opt, epoch)
        

if __name__ == '__main__':
    main()
