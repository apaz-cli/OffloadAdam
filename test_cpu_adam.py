import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from cpu_adam import CPUAdam
from torch.optim import Adam
import math

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
        
        # Forward/backward pass
        output_cpu = model_cpu(data)
        loss_cpu = F.nll_loss(output_cpu, target)
        loss_cpu.backward()
        output_torch = model_torch(data)
        loss_torch = F.nll_loss(output_torch, target)
        loss_torch.backward()
            
        # Optimizer steps
        cpu_opt.step()
        torch_opt.step()
        
        # Verify parameters are close
        for param_cpu, param_torch in zip(model_cpu.parameters(), model_torch.parameters()):    
            max_diff = torch.max(torch.abs(param_cpu - param_torch))
            if max_diff > 1e-4:
                raise AssertionError(f"Parameters diverged! Max difference: {max_diff}")
        
        # Copy CPU parameters to torch model to prevent accumulation
        param_torch.data.copy_(param_cpu.data)

        # Print epoch progress
        print(f'Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}]')

def main():
    # Training settings
    batch_size = 32
    epochs = 5
    lr = 0.01
    betas = (0.9, 0.999)
    eps = 1e-8
    
    # Load digits dataset
    digits = load_digits()
    images = StandardScaler().fit_transform(digits.data).astype(np.float32)
    labels = digits.target.astype(np.int64)
    
    # Create DataLoader
    images = torch.from_numpy(images)
    labels = torch.from_numpy(labels)
    dataset = torch.utils.data.TensorDataset(images, labels)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize two identical models
    model_cpu = SimpleNet()
    model_torch = SimpleNet()
    model_torch.load_state_dict(model_cpu.state_dict())

    print("Using vector width: ", CPUAdam.vector_width())
    print("Total params: ", sum(p.numel() for p in model_cpu.parameters()))
    print("Params:")
    for name, param in model_cpu.named_parameters():
        print(name, param.shape)

    # Create optimizers
    cpu_opt = CPUAdam(model_cpu.parameters(), lr=lr, betas=betas, eps=eps)
    torch_opt = Adam(model_torch.parameters(), lr=lr, betas=betas, eps=eps)
    
    # Train both models
    for epoch in range(1, epochs + 1):
        train_epoch(model_cpu, model_torch, train_loader, cpu_opt, torch_opt, epoch)
        

if __name__ == '__main__':
    main()
