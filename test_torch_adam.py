import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
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

def main():
    # Training settings
    batch_size = 32
    epochs = 50
    lr = 0.01
    betas = (0.9, 0.999)
    eps = 1e-8
    
    # Load digits dataset
    digits = load_digits()
    data, target = digits.data, digits.target # type: ignore
    images = StandardScaler().fit_transform(data).astype(np.float32)
    labels = target.astype(np.int64)
    
    # Create DataLoader
    images = torch.from_numpy(images)
    labels = torch.from_numpy(labels)
    dataset = torch.utils.data.TensorDataset(images, labels)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = SimpleNet()
    model.train()

    print("Total params:", sum(p.numel() for p in model.parameters()))
    print("Params:", list(name for name, _ in model.named_parameters()))
    print()

    # Create optimizer
    optimizer = Adam(model.parameters(), lr=lr, betas=betas, eps=eps)
    
    # Train model
    for epoch in range(1, epochs + 1):
        if epoch % 10 == 0:
            # Serialize and deserialize optimizer
            torch.save(optimizer.state_dict(), "/tmp/torch_opt.pth")
            optimizer.load_state_dict(torch.load("/tmp/torch_opt.pth"))

        sample_number = 0
        for data, target in train_loader:
            sample_number += len(data)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward/backward pass
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
                
            # Optimizer step
            optimizer.step()
            
            # Print epoch progress
            print(f'\rEpoch: {epoch} [{sample_number}/{len(dataset)}] ', end="")
        print()

if __name__ == '__main__':
    main()
