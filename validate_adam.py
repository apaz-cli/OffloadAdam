import torch
import numpy as np

def main():
    # Initialize with same parameters as C version
    param_count = 3
    learning_rate = 0.01
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8

    # Create parameters and gradients as tensors
    params = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    gradients = torch.tensor([0.1, -0.2, 0.3])

    # Print initial parameters
    print("Before optimization:")
    print("PyTorch params:", params.detach().numpy())

    # Create PyTorch Adam optimizer
    optimizer = torch.optim.Adam([params], lr=learning_rate, betas=(beta1, beta2), eps=epsilon)
    
    # Manually set gradients (simulating backward pass)
    params.grad = gradients

    # Perform one optimization step
    optimizer.step()

    # Print results
    print("\nAfter optimization:")
    print("PyTorch params:", params.detach().numpy())
    
    # Print C implementation results for comparison
    print("\nC implementation results (from previous output):")
    print("param[0] = 0.999000")
    print("param[1] = 2.001000")
    print("param[2] = 2.999000")
    
    # Calculate and print differences
    c_params = np.array([0.999000, 2.001000, 2.999000])
    pytorch_params = params.detach().numpy()
    diff = np.abs(c_params - pytorch_params)
    
    print("\nAbsolute differences between PyTorch and C implementations:")
    for i, d in enumerate(diff):
        print(f"param[{i}] diff: {d:.6f}")

if __name__ == "__main__":
    main()
