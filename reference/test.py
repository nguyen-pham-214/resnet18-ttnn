import torch
from resnet18 import create_torch_model

def main():
    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Create model
    model = create_torch_model(device)

    # Create 4 fake input: batch_size=4, 3 channels, height x width 32x32
    x = torch.randn(4, 3, 32, 32).to(device)

    # Disable gradient computation to only test the forward pass
    with torch.no_grad():
        output = model(x)

    print("Input shape :", x.shape)
    print("Output shape:", output.shape)
    print("Output:")
    print(output)

if __name__ == "__main__":
    main()