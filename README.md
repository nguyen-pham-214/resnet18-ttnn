This project implements the ResNet18 model using TTNN and validates it against a PyTorch reference implementation.
The goal is to ensure correctness by comparing outputs between the two implementations.

```bash
resnet18-ttnn/
│
├── reference/                # PyTorch reference implementation
│   ├── outputs/              # Contains exported model weights
│   │   └── resnet18_weights.pth
│   ├── export-weights.py     # Script to export weights from PyTorch model
│   ├── resnet18_torch.py     # Defines the ResNet18 model in PyTorch
│   └── test.py               # Runs PyTorch model with fake input tensor for testing
│
├── test/
│   └── test.py               # Runs both PyTorch and TTNN models and compares outputs (PCC)
│
├── ttnn/                     # TTNN implementation
│   ├── utils/                # Building blocks (layers)
│   │   ├── Head/             
│   │   ├── InputStem/        
│   │   └── Layer/            
│   └── resnet18_ttnn.py      # Assembles all layers into full ResNet18 model
```

Run the comparison test between PyTorch and TTNN:
`python test/test.py`