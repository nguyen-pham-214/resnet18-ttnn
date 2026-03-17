import torch
from resnet18 import create_torch_model

def main():
    device = torch.device("cpu")

    model = create_torch_model(device)

    # get weights
    state_dict = model.state_dict()

    # save to file
    torch.save(state_dict, "resnet18_weights.pth")

    print("Saved weights to resnet18_weights.pth")

    # print some keys
    print("\nFirst 10 weight keys:")
    for i, key in enumerate(state_dict.keys()):
        print(key)
        if i > 10:
            break

if __name__ == "__main__":
    main()