import torch
def test_gpu():
    if torch.cuda.is_available():
        print("CUDA is available. Testing GPU...")
        device = torch.device("cuda")
        x = torch.rand(1000, 1000, device=device)
        y = torch.rand(1000, 1000, device=device)
        z = x + y
        print(f"GPU test successful: {z.device}")
    else:
        print("CUDA is not available. Running on CPU.")
        device = torch.device("cpu")
    return device

if __name__ == "__main__":
    device = test_gpu()
    print(f"Using device: {device}")
    # You can add more tests or functionality here if needed