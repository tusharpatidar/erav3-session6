import torch
from network import Network
from train import prepare_data
from train import DEVICE, BATCH_SIZE

def test_parameter_count():
    model = Network()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel Parameter Count Test:")
    print(f"Total parameters in model: {total_params:,}")
    print(f"Parameter limit: 20,000")
    assert total_params < 20000, f"Model has {total_params:,} parameters, should be < 20,000"
    
    # Test model accuracy
    model = model.to(DEVICE)
    try:
        # Load the trained model weights
        model.load_state_dict(torch.load('models/best_model.pth'))
        print("Loaded trained model weights successfully")
    except:
        print("Warning: Could not load trained model weights. Testing with untrained model.")
    
    _, test_loader = prepare_data(BATCH_SIZE)
    
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    accuracy = 100. * correct / total
    print(f"Current model accuracy: {accuracy:.2f}%")
    print(f"Required accuracy: 99.4%")
    assert accuracy >= 99.4, f"Model accuracy {accuracy:.2f}% is below required 99.4%"

def test_model_components():
    model = Network()
    print("\nModel Components Test:")
    
    # Test for Batch Normalization
    has_bn = any(isinstance(m, torch.nn.BatchNorm2d) for m in model.modules())
    assert has_bn, "Model should use Batch Normalization"
    
    # Test for Dropout
    has_dropout = any(isinstance(m, torch.nn.Dropout) for m in model.modules())
    assert has_dropout, "Model should use Dropout"
    
    # Test for GAP or FC
    has_gap = any(isinstance(m, torch.nn.AdaptiveAvgPool2d) for m in model.modules())
    has_fc = any(isinstance(m, torch.nn.Linear) for m in model.modules())
    print(f"Has Global Average Pooling: {has_gap}")
    print(f"Has Fully Connected layer: {has_fc}")
    assert has_gap or has_fc, "Model should use either GAP or FC layer"

    # Print model architecture summary
    print("\nModel Architecture:")
    for name, module in model.named_children():
        print(f"{name}: {module}")