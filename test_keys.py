import torch
from backend.model.model_loader import build_model, CLASSES
import traceback

with open("out_keys.txt", "w") as f:
    try:
        path = r'e:\projects\major\skin cancer\backend\model\skin_cancer_densenet_v2_final.pth'
        f.write(f"Loading weights from: {path}\n")
        checkpoint = torch.load(path, map_location='cpu')
        
        if 'model_state' in checkpoint:
            state_dict = checkpoint['model_state']
        else:
            state_dict = checkpoint
            
        model = build_model(len(CLASSES))
        expected_keys = set(model.state_dict().keys())
        saved_keys = set(state_dict.keys())

        missing = list(expected_keys - saved_keys)
        unexpected = list(saved_keys - expected_keys)
        
        f.write(f"Missing keys (in model not in file): {len(missing)}\n")
        f.write(f"Examples: {missing[:10]}\n\n")
        
        f.write(f"Unexpected keys (in file not in model): {len(unexpected)}\n")
        f.write(f"Examples: {unexpected[:10]}\n\n")

        # Create a mapping wrapper
        # Let's inspect differences if prefixes are just off
        if unexpected:
            f.write(f"First unexpected key: {unexpected[0]}\n")
        if missing:
            f.write(f"First missing key: {missing[0]}\n")
            
        f.write("DONE_INSPECTING\n")
    except Exception as e:
        f.write("ERROR:\n" + traceback.format_exc())
