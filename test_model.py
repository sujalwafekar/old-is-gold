from backend.model.model_loader import load_model, predict
from PIL import Image
import torch
import numpy as np
import time

try:
    print("Testing model load...")
    start = time.time()
    model = load_model()
    print(f"Model loaded successfully in {time.time() - start:.2f}s!")
    
    # Create a random dummy image
    img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    
    device = next(model.parameters()).device
    
    print("Testing prediction...")
    res = predict(img, model, device)
    print("Prediction result:", res)
    print("ALL TESTS PASSED!")
except Exception as e:
    print(f"ERROR: {e}")
