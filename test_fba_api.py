#!/usr/bin/env python3
"""
Test script to verify FBA model API works correctly
"""

import sys
import os
import numpy as np
import torch

def test_fba_api():
    """Test the FBA model API"""
    try:
        # Add the FBA repository to path
        fba_repo = os.path.join(os.path.dirname(__file__), "third_party", "FBA_Matting")
        if not os.path.exists(fba_repo):
            print(f"❌ FBA repository not found at: {fba_repo}")
            return False
        
        sys.path.insert(0, fba_repo)
        
        # Test 1: Import FBA model
        print("1. Testing FBA model import...")
        try:
            from networks.models import build_model
            print("✅ FBA model import successful")
        except ImportError as e:
            print(f"❌ FBA model import failed: {e}")
            return False
        
        # Test 2: Import FBA transforms
        print("2. Testing FBA transforms import...")
        try:
            from networks.transforms import trimap_transform, normalise_image
            print("✅ FBA transforms import successful")
        except ImportError as e:
            print(f"❌ FBA transforms import failed: {e}")
            return False
        
        # Test 3: Load FBA model
        print("3. Testing FBA model loading...")
        try:
            weights_path = os.path.join(fba_repo, "FBA.pth")
            if not os.path.exists(weights_path):
                print(f"❌ FBA weights not found at: {weights_path}")
                return False
            
            model = build_model(weights_path)
            model.eval()
            print("✅ FBA model loaded successfully")
        except Exception as e:
            print(f"❌ FBA model loading failed: {e}")
            return False
        
        # Test 4: Test FBA forward pass
        print("4. Testing FBA forward pass...")
        try:
            # Create dummy inputs
            batch_size = 1
            channels = 3
            height = 64  # Multiple of 8
            width = 64   # Multiple of 8
            
            # Create dummy image and trimap
            image = torch.randn(batch_size, channels, height, width)
            trimap = torch.randn(batch_size, 2, height, width)  # 2-channel trimap
            
            # Apply transforms
            image_transformed = normalise_image(image.clone())
            trimap_transformed = torch.from_numpy(trimap_transform(trimap.squeeze(0).permute(1, 2, 0).numpy())).unsqueeze(0)
            
            # Move to device
            device = next(model.parameters()).device
            image = image.to(device)
            trimap = trimap.to(device)
            image_transformed = image_transformed.to(device)
            trimap_transformed = trimap_transformed.to(device)
            
            # Test forward pass
            with torch.no_grad():
                output = model(image, trimap, image_transformed, trimap_transformed)
                print(f"✅ FBA forward pass successful, output shape: {output.shape}")
                
        except Exception as e:
            print(f"❌ FBA forward pass failed: {e}")
            return False
        
        print("\n🎉 All FBA tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

if __name__ == "__main__":
    print("=== FBA API Test ===\n")
    
    success = test_fba_api()
    
    if success:
        print("\n✅ FBA is working correctly!")
    else:
        print("\n❌ FBA has issues. Check the error messages above.")
