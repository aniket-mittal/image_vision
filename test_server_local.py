#!/usr/bin/env python3
"""
Test script to verify model_server.py functionality locally
"""

import requests
import time
import json

def test_server_health():
    """Test the health endpoint"""
    try:
        response = requests.get("http://localhost:8765/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print("✅ Health check passed")
            print(f"   Status: {data.get('status')}")
            print(f"   Models: {json.dumps(data.get('models', {}), indent=6)}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Server not running on localhost:8765")
        return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_server_startup():
    """Test if the server module can be imported and key components loaded"""
    try:
        import model_server
        print("✅ Server module imported successfully")
        
        # Check if key components are available
        if hasattr(model_server, 'CLIP_GEN') and model_server.CLIP_GEN is not None:
            print("✅ CLIP model loaded")
        else:
            print("⚠️  CLIP model not loaded")
            
        if hasattr(model_server, 'DETECTOR') and model_server.DETECTOR is not None:
            print("✅ GroundedSAM detector loaded")
        else:
            print("⚠️  GroundedSAM detector not loaded")
            
        if hasattr(model_server, 'SAM_AMG') and model_server.SAM_AMG is not None:
            print("✅ SAM AutomaticMaskGenerator imported")
        else:
            print("⚠️  SAM AutomaticMaskGenerator not imported")
            
        return True
    except Exception as e:
        print(f"❌ Server module import failed: {e}")
        return False

def test_sdxl_lazy_loading():
    """Test SDXL lazy loading functionality"""
    try:
        import model_server
        
        # Check if SDXL is in the environment
        if hasattr(model_server, '_DIFFUSERS_ENV'):
            sdxl_status = model_server._DIFFUSERS_ENV.get("pipe_sdxl")
            if sdxl_status is None:
                print("✅ SDXL lazy loading ready (not preloaded)")
            else:
                print("✅ SDXL already loaded")
            return True
        else:
            print("❌ _DIFFUSERS_ENV not found")
            return False
    except Exception as e:
        print(f"❌ SDXL lazy loading test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Model Server Functionality")
    print("=" * 50)
    
    # Test 1: Module import and component loading
    print("\n1. Testing module import...")
    test_server_startup()
    
    # Test 2: SDXL lazy loading
    print("\n2. Testing SDXL lazy loading...")
    test_sdxl_lazy_loading()
    
    # Test 3: Server health (if running)
    print("\n3. Testing server health...")
    test_server_health()
    
    print("\n" + "=" * 50)
    print("✅ All tests completed!")
    print("\nTo start the server locally, run:")
    print("   python model_server.py")
    print("\nOr test with your Next.js app by setting:")
    print("   MODEL_SERVER_URL=http://localhost:8765")
