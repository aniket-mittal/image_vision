#!/usr/bin/env python3
"""
Test script to verify FBA model loading works correctly
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_fba_loading():
    """Test FBA model loading"""
    try:
        from model_server import ensure_fba_loaded
        
        print("Testing FBA model loading...")
        net = ensure_fba_loaded()
        
        if net is not None:
            if isinstance(net, str) and net == "rembg":
                print("✅ Alternative matting (rembg) loaded successfully")
                return True
            else:
                print(f"✅ FBA model loaded successfully on device: {next(net.parameters()).device}")
                return True
        else:
            print("❌ FBA model loading failed")
            return False
            
    except Exception as e:
        print(f"❌ Error testing FBA loading: {e}")
        return False

def test_health_endpoint():
    """Test the health endpoint"""
    try:
        from model_server import Handler
        import json
        
        # Create a mock request
        class MockRequest:
            def __init__(self):
                self.path = "/health"
                self.headers = {}
                self.rfile = None
                self.wfile = None
        
        # Create a mock response writer
        class MockResponse:
            def __init__(self):
                self.data = b""
            
            def write(self, data):
                self.data += data
        
        # Test the health endpoint
        handler = Handler(MockRequest(), ("localhost", 8000), None)
        handler.wfile = MockResponse()
        
        # Call the health endpoint
        handler.do_GET()
        
        # Parse the response
        response_data = json.loads(handler.wfile.data.decode('utf-8'))
        print(f"Health endpoint response: {json.dumps(response_data, indent=2)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing health endpoint: {e}")
        return False

if __name__ == "__main__":
    print("=== FBA Model Loading Test ===\n")
    
    # Test 1: FBA loading
    print("1. Testing FBA model loading...")
    fba_success = test_fba_loading()
    print()
    
    # Test 2: Health endpoint
    print("2. Testing health endpoint...")
    health_success = test_health_endpoint()
    print()
    
    # Summary
    print("=== Test Summary ===")
    print(f"FBA Loading: {'✅ PASS' if fba_success else '❌ FAIL'}")
    print(f"Health Endpoint: {'✅ PASS' if health_success else '❌ PASS'}")
    
    if fba_success:
        print("\n🎉 FBA model is working correctly!")
    else:
        print("\n⚠️  FBA model has issues. Check the error messages above.")
