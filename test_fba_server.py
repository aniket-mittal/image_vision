#!/usr/bin/env python3
"""
Test script to verify FBA functionality on the H100 server
"""

import requests
import json
import sys

def test_server_health(server_url):
    """Test the server health endpoint"""
    try:
        print(f"Testing server health at: {server_url}/health")
        response = requests.get(f"{server_url}/health", timeout=10)
        
        if response.status_code == 200:
            health_data = response.json()
            print("✅ Health check passed")
            print(f"Models status: {json.dumps(health_data['models'], indent=2)}")
            return health_data
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return None

def test_fba_functionality(server_url):
    """Test the FBA test endpoint"""
    try:
        print(f"\nTesting FBA functionality at: {server_url}/test_fba")
        response = requests.get(f"{server_url}/test_fba", timeout=30)
        
        if response.status_code == 200:
            fba_data = response.json()
            print("✅ FBA test passed")
            print(f"FBA status: {json.dumps(fba_data, indent=2)}")
            return fba_data
        else:
            print(f"❌ FBA test failed: {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ FBA test error: {e}")
        return None

def main():
    # Get server URL from command line or use default
    if len(sys.argv) > 1:
        server_url = sys.argv[1].rstrip('/')
    else:
        server_url = "http://localhost:8000"
    
    print("=== FBA Server Test ===\n")
    print(f"Server URL: {server_url}\n")
    
    # Test 1: Server health
    health_data = test_server_health(server_url)
    if not health_data:
        print("\n❌ Cannot connect to server. Make sure it's running.")
        return
    
    # Test 2: FBA functionality
    fba_data = test_fba_functionality(server_url)
    
    # Summary
    print("\n=== Test Summary ===")
    print(f"Server Health: ✅ PASS")
    print(f"FBA Functionality: {'✅ PASS' if fba_data else '❌ FAIL'}")
    
    if fba_data and fba_data.get('status') == 'fba_loaded':
        print("\n🎉 FBA model is working correctly on the server!")
    elif fba_data and fba_data.get('status') == 'alternative_matting':
        print(f"\n⚠️  Using alternative matting method: {fba_data.get('method')}")
        print("FBA weights may not be loaded correctly.")
    else:
        print("\n⚠️  FBA model has issues. Check the server logs for details.")

if __name__ == "__main__":
    main()
