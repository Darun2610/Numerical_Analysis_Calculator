#!/usr/bin/env python3
"""
Test script for the Numerical Methods Calculator
"""

import requests
import json

def test_bisection_method():
    """Test the bisection method API"""
    url = "http://localhost:5000/api/bisection"
    data = {
        "function": "x**2 - 4",
        "a": 0,
        "b": 3,
        "tolerance": 0.0001,
        "max_iterations": 100
    }
    
    try:
        response = requests.post(url, json=data)
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                print("✅ Bisection method test PASSED")
                print(f"   Root: {result['result']['root']:.6f}")
                print(f"   Iterations: {result['result']['final_iteration']}")
                print(f"   Converged: {result['result']['converged']}")
                return True
            else:
                print("❌ Bisection method test FAILED")
                print(f"   Error: {result.get('error', 'Unknown error')}")
                return False
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False

def test_newton_raphson_method():
    """Test the Newton-Raphson method API"""
    url = "http://localhost:5000/api/newton_raphson"
    data = {
        "function": "x**2 - 4",
        "derivative": "2*x",
        "x0": 2,
        "tolerance": 0.0001,
        "max_iterations": 100
    }
    
    try:
        response = requests.post(url, json=data)
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                print("✅ Newton-Raphson method test PASSED")
                print(f"   Root: {result['result']['root']:.6f}")
                print(f"   Iterations: {result['result']['final_iteration']}")
                print(f"   Converged: {result['result']['converged']}")
                return True
            else:
                print("❌ Newton-Raphson method test FAILED")
                print(f"   Error: {result.get('error', 'Unknown error')}")
                return False
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False

def test_homepage():
    """Test the homepage"""
    try:
        response = requests.get("http://localhost:5000")
        if response.status_code == 200:
            print("✅ Homepage test PASSED")
            return True
        else:
            print(f"❌ Homepage test FAILED: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Homepage connection error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Numerical Methods Calculator")
    print("=" * 50)
    
    tests = [
        ("Homepage", test_homepage),
        ("Bisection Method", test_bisection_method),
        ("Newton-Raphson Method", test_newton_raphson_method),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Testing {test_name}...")
        if test_func():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The application is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the application.")
    
    return passed == total

if __name__ == "__main__":
    main() 