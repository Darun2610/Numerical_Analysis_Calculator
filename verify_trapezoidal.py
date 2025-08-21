#!/usr/bin/env python3

from na_methods import trapezoidal_rule_tabular

def test_user_data():
    """Test function for user to input their own data"""
    print("=== Trapezoidal Rule Verification ===")
    print("Enter your x and y points to verify the calculation")
    print()
    
    # Get x points
    x_input = input("Enter x points (comma-separated, e.g., 0,1,2,3,4): ").strip()
    x_points = [float(x.strip()) for x in x_input.split(',')]
    
    # Get y points
    y_input = input("Enter y points (comma-separated, e.g., 0,2,4,6,8): ").strip()
    y_points = [float(y.strip()) for y in y_input.split(',')]
    
    print(f"\nInput data:")
    print(f"x_points: {x_points}")
    print(f"y_points: {y_points}")
    print()
    
    # Calculate result
    result = trapezoidal_rule_tabular(x_points, y_points)
    
    print(f"Result: {result['integral']}")
    print()
    print("Step-by-step calculation:")
    for step in result['steps']:
        print(f"  {step}")

if __name__ == "__main__":
    test_user_data() 