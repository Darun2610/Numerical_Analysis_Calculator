#!/usr/bin/env python3
from na_methods import euler_method, eval_ode_function

def test_euler_method():
    """Test the Euler method with a simple ODE"""
    print("=== Testing Euler Method ===")
    
    # Test case: dy/dx = x + y, y(0) = 1
    # This should give us a solution that grows exponentially
    function_str = "x + y"
    x0 = 0.0
    y0 = 1.0
    h = 0.1
    n = 10
    
    print(f"ODE: dy/dx = {function_str}")
    print(f"Initial condition: y({x0}) = {y0}")
    print(f"Step size: h = {h}")
    print(f"Number of steps: n = {n}")
    print()
    
    # Test the ODE function evaluator
    print("Testing ODE function evaluator:")
    test_x, test_y = 0.0, 1.0
    result = eval_ode_function(function_str, test_x, test_y)
    print(f"f({test_x}, {test_y}) = {result}")
    
    test_x, test_y = 0.1, 1.1
    result = eval_ode_function(function_str, test_x, test_y)
    print(f"f({test_x}, {test_y}) = {result}")
    print()
    
    # Run Euler method
    print("Running Euler method:")
    result = euler_method(function_str, x0, y0, h, n)
    
    print("Results:")
    print("x_values:", result['x_values'])
    print("y_values:", result['y_values'])
    print()
    
    # Show step-by-step calculation
    print("Step-by-step calculation:")
    for i in range(len(result['x_values'])):
        x = result['x_values'][i]
        y = result['y_values'][i]
        if i < len(result['x_values']) - 1:
            dy_dx = eval_ode_function(function_str, x, y)
            print(f"Step {i}: x = {x:.3f}, y = {y:.6f}, dy/dx = {dy_dx:.6f}")
        else:
            print(f"Step {i}: x = {x:.3f}, y = {y:.6f} (final)")
    
    return result

if __name__ == "__main__":
    test_euler_method() 