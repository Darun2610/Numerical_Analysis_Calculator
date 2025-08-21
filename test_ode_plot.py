#!/usr/bin/env python3
from na_methods import euler_method
from app import create_ode_plot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
import io

def test_ode_plot():
    """Test ODE plot generation"""
    print("=== Testing ODE Plot Generation ===")
    
    # Test case: dy/dx = x + y, y(0) = 1
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
    
    # Run Euler method
    result = euler_method(function_str, x0, y0, h, n)
    
    print("Results:")
    print("x_values:", result['x_values'])
    print("y_values:", result['y_values'])
    print()
    
    # Test plot generation
    print("Testing plot generation...")
    try:
        plot_data = create_ode_plot(result['x_values'], result['y_values'], 'Euler Method')
        print("✓ Plot generated successfully!")
        print(f"Plot data length: {len(plot_data)} characters")
        
        # Try to decode and verify it's valid base64
        decoded_data = base64.b64decode(plot_data)
        print(f"✓ Decoded plot data length: {len(decoded_data)} bytes")
        
        # Try to create a simple plot manually to verify matplotlib is working
        plt.figure(figsize=(8, 6))
        plt.plot(result['x_values'], result['y_values'], 'b-o', markersize=4, label='Euler Solution')
        plt.grid(True, alpha=0.3)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.title('Euler Method Test Plot')
        
        # Save to test file
        plt.savefig('test_ode_plot.png', dpi=100, bbox_inches='tight')
        plt.close()
        print("✓ Test plot saved as 'test_ode_plot.png'")
        
    except Exception as e:
        print(f"✗ Error generating plot: {e}")
        import traceback
        traceback.print_exc()
    
    return result

if __name__ == "__main__":
    test_ode_plot() 