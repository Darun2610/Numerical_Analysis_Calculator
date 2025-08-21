from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import json
import base64
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# Ensure matplotlib doesn't use any interactive backend
plt.ioff()
import numpy as np
from na_methods import *

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/method/<method_name>')
def method_page(method_name):
    return render_template('method.html', method_name=method_name)

@app.route('/api/bisection', methods=['POST'])
def bisection_api():
    try:
        data = request.get_json()
        function_str = data['function']
        a = float(data['a'])
        b = float(data['b'])
        tolerance = float(data['tolerance'])
        max_iterations = int(data['max_iterations'])
        
        # Validate tolerance
        if tolerance <= 0:
            return jsonify({'success': False, 'error': 'Tolerance must be a positive number'})
        
        result = bisection_method(function_str, a, b, tolerance, max_iterations)
        
        # Create plot
        plot_data = create_bisection_plot(function_str, a, b, result['iterations'])
        
        return jsonify({
            'success': True,
            'result': result,
            'plot': plot_data
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/newton_raphson', methods=['POST'])
def newton_raphson_api():
    try:
        data = request.get_json()
        function_str = data['function']
        derivative_str = data['derivative']
        x0 = float(data['x0'])
        tolerance = float(data['tolerance'])
        max_iterations = int(data['max_iterations'])
        
        # Validate tolerance
        if tolerance <= 0:
            return jsonify({'success': False, 'error': 'Tolerance must be a positive number'})
        
        result = newton_raphson_method(function_str, derivative_str, x0, tolerance, max_iterations)
        
        # Create plot
        plot_data = create_newton_plot(function_str, derivative_str, x0, result['iterations'])
        
        return jsonify({
            'success': True,
            'result': result,
            'plot': plot_data
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/regula_falsi', methods=['POST'])
def regula_falsi_api():
    try:
        data = request.get_json()
        function_str = data['function']
        a = float(data['a'])
        b = float(data['b'])
        tolerance = float(data['tolerance'])
        max_iterations = int(data['max_iterations'])
        
        result = regula_falsi_method(function_str, a, b, tolerance, max_iterations)
        
        # Create plot
        plot_data = create_regula_falsi_plot(function_str, a, b, result['iterations'])
        
        return jsonify({
            'success': True,
            'result': result,
            'plot': plot_data
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/secant', methods=['POST'])
def secant_api():
    try:
        data = request.get_json()
        function_str = data['function']
        x0 = float(data['x0'])
        x1 = float(data['x1'])
        tolerance = float(data['tolerance'])
        max_iterations = int(data['max_iterations'])
        
        result = secant_method(function_str, x0, x1, tolerance, max_iterations)
        
        # Create plot
        plot_data = create_secant_plot(function_str, x0, x1, result['iterations'])
        
        return jsonify({
            'success': True,
            'result': result,
            'plot': plot_data
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/gauss_jacobi', methods=['POST'])
def gauss_jacobi_api():
    try:
        data = request.get_json()
        matrix = data['matrix']
        b_vector = data['b_vector']
        tolerance = float(data['tolerance'])
        max_iterations = int(data['max_iterations'])
        
        result = gauss_jacobi_method(matrix, b_vector, tolerance, max_iterations)
        
        return jsonify({
            'success': True,
            'result': result
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/gauss_seidel', methods=['POST'])
def gauss_seidel_api():
    try:
        data = request.get_json()
        matrix = data['matrix']
        b_vector = data['b_vector']
        tolerance = float(data['tolerance'])
        max_iterations = int(data['max_iterations'])
        
        result = gauss_seidel_method(matrix, b_vector, tolerance, max_iterations)
        
        return jsonify({
            'success': True,
            'result': result
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/thomas_algorithm', methods=['POST'])
def thomas_algorithm_api():
    try:
        data = request.get_json()
        a = data['a']  # subdiagonal
        b = data['b']  # main diagonal
        c = data['c']  # superdiagonal
        d = data['d']  # right-hand side
        
        result = thomas_algorithm(a, b, c, d)
        
        return jsonify({
            'success': True,
            'result': result
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/newton_interpolation', methods=['POST'])
def newton_interpolation_api():
    try:
        data = request.get_json()
        # Robust parsing to handle values like "[1,2,3]" or "1,2,3" or ["[1","2","3]"]
        if isinstance(data['x_points'], str):
            try:
                import ast
                x_points = ast.literal_eval(data['x_points'])
                y_points = ast.literal_eval(data['y_points'])
            except Exception:
                x_points = [x.strip() for x in data['x_points'].strip('[]').split(',')]
                y_points = [y.strip() for y in data['y_points'].strip('[]').split(',')]
        else:
            x_points = data['x_points']
            y_points = data['y_points']

        # Clean leftover brackets from individual elements
        x_points = [str(x).strip('[]') for x in x_points]
        y_points = [str(y).strip('[]') for y in y_points]

        x_points = [float(x) for x in x_points]
        y_points = [float(y) for y in y_points]
        x_eval = float(data['x_eval'])
        
        result = newton_interpolation(x_points, y_points, x_eval)
        
        # Create plot
        plot_data = create_interpolation_plot(x_points, y_points, x_eval, result['interpolated_value'])
        
        return jsonify({
            'success': True,
            'result': result,
            'plot': plot_data
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/lagrange_interpolation', methods=['POST'])
def lagrange_interpolation_api():
    try:
        data = request.get_json()
        # Robust parsing to handle values like "[1,2,3]" or "1,2,3" or ["[1","2","3]"]
        if isinstance(data['x_points'], str):
            try:
                import ast
                x_points = ast.literal_eval(data['x_points'])
                y_points = ast.literal_eval(data['y_points'])
            except Exception:
                x_points = [x.strip() for x in data['x_points'].strip('[]').split(',')]
                y_points = [y.strip() for y in data['y_points'].strip('[]').split(',')]
        else:
            x_points = data['x_points']
            y_points = data['y_points']

        # Clean leftover brackets from individual elements
        x_points = [str(x).strip('[]') for x in x_points]
        y_points = [str(y).strip('[]') for y in y_points]

        x_points = [float(x) for x in x_points]
        y_points = [float(y) for y in y_points]
        x_eval = float(data['x_eval'])
        
        result = lagrange_interpolation(x_points, y_points, x_eval)
        
        # Create plot
        plot_data = create_interpolation_plot(x_points, y_points, x_eval, result['interpolated_value'])
        
        return jsonify({
            'success': True,
            'result': result,
            'plot': plot_data
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/trapezoidal', methods=['POST'])
def trapezoidal_api():
    try:
        data = request.get_json()
        function_str = data['function']
        a = float(data['a'])
        b = float(data['b'])
        n = int(data['n'])
        
        result = trapezoidal_rule(function_str, a, b, n)
        
        # Create plot
        plot_data = create_integration_plot(function_str, a, b, n, 'trapezoidal')
        
        return jsonify({
            'success': True,
            'result': result,
            'plot': plot_data
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/simpson', methods=['POST'])
def simpson_api():
    try:
        data = request.get_json()
        function_str = data['function']
        a = float(data['a'])
        b = float(data['b'])
        n = int(data['n'])
        
        result = simpson_rule(function_str, a, b, n)
        
        # Create plot
        plot_data = create_integration_plot(function_str, a, b, n, 'simpson')
        
        return jsonify({
            'success': True,
            'result': result,
            'plot': plot_data
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/midpoint', methods=['POST'])
def midpoint_api():
    try:
        data = request.get_json()
        function_str = data['function']
        a = float(data['a'])
        b = float(data['b'])
        n = int(data['n'])
        
        result = midpoint_rule(function_str, a, b, n)
        
        # Create plot
        plot_data = create_integration_plot(function_str, a, b, n, 'midpoint')
        
        return jsonify({
            'success': True,
            'result': result,
            'plot': plot_data
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/trapezoidal_tabular', methods=['POST'])
def trapezoidal_tabular_api():
    try:
        data = request.get_json()
        print(f"Received data: {data}")
        print(f"x_points type: {type(data['x_points'])}, value: {data['x_points']}")
        print(f"y_points type: {type(data['y_points'])}, value: {data['y_points']}")
        
        # Handle case where data might be sent as string representation of list
        if isinstance(data['x_points'], str):
            try:
                import ast
                x_points = ast.literal_eval(data['x_points'])
                y_points = ast.literal_eval(data['y_points'])
            except:
                # Fallback to comma-separated parsing
                x_points = [x.strip() for x in data['x_points'].strip('[]').split(',')]
                y_points = [y.strip() for y in data['y_points'].strip('[]').split(',')]
        else:
            x_points = data['x_points']
            y_points = data['y_points']
            
        # Clean up any remaining bracket characters
        x_points = [x.strip('[]') for x in x_points]
        y_points = [y.strip('[]') for y in y_points]
        
        x_points = [float(x) for x in x_points]
        y_points = [float(y) for y in y_points]
        
        print(f"After conversion - x_points: {x_points}")
        print(f"After conversion - y_points: {y_points}")
        
        result = trapezoidal_rule_tabular(x_points, y_points)
        
        print(f"Trapezoidal result: {result['integral']}")
        
        # Create plot for tabular data
        plot_data = create_tabular_integration_plot(x_points, y_points, 'trapezoidal')
        
        return jsonify({
            'success': True,
            'result': result,
            'plot': plot_data
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/simpson_tabular', methods=['POST'])
def simpson_tabular_api():
    try:
        data = request.get_json()
        print(f"Simpson tabular - Received data: {data}")
        print(f"x_points type: {type(data['x_points'])}, value: {data['x_points']}")
        print(f"y_points type: {type(data['y_points'])}, value: {data['y_points']}")
        
        # Handle case where data might be sent as string representation of list
        if isinstance(data['x_points'], str):
            try:
                import ast
                x_points = ast.literal_eval(data['x_points'])
                y_points = ast.literal_eval(data['y_points'])
            except:
                # Fallback to comma-separated parsing
                x_points = [x.strip() for x in data['x_points'].strip('[]').split(',')]
                y_points = [y.strip() for y in data['y_points'].strip('[]').split(',')]
        else:
            x_points = data['x_points']
            y_points = data['y_points']
            
        # Clean up any remaining bracket characters
        x_points = [x.strip('[]') for x in x_points]
        y_points = [y.strip('[]') for y in y_points]
        
        x_points = [float(x) for x in x_points]
        y_points = [float(y) for y in y_points]
        
        print(f"After conversion - x_points: {x_points}")
        print(f"After conversion - y_points: {y_points}")
        
        result = simpson_rule_tabular(x_points, y_points)
        
        # Create plot for tabular data
        plot_data = create_tabular_integration_plot(x_points, y_points, 'simpson')
        
        return jsonify({
            'success': True,
            'result': result,
            'plot': plot_data
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/midpoint_tabular', methods=['POST'])
def midpoint_tabular_api():
    try:
        data = request.get_json()
        print(f"Midpoint tabular - Received data: {data}")
        print(f"x_points type: {type(data['x_points'])}, value: {data['x_points']}")
        print(f"y_points type: {type(data['y_points'])}, value: {data['y_points']}")
        
        # Handle case where data might be sent as string representation of list
        if isinstance(data['x_points'], str):
            try:
                import ast
                x_points = ast.literal_eval(data['x_points'])
                y_points = ast.literal_eval(data['y_points'])
            except:
                # Fallback to comma-separated parsing
                x_points = [x.strip() for x in data['x_points'].strip('[]').split(',')]
                y_points = [y.strip() for y in data['y_points'].strip('[]').split(',')]
        else:
            x_points = data['x_points']
            y_points = data['y_points']
            
        # Clean up any remaining bracket characters
        x_points = [x.strip('[]') for x in x_points]
        y_points = [y.strip('[]') for y in y_points]
        
        x_points = [float(x) for x in x_points]
        y_points = [float(y) for y in y_points]
        
        print(f"After conversion - x_points: {x_points}")
        print(f"After conversion - y_points: {y_points}")
        
        result = midpoint_rule_tabular(x_points, y_points)
        
        # Create plot for tabular data
        plot_data = create_tabular_integration_plot(x_points, y_points, 'midpoint')
        
        return jsonify({
            'success': True,
            'result': result,
            'plot': plot_data
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/euler', methods=['POST'])
def euler_api():
    try:
        data = request.get_json()
        function_str = data['function']
        x0 = float(data['x0'])
        y0 = float(data['y0'])
        h = float(data['h'])
        n = int(data['n'])
        
        result = euler_method(function_str, x0, y0, h, n)
        
        # Create plot
        plot_data = create_ode_plot(result['x_values'], result['y_values'], 'Euler Method')
        print(f"Euler plot data length: {len(plot_data)}")
        print(f"Euler plot data preview: {plot_data[:100]}...")
        
        return jsonify({
            'success': True,
            'result': result,
            'plot': plot_data
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/modified_euler', methods=['POST'])
def modified_euler_api():
    try:
        data = request.get_json()
        function_str = data['function']
        x0 = float(data['x0'])
        y0 = float(data['y0'])
        h = float(data['h'])
        n = int(data['n'])
        
        result = modified_euler_method(function_str, x0, y0, h, n)
        
        # Create plot
        plot_data = create_ode_plot(result['x_values'], result['y_values'], 'Modified Euler Method')
        
        return jsonify({
            'success': True,
            'result': result,
            'plot': plot_data
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/rk2', methods=['POST'])
def rk2_api():
    try:
        data = request.get_json()
        function_str = data['function']
        x0 = float(data['x0'])
        y0 = float(data['y0'])
        h = float(data['h'])
        n = int(data['n'])
        
        result = rk2_method(function_str, x0, y0, h, n)
        
        # Create plot
        plot_data = create_ode_plot(result['x_values'], result['y_values'], 'RK2 Method')
        
        return jsonify({
            'success': True,
            'result': result,
            'plot': plot_data
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

def create_bisection_plot(function_str, a, b, iterations):
    """Create plot for bisection method"""
    # Clear any existing plots and create a new figure
    plt.clf()
    plt.close('all')
    fig = plt.figure(figsize=(10, 6))
    
    x = np.linspace(a-1, b+1, 1000)
    y = [eval_function(function_str, xi) for xi in x]
    
    plt.plot(x, y, 'b-', label='f(x)')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=a, color='r', linestyle='--', alpha=0.7, label=f'a={a}')
    plt.axvline(x=b, color='g', linestyle='--', alpha=0.7, label=f'b={b}')
    
    # Plot iterations
    for i, iteration in enumerate(iterations):
        c = iteration['c']
        plt.plot(c, 0, 'ro', markersize=8, alpha=0.7)
        plt.annotate(f'c{i+1}={c:.4f}', (c, 0), xytext=(10, 10), 
                    textcoords='offset points', fontsize=8)
    
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title('Bisection Method')
    
    # Save to base64
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=100, bbox_inches='tight')
    img.seek(0)
    plt.close(fig)
    return base64.b64encode(img.getvalue()).decode()

def create_newton_plot(function_str, derivative_str, x0, iterations):
    """Create plot for Newton-Raphson method"""
    # Clear any existing plots and create a new figure
    plt.clf()
    plt.close('all')
    fig = plt.figure(figsize=(10, 6))
    
    x = np.linspace(x0-2, x0+2, 1000)
    y = [eval_function(function_str, xi) for xi in x]
    
    plt.plot(x, y, 'b-', label='f(x)')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    # Plot iterations
    for i, iteration in enumerate(iterations):
        x_i = iteration['x']
        f_x = iteration['f_x']
        f_prime_x = iteration['f_prime_x']
        
        plt.plot(x_i, f_x, 'ro', markersize=8, alpha=0.7)
        plt.annotate(f'x{i+1}={x_i:.4f}', (x_i, f_x), xytext=(10, 10), 
                    textcoords='offset points', fontsize=8)
        
        # Plot tangent line
        if i < len(iterations) - 1:
            x_next = iterations[i+1]['x']
            x_tangent = np.linspace(x_i-0.5, x_next+0.5, 100)
            y_tangent = f_x + f_prime_x * (x_tangent - x_i)
            plt.plot(x_tangent, y_tangent, 'g--', alpha=0.5)
    
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title('Newton-Raphson Method')
    
    # Save to base64
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=100, bbox_inches='tight')
    img.seek(0)
    plt.close(fig)
    return base64.b64encode(img.getvalue()).decode()

def create_regula_falsi_plot(function_str, a, b, iterations):
    """Create plot for Regula Falsi method"""
    # Clear any existing plots and create a new figure
    plt.clf()
    plt.close('all')
    fig = plt.figure(figsize=(10, 6))
    
    x = np.linspace(a-1, b+1, 1000)
    y = [eval_function(function_str, xi) for xi in x]
    
    plt.plot(x, y, 'b-', label='f(x)')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    # Plot iterations
    for i, iteration in enumerate(iterations):
        x_i = iteration['c']
        f_x = iteration['f_c']
        plt.plot(x_i, f_x, 'ro', markersize=8, alpha=0.7)
        plt.annotate(f'x{i+1}={x_i:.4f}', (x_i, f_x), xytext=(10, 10), 
                    textcoords='offset points', fontsize=8)
    
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title('Regula Falsi Method')
    
    # Save to base64
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=100, bbox_inches='tight')
    img.seek(0)
    plt.close(fig)
    return base64.b64encode(img.getvalue()).decode()

def create_secant_plot(function_str, x0, x1, iterations):
    """Create plot for Secant method"""
    # Clear any existing plots and create a new figure
    plt.clf()
    plt.close('all')
    fig = plt.figure(figsize=(10, 6))
    
    x = np.linspace(min(x0, x1)-1, max(x0, x1)+1, 1000)
    y = [eval_function(function_str, xi) for xi in x]
    
    plt.plot(x, y, 'b-', label='f(x)')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    # Plot iterations
    for i, iteration in enumerate(iterations):
        x_i = iteration['x_curr']
        f_x = iteration['f_curr']
        plt.plot(x_i, f_x, 'ro', markersize=8, alpha=0.7)
        plt.annotate(f'x{i+1}={x_i:.4f}', (x_i, f_x), xytext=(10, 10), 
                    textcoords='offset points', fontsize=8)
    
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title('Secant Method')
    
    # Save to base64
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=100, bbox_inches='tight')
    img.seek(0)
    plt.close(fig)
    return base64.b64encode(img.getvalue()).decode()

def create_interpolation_plot(x_points, y_points, x_eval, interpolated_value):
    """Create plot for interpolation methods"""
    # Clear any existing plots and create a new figure
    plt.clf()
    plt.close('all')
    fig = plt.figure(figsize=(10, 6))
    
    # Plot original points
    plt.plot(x_points, y_points, 'bo', markersize=8, label='Data Points')
    
    # Plot interpolated point
    plt.plot(x_eval, interpolated_value, 'ro', markersize=10, label='Interpolated Point')
    
    # Create smooth curve for visualization
    x_smooth = np.linspace(min(x_points), max(x_points), 1000)
    y_smooth = []
    for x in x_smooth:
        y_smooth.append(newton_interpolation(x_points, y_points, x)['interpolated_value'])
    
    plt.plot(x_smooth, y_smooth, 'g-', alpha=0.7, label='Interpolation Curve')
    
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title('Interpolation')
    
    # Save to base64
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=100, bbox_inches='tight')
    img.seek(0)
    plt.close(fig)
    return base64.b64encode(img.getvalue()).decode()

def create_integration_plot(function_str, a, b, n, method):
    """Create plot for integration methods"""
    # Clear any existing plots and create a new figure
    plt.clf()
    plt.close('all')
    fig = plt.figure(figsize=(10, 6))
    
    x = np.linspace(a, b, 1000)
    y = [eval_function(function_str, xi) for xi in x]
    
    plt.plot(x, y, 'b-', label='f(x)')
    plt.fill_between(x, y, alpha=0.3, color='blue')
    
    # Plot integration points
    if method == 'trapezoidal':
        x_points = np.linspace(a, b, n+1)
        y_points = [eval_function(function_str, xi) for xi in x_points]
        plt.plot(x_points, y_points, 'ro', markersize=6)
        
        # Draw trapezoids
        for i in range(n):
            plt.plot([x_points[i], x_points[i+1]], [y_points[i], y_points[i+1]], 'r-', alpha=0.7)
    
    elif method == 'simpson':
        x_points = np.linspace(a, b, n+1)
        y_points = [eval_function(function_str, xi) for xi in x_points]
        plt.plot(x_points, y_points, 'go', markersize=6)
        
        # Draw Simpson's rule approximation
        for i in range(0, n, 2):
            if i+2 <= n:
                x_quad = np.linspace(x_points[i], x_points[i+2], 100)
                # Quadratic interpolation
                y_quad = []
                for x in x_quad:
                    # Simple quadratic interpolation
                    h = x_points[i+1] - x_points[i]
                    y_quad.append(y_points[i] + (y_points[i+1] - y_points[i]) * (x - x_points[i]) / h)
                plt.plot(x_quad, y_quad, 'g-', alpha=0.7)
    
    elif method == 'midpoint':
        x_points = np.linspace(a, b, n)
        y_points = [eval_function(function_str, (x_points[i] + x_points[i+1])/2) for i in range(n-1)]
        x_mid = [(x_points[i] + x_points[i+1])/2 for i in range(n-1)]
        plt.plot(x_mid, y_points, 'mo', markersize=6)
    
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title(f'{method.title()} Rule Integration')
    
    # Save to base64
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=100, bbox_inches='tight')
    img.seek(0)
    plt.close(fig)
    return base64.b64encode(img.getvalue()).decode()

def create_ode_plot(x_values, y_values, method_name):
    """Create plot for ODE methods"""
    # Clear any existing plots and create a new figure
    plt.clf()
    plt.close('all')
    fig = plt.figure(figsize=(10, 6))
    
    plt.plot(x_values, y_values, 'b-o', markersize=4, label=f'{method_name} Solution')
    plt.grid(True, alpha=0.3)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title(f'{method_name}')
    
    # Save to base64
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=100, bbox_inches='tight')
    img.seek(0)
    plt.close(fig)
    return base64.b64encode(img.getvalue()).decode()

def create_tabular_integration_plot(x_points, y_points, method):
    """Create plot for tabular integration methods"""
    # Clear any existing plots and create a new figure
    plt.clf()
    plt.close('all')
    fig = plt.figure(figsize=(10, 6))
    
    # Plot the data points
    plt.plot(x_points, y_points, 'bo', markersize=8, label='Data Points')
    
    # Connect points with lines
    plt.plot(x_points, y_points, 'b-', alpha=0.7)
    
    # Fill area under the curve
    plt.fill_between(x_points, y_points, alpha=0.3, color='blue')
    
    # Add method-specific visualization
    if method == 'trapezoidal':
        # Draw trapezoids
        for i in range(len(x_points) - 1):
            plt.plot([x_points[i], x_points[i+1]], [y_points[i], y_points[i+1]], 'r-', alpha=0.7, linewidth=2)
        plt.title('Trapezoidal Rule Integration (Tabular Data)')
        
    elif method == 'simpson':
        # Draw Simpson's rule approximation
        if len(x_points) >= 3:
            for i in range(0, len(x_points) - 2, 2):
                if i + 2 < len(x_points):
                    x_quad = np.linspace(x_points[i], x_points[i+2], 100)
                    # Simple quadratic interpolation
                    y_quad = []
                    for x in x_quad:
                        # Linear interpolation between points
                        if x <= x_points[i+1]:
                            t = (x - x_points[i]) / (x_points[i+1] - x_points[i])
                            y_quad.append(y_points[i] * (1-t) + y_points[i+1] * t)
                        else:
                            t = (x - x_points[i+1]) / (x_points[i+2] - x_points[i+1])
                            y_quad.append(y_points[i+1] * (1-t) + y_points[i+2] * t)
                    plt.plot(x_quad, y_quad, 'g-', alpha=0.7, linewidth=2)
        plt.title('Simpson\'s Rule Integration (Tabular Data)')
        
    elif method == 'midpoint':
        # Draw midpoint rectangles
        for i in range(len(x_points) - 1):
            x_mid = (x_points[i] + x_points[i+1]) / 2
            y_mid = (y_points[i] + y_points[i+1]) / 2
            width = x_points[i+1] - x_points[i]
            rect = plt.Rectangle((x_points[i], 0), width, y_mid, alpha=0.3, color='m')
            plt.gca().add_patch(rect)
        plt.title('Midpoint Rule Integration (Tabular Data)')
    
    plt.grid(True, alpha=0.3)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    
    # Save to base64
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=100, bbox_inches='tight')
    img.seek(0)
    plt.close(fig)
    return base64.b64encode(img.getvalue()).decode()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 