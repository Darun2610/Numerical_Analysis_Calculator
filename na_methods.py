import numpy as np
import math
from typing import List, Dict, Any

def eval_function(function_str: str, x: float) -> float:
    """Safely evaluate a mathematical function string"""
    import math
    
    # Create a safe namespace with mathematical functions
    safe_dict = {
        'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
        'asin': math.asin, 'acos': math.acos, 'atan': math.atan,
        'sinh': math.sinh, 'cosh': math.cosh, 'tanh': math.tanh,
        'exp': math.exp, 'log': math.log, 'log10': math.log10,
        'sqrt': math.sqrt, 'pow': math.pow,
        'pi': math.pi, 'e': math.e,
        'abs': abs, 'round': round, 'floor': math.floor, 'ceil': math.ceil,
        'x': x  # The variable x
    }
    
    # Clean the function string
    function_str = function_str.strip()
    
    # Handle common syntax variations
    function_str = function_str.replace('^', '**')  # Handle ^ as power
    function_str = function_str.replace('**2', '**2')  # Ensure proper power syntax
    
    try:
        # Evaluate the function
        result = eval(function_str, {"__builtins__": {}}, safe_dict)
        return float(result)
    except NameError as e:
        # Provide helpful error message for undefined variables
        raise ValueError(f"Invalid function: {function_str}. Error: {str(e)}. Make sure to use 'x' as the variable.")
    except SyntaxError as e:
        # Provide helpful error message for syntax errors
        raise ValueError(f"Invalid function syntax: {function_str}. Error: {str(e)}. Check your mathematical expression.")
    except ZeroDivisionError:
        raise ValueError(f"Division by zero in function: {function_str}")
    except Exception as e:
        # General error handling
        raise ValueError(f"Invalid function: {function_str}. Error: {str(e)}")

def eval_ode_function(function_str: str, x: float, y: float) -> float:
    """Safely evaluate an ODE function string with variables x and y"""
    import math
    
    # Create a safe namespace with mathematical functions
    safe_dict = {
        'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
        'asin': math.asin, 'acos': math.acos, 'atan': math.atan,
        'sinh': math.sinh, 'cosh': math.cosh, 'tanh': math.tanh,
        'exp': math.exp, 'log': math.log, 'log10': math.log10,
        'sqrt': math.sqrt, 'pow': math.pow,
        'pi': math.pi, 'e': math.e,
        'abs': abs, 'round': round, 'floor': math.floor, 'ceil': math.ceil,
        'x': x, 'y': y  # The variables x and y
    }
    
    # Clean the function string
    function_str = function_str.strip()
    
    # Handle common syntax variations
    function_str = function_str.replace('^', '**')  # Handle ^ as power
    function_str = function_str.replace('**2', '**2')  # Ensure proper power syntax
    
    try:
        # Evaluate the function
        result = eval(function_str, {"__builtins__": {}}, safe_dict)
        return float(result)
    except NameError as e:
        # Provide helpful error message for undefined variables
        raise ValueError(f"Invalid ODE function: {function_str}. Error: {str(e)}. Make sure to use 'x' and 'y' as the variables.")
    except SyntaxError as e:
        # Provide helpful error message for syntax errors
        raise ValueError(f"Invalid ODE function syntax: {function_str}. Error: {str(e)}. Check your mathematical expression.")
    except ZeroDivisionError:
        raise ValueError(f"Division by zero in ODE function: {function_str}")
    except Exception as e:
        # General error handling
        raise ValueError(f"Invalid ODE function: {function_str}. Error: {str(e)}")

def eval_derivative(derivative_str: str, x: float) -> float:
    """Safely evaluate a derivative function string"""
    return eval_function(derivative_str, x)

# ==================== ROOT FINDING METHODS ====================

def bisection_method(function_str: str, a: float, b: float, tolerance: float, max_iterations: int) -> Dict[str, Any]:
    """Bisection method for finding roots"""
    iterations = []
    fa = eval_function(function_str, a)
    fb = eval_function(function_str, b)
    
    if fa * fb > 0:
        raise ValueError("f(a) and f(b) must have opposite signs")
    
    c = None
    fc = None
    
    for i in range(max_iterations):
        c = (a + b) / 2
        fc = eval_function(function_str, c)
        
        # Calculate errors
        absolute_error = abs(b - a) / 2
        relative_error = absolute_error / abs(c) if c != 0 else float('inf')
        
        iteration_data = {
            'iteration': i + 1,
            'a': a,
            'b': b,
            'c': c,
            'f_a': fa,
            'f_b': fb,
            'f_c': fc,
            'absolute_error': absolute_error,
            'relative_error': relative_error
        }
        iterations.append(iteration_data)
        
        # Check convergence: either function value is close to zero or interval is small enough
        if abs(fc) < tolerance or absolute_error < tolerance:
            break
            
        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc
    
    # Ensure we have the final values
    if c is None:
        c = (a + b) / 2
        fc = eval_function(function_str, c)
    
    return {
        'root': c,
        'iterations': iterations,
        'final_iteration': len(iterations),
        'converged': abs(fc) < tolerance or absolute_error < tolerance
    }

def newton_raphson_method(function_str: str, derivative_str: str, x0: float, tolerance: float, max_iterations: int) -> Dict[str, Any]:
    """Newton-Raphson method for finding roots"""
    iterations = []
    x = x0
    
    for i in range(max_iterations):
        f_x = eval_function(function_str, x)
        f_prime_x = eval_derivative(derivative_str, x)
        
        if abs(f_prime_x) < 1e-10:
            raise ValueError("Derivative is zero - method fails")
        
        x_new = x - f_x / f_prime_x
        
        # Calculate errors
        absolute_error = abs(x_new - x)
        relative_error = absolute_error / abs(x_new) if x_new != 0 else float('inf')
        
        iteration_data = {
            'iteration': i + 1,
            'x': x,
            'f_x': f_x,
            'f_prime_x': f_prime_x,
            'x_new': x_new,
            'absolute_error': absolute_error,
            'relative_error': relative_error
        }
        iterations.append(iteration_data)
        
        # Check convergence: either function value is close to zero or step size is small enough
        if abs(f_x) < tolerance or absolute_error < tolerance:
            break
            
        x = x_new
    
    # Calculate final function value for convergence check
    final_f_x = eval_function(function_str, x)
    
    return {
        'root': x,
        'iterations': iterations,
        'final_iteration': len(iterations),
        'converged': abs(final_f_x) < tolerance or (len(iterations) > 0 and iterations[-1]['absolute_error'] < tolerance)
    }

def regula_falsi_method(function_str: str, a: float, b: float, tolerance: float, max_iterations: int) -> Dict[str, Any]:
    """Regula Falsi (False Position) method for finding roots"""
    iterations = []
    fa = eval_function(function_str, a)
    fb = eval_function(function_str, b)
    
    if fa * fb > 0:
        raise ValueError("f(a) and f(b) must have opposite signs")
    
    for i in range(max_iterations):
        # False position formula
        c = b - fb * (b - a) / (fb - fa)
        fc = eval_function(function_str, c)
        
        # Calculate errors
        absolute_error = abs(c - b)
        relative_error = absolute_error / abs(c) if c != 0 else float('inf')
        
        iteration_data = {
            'iteration': i + 1,
            'a': a,
            'b': b,
            'c': c,
            'f_a': fa,
            'f_b': fb,
            'f_c': fc,
            'absolute_error': absolute_error,
            'relative_error': relative_error
        }
        iterations.append(iteration_data)
        
        if abs(fc) < tolerance or absolute_error < tolerance:
            break
            
        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc
    
    return {
        'root': c,
        'iterations': iterations,
        'final_iteration': len(iterations),
        'converged': abs(fc) < tolerance or absolute_error < tolerance
    }

def secant_method(function_str: str, x0: float, x1: float, tolerance: float, max_iterations: int) -> Dict[str, Any]:
    """Secant method for finding roots"""
    iterations = []
    x_prev = x0
    x_curr = x1
    
    for i in range(max_iterations):
        f_prev = eval_function(function_str, x_prev)
        f_curr = eval_function(function_str, x_curr)
        
        if abs(f_curr - f_prev) < 1e-10:
            raise ValueError("Function values are too close - method fails")
        
        x_new = x_curr - f_curr * (x_curr - x_prev) / (f_curr - f_prev)
        
        # Calculate errors
        absolute_error = abs(x_new - x_curr)
        relative_error = absolute_error / abs(x_new) if x_new != 0 else float('inf')
        
        iteration_data = {
            'iteration': i + 1,
            'x_prev': x_prev,
            'x_curr': x_curr,
            'f_prev': f_prev,
            'f_curr': f_curr,
            'x_new': x_new,
            'absolute_error': absolute_error,
            'relative_error': relative_error
        }
        iterations.append(iteration_data)
        
        if absolute_error < tolerance:
            break
            
        x_prev = x_curr
        x_curr = x_new
    
    return {
        'root': x_curr,
        'iterations': iterations,
        'final_iteration': len(iterations),
        'converged': absolute_error < tolerance
    }

# ==================== LINEAR EQUATIONS METHODS ====================

def gauss_jacobi_method(matrix: List[List[float]], b_vector: List[float], tolerance: float, max_iterations: int) -> Dict[str, Any]:
    """Gauss-Jacobi iterative method for solving linear equations"""
    n = len(matrix)
    x = [0.0] * n  # Initial guess
    iterations = []
    
    for k in range(max_iterations):
        x_new = [0.0] * n
        max_error = 0.0
        residuals = []
        
        for i in range(n):
            sum_val = 0.0
            for j in range(n):
                if i != j:
                    sum_val += matrix[i][j] * x[j]
            
            x_new[i] = (b_vector[i] - sum_val) / matrix[i][i]
            max_error = max(max_error, abs(x_new[i] - x[i]))
            
            # Calculate residual for this equation
            residual = b_vector[i] - sum(matrix[i][j] * x_new[j] for j in range(n))
            residuals.append(residual)
        
        iteration_data = {
            'iteration': k + 1,
            'x_old': x.copy(),
            'x_new': x_new.copy(),
            'max_error': max_error,
            'residuals': residuals,
            'max_residual': max(abs(r) for r in residuals)
        }
        iterations.append(iteration_data)
        
        if max_error < tolerance:
            break
            
        x = x_new
    
    return {
        'solution': x_new,
        'iterations': iterations,
        'final_iteration': len(iterations),
        'converged': max_error < tolerance,
        'method': 'Gauss-Jacobi Method',
        'matrix_size': n,
        'tolerance': tolerance
    }

def gauss_seidel_method(matrix: List[List[float]], b_vector: List[float], tolerance: float, max_iterations: int) -> Dict[str, Any]:
    """Gauss-Seidel iterative method for solving linear equations"""
    n = len(matrix)
    x = [0.0] * n  # Initial guess
    iterations = []
    
    for k in range(max_iterations):
        x_old = x.copy()
        max_error = 0.0
        residuals = []
        
        for i in range(n):
            sum_val = 0.0
            for j in range(n):
                if i != j:
                    sum_val += matrix[i][j] * x[j]
            
            x[i] = (b_vector[i] - sum_val) / matrix[i][i]
            max_error = max(max_error, abs(x[i] - x_old[i]))
            
            # Calculate residual for this equation
            residual = b_vector[i] - sum(matrix[i][j] * x[j] for j in range(n))
            residuals.append(residual)
        
        iteration_data = {
            'iteration': k + 1,
            'x': x.copy(),
            'max_error': max_error,
            'residuals': residuals,
            'max_residual': max(abs(r) for r in residuals)
        }
        iterations.append(iteration_data)
        
        if max_error < tolerance:
            break
    
    return {
        'solution': x,
        'iterations': iterations,
        'final_iteration': len(iterations),
        'converged': max_error < tolerance,
        'method': 'Gauss-Seidel Method',
        'matrix_size': n,
        'tolerance': tolerance
    }

def thomas_algorithm(a: List[float], b: List[float], c: List[float], d: List[float]) -> Dict[str, Any]:
    """Thomas algorithm for solving tridiagonal systems"""
    n = len(d)
    
    # Forward elimination
    c_prime = [0.0] * n
    d_prime = [0.0] * n
    forward_steps = []
    
    c_prime[0] = c[0] / b[0]
    d_prime[0] = d[0] / b[0]
    forward_steps.append({
        'step': 'Forward Elimination - Step 1',
        'i': 0,
        'c_prime[0]': f"c[0]/b[0] = {c[0]}/{b[0]} = {c_prime[0]:.6f}",
        'd_prime[0]': f"d[0]/b[0] = {d[0]}/{b[0]} = {d_prime[0]:.6f}"
    })
    
    for i in range(1, n):
        denominator = b[i] - a[i] * c_prime[i-1]
        c_prime[i] = c[i] / denominator
        d_prime[i] = (d[i] - a[i] * d_prime[i-1]) / denominator
        
        forward_steps.append({
            'step': f'Forward Elimination - Step {i+1}',
            'i': i,
            'denominator': f"b[{i}] - a[{i}] * c_prime[{i-1}] = {b[i]} - {a[i]} * {c_prime[i-1]:.6f} = {denominator:.6f}",
            'c_prime[i]': f"c[{i}]/denominator = {c[i]}/{denominator:.6f} = {c_prime[i]:.6f}",
            'd_prime[i]': f"(d[{i}] - a[{i}] * d_prime[{i-1}])/denominator = ({d[i]} - {a[i]} * {d_prime[i-1]:.6f})/{denominator:.6f} = {d_prime[i]:.6f}"
        })
    
    # Backward substitution
    x = [0.0] * n
    x[n-1] = d_prime[n-1]
    backward_steps = []
    
    backward_steps.append({
        'step': f'Backward Substitution - Step 1',
        'i': n-1,
        'x[n-1]': f"d_prime[n-1] = {d_prime[n-1]:.6f}"
    })
    
    for i in range(n-2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i+1]
        backward_steps.append({
            'step': f'Backward Substitution - Step {n-i}',
            'i': i,
            'x[i]': f"d_prime[{i}] - c_prime[{i}] * x[{i+1}] = {d_prime[i]:.6f} - {c_prime[i]:.6f} * {x[i+1]:.6f} = {x[i]:.6f}"
        })
    
    return {
        'solution': x,
        'method': 'Thomas Algorithm (Direct)',
        'matrix_size': n,
        'forward_steps': forward_steps,
        'backward_steps': backward_steps,
        'c_prime': c_prime,
        'd_prime': d_prime
    }

# ==================== INTERPOLATION METHODS ====================

def newton_interpolation(x_points: List[float], y_points: List[float], x_eval: float) -> Dict[str, Any]:
    """Newton's divided difference interpolation"""
    n = len(x_points)
    
    # Create divided difference table
    dd_table = [[0.0] * n for _ in range(n)]
    
    # First column is y values
    for i in range(n):
        dd_table[i][0] = y_points[i]
    
    # Calculate divided differences with steps
    steps = []
    steps.append("Step 1: Initialize the divided difference table")
    steps.append(f"   First column (f[x_i]): {[round(y, 6) for y in y_points]}")
    
    for j in range(1, n):
        step_desc = f"Step {j+1}: Calculate {j}st divided differences"
        step_calc = []
        for i in range(n - j):
            numerator = dd_table[i+1][j-1] - dd_table[i][j-1]
            denominator = x_points[i+j] - x_points[i]
            dd_table[i][j] = numerator / denominator
            step_calc.append(f"   f[x_{i},x_{i+j}] = (f[x_{i+1},...,x_{i+j-1}] - f[x_{i},...,x_{i+j-1}]) / (x_{i+j} - x_{i})")
            step_calc.append(f"   f[x_{i},x_{i+j}] = ({dd_table[i+1][j-1]:.6f} - {dd_table[i][j-1]:.6f}) / ({x_points[i+j]:.6f} - {x_points[i]:.6f}) = {dd_table[i][j]:.6f}")
        steps.append(step_desc)
        steps.extend(step_calc)
    
    # Evaluate polynomial with steps
    steps.append("Step 3: Evaluate the Newton polynomial")
    result = dd_table[0][0]
    term = 1.0
    steps.append(f"   P(x) = {result:.6f}")
    
    for i in range(1, n):
        term *= (x_eval - x_points[i-1])
        result += dd_table[0][i] * term
        term_str = " * ".join([f"(x - {x_points[k]:.6f})" for k in range(i)])
        steps.append(f"   + {dd_table[0][i]:.6f} * {term_str}")
        steps.append(f"   = {result:.6f}")
    
    steps.append(f"Final result: P({x_eval:.6f}) = {result:.6f}")
    
    return {
        'interpolated_value': result,
        'divided_differences': dd_table,
        'steps': steps,
        'method': 'Newton Divided Difference'
    }

def lagrange_interpolation(x_points: List[float], y_points: List[float], x_eval: float) -> Dict[str, Any]:
    """Lagrange interpolation"""
    n = len(x_points)
    result = 0.0
    steps = []
    
    steps.append("Step 1: Lagrange interpolation formula")
    steps.append("   P(x) = Σ(y_i * L_i(x)) where L_i(x) = Π((x - x_j)/(x_i - x_j)) for j≠i")
    steps.append(f"   Data points: x = {[round(x, 6) for x in x_points]}, y = {[round(y, 6) for y in y_points]}")
    steps.append(f"   Evaluate at x = {x_eval:.6f}")
    
    for i in range(n):
        steps.append(f"Step {i+2}: Calculate L_{i}(x) and term {i+1}")
        steps.append(f"   L_{i}(x) = y_{i} * Π((x - x_j)/(x_{i} - x_j)) for j≠{i}")
        
        term = y_points[i]
        numerator_factors = []
        denominator_factors = []
        
        for j in range(n):
            if i != j:
                num_factor = x_eval - x_points[j]
                den_factor = x_points[i] - x_points[j]
                term *= num_factor / den_factor
                numerator_factors.append(f"(x - x_{j}) = ({x_eval:.6f} - {x_points[j]:.6f}) = {num_factor:.6f}")
                denominator_factors.append(f"(x_{i} - x_{j}) = ({x_points[i]:.6f} - {x_points[j]:.6f}) = {den_factor:.6f}")
        
        steps.append(f"   Numerator factors: {' * '.join(numerator_factors)}")
        steps.append(f"   Denominator factors: {' * '.join(denominator_factors)}")
        steps.append(f"   L_{i}(x) = {y_points[i]:.6f} * ({' * '.join([f'({x_eval:.6f} - {x_points[j]:.6f})/({x_points[i]:.6f} - {x_points[j]:.6f})' for j in range(n) if j != i])})")
        steps.append(f"   L_{i}(x) = {term:.6f}")
        steps.append(f"   Term {i+1} = {term:.6f}")
        
        result += term
    
    steps.append("Step 3: Sum all terms")
    steps.append(f"   P(x) = {' + '.join([f'{y_points[i]:.6f} * L_{i}(x)' for i in range(n)])}")
    steps.append(f"   P({x_eval:.6f}) = {result:.6f}")
    
    return {
        'interpolated_value': result,
        'steps': steps,
        'method': 'Lagrange Interpolation'
    }

# ==================== INTEGRATION METHODS ====================

def trapezoidal_rule(function_str: str, a: float, b: float, n: int) -> Dict[str, Any]:
    """Trapezoidal rule for numerical integration"""
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = [eval_function(function_str, xi) for xi in x]
    
    steps = []
    steps.append("Step 1: Calculate step size and generate points")
    steps.append(f"   h = (b - a) / n = ({b:.6f} - {a:.6f}) / {n} = {h:.6f}")
    steps.append(f"   x points: {[round(xi, 6) for xi in x]}")
    steps.append(f"   y values: {[round(yi, 6) for yi in y]}")
    
    steps.append("Step 2: Apply Trapezoidal Rule formula")
    steps.append("   ∫f(x)dx ≈ h/2 * [f(x₀) + 2f(x₁) + 2f(x₂) + ... + 2f(xₙ₋₁) + f(xₙ)]")
    steps.append(f"   ∫f(x)dx ≈ {h:.6f}/2 * [f({x[0]:.6f}) + 2f({x[1]:.6f}) + ... + 2f({x[-2]:.6f}) + f({x[-1]:.6f})]")
    
    # Calculate step by step
    integral = (y[0] + y[-1]) / 2
    steps.append(f"   First term: (f(x₀) + f(xₙ))/2 = ({y[0]:.6f} + {y[-1]:.6f})/2 = {integral:.6f}")
    
    middle_terms = []
    for i in range(1, n):
        integral += y[i]
        middle_terms.append(f"f({x[i]:.6f}) = {y[i]:.6f}")
    
    steps.append(f"   Middle terms: {' + '.join(middle_terms)}")
    steps.append(f"   Sum of middle terms: {sum(y[1:-1]):.6f}")
    
    integral *= h
    steps.append(f"   Final result: {integral/h:.6f} * {h:.6f} = {integral:.6f}")
    
    return {
        'integral': integral,
        'n_intervals': n,
        'h': h,
        'x_points': x.tolist(),
        'y_points': y,
        'method': 'Trapezoidal Rule',
        'steps': steps
    }

def simpson_rule(function_str: str, a: float, b: float, n: int) -> Dict[str, Any]:
    """Simpson's rule for numerical integration"""
    if n % 2 != 0:
        n += 1  # Simpson's rule requires even number of intervals
    
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = [eval_function(function_str, xi) for xi in x]
    
    steps = []
    steps.append("Step 1: Calculate step size and generate points")
    steps.append(f"   h = (b - a) / n = ({b:.6f} - {a:.6f}) / {n} = {h:.6f}")
    steps.append(f"   x points: {[round(xi, 6) for xi in x]}")
    steps.append(f"   y values: {[round(yi, 6) for yi in y]}")
    
    steps.append("Step 2: Apply Simpson's Rule formula")
    steps.append("   ∫f(x)dx ≈ h/3 * [f(x₀) + 4f(x₁) + 2f(x₂) + 4f(x₃) + 2f(x₄) + ... + 4f(xₙ₋₁) + f(xₙ)]")
    steps.append(f"   ∫f(x)dx ≈ {h:.6f}/3 * [f({x[0]:.6f}) + 4f({x[1]:.6f}) + 2f({x[2]:.6f}) + ... + 4f({x[-2]:.6f}) + f({x[-1]:.6f})]")
    
    # Simpson's formula with steps
    integral = y[0] + y[-1]
    steps.append(f"   First and last terms: f(x₀) + f(xₙ) = {y[0]:.6f} + {y[-1]:.6f} = {integral:.6f}")
    
    odd_terms = []
    even_terms = []
    
    for i in range(1, n, 2):
        integral += 4 * y[i]
        odd_terms.append(f"4f({x[i]:.6f}) = 4({y[i]:.6f}) = {4*y[i]:.6f}")
    
    for i in range(2, n, 2):
        integral += 2 * y[i]
        even_terms.append(f"2f({x[i]:.6f}) = 2({y[i]:.6f}) = {2*y[i]:.6f}")
    
    steps.append(f"   Odd-indexed terms (multiplied by 4): {' + '.join(odd_terms)}")
    steps.append(f"   Even-indexed terms (multiplied by 2): {' + '.join(even_terms)}")
    steps.append(f"   Sum of all terms: {integral:.6f}")
    
    integral *= h / 3
    steps.append(f"   Final result: {integral/(h/3):.6f} * {h:.6f}/3 = {integral:.6f}")
    
    return {
        'integral': integral,
        'n_intervals': n,
        'h': h,
        'x_points': x.tolist(),
        'y_points': y,
        'method': 'Simpson\'s Rule',
        'steps': steps
    }

def midpoint_rule(function_str: str, a: float, b: float, n: int) -> Dict[str, Any]:
    """Midpoint rule for numerical integration"""
    h = (b - a) / n
    x_midpoints = [a + h/2 + i*h for i in range(n)]
    y_midpoints = [eval_function(function_str, xi) for xi in x_midpoints]
    
    steps = []
    steps.append("Step 1: Calculate step size and generate midpoint points")
    steps.append(f"   h = (b - a) / n = ({b:.6f} - {a:.6f}) / {n} = {h:.6f}")
    steps.append(f"   Midpoint x values: {[round(xi, 6) for xi in x_midpoints]}")
    steps.append(f"   Midpoint y values: {[round(yi, 6) for yi in y_midpoints]}")
    
    steps.append("Step 2: Apply Midpoint Rule formula")
    steps.append("   ∫f(x)dx ≈ h * [f(x₁/₂) + f(x₃/₂) + f(x₅/₂) + ... + f(xₙ₋₁/₂)]")
    steps.append(f"   ∫f(x)dx ≈ {h:.6f} * [f({x_midpoints[0]:.6f}) + f({x_midpoints[1]:.6f}) + ... + f({x_midpoints[-1]:.6f})]")
    
    # Calculate step by step
    integral = h * sum(y_midpoints)
    steps.append(f"   Sum of function values at midpoints: {sum(y_midpoints):.6f}")
    steps.append(f"   Final result: {h:.6f} * {sum(y_midpoints):.6f} = {integral:.6f}")
    
    return {
        'integral': integral,
        'n_intervals': n,
        'h': h,
        'x_midpoints': x_midpoints,
        'y_midpoints': y_midpoints,
        'method': 'Midpoint Rule',
        'steps': steps
    }

# ==================== TABULAR INTEGRATION METHODS ====================

def trapezoidal_rule_tabular(x_points: List[float], y_points: List[float]) -> Dict[str, Any]:
    """Trapezoidal rule for numerical integration using tabular data"""
    if len(x_points) != len(y_points):
        raise ValueError("Number of x and y points must be equal")
    
    n = len(x_points) - 1
    steps = []
    steps.append("Step 1: Tabular data integration using Trapezoidal Rule")
    steps.append(f"   x points: {[round(xi, 6) for xi in x_points]}")
    steps.append(f"   y points: {[round(yi, 6) for yi in y_points]}")
    steps.append(f"   Number of intervals: {n}")
    
    steps.append("Step 2: Apply Trapezoidal Rule formula")
    steps.append("   ∫f(x)dx ≈ h/2 * [f(x₀) + 2f(x₁) + 2f(x₂) + ... + 2f(xₙ₋₁) + f(xₙ)]")
    
    # Calculate uniform step size
    h = (x_points[-1] - x_points[0]) / n
    steps.append(f"   Step size: h = {h:.6f}")
    
    # Calculate integral using standard trapezoidal rule formula
    integral = y_points[0] + y_points[-1]  # First and last terms
    
    middle_terms = []
    for i in range(1, n):
        integral += 2 * y_points[i]
        middle_terms.append(f"2f({x_points[i]:.6f}) = 2({y_points[i]:.6f}) = {2*y_points[i]:.6f}")
    
    steps.append(f"   First and last terms: f(x₀) + f(xₙ) = {y_points[0]:.6f} + {y_points[-1]:.6f} = {y_points[0] + y_points[-1]:.6f}")
    steps.append(f"   Middle terms (multiplied by 2): {' + '.join(middle_terms)}")
    steps.append(f"   Sum of all terms: {integral:.6f}")
    
    integral *= h / 2
    steps.append(f"   Final result: {integral/(h/2):.6f} * {h:.6f}/2 = {integral:.6f}")
    
    return {
        'integral': integral,
        'n_intervals': n,
        'h': h,
        'x_points': x_points,
        'y_points': y_points,
        'method': 'Trapezoidal Rule (Tabular)',
        'steps': steps
    }

def simpson_rule_tabular(x_points: List[float], y_points: List[float]) -> Dict[str, Any]:
    """Simpson's rule for numerical integration using tabular data"""
    if len(x_points) != len(y_points):
        raise ValueError("Number of x and y points must be equal")
    
    n = len(x_points) - 1
    if n % 2 != 0:
        raise ValueError("Simpson's rule requires even number of intervals")
    
    steps = []
    steps.append("Step 1: Tabular data integration using Simpson's Rule")
    steps.append(f"   x points: {[round(xi, 6) for xi in x_points]}")
    steps.append(f"   y points: {[round(yi, 6) for yi in y_points]}")
    steps.append(f"   Number of intervals: {n}")
    
    steps.append("Step 2: Apply Simpson's Rule formula")
    steps.append("   ∫f(x)dx ≈ h/3 * [f(x₀) + 4f(x₁) + 2f(x₂) + 4f(x₃) + 2f(x₄) + ... + 4f(xₙ₋₁) + f(xₙ)]")
    
    # Calculate step size (assuming uniform step size for Simpson's rule)
    h = (x_points[-1] - x_points[0]) / n
    steps.append(f"   Step size: h = {h:.6f}")
    
    # Calculate integral using Simpson's rule
    integral = y_points[0] + y_points[-1]  # First and last terms
    
    odd_terms = []
    even_terms = []
    
    for i in range(1, n, 2):
        integral += 4 * y_points[i]
        odd_terms.append(f"4f({x_points[i]:.6f}) = 4({y_points[i]:.6f}) = {4*y_points[i]:.6f}")
    
    for i in range(2, n, 2):
        integral += 2 * y_points[i]
        even_terms.append(f"2f({x_points[i]:.6f}) = 2({y_points[i]:.6f}) = {2*y_points[i]:.6f}")
    
    steps.append(f"   First and last terms: f(x₀) + f(xₙ) = {y_points[0]:.6f} + {y_points[-1]:.6f} = {y_points[0] + y_points[-1]:.6f}")
    steps.append(f"   Odd-indexed terms (multiplied by 4): {' + '.join(odd_terms)}")
    steps.append(f"   Even-indexed terms (multiplied by 2): {' + '.join(even_terms)}")
    steps.append(f"   Sum of all terms: {integral:.6f}")
    
    integral *= h / 3
    steps.append(f"   Final result: {integral/(h/3):.6f} * {h:.6f}/3 = {integral:.6f}")
    
    return {
        'integral': integral,
        'n_intervals': n,
        'h': h,
        'x_points': x_points,
        'y_points': y_points,
        'method': 'Simpson\'s Rule (Tabular)',
        'steps': steps
    }

def midpoint_rule_tabular(x_points: List[float], y_points: List[float]) -> Dict[str, Any]:
    """Midpoint rule for numerical integration using tabular data"""
    if len(x_points) != len(y_points):
        raise ValueError("Number of x and y points must be equal")
    
    n = len(x_points) - 1
    steps = []
    steps.append("Step 1: Tabular data integration using Midpoint Rule")
    steps.append(f"   x points: {[round(xi, 6) for xi in x_points]}")
    steps.append(f"   y points: {[round(yi, 6) for yi in y_points]}")
    steps.append(f"   Number of intervals: {n}")
    
    steps.append("Step 2: Apply Midpoint Rule formula")
    steps.append("   ∫f(x)dx ≈ h * [f(x₁/₂) + f(x₃/₂) + f(x₅/₂) + ... + f(xₙ₋₁/₂)]")
    
    # Calculate midpoints and their values
    x_midpoints = [(x_points[i] + x_points[i+1]) / 2 for i in range(n)]
    y_midpoints = [(y_points[i] + y_points[i+1]) / 2 for i in range(n)]  # Average of adjacent points
    
    steps.append(f"   Midpoint x values: {[round(xi, 6) for xi in x_midpoints]}")
    steps.append(f"   Midpoint y values (averaged): {[round(yi, 6) for yi in y_midpoints]}")
    
    # Calculate step sizes
    h_values = [x_points[i+1] - x_points[i] for i in range(n)]
    steps.append(f"   Step sizes: {[round(hi, 6) for hi in h_values]}")
    
    # Calculate integral
    integral = 0
    for i in range(n):
        h = h_values[i]
        y_mid = y_midpoints[i]
        area = h * y_mid
        integral += area
        steps.append(f"   Interval {i+1}: h = {h:.6f}, f(x_{i+0.5}) = {y_mid:.6f}")
        steps.append(f"   Area = {h:.6f} * {y_mid:.6f} = {area:.6f}")
    
    steps.append(f"   Total integral: {integral:.6f}")
    
    return {
        'integral': integral,
        'n_intervals': n,
        'x_midpoints': x_midpoints,
        'y_midpoints': y_midpoints,
        'h_values': h_values,
        'method': 'Midpoint Rule (Tabular)',
        'steps': steps
    }

# ==================== ODE METHODS ====================

def euler_method(function_str: str, x0: float, y0: float, h: float, n: int) -> Dict[str, Any]:
    """Euler's method for solving ODEs"""
    x_values = [x0]
    y_values = [y0]
    
    for i in range(n):
        x = x_values[-1]
        y = y_values[-1]
        
        # dy/dx = f(x, y)
        dy_dx = eval_ode_function(function_str, x, y)
        
        x_new = x + h
        y_new = y + h * dy_dx
        
        x_values.append(x_new)
        y_values.append(y_new)
    
    return {
        'x_values': x_values,
        'y_values': y_values,
        'h': h,
        'n_steps': n,
        'final_iteration': n,
        'converged': True,
        'method': 'Euler Method'
    }

def modified_euler_method(function_str: str, x0: float, y0: float, h: float, n: int) -> Dict[str, Any]:
    """Modified Euler method for solving ODEs"""
    x_values = [x0]
    y_values = [y0]
    
    for i in range(n):
        x = x_values[-1]
        y = y_values[-1]
        
        # Predictor step
        dy_dx = eval_ode_function(function_str, x, y)
        y_pred = y + h * dy_dx
        
        # Corrector step
        dy_dx_corr = eval_ode_function(function_str, x + h, y_pred)
        y_new = y + h * (dy_dx + dy_dx_corr) / 2
        
        x_new = x + h
        
        x_values.append(x_new)
        y_values.append(y_new)
    
    return {
        'x_values': x_values,
        'y_values': y_values,
        'h': h,
        'n_steps': n,
        'final_iteration': n,
        'converged': True,
        'method': 'Modified Euler Method'
    }

def rk2_method(function_str: str, x0: float, y0: float, h: float, n: int) -> Dict[str, Any]:
    """RK2 (Runge-Kutta 2nd order) method for solving ODEs"""
    x_values = [x0]
    y_values = [y0]
    
    for i in range(n):
        x = x_values[-1]
        y = y_values[-1]
        
        # RK2 steps
        k1 = eval_ode_function(function_str, x, y)
        k2 = eval_ode_function(function_str, x + h, y + h * k1)
        
        y_new = y + h * (k1 + k2) / 2
        x_new = x + h
        
        x_values.append(x_new)
        y_values.append(y_new)
    
    return {
        'x_values': x_values,
        'y_values': y_values,
        'h': h,
        'n_steps': n,
        'final_iteration': n,
        'converged': True,
        'method': 'RK2 Method'
    } 