# Numerical Methods Calculator

A comprehensive full-stack web application for exploring various numerical methods with step-by-step solutions, error analysis, and visualizations. Built with Python Flask backend and modern HTML/CSS/JS frontend.

## Features

### Root Finding Methods
- **Bisection Method**: Interval halving with guaranteed convergence
- **Newton-Raphson Method**: Fast convergence using derivatives
- **Regula Falsi Method**: False position combining bisection and secant
- **Secant Method**: Newton-like method without derivatives

### Linear Equations Methods
- **Gauss-Jacobi Method**: Iterative method for linear systems
- **Gauss-Seidel Method**: Improved iterative method
- **Thomas Algorithm**: Direct method for tridiagonal systems

### Interpolation Methods
- **Newton Interpolation**: Divided difference method
- **Lagrange Interpolation**: Classical polynomial interpolation

### Integration Methods
- **Trapezoidal Rule**: Approximate integrals using trapezoids
- **Simpson's Rule**: Higher accuracy with quadratic approximations
- **Midpoint Rule**: Simple integration using midpoints

### Differential Equations Methods
- **Euler Method**: Basic first-order ODE solver
- **Modified Euler Method**: Improved predictor-corrector approach
- **RK2 Method**: Second-order Runge-Kutta method

## Key Features

- **Step-by-step iterations**: Detailed tables showing each iteration
- **Error analysis**: Absolute and relative errors per iteration
- **Visualizations**: Interactive plots using matplotlib
- **Modern UI**: Beautiful, responsive design with Bootstrap
- **Educational focus**: Perfect for students and teachers
- **Lovable ready**: Easy to customize with Lovable

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd numerical-analysis-calculator
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   python app.py
   ```

4. **Open your browser** and navigate to:
   ```
   http://localhost:5000
   ```

## Usage

### Getting Started

1. **Choose a method** from the homepage
2. **Enter parameters** in the form (examples provided)
3. **Click "Calculate"** to run the method
4. **View results** including:
   - Final solution
   - Step-by-step iteration table
   - Error analysis
   - Visualization plot

### Example: Bisection Method

1. Navigate to `/method/bisection`
2. Enter parameters:
   - Function: `x**2 - 4`
   - Lower bound: `0`
   - Upper bound: `3`
   - Tolerance: `0.0001`
   - Max iterations: `100`
3. Click "Calculate"
4. View the root (approximately 2.0) with iteration details

### Example: Newton-Raphson Method

1. Navigate to `/method/newton_raphson`
2. Enter parameters:
   - Function: `x**2 - 4`
   - Derivative: `2*x`
   - Initial guess: `2`
   - Tolerance: `0.0001`
   - Max iterations: `100`
3. Click "Calculate"
4. View the root with tangent line visualization

## API Endpoints

All methods are available via REST API endpoints:

- `POST /api/bisection` - Bisection method
- `POST /api/newton_raphson` - Newton-Raphson method
- `POST /api/regula_falsi` - Regula Falsi method
- `POST /api/secant` - Secant method
- `POST /api/gauss_jacobi` - Gauss-Jacobi method
- `POST /api/gauss_seidel` - Gauss-Seidel method
- `POST /api/thomas_algorithm` - Thomas algorithm
- `POST /api/newton_interpolation` - Newton interpolation
- `POST /api/lagrange_interpolation` - Lagrange interpolation
- `POST /api/trapezoidal` - Trapezoidal rule
- `POST /api/simpson` - Simpson's rule
- `POST /api/midpoint` - Midpoint rule
- `POST /api/euler` - Euler method
- `POST /api/modified_euler` - Modified Euler method
- `POST /api/rk2` - RK2 method

## File Structure

```
numerical-analysis-calculator/
├── app.py                 # Main Flask application
├── na_methods.py         # Numerical methods implementations
├── requirements.txt      # Python dependencies
├── README.md            # This file
└── templates/
    ├── index.html       # Homepage with method selection
    └── method.html      # Dynamic method page template
```

## Mathematical Functions

The application supports various mathematical functions in input:

- **Basic operations**: `+`, `-`, `*`, `/`, `**`
- **Trigonometric**: `sin`, `cos`, `tan`
- **Exponential/Logarithmic**: `exp`, `log`
- **Constants**: `pi`, `e`
- **Square root**: `sqrt`

Examples:
- `x**2 - 4` (quadratic function)
- `sin(x)` (sine function)
- `exp(-x**2)` (Gaussian function)
- `x**3 + 2*x - 5` (cubic function)

## Error Analysis

Each method provides comprehensive error analysis:

- **Absolute Error**: `|x_new - x_old|`
- **Relative Error**: `|x_new - x_old| / |x_new|`
- **Convergence Status**: Whether the method converged
- **Iteration Count**: Number of iterations performed

## Visualization

The application generates plots for:

- **Root finding**: Function plot with iteration points
- **Integration**: Function plot with integration regions
- **ODE methods**: Solution trajectory plots
- **Interpolation**: Data points with interpolated curves

## Customization with Lovable

The application is designed to be easily customizable with Lovable:

1. **Frontend**: Modern HTML/CSS/JS structure
2. **Modular design**: Separate method implementations
3. **RESTful API**: Clean API endpoints
4. **Responsive design**: Works on all devices
5. **Educational focus**: Clear explanations and examples

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add new numerical methods to `na_methods.py`
4. Add corresponding API endpoints to `app.py`
5. Update the frontend templates
6. Submit a pull request

## License

This project is open source and available under the MIT License.

## Support

For questions or issues, please open an issue on the GitHub repository.

---

**Built for educational purposes** - Perfect for numerical analysis courses and self-study! 