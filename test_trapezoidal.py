#!/usr/bin/env python3

from na_methods import trapezoidal_rule_tabular

# Test case 1: Simple linear function y = 2x from x=0 to x=4
# x_points = [0, 1, 2, 3, 4]
# y_points = [0, 2, 4, 6, 8]
# Expected integral: 16 (area under y=2x from 0 to 4)

print("Test Case 1: Linear function y = 2x from x=0 to x=4")
x_points = [0, 1, 2, 3, 4]
y_points = [0, 2, 4, 6, 8]
result = trapezoidal_rule_tabular(x_points, y_points)
print(f"Result: {result['integral']}")
print(f"Expected: 16.0")
print(f"Steps:")
for step in result['steps']:
    print(f"  {step}")
print()

# Test case 2: Quadratic function y = x^2 from x=0 to x=4
# x_points = [0, 1, 2, 3, 4]
# y_points = [0, 1, 4, 9, 16]
# Expected integral: 21.33... (area under y=x^2 from 0 to 4)

print("Test Case 2: Quadratic function y = x^2 from x=0 to x=4")
x_points = [0, 1, 2, 3, 4]
y_points = [0, 1, 4, 9, 16]
result = trapezoidal_rule_tabular(x_points, y_points)
print(f"Result: {result['integral']}")
print(f"Expected: ~21.33")
print(f"Steps:")
for step in result['steps']:
    print(f"  {step}")
print()

# Test case 3: Exponential function y = 2^x from x=0 to x=4
# x_points = [0, 1, 2, 3, 4]
# y_points = [1, 2, 4, 8, 16]
# Expected integral: ~22.5 (approximate area under y=2^x from 0 to 4)

print("Test Case 3: Exponential function y = 2^x from x=0 to x=4")
x_points = [0, 1, 2, 3, 4]
y_points = [1, 2, 4, 8, 16]
result = trapezoidal_rule_tabular(x_points, y_points)
print(f"Result: {result['integral']}")
print(f"Expected: ~22.5")
print(f"Steps:")
for step in result['steps']:
    print(f"  {step}") 