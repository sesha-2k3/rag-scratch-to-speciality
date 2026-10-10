"""Is pipeline A really better than B? McNemar's test from scratch.

Concepts: notes/04-evaluation.md
Run:      python code/evaluation/compare_pipelines_scratch.py

Example: 50 questions, A gets 41 right (82%), B gets 39 right (78%).
"""
import math

# True = answered correctly. Built so both=35, A only=6, B only=4, neither=5.
a_correct = [True] * 35 + [True] * 6 + [False] * 4 + [False] * 5
b_correct = [True] * 35 + [False] * 6 + [True] * 4 + [False] * 5

a_only = sum(a and not b for a, b in zip(a_correct, b_correct))
b_only = sum(b and not a for a, b in zip(a_correct, b_correct))
print(f"A: {sum(a_correct)}/50, B: {sum(b_correct)}/50")
print(f"Disagreements: A right & B wrong = {a_only}, B right & A wrong = {b_only}")

# Chi-square version (no continuity correction), 1 degree of freedom
chi2 = (a_only - b_only) ** 2 / (a_only + b_only)
p_chi2 = math.erfc(math.sqrt(chi2 / 2))
print(f"Chi-square = {chi2:.2f}, p = {p_chi2:.2f}")

# Exact version: under "equal systems", each disagreement is a coin flip
n, x = a_only + b_only, max(a_only, b_only)
p_exact = min(1.0, 2 * sum(math.comb(n, i) for i in range(x, n + 1)) / 2 ** n)
print(f"Exact binomial p = {p_exact:.2f}")
print("Not significant: the 4-point gap is consistent with random noise.")
