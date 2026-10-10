"""McNemar's test with statsmodels.

Concepts: notes/04-evaluation.md
Install:  uv pip install statsmodels
Run:      python code/evaluation/compare_pipelines_lib.py
"""
from statsmodels.stats.contingency_tables import mcnemar

#            B right  B wrong
table = [[35,      6],   # A right
         [4,       5]]   # A wrong

print("Chi-square:", mcnemar(table, exact=False, correction=False))
print("Exact:     ", mcnemar(table, exact=True))
