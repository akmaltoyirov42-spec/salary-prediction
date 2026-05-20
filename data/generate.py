"""
Generates a realistic salary dataset.
Based on real salary survey data patterns from Stack Overflow & Levels.fyi.

Real dataset alternative: kaggle.com/datasets/mrsimple07/salary-prediction-data
"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 1000

experience = rng.integers(0, 21, n)

roles = rng.choice(
    ["Data Analyst", "Data Scientist", "ML Engineer", "Data Engineer", "Research Scientist"],
    p=[0.25, 0.30, 0.20, 0.15, 0.10],
    size=n,
)

role_base = {
    "Data Analyst":       55000,
    "Data Scientist":     75000,
    "ML Engineer":        90000,
    "Data Engineer":      80000,
    "Research Scientist": 85000,
}

education = rng.choice(["Bachelor", "Master", "PhD"], p=[0.50, 0.35, 0.15], size=n)
edu_bonus = {"Bachelor": 0, "Master": 8000, "PhD": 18000}

company_size = rng.choice(["Startup", "Mid-size", "Large"], p=[0.30, 0.40, 0.30], size=n)
company_bonus = {"Startup": -5000, "Mid-size": 5000, "Large": 15000}

python_skill = rng.integers(1, 6, n)   # 1-5 scale
sql_skill    = rng.integers(1, 6, n)
ml_skill     = rng.integers(1, 6, n)

rows = []
for i in range(n):
    base = role_base[roles[i]]
    exp_bump = experience[i] * rng.uniform(3000, 5000)
    skill_bump = (python_skill[i] + sql_skill[i] + ml_skill[i]) * 500
    edu_b = edu_bonus[education[i]]
    comp_b = company_bonus[company_size[i]]
    noise = rng.normal(0, 6000)

    salary = max(30000, base + exp_bump + skill_bump + edu_b + comp_b + noise)

    rows.append({
        "experience_years": experience[i],
        "role":             roles[i],
        "education":        education[i],
        "company_size":     company_size[i],
        "python_skill":     python_skill[i],
        "sql_skill":        sql_skill[i],
        "ml_skill":         ml_skill[i],
        "salary_usd":       int(round(salary, -2)),
    })

df = pd.DataFrame(rows)
df.to_csv("data/salaries.csv", index=False)
print(f"Saved {len(df)} rows")
print(df.describe().round(0))
