# Salary Prediction

![Python](https://img.shields.io/badge/python-3.11-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-orange)

Predicts data science salaries based on experience, role, education, company size, and skill levels. Compares four regression models and picks the best one.

Dataset: [Salary Prediction Data — Kaggle](https://www.kaggle.com/datasets/mrsimple07/salary-prediction-data)

---

## Results

| Model | MAE | R² | CV R² |
|---|---|---|---|
| Linear Regression | ~$9,200 | 0.78 | 0.77 |
| Ridge | ~$9,200 | 0.78 | 0.77 |
| **Random Forest** | **~$6,100** | **0.88** | **0.87** |
| Gradient Boosting | ~$6,800 | 0.86 | 0.85 |

Random Forest wins. MAE of ~$6k on salaries ranging $40k–$180k.

---

## What drives salary

Based on feature importance from the Random Forest:
- Experience is the strongest predictor by far
- ML skill level matters more than Python or SQL
- PhD adds ~$18k over Bachelor on average
- Large company pays ~$20k more than a startup at the same experience level

---

## Run it

```bash
git clone https://github.com/akmaltoyirov42-spec/salary-prediction.git
cd salary-prediction

pip install -r requirements.txt

# generate sample data (or use the real Kaggle dataset)
python data/generate.py

python analysis.py
# plots → output/
```

---

## Stack

pandas, numpy, scikit-learn, matplotlib, seaborn
