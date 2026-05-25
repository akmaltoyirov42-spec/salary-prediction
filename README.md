# salary prediction

![Python](https://img.shields.io/badge/python-3.11-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-orange)

predicts data science salaries based on experience, role, education, company size, and skill levels. compared 4 regression models to pick the best one.

dataset: [Salary Prediction Data — Kaggle](https://www.kaggle.com/datasets/mrsimple07/salary-prediction-data)

---

## results

| model | MAE | R² | CV R² |
|---|---|---|---|
| Linear Regression | ~$9,200 | 0.78 | 0.77 |
| Ridge | ~$9,200 | 0.78 | 0.77 |
| **Random Forest** | **~$6,100** | **0.88** | **0.87** |
| Gradient Boosting | ~$6,800 | 0.86 | 0.85 |

random forest wins — about $6k off on salaries that range from $40k to $180k.

---

## what drives salary

- experience is by far the biggest factor
- ML skill matters more than python or SQL
- PhD adds ~$18k over bachelor on average
- big company pays ~$20k more than startup for the same experience
- tree-based models catch the non-linear interactions linear models miss

---

## run it

```bash
git clone https://github.com/akmaltoyirov42-spec/salary-prediction.git
cd salary-prediction
pip install -r requirements.txt

python data/generate.py
python analysis.py
# plots show up in output/
```

---

## what's next

planning to add XGBoost with optuna for hyperparameter tuning, and use SHAP for proper feature importance instead of the default random forest one.

---

pandas, numpy, scikit-learn, matplotlib, seaborn
