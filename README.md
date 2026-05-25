# salary prediction

![Python](https://img.shields.io/badge/python-3.11-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-orange)

still learning ML so i tried building a model that predicts data science salaries. tried 4 different regression models to see which one works best.

dataset: [Salary Prediction Data — Kaggle](https://www.kaggle.com/datasets/mrsimple07/salary-prediction-data)

---

## results

| model | MAE | R² | CV R² |
|---|---|---|---|
| Linear Regression | ~$9,200 | 0.78 | 0.77 |
| Ridge | ~$9,200 | 0.78 | 0.77 |
| **Random Forest** | **~$6,100** | **0.88** | **0.87** |
| Gradient Boosting | ~$6,800 | 0.86 | 0.85 |

random forest won. about $6k off on salaries that range from $40k to $180k. not bad.

---

## what i noticed

- experience is by far the biggest factor (no surprise)
- ML skill matters more than python or SQL
- PhD adds ~$18k over bachelor on average
- big company pays ~$20k more than startup for same experience
- linear models work but tree-based ones catch the non-linear stuff better

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

pandas, numpy, scikit-learn, matplotlib, seaborn
