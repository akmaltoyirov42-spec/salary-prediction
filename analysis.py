import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score

OUTPUT = Path("output")
OUTPUT.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams["figure.dpi"] = 130

df = pd.read_csv("data/salaries.csv")
print(f"Shape: {df.shape}")
print(df.describe().round(0))


# salary distribution
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].hist(df["salary_usd"] / 1000, bins=40, color="#3498db", edgecolor="white", linewidth=0.4)
axes[0].axvline(df["salary_usd"].median() / 1000, color="red", linestyle="--", linewidth=1.5,
                label=f"Median ${df['salary_usd'].median()/1000:.0f}k")
axes[0].set_title("Salary Distribution")
axes[0].set_xlabel("Salary (USD thousands)")
axes[0].legend()

sns.boxplot(data=df, x="role", y="salary_usd", ax=axes[1], palette="Set2")
axes[1].set_title("Salary by Role")
axes[1].set_xlabel("")
axes[1].set_ylabel("Salary (USD)")
axes[1].tick_params(axis="x", rotation=20)
axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))

plt.suptitle("Salary Overview", fontweight="bold")
plt.tight_layout()
plt.savefig(OUTPUT / "1_overview.png")
plt.close()
print("Saved 1_overview.png")


# experience vs salary
fig, ax = plt.subplots(figsize=(11, 5))
for role in df["role"].unique():
    sub = df[df["role"] == role]
    exp_avg = sub.groupby("experience_years")["salary_usd"].mean()
    ax.plot(exp_avg.index, exp_avg.values / 1000, marker="o", markersize=4, label=role, linewidth=1.8)

ax.set_title("Average Salary by Experience — per Role", fontweight="bold")
ax.set_xlabel("Years of experience")
ax.set_ylabel("Salary (USD thousands)")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(OUTPUT / "2_experience_salary.png")
plt.close()
print("Saved 2_experience_salary.png")


# education & company size
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

edu_order = ["Bachelor", "Master", "PhD"]
edu_med = df.groupby("education")["salary_usd"].median().reindex(edu_order) / 1000
axes[0].bar(edu_med.index, edu_med.values, color=["#95a5a6", "#3498db", "#2ecc71"])
axes[0].set_title("Median Salary by Education")
axes[0].set_ylabel("USD thousands")
for i, v in enumerate(edu_med.values):
    axes[0].text(i, v + 0.5, f"${v:.0f}k", ha="center", fontsize=10)

comp_order = ["Startup", "Mid-size", "Large"]
comp_med = df.groupby("company_size")["salary_usd"].median().reindex(comp_order) / 1000
axes[1].bar(comp_med.index, comp_med.values, color=["#e74c3c", "#f39c12", "#27ae60"])
axes[1].set_title("Median Salary by Company Size")
axes[1].set_ylabel("USD thousands")
for i, v in enumerate(comp_med.values):
    axes[1].text(i, v + 0.5, f"${v:.0f}k", ha="center", fontsize=10)

plt.suptitle("What affects salary?", fontweight="bold")
plt.tight_layout()
plt.savefig(OUTPUT / "3_edu_company.png")
plt.close()
print("Saved 3_edu_company.png")


# correlation heatmap
fig, ax = plt.subplots(figsize=(8, 6))
num_cols = ["experience_years", "python_skill", "sql_skill", "ml_skill", "salary_usd"]
corr = df[num_cols].corr().round(2)
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, ax=ax, linewidths=0.5)
ax.set_title("Feature Correlations", fontweight="bold")
plt.tight_layout()
plt.savefig(OUTPUT / "4_correlations.png")
plt.close()
print("Saved 4_correlations.png")


# ML — encode and train models
le = LabelEncoder()
df_ml = df.copy()
for col in ["role", "education", "company_size"]:
    df_ml[col] = le.fit_transform(df_ml[col])

X = df_ml.drop("salary_usd", axis=1)
y = df_ml["salary_usd"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Linear Regression": LinearRegression(),
    "Ridge":             Ridge(alpha=1.0),
    "Random Forest":     RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
}

print("\nModel results:")
results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    cv   = cross_val_score(model, X, y, cv=5, scoring="r2").mean()
    results.append({"Model": name, "MAE": int(mae), "R2": round(r2, 3), "CV R2": round(cv, 3)})
    print(f"  {name:<22} MAE=${mae:,.0f}   R2={r2:.3f}   CV R2={cv:.3f}")

results_df = pd.DataFrame(results)
best = results_df.loc[results_df["R2"].idxmax(), "Model"]
print(f"\nBest: {best}")

# feature importance from best tree model
rf = models["Random Forest"]
importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(9, 5))
importance.plot.barh(ax=ax, color="#3498db", alpha=0.85)
ax.set_title("Feature Importance — Random Forest", fontweight="bold")
ax.set_xlabel("Importance")
plt.tight_layout()
plt.savefig(OUTPUT / "5_feature_importance.png")
plt.close()
print("Saved 5_feature_importance.png")

# actual vs predicted
best_model = models[best]
y_pred_best = best_model.predict(X_test)

fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(y_test / 1000, y_pred_best / 1000, alpha=0.4, s=20, color="#e74c3c")
mn, mx = y_test.min() / 1000, y_test.max() / 1000
ax.plot([mn, mx], [mn, mx], "k--", linewidth=1, label="Perfect prediction")
ax.set_title(f"Actual vs Predicted — {best}", fontweight="bold")
ax.set_xlabel("Actual salary ($k)")
ax.set_ylabel("Predicted salary ($k)")
ax.legend()
plt.tight_layout()
plt.savefig(OUTPUT / "6_actual_vs_predicted.png")
plt.close()
print("Saved 6_actual_vs_predicted.png")

print("\nDone — plots in output/")
