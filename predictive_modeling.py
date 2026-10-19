"""
Week 4 - Predictive Modeling and Optimization in Logistics Systems
Author: Yash Zujam | Logistics Data Analyst Intern

Predicts shipment delivery time (hours) and uses the model to optimize
transport-mode selection and carrier allocation.
Run from the project root:  python scripts/predictive_modeling.py
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import linprog
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SEED = 42
rng = np.random.default_rng(SEED)
sns.set_theme(style="whitegrid", palette="deep")
IMG = "images/"
results = {}

# ------------------------------------------------------------------
# 1. DATA SIMULATION  (target = delivery_time_hrs)
# ------------------------------------------------------------------
n = 2500
modes = ["Road", "Rail", "Air", "Sea"]
carriers = ["FastTrack", "SwiftMove", "CargoPrime", "EcoFreight"]
regions = ["North", "South", "East", "West", "Central"]

df = pd.DataFrame({
    "distance_km": rng.integers(50, 3000, n),
    "weight_kg": np.round(rng.gamma(4, 120, n), 1),
    "transport_mode": rng.choice(modes, n, p=[0.5, 0.2, 0.1, 0.2]),
    "carrier": rng.choice(carriers, n),
    "region": rng.choice(regions, n),
    "traffic_index": np.round(rng.uniform(0, 10, n), 1),
    "weather_severity": np.round(rng.uniform(0, 10, n), 1),
    "num_stops": rng.integers(0, 6, n),
    "is_peak_season": rng.choice([0, 1], n, p=[0.8, 0.2]),
})

speed = {"Road": 55, "Rail": 45, "Air": 600, "Sea": 30}
handling = {"Road": 6, "Rail": 12, "Air": 8, "Sea": 36}
carrier_delay = {"FastTrack": 1.0, "SwiftMove": 2.5, "CargoPrime": 1.5, "EcoFreight": 0.0}
cost_km = {"Road": 1.2, "Rail": 0.8, "Air": 4.5, "Sea": 0.5}


def true_time(d, noise=True):
    """Ground-truth process used to simulate delivery time."""
    t = (d["distance_km"] / d["transport_mode"].map(speed)
         + d["transport_mode"].map(handling)
         + np.where(d["transport_mode"] == "Road", 0.9 * d["traffic_index"], 0.1 * d["traffic_index"])
         + 0.12 * d["weather_severity"] ** 2
         + 1.3 * d["num_stops"]
         + 0.004 * d["weight_kg"]
         + d["carrier"].map(carrier_delay)
         + 6 * d["is_peak_season"])
    if noise:
        t = t + rng.normal(0, 3.0, len(d))
    return np.clip(t, 2, None)


df["delivery_time_hrs"] = true_time(df).round(1)
df["shipping_cost"] = (df["distance_km"] * df["transport_mode"].map(cost_km)
                       + df["weight_kg"] * 0.45 + rng.normal(0, 30, n)).clip(lower=20).round(2)
df.to_csv("data/logistics_prediction_data.csv", index=False)

# ------------------------------------------------------------------
# 2. DATA PREPARATION
# ------------------------------------------------------------------
target = "delivery_time_hrs"
cat = ["transport_mode", "carrier", "region"]
num = ["distance_km", "weight_kg", "traffic_index", "weather_severity", "num_stops", "is_peak_season"]
X, y = df[cat + num], df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

prep = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
    ("num", StandardScaler(), num),
])

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=8, random_state=SEED),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(random_state=SEED),
}

# ------------------------------------------------------------------
# 3. TRAINING, CROSS-VALIDATION AND TEST EVALUATION
# ------------------------------------------------------------------
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
rows = []
for name, mdl in models.items():
    pipe = Pipeline([("prep", prep), ("model", mdl)])
    cv_r2 = cross_val_score(pipe, X_train, y_train, cv=kf, scoring="r2")
    cv_rmse = -cross_val_score(pipe, X_train, y_train, cv=kf, scoring="neg_root_mean_squared_error")
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    rows.append({
        "Model": name,
        "CV R2 (mean)": cv_r2.mean(), "CV RMSE (mean)": cv_rmse.mean(),
        "Test MAE": mean_absolute_error(y_test, pred),
        "Test RMSE": np.sqrt(mean_squared_error(y_test, pred)),
        "Test R2": r2_score(y_test, pred),
    })
comp = pd.DataFrame(rows).round(3)
print(comp.to_string(index=False))
results["comparison"] = comp.to_dict(orient="records")

# ------------------------------------------------------------------
# 4. HYPERPARAMETER TUNING (GridSearchCV on the best baseline family)
# ------------------------------------------------------------------
pipe = Pipeline([("prep", prep), ("model", GradientBoostingRegressor(random_state=SEED))])
grid = {
    "model__n_estimators": [100, 200, 300],
    "model__learning_rate": [0.05, 0.1],
    "model__max_depth": [2, 3, 4],
}
search = GridSearchCV(pipe, grid, cv=kf, scoring="neg_root_mean_squared_error", n_jobs=-1)
search.fit(X_train, y_train)
best = search.best_estimator_
pred = best.predict(X_test)
tuned = {
    "best_params": {k.replace("model__", ""): v for k, v in search.best_params_.items()},
    "cv_rmse": float(-search.best_score_),
    "test_mae": float(mean_absolute_error(y_test, pred)),
    "test_rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
    "test_r2": float(r2_score(y_test, pred)),
}
print("Tuned:", tuned)
results["tuned"] = tuned

# ------------------------------------------------------------------
# 5. VISUALIZATIONS
# ------------------------------------------------------------------
plt.figure(figsize=(8, 5))
sns.histplot(y, bins=40, kde=True, color="#2a6fb0")
plt.title("Target Variable: Delivery Time Distribution"); plt.xlabel("Delivery time (hours)")
plt.tight_layout(); plt.savefig(IMG + "fig1_target_distribution.png", dpi=150); plt.close()

plt.figure(figsize=(8, 5))
cm = comp.copy()
cm.loc[len(cm)] = ["Tuned Gradient Boosting", np.nan, tuned["cv_rmse"], tuned["test_mae"],
                   tuned["test_rmse"], tuned["test_r2"]]
ax = sns.barplot(data=cm, y="Model", x="Test RMSE")
for i, v in enumerate(cm["Test RMSE"]):
    ax.text(v + 0.2, i, f"{v:.2f}", va="center")
plt.title("Model Comparison: Test RMSE (lower is better)"); plt.xlabel("RMSE (hours)")
plt.tight_layout(); plt.savefig(IMG + "fig2_model_comparison.png", dpi=150); plt.close()

plt.figure(figsize=(6.5, 6))
plt.scatter(y_test, pred, alpha=0.5, s=18)
lim = [0, max(y_test.max(), pred.max()) + 5]
plt.plot(lim, lim, "r--", label="Perfect prediction")
plt.xlabel("Actual delivery time (hrs)"); plt.ylabel("Predicted delivery time (hrs)")
plt.title("Actual vs Predicted (Tuned Gradient Boosting)"); plt.legend()
plt.tight_layout(); plt.savefig(IMG + "fig3_actual_vs_predicted.png", dpi=150); plt.close()

resid = y_test - pred
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
sns.histplot(resid, bins=35, kde=True, ax=axes[0], color="#2a6fb0")
axes[0].set_title("Residual Distribution"); axes[0].set_xlabel("Residual (hrs)")
axes[1].scatter(pred, resid, alpha=0.5, s=18); axes[1].axhline(0, color="red", ls="--")
axes[1].set_title("Residuals vs Predicted"); axes[1].set_xlabel("Predicted (hrs)")
axes[1].set_ylabel("Residual (hrs)")
plt.tight_layout(); plt.savefig(IMG + "fig4_residuals.png", dpi=150); plt.close()

names = best.named_steps["prep"].get_feature_names_out()
imp = pd.Series(best.named_steps["model"].feature_importances_, index=names)
imp.index = [s.replace("cat__", "").replace("num__", "") for s in imp.index]
imp = imp.sort_values(ascending=False).head(10)
plt.figure(figsize=(8, 5))
sns.barplot(x=imp.values, y=imp.index, color="#2a6fb0")
plt.title("Top 10 Feature Importances"); plt.xlabel("Importance")
plt.tight_layout(); plt.savefig(IMG + "fig5_feature_importance.png", dpi=150); plt.close()
results["top_features"] = {k: round(float(v), 3) for k, v in imp.items()}

# ------------------------------------------------------------------
# 6. OPTIMIZATION A - Mode selection using model predictions
#    For each shipment choose the cheapest mode whose PREDICTED delivery
#    time meets a fixed SLA deadline. If no mode meets it, use the fastest.
# ------------------------------------------------------------------
batch = X_test.copy().reset_index(drop=True)
batch["shipping_cost"] = df.loc[X_test.index, "shipping_cost"].values
SLA_HRS = 60
pred_time, est_cost = {}, {}
for m in modes:
    tmp = batch.copy()
    tmp["transport_mode"] = m
    pred_time[m] = best.predict(tmp[cat + num])
    est_cost[m] = (tmp["distance_km"] * cost_km[m] + tmp["weight_kg"] * 0.45).values
pt = pd.DataFrame(pred_time); ec = pd.DataFrame(est_cost)

current = batch["transport_mode"]
cur_cost = np.array([ec.loc[i, current[i]] for i in range(len(batch))])
cur_time = np.array([pt.loc[i, current[i]] for i in range(len(batch))])

opt_mode, opt_cost, opt_time = [], [], []
for i in range(len(batch)):
    ok = [m for m in modes if pt.loc[i, m] <= SLA_HRS]
    if ok:
        m = min(ok, key=lambda k: ec.loc[i, k])
    else:  # nothing meets SLA -> choose fastest
        m = min(modes, key=lambda k: pt.loc[i, k])
    opt_mode.append(m); opt_cost.append(ec.loc[i, m]); opt_time.append(pt.loc[i, m])
opt_cost, opt_time = np.array(opt_cost), np.array(opt_time)

air_cost = ec["Air"].values
air_time = pt["Air"].values
ok_now = cur_time <= SLA_HRS            # shipments already meeting the SLA
res_a = {
    "shipments": len(batch), "sla_hrs": SLA_HRS,
    "baseline_cost": float(cur_cost.sum()), "optimized_cost": float(opt_cost.sum()),
    "all_air_cost": float(air_cost.sum()),
    "baseline_sla_breach_pct": float((cur_time > SLA_HRS).mean() * 100),
    "optimized_sla_breach_pct": float((opt_time > SLA_HRS).mean() * 100),
    "all_air_sla_breach_pct": float((air_time > SLA_HRS).mean() * 100),
    "opt_vs_all_air_saving_pct": float((1 - opt_cost.sum() / air_cost.sum()) * 100),
    "on_time_n": int(ok_now.sum()), "late_n": int((~ok_now).sum()),
    "on_time_group_cost_change_pct": float((opt_cost[ok_now].sum() / cur_cost[ok_now].sum() - 1) * 100),
    "late_group_cost_change_pct": float((opt_cost[~ok_now].sum() / cur_cost[~ok_now].sum() - 1) * 100),
    "baseline_mix": current.value_counts(normalize=True).mul(100).round(1).to_dict(),
    "optimized_mix": pd.Series(opt_mode).value_counts(normalize=True).mul(100).round(1).to_dict(),
}
print("Optimization A:", res_a)
results["opt_mode"] = res_a

# ------------------------------------------------------------------
# 7. OPTIMIZATION B - Carrier allocation as a linear program
#    minimise  sum(cost_ij + penalty * predicted_time_ij) * x_ij
#    s.t. each shipment assigned once; carrier capacity limits.
# ------------------------------------------------------------------
m = len(batch)
pen = 2.0   # USD penalty per predicted hour of transit
cap = {"FastTrack": 0.30, "SwiftMove": 0.25, "CargoPrime": 0.25, "EcoFreight": 0.20}
cap_n = {k: int(np.ceil(v * m)) + 10 for k, v in cap.items()}   # capacity per carrier
price_factor = {"FastTrack": 1.05, "SwiftMove": 0.95, "CargoPrime": 1.00, "EcoFreight": 1.08}

base_cost = ec.values[np.arange(m), [modes.index(x) for x in opt_mode]]
C = np.zeros((m, len(carriers)))
for j, c in enumerate(carriers):
    tmp = batch.copy(); tmp["transport_mode"] = opt_mode; tmp["carrier"] = c
    p_time = best.predict(tmp[cat + num])
    C[:, j] = base_cost * price_factor[c] + pen * p_time

c_vec = C.flatten()
A_eq = np.zeros((m, m * len(carriers)))
for i in range(m):
    A_eq[i, i * len(carriers):(i + 1) * len(carriers)] = 1
A_ub = np.zeros((len(carriers), m * len(carriers)))
for j in range(len(carriers)):
    A_ub[j, j::len(carriers)] = 1
b_ub = [cap_n[c] for c in carriers]
lp = linprog(c_vec, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=np.ones(m), bounds=(0, 1), method="highs")
X_alloc = lp.x.reshape(m, len(carriers)).argmax(axis=1)

# baseline: random carrier assignment respecting the same data (original carriers)
orig = [carriers.index(x) for x in batch["carrier"]]
base_total = C[np.arange(m), orig].sum()
opt_total = C[np.arange(m), X_alloc].sum()
res_b = {
    "status": lp.message, "baseline_objective": float(base_total), "optimized_objective": float(opt_total),
    "improvement_pct": float((1 - opt_total / base_total) * 100),
    "baseline_split": pd.Series(batch["carrier"]).value_counts(normalize=True).mul(100).round(1).to_dict(),
    "optimized_split": pd.Series([carriers[k] for k in X_alloc]).value_counts(normalize=True).mul(100).round(1).to_dict(),
    "capacity": cap_n,
}
print("Optimization B:", res_b)
results["opt_carrier"] = res_b

# Fig 6: optimization results (three policies)
pol = ["Current mix", "Optimized", "All Air"]
costs = [res_a["baseline_cost"] / 1000, res_a["optimized_cost"] / 1000, res_a["all_air_cost"] / 1000]
breach = [res_a["baseline_sla_breach_pct"], res_a["optimized_sla_breach_pct"], res_a["all_air_sla_breach_pct"]]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].bar(pol, costs, color=["#999999", "#2a6fb0", "#c0504d"])
axes[0].set_title("Total Shipping Cost by Policy"); axes[0].set_ylabel("Cost (thousand USD)")
for i, v in enumerate(costs):
    axes[0].text(i, v + 15, f"{v:,.0f}k", ha="center")
axes[1].bar(pol, breach, color=["#999999", "#2a6fb0", "#c0504d"])
axes[1].set_title(f"Predicted SLA Breaches (> {SLA_HRS} hrs)"); axes[1].set_ylabel("% of shipments")
for i, v in enumerate(breach):
    axes[1].text(i, v + 0.5, f"{v:.1f}%", ha="center")
plt.tight_layout(); plt.savefig(IMG + "fig6_optimization_results.png", dpi=150); plt.close()

# Fig 7: carrier allocation before vs after
split = pd.DataFrame({"Baseline": res_b["baseline_split"], "Optimized": res_b["optimized_split"]}).fillna(0).loc[carriers]
split.plot(kind="bar", figsize=(8, 5))
plt.title("Carrier Allocation: Baseline vs LP-Optimized"); plt.ylabel("% of shipments"); plt.xticks(rotation=0)
plt.tight_layout(); plt.savefig(IMG + "fig7_carrier_allocation.png", dpi=150); plt.close()

json.dump(results, open("data/results.json", "w"), indent=2, default=float)
print("Done.")
