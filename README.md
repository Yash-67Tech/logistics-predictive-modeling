# Logistics Predictive Modeling and Optimization

**Author:** Yash Zujam | Logistics Data Analyst Intern
**Task:** Week 4 - Predictive Modeling and Optimization in Logistics Systems

Python project that predicts shipment delivery time (hours) with machine learning and uses the predictions to optimize transport-mode selection and carrier allocation.

## Project structure
```
logistics-predictive-modeling/
├── scripts/predictive_modeling.py                 # simulation, models, tuning, optimization
├── data/logistics_prediction_data.csv             # simulated dataset (2,500 shipments)
├── data/model_output.txt                          # printed results
├── data/results.json                              # saved metrics and optimization results
├── images/                                        # 7 generated figures
├── Predictive_Modeling_Report_Yash_Zujam.docx     # full written report
└── requirements.txt
```

## How to run
```bash
pip install -r requirements.txt
python scripts/predictive_modeling.py
```
Run it from the project root so the `data/` and `images/` folders are found.

## Method
- **Models compared:** Linear Regression, Decision Tree, Random Forest, Gradient Boosting
- **Validation:** 80/20 train-test split and 5-fold cross-validation
- **Tuning:** GridSearchCV on Gradient Boosting (18 combinations)
- **Metrics:** MAE, RMSE, R-squared
- **Optimization:** SLA-aware mode selection and linear-programming carrier allocation (SciPy HiGHS)

## Key results
- Tuned Gradient Boosting: test MAE 2.65 hrs, RMSE 3.29 hrs, R2 0.987
- Mode selection removes all predicted SLA breaches (39.2% to 0%) and costs 37.8% less than shipping everything by air
- Linear-programming carrier allocation improves the cost-plus-delay objective by 5.3%
- Data is simulated, so accuracy figures are optimistic compared with real-world data

## Tools
Python, pandas, NumPy, scikit-learn, SciPy, matplotlib, seaborn
