# IOT-based-Food-Monitoring
# Meat Spoilage Prediction Pipeline

A machine learning pipeline that predicts **time-to-spoilage** of meat using sensor data (pH, gas, temperature, humidity). Combines Gaussian Process Regression, Support Vector Regression, and a Genetic Algorithm optimizer into a stacked ensemble.

---

## How It Works

1. **Load & Resample** — reads `data.csv`, resamples to uniform 21-second intervals via linear interpolation
2. **Preprocess** — IQR clipping, log transforms, Savitzky-Golay smoothing, 1st/2nd derivatives, composite spoilage index
3. **Feature Engineering** — lag features, rolling mean/std, time-elapsed
4. **Genetic Algorithm** — optimizes GPR kernel, SVR hyperparameters, lag steps, and window sizes over 25 generations
5. **Train Ensemble** — GPR + SVR base learners stacked with a Ridge meta-learner via time-series cross-validation
6. **Evaluate** — RMSE, MAE, R² on held-out eval split; outputs 12 publication-quality graphs

---

## Requirements

```bash
pip install numpy pandas scipy scikit-learn matplotlib
```

---


---

## Input Data

`data.csv` must contain:

| Column | Description |
|---|---|
| `Time-Date` | UTC timestamp |
| `PH` | pH reading |
| `MQ135-AirQuality` | Air quality sensor |
| `MQ136-H2S` | Hydrogen sulfide sensor |
| `MQ137-NH3` | Ammonia sensor |
| `Food_Temp` | Food surface temperature (°C) |
| `Food_Humid` | Food humidity (%) |
| `Amb_Temp` | Ambient temperature (°C) |
| `Amb_Humid` | Ambient humidity (%) |

---

## Output Graphs

| File | Description |
|---|---|
| `01_ph_trajectory.png` | pH over time with Fresh / Warning / Spoiled zones |
| `02_sensor_dashboard.png` | MQ135, MQ136, food temp, humidity signals |
| `03_spoilage_derivatives.png` | Composite spoilage index + 1st/2nd derivatives |
| `04_ga_convergence.png` | Genetic algorithm RMSE convergence curve |
| `05_time_to_spoil_timeline.png` | GPR / SVR / Ensemble predictions vs ground truth |
| `06_density_actual_vs_predicted.png` | Log-density scatter plots for all three models |
| `07_residual_analysis.png` | Histogram, Q-Q plot, residuals vs actual |
| `08_gpr_uncertainty.png` | GPR ±1σ / ±2σ uncertainty bands |
| `09_model_comparison.png` | RMSE / MAE / R² bar chart across models |
| `10_alert_dashboard.png` | Real-time alert status at query time points |
| `11_feature_correlation.png` | Pearson correlation heatmap of engineered features |
| `12_spoilage_class.png` | Pie chart + timeline of Fresh / Warning / Spoiled labels |

---

## Key Config

```python
SPOILAGE_PH       = 3.2   # pH threshold for spoilage
ALERT_MINUTES     = 60    # alert if predicted time-to-spoil < 60 min
GA_POP_SIZE       = 20
GA_GENERATIONS    = 25
GA_MUTATION_RATE  = 0.15
TSCV_SPLITS       = 5
```

---

## Spoilage Classes

| Class | pH Range |
|---|---|
| Fresh | > 3.8(from meat) |
| Warning | 3.2 – 3.8 (meat) |
| Spoiled | ≤ 3.2(meat) |
