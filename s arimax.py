import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

df = pd.read_csv(r"E:\FINAL OF FINAL\data\merged_data.csv", sep=';')
df.columns = df.columns.str.strip()
df = df.dropna()

selected_cols = [
    'Yield (kg/ha)', 'Year', 'Pesticides (total)',
    'Average Mean Surface Air Temperature (Annual Mean)',
    'Precipitation (mm)', 'Fertilizer (kg/ha)'
]
df = df[selected_cols].sort_values('Year').reset_index(drop=True)

df['Log_Yield'] = np.log(df['Yield (kg/ha)'])

#tạo lag cho biến ngoại sinh và biến phi tuyến
exog_base = [
    'Pesticides (total)',
    'Average Mean Surface Air Temperature (Annual Mean)',
    'Precipitation (mm)',
    'Fertilizer (kg/ha)'
]

for col in exog_base:
    df[f'{col}_lag1'] = df[col].shift(1)
    df[f'{col}_lag2'] = df[col].shift(2)

df['Temp_sq'] = df['Average Mean Surface Air Temperature (Annual Mean)'] ** 2
df['Rain_sq'] = df['Precipitation (mm)'] ** 2

df = df.dropna().reset_index(drop=True)

y = df['Log_Yield']
y_actual = df['Yield (kg/ha)']

exog_cols = []
for col in exog_base:
    exog_cols += [col, f'{col}_lag1', f'{col}_lag2']
exog_cols += ['Temp_sq', 'Rain_sq']

X = df[exog_cols].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=exog_cols, index=df.index)


model = SARIMAX(
    endog=y,
    exog=X_scaled,
    order=(2, 1, 1),         # ARIMA(p,d,q)
    trend='c',               
    enforce_stationarity=False,
    enforce_invertibility=False
)

res = model.fit(disp=False)

pred = res.get_prediction(start=0, end=len(y)-1, exog=X_scaled, dynamic=False)
y_pred_log = pred.predicted_mean

sigma2 = res.filter_results.obs_cov[0, 0] if hasattr(res.filter_results, "obs_cov") else np.var(res.resid.dropna())
y_pred_original = np.exp(y_pred_log + 0.5 * sigma2)

r2 = r2_score(y_actual, y_pred_original)
rmse = np.sqrt(mean_squared_error(y_actual, y_pred_original))

print("\n" + "="*50)
print("--- KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH ARIMAX TỐI ƯU ---")
print(f"R-squared: {r2:.4f}")
print(f"RMSE:      {rmse:.2f} kg/ha")
print("="*50)
print(res.summary())

res.plot_diagnostics(figsize=(12, 8))
plt.show()

    # Plot fit values vs actual values
plt.figure(figsize=(12,5))
plt.plot(df['Year'], y_actual, label='Actual')
plt.plot(df['Year'], y_pred_original, label='Fitted')
plt.legend()
plt.title("Actual vs Fitted (kg/ha)")
plt.show()
