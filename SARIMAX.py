import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# =========================
# 1) Load & chuẩn bị dữ liệu
# =========================
df = pd.read_csv(r"E:\FINAL OF FINAL\data\merged_data.csv", sep=';')
df.columns = df.columns.str.strip()
df = df.dropna()

selected_cols = [
    'Yield (kg/ha)', 'Year', 'Pesticides (total)',
    'Average Mean Surface Air Temperature (Annual Mean)',
    'Precipitation (mm)', 'Fertilizer (kg/ha)'
]
df = df[selected_cols].sort_values('Year').reset_index(drop=True)

# Log Yield
df['Log_Yield'] = np.log(df['Yield (kg/ha)'])

# =========================
# 2) Tạo lag cho biến ngoại sinh (rất quan trọng)
# =========================
exog_base = [
    'Pesticides (total)',
    'Average Mean Surface Air Temperature (Annual Mean)',
    'Precipitation (mm)',
    'Fertilizer (kg/ha)'
]

# Lag 1 và lag 2 (bạn có thể thử thêm lag 3)
for col in exog_base:
    df[f'{col}_lag1'] = df[col].shift(1)
    df[f'{col}_lag2'] = df[col].shift(2)

# (Tuỳ chọn) phi tuyến nhiệt độ/mưa
df['Temp_sq'] = df['Average Mean Surface Air Temperature (Annual Mean)'] ** 2
df['Rain_sq'] = df['Precipitation (mm)'] ** 2

# BỎ Year khỏi exog (khuyến nghị mạnh khi d=1)
# Nếu bạn muốn giữ trend, làm bằng drift/trend chứ không phải Year trong exog

df = df.dropna().reset_index(drop=True)

y = df['Log_Yield']
y_actual = df['Yield (kg/ha)']

exog_cols = []
for col in exog_base:
    exog_cols += [col, f'{col}_lag1', f'{col}_lag2']
exog_cols += ['Temp_sq', 'Rain_sq']

X = df[exog_cols].copy()

# =========================
# 3) Chuẩn hoá exog (ổn định hệ số + giảm singular)
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=exog_cols, index=df.index)

# Không cần add_constant nếu trend dùng ở SARIMAX (trend='c' hoặc 't')
# Nếu bạn muốn constant trong exog thì add_constant, nhưng SARIMAX có trend xử lý sạch hơn.

# =========================
# 4) Fit SARIMAX (ARIMAX) + dự báo đúng kiểu one-step in-sample
# =========================
# Bắt đầu từ mô hình bạn có, nhưng dùng SARIMAX để forecast chuẩn
model = SARIMAX(
    endog=y,
    exog=X_scaled,
    order=(2, 1, 1),         # gợi ý bắt đầu đơn giản hơn (thường tốt hơn 5,1,1)
    trend='c',               # có intercept (drift) thay vì Year
    enforce_stationarity=False,
    enforce_invertibility=False
)

res = model.fit(disp=False)

# Dự báo one-step in-sample (dynamic=False) -> phù hợp để tính R2/RMSE in-sample
pred = res.get_prediction(start=0, end=len(y)-1, exog=X_scaled, dynamic=False)
y_pred_log = pred.predicted_mean

# =========================
# 5) Bias correction khi exp(log) (quan trọng!)
# =========================
# Khi log-normal: E[Y] ≈ exp(mu + 0.5*sigma^2)
sigma2 = res.filter_results.obs_cov[0, 0] if hasattr(res.filter_results, "obs_cov") else np.var(res.resid.dropna())
y_pred_original = np.exp(y_pred_log + 0.5 * sigma2)

# =========================
# 6) Metrics
# =========================
r2 = r2_score(y_actual, y_pred_original)
rmse = np.sqrt(mean_squared_error(y_actual, y_pred_original))

print("\n" + "="*50)
print("--- KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH ARIMAX TỐI ƯU ---")
print(f"R-squared: {r2:.4f}")
print(f"RMSE:      {rmse:.2f} kg/ha")
print("="*50)
print(res.summary())

# Diagnostics
res.plot_diagnostics(figsize=(12, 8))
plt.show()

# Plot fit
plt.figure(figsize=(12,5))
plt.plot(df['Year'], y_actual, label='Actual')
plt.plot(df['Year'], y_pred_original, label='Fitted')
plt.legend()
plt.title("Actual vs Fitted (kg/ha)")
plt.show()
