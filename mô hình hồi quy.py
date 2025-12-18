import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

# Load the data
df = pd.read_csv(r"E:\FINAL OF FINAL\data\merged_data.csv", delimiter=';')

# 1. Data Preparation: Ensure data is sorted by Year
df = df.sort_values(by='Year').reset_index(drop=True)

# 2. Define Variables (Y: Yield, X: Time)
Y = df['Yield (kg/ha)']
X = df['Year']

# 3. OLS Model for Trend: Yield = b0 + b1 * Year
X = sm.add_constant(X)
trend_model = sm.OLS(Y, X).fit()

# 4. Print Summary
print("--- KẾT QUẢ MÔ HÌNH HỒI QUY XU HƯỚNG ---")
print(trend_model.summary().as_text())

# 5. Calculate Residuals and Fitted Values
df['Fitted_Trend'] = trend_model.predict(X)
df['Residuals'] = trend_model.resid

# 6. Plotting - Part A: Trend and Residuals
plt.figure(figsize=(16, 6))

# Plot 1: Yield vs Year with Trend Line
plt.subplot(1, 2, 1)
plt.scatter(df['Year'], df['Yield (kg/ha)'], alpha=0.5, label='Yield Thực tế')
plt.plot(df['Year'], df['Fitted_Trend'], color='red', linestyle='-', linewidth=2, label='Đường Trend (OLS)')
plt.title('Xu hướng Năng suất (Yield) và Đường Hồi quy Trend', fontsize=14)
plt.xlabel('Năm', fontsize=12)
plt.ylabel('Năng suất (kg/ha)', fontsize=12)
plt.legend()
plt.grid(axis='y', linestyle='--')

# Plot 2: Residuals vs Year
plt.subplot(1, 2, 2)
plt.scatter(df['Year'], df['Residuals'], alpha=0.5, color='gray')
plt.axhline(0, color='red', linestyle='--', linewidth=1)
plt.title('Biểu đồ Phần dư theo Thời gian', fontsize=14)
plt.xlabel('Năm', fontsize=12)
plt.ylabel('Phần dư', fontsize=12)
plt.grid(axis='y', linestyle='--')

plt.tight_layout()
plt.savefig('trend_and_residuals_plot.png')
plt.close()

# 7. Plotting - Part B: ACF of Residuals
plt.figure(figsize=(8, 5))
# Since the data is not a single time series but panel data, the ACF plot needs special care.
# The `Year` column is not unique (multiple countries/items per year).
# For the purpose of checking for general autocorrelation in the *errors* of the trend model,
# we still plot the ACF of the residuals assuming the order is chronological by Year.
plot_acf(df['Residuals'], lags=50, alpha=0.05, title='ACF của Phần dư (Residuals)')
plt.xlabel('Độ trễ (Lag)')
plt.savefig('acf_residuals_plot.png')
plt.close()

print("Đã tạo 'trend_and_residuals_plot.png' và 'acf_residuals_plot.png'")