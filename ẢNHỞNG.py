import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# 1. Tải và làm sạch dữ liệu
df = pd.read_csv(r"E:\FINAL OF FINAL\data\merged_data.csv", sep=';')
df.columns = df.columns.str.strip()
df = df.dropna().sort_values(by=['Year']).reset_index(drop=True)

# 2. Chuẩn bị biến
# Log biến mục tiêu để ổn định phương sai
y = np.log(df['Yield (kg/ha)'])

# Các biến ngoại sinh (loại bỏ Year để tránh trùng lặp với đặc tính thời gian của ARIMA)
exog_cols = [
    'Pesticides (total)', 
    'Average Mean Surface Air Temperature (Annual Mean)',
    'Precipitation (mm)', 
    'Fertilizer (kg/ha)'
]
X = df[exog_cols]

# CHUẨN HÓA X: Giúp các hệ số (coefficients) có thể so sánh trực tiếp với nhau
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=exog_cols)
X_scaled = sm.add_constant(X_scaled)

# 3. Huấn luyện mô hình ARIMAX
# Sử dụng order=(1,1,1) thường ổn định hơn cho dữ liệu ngắn; bạn có thể chỉnh lại (5,1,1) nếu cần
model = SARIMAX(y, exog=X_scaled, order=(2, 1, 1), trend='n')
results = model.fit()

# 4. Trích xuất hệ số để vẽ biểu đồ
all_params = results.params
exog_vars_only = [c for c in X_scaled.columns if c != 'const']
coef_values = all_params[exog_vars_only]

clean_names = {
    'Pesticides (total)': 'Thuốc trừ sâu',
    'Average Mean Surface Air Temperature (Annual Mean)': 'Nhiệt độ TB',
    'Precipitation (mm)': 'Lượng mưa',
    'Fertilizer (kg/ha)': 'Phân bón'
}

plot_data = pd.DataFrame({
    'Yếu tố': [clean_names.get(name, name) for name in exog_vars_only],
    'Hệ số': coef_values.values
}).sort_values(by='Hệ số', ascending=True)

# 5. Vẽ biểu đồ "điểm nhấn" chuyên nghiệp
plt.figure(figsize=(12, 7))
sns.set_style("whitegrid", {'axes.grid' : False})

# Màu sắc phân biệt Tốt/Xấu
colors = ['#e74c3c' if x < 0 else '#3498db' for x in plot_data['Hệ số']]

# Vẽ thanh ngang
bars = plt.barh(plot_data['Yếu tố'], plot_data['Hệ số'], color=colors, edgecolor='black', height=0.6)

# Đường gốc 0
plt.axvline(x=0, color='black', linestyle='-', linewidth=2, alpha=0.6)

# Thêm giá trị số lên thanh
for bar in bars:
    width = bar.get_width()
    label_x = width + (0.005 if width > 0 else -0.005)
    plt.text(label_x, bar.get_y() + bar.get_height()/2, 
             f'{width:.4f}', 
             va='center', ha='left' if width > 0 else 'right',
             fontweight='bold', fontsize=12)

plt.title('MỨC ĐỘ ẢNH HƯỞNG CỦA CÁC YẾU TỐ ĐẾN NĂNG SUẤT\n(Dựa trên mô hình SARIMAX đã chuẩn hóa)', 
          fontsize=16, fontweight='bold', pad=25)
plt.xlabel('Trọng số tác động (Càng xa 0 tác động càng mạnh)', fontsize=13)
sns.despine(left=True, bottom=True)

plt.tight_layout()
plt.show()

print(results.summary())