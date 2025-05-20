# ------------------ Step 1: Import Required Libraries -----------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam, RMSprop
from scipy.stats import f_oneway, probplot

# ------------------ Step 2: Load Dataset ------------------
df = pd.read_csv("Airline_Revenue_Management_Dataset.csv")
print(df['Departure Date'].min())
print(df['Departure Date'].max())

# ------------------ Step 3: Exploratory Data Analysis (EDA) ------------------
print("Dataset Head:\n", df.head())
print("\nData Types:\n", df.dtypes)
print("\nMissing Values:\n", df.isnull().sum())
print("\nSummary Statistics:\n", df.describe())

# Histograms
df.hist(bins=30, figsize=(14, 10))
plt.suptitle("Feature Distributions")
plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.select_dtypes(include=np.number).corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

# ------------------ Step 4: Encode Categorical Variables ------------------
le = LabelEncoder()
df['Fare_Class'] = le.fit_transform(df['Fare_Class'])
df['Weather_Impact'] = le.fit_transform(df['Weather_Impact'])

# ------------------ Step 5: Feature Selection ------------------
features = ['Booking_Lead_Time', 'Fare_Class', 'Seats_Available', 'Seats_Booked',
            'Competitor_Price', 'Fuel_Price', 'Public_Holiday', 'Weather_Impact']
X = df[features]
y = df['Ticket_Price']

# ------------------ Step 6: Train-Test Split & Scaling ------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save splits
X_train.to_csv("train_features.csv", index=False)
y_train.to_csv("train_labels.csv", index=False)
X_test.to_csv("test_features.csv", index=False)
y_test.to_csv("test_labels.csv", index=False)

# ------------------ Step 7: Hyperparameter Tuning ------------------
# Random Forest
rf = RandomForestRegressor(random_state=42)
rf_params = {'n_estimators': [100, 150], 'max_depth': [None, 10, 20]}
rf_grid = GridSearchCV(rf, rf_params, cv=3, scoring='neg_mean_squared_error')
rf_grid.fit(X_train_scaled, y_train)
best_rf = rf_grid.best_estimator_

# XGBoost
xgb_model = xgb.XGBRegressor(random_state=42)
xgb_params = {'n_estimators': [100, 150], 'max_depth': [3, 5], 'learning_rate': [0.05, 0.1]}
xgb_grid = GridSearchCV(xgb_model, xgb_params, cv=3, scoring='neg_mean_squared_error')
xgb_grid.fit(X_train_scaled, y_train)
best_xgb = xgb_grid.best_estimator_

# ------------------ Step 8: LSTM Hyperparameter Tuning (Optimized) ------------------

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Reshape for LSTM input
X_train_lstm = np.reshape(X_train_scaled, (X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_lstm = np.reshape(X_test_scaled, (X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# Model builder
def build_lstm_model(units=64, optimizer='adam', activation='relu'):
    model = Sequential()
    model.add(LSTM(units, input_shape=(1, X_train_scaled.shape[1]), activation=activation))
    model.add(Dense(1))
    model.compile(optimizer=optimizer, loss='mse')
    return model

# Reduced search grid
lstm_param_grid = {
    'units': [64],
    'activation': ['relu', 'tanh'],
    'optimizer': ['adam'],
    'batch_size': [32]
}

# Early stopping callback
early_stop = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)

best_rmse = float('inf')
best_lstm_model = None
best_lstm_params = {}

# Tuning loop
for units in lstm_param_grid['units']:
    for activation in lstm_param_grid['activation']:
        for optimizer in lstm_param_grid['optimizer']:
            for batch_size in lstm_param_grid['batch_size']:
                print(f"Training LSTM with {units} units, {activation} activation, {optimizer} optimizer, batch size {batch_size}")
                model = build_lstm_model(units=units, optimizer=optimizer, activation=activation)
                model.fit(X_train_lstm, y_train, epochs=20, batch_size=batch_size, verbose=0, callbacks=[early_stop])
                preds = model.predict(X_test_lstm).flatten()
                rmse = np.sqrt(mean_squared_error(y_test, preds))
                print(f"RMSE: {rmse:.4f}")

                if rmse < best_rmse:
                    best_rmse = rmse
                    best_lstm_model = model
                    best_lstm_params = {
                        'units': units,
                        'activation': activation,
                        'optimizer': optimizer,
                        'batch_size': batch_size
                    }

print("\n Best LSTM Model Found:")
print(best_lstm_params)
print(f"Best LSTM RMSE: {best_rmse:.4f}")

# Predict using best LSTM
lstm_preds = best_lstm_model.predict(X_test_lstm).flatten()

# ------------------ Step 9: Model Evaluation ------------------

rf_preds = best_rf.predict(X_test_scaled)
xgb_preds = best_xgb.predict(X_test_scaled)

print("\nModel Performance:\n")
print("Random Forest RMSE:", np.sqrt(mean_squared_error(y_test, rf_preds)))
print("Random Forest R²:", r2_score(y_test, rf_preds))

print("\nXGBoost RMSE:", np.sqrt(mean_squared_error(y_test, xgb_preds)))
print("XGBoost R²:", r2_score(y_test, xgb_preds))

print("\nTuned LSTM RMSE:", np.sqrt(mean_squared_error(y_test, lstm_preds)))
print("Tuned LSTM R²:", r2_score(y_test, lstm_preds))

# ------------------ Step 10: Feature Importance ------------------

# Random Forest and XGBoost importances
rf_importance = best_rf.feature_importances_
xgb_importance = best_xgb.feature_importances_

# (Assuming you already computed lstm_feature_importances using permutation importance)

plt.figure(figsize=(18, 5))

# Random Forest
plt.subplot(1, 3, 1)
sns.barplot(x=rf_importance, y=features, color='skyblue')
plt.title("Random Forest Feature Importance")

# XGBoost
plt.subplot(1, 3, 2)
sns.barplot(x=xgb_importance, y=features, color='lightgreen')
plt.title("XGBoost Feature Importance")

# LSTM
plt.subplot(1, 3, 3)
sns.barplot(x=lstm_feature_importances, y=features, color='lightcoral')
plt.title("LSTM (Permutation) Feature Importance")

plt.tight_layout()
plt.show()

# ------------------ Step 10B: Combined Feature Importance Comparison Plot ------------------

importance_df = pd.DataFrame({
    'Feature': features,
    'Random Forest': rf_importance,
    'XGBoost': xgb_importance,
    'LSTM (Permutation)': lstm_feature_importances
})

importance_melted = importance_df.melt(id_vars='Feature',
                                       value_vars=['Random Forest', 'XGBoost', 'LSTM (Permutation)'],
                                       var_name='Model', value_name='Importance')

plt.figure(figsize=(14, 8))
sns.barplot(data=importance_melted, x='Importance', y='Feature', hue='Model')
plt.title("🔍 Feature Importance Comparison Across Models", fontsize=16, fontweight='bold')
plt.xlabel("Importance Score", fontsize=14)
plt.ylabel("Feature", fontsize=14)
plt.legend(title="Model", fontsize=12, title_fontsize=13)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# ------------------ Step 10C: Top 5 Important Features Table ------------------

top_rf = importance_df[['Feature', 'Random Forest']].sort_values('Random Forest', ascending=False).head(5)
top_xgb = importance_df[['Feature', 'XGBoost']].sort_values('XGBoost', ascending=False).head(5)
top_lstm = importance_df[['Feature', 'LSTM (Permutation)']].sort_values('LSTM (Permutation)', ascending=False).head(5)

print("\n Top 5 Important Features - Random Forest:")
print(top_rf)

print("\n Top 5 Important Features - XGBoost:")
print(top_xgb)

print("\n Top 5 Important Features - LSTM (Permutation Importance):")
print(top_lstm)

# ------------------ Step 11: Prediction Visualization ------------------

plt.figure(figsize=(10, 6))
plt.plot(y_test.values[:100], label='Actual', linewidth=2)
plt.plot(rf_preds[:100], label='Random Forest', linestyle='--')
plt.plot(xgb_preds[:100], label='XGBoost', linestyle='-.')
plt.plot(lstm_preds[:100], label='LSTM', linestyle=':')

plt.title("Actual vs Predicted Ticket Prices (First 100 Samples)", fontsize=16, fontweight='bold')
plt.xlabel("Sample Number", fontsize=14)
plt.ylabel("Ticket Price (£)", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

# Save the plot
plt.savefig("prediction_comparison_plot.png", dpi=300)

# Show the plot
plt.show()

# ------------------ Step 12: Improved Dynamic Pricing Simulation (Fixed & Upgraded) ------------------

# 1. Redefine X and y for seats prediction
features_seats = ['Booking_Lead_Time', 'Fare_Class', 'Seats_Available',
                  'Competitor_Price', 'Fuel_Price', 'Public_Holiday', 'Weather_Impact']

X_seats = df[features_seats]
y_seats = df['Seats_Booked']

# 2. Train-test split and scale
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_seats, y_seats, test_size=0.2, random_state=42)

scaler_seats = StandardScaler()
X_train_scaled_s = scaler_seats.fit_transform(X_train_s)
X_test_scaled_s = scaler_seats.transform(X_test_s)

# 3. Train model to predict seats booked
xgb_seat_model = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
xgb_seat_model.fit(X_train_scaled_s, y_train_s)

# 4. Predict seats
predicted_seats = np.maximum(xgb_seat_model.predict(X_test_scaled_s), 0)  # Avoid negative seats

# 5. Get corresponding ticket prices
ticket_prices = df.loc[X_test_s.index, 'Ticket_Price'].reset_index(drop=True)

# 6. Calculate simulated revenue
revenue_df = pd.DataFrame({
    'Predicted_Seats_Booked': predicted_seats,
    'Ticket_Price': ticket_prices
})
revenue_df['Predicted_Revenue'] = revenue_df['Predicted_Seats_Booked'] * revenue_df['Ticket_Price']
total_revenue = revenue_df['Predicted_Revenue'].sum()

print(f"\n Corrected Simulated Total Revenue: £{total_revenue:,.2f}")

# Optional: Top 5 revenue samples
print("\nTop 5 Revenue Samples:")
print(revenue_df.sort_values('Predicted_Revenue', ascending=False).head())

# ---  First Plot: Distribution of Predicted Revenues ---
plt.figure(figsize=(10, 6))
sns.histplot(revenue_df['Predicted_Revenue'], bins=50, kde=True)
plt.title("Distribution of Simulated Revenues per Flight", fontsize=16, fontweight='bold')
plt.xlabel("Revenue (£)", fontsize=14)
plt.ylabel("Count", fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.show()

# ---  Second Plot: Monthly Revenue Bar Chart (Upgraded) ---

# Attach Flight Dates
flight_dates = df.loc[X_test_s.index, 'Flight_Date'].reset_index(drop=True)

# Add Flight_Date to revenue_df
revenue_df['Flight_Date'] = pd.to_datetime(flight_dates, dayfirst=True)

# Extract Month Names
revenue_df['Month'] = revenue_df['Flight_Date'].dt.strftime('%b')  # 'Jan', 'Feb', etc.
revenue_df['Month_Num'] = revenue_df['Flight_Date'].dt.month       # For sorting

# Group by Month
monthly_revenue = revenue_df.groupby(['Month_Num', 'Month'])['Predicted_Revenue'].sum().reset_index()
monthly_revenue = monthly_revenue.sort_values('Month_Num')

# Plot Monthly Revenue
plt.figure(figsize=(12, 6))
bars = plt.bar(monthly_revenue['Month'], monthly_revenue['Predicted_Revenue'], color=sns.color_palette("pastel"))

# Add value labels on top
for bar in bars:
    height = bar.get_height()
    plt.annotate(f'£{height/1e6:.1f}M',  # Shows in Millions
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 5),  # 5 points vertical offset
                 textcoords="offset points",
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

# Final Touches
plt.title('Monthly Predicted Revenue in 2022', fontsize=18, fontweight='bold')
plt.xlabel('Month', fontsize=14)
plt.ylabel('Revenue (£)', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Save the plot
plt.savefig('monthly_predicted_revenue_upgraded.png', dpi=300)

# Show plot
plt.show()


# Show descriptive statistics
revenue_stats = revenue_df['Predicted_Revenue'].describe()
print(revenue_stats)


# ------------------  Step 12 Completed ------------------


# ------------------ Step 13: ANOVA Test  ------------------

from sklearn.metrics import r2_score

# --- Part 1: ANOVA Test on Residuals ---
rf_residuals = y_test.values - rf_preds
xgb_residuals = y_test.values - xgb_preds
lstm_residuals = y_test.values - lstm_preds

f_stat_res, p_val_res = f_oneway(rf_residuals, xgb_residuals, lstm_residuals)
print("\nANOVA Test on Residuals:")
print(f"F-statistic: {f_stat_res:.4f}")
print(f"P-value: {p_val_res:.4f}")
if p_val_res < 0.05:
    print(" There is a statistically significant difference between model residuals.")
else:
    print(" No statistically significant difference found between model residuals.")

plt.figure(figsize=(8, 6))
plt.boxplot([rf_residuals, xgb_residuals, lstm_residuals], labels=['Random Forest', 'XGBoost', 'LSTM'])
plt.title("Model Residuals Comparison")
plt.ylabel("Residuals (Actual - Predicted)")
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Part 2: ANOVA Test on R² Values per batch ---

# Divide test set into batches
batch_size = 100
n_batches = len(y_test) // batch_size

rf_r2_batches = []
xgb_r2_batches = []
lstm_r2_batches = []

for i in range(n_batches):
    start = i * batch_size
    end = start + batch_size
    
    rf_r2_batches.append(r2_score(y_test.values[start:end], rf_preds[start:end]))
    xgb_r2_batches.append(r2_score(y_test.values[start:end], xgb_preds[start:end]))
    lstm_r2_batches.append(r2_score(y_test.values[start:end], lstm_preds[start:end]))

f_stat_r2, p_val_r2 = f_oneway(rf_r2_batches, xgb_r2_batches, lstm_r2_batches)
print("\nANOVA Test on R² Values:")
print(f"F-statistic: {f_stat_r2:.4f}")
print(f"P-value: {p_val_r2:.4f}")
if p_val_r2 < 0.05:
    print(" Statistically significant difference between model R² scores across batches.")
else:
    print(" No statistically significant difference found between model R² scores across batches.")

plt.figure(figsize=(8, 6))
plt.boxplot([rf_r2_batches, xgb_r2_batches, lstm_r2_batches], labels=['Random Forest', 'XGBoost', 'LSTM'])
plt.title("Model R² Scores Across Test Batches")
plt.ylabel("R² Score")
plt.grid(True)
plt.tight_layout()
plt.show()



# ------------------ Step 14: Identify Best Model (with RMSE + R2 Score) ------------------

# a. RMSE Comparison
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_preds))
lstm_rmse = np.sqrt(mean_squared_error(y_test, lstm_preds))

# b. R2 Score Comparison
rf_r2 = r2_score(y_test, rf_preds)
xgb_r2 = r2_score(y_test, xgb_preds)
lstm_r2 = r2_score(y_test, lstm_preds)

# c. Print Results
print("\nRMSE Comparison:")
print(f"Random Forest RMSE: {rf_rmse:.2f}")
print(f"XGBoost RMSE: {xgb_rmse:.2f}")
print(f"LSTM RMSE: {lstm_rmse:.2f}")

print("\nR² Score Comparison:")
print(f"Random Forest R²: {rf_r2:.3f}")
print(f"XGBoost R²: {xgb_r2:.3f}")
print(f"LSTM R²: {lstm_r2:.3f}")

# d. Identify Best Model Based on RMSE
best_model_rmse = min({"Random Forest": rf_rmse, "XGBoost": xgb_rmse, "LSTM": lstm_rmse},
                      key=lambda k: {"Random Forest": rf_rmse, "XGBoost": xgb_rmse, "LSTM": lstm_rmse}[k])

# e. Identify Best Model Based on R2
best_model_r2 = max({"Random Forest": rf_r2, "XGBoost": xgb_r2, "LSTM": lstm_r2},
                    key=lambda k: {"Random Forest": rf_r2, "XGBoost": xgb_r2, "LSTM": lstm_r2}[k])

print(f"\n🏆 Best Model Based on RMSE: {best_model_rmse}")
print(f"🏆 Best Model Based on R² Score: {best_model_r2}")

# ------------------ Step 15: QQ Plots ------------------

plt.figure(figsize=(15, 4))
for i, res in enumerate([rf_residuals, xgb_residuals, lstm_residuals]):
    plt.subplot(1, 3, i+1)
    probplot(res, dist="norm", plot=plt)
    plt.title(['Random Forest', 'XGBoost', 'LSTM'][i] + " Residuals QQ Plot")
plt.tight_layout()
plt.show()


# ------------------ Step 16: Create and Display Comparison Table ------------------

# 1. Create dictionary
comparison_data = {
    "Model": ["Random Forest", "XGBoost", "LSTM"],
    "RMSE": [rf_rmse, xgb_rmse, lstm_rmse],
    "R² Score": [rf_r2, xgb_r2, lstm_r2]
}

# 2. Create DataFrame
comparison_df = pd.DataFrame(comparison_data)

# 3. Sort by RMSE (optional, cleaner view)
comparison_df = comparison_df.sort_values(by="RMSE")

# 4. Show Table
print("\n Model Comparison Table:")
print(comparison_df.to_string(index=False))


