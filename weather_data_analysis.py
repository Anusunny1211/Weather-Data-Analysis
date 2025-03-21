import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Step 1: Load the Data
df = pd.read_csv(r'C:\Users\hp\Downloads\basic\basic\weather.csv')

# Step 2: Handle Missing Values (Removed 'Date' since it's not present)
df.dropna(subset=['MinTemp', 'MaxTemp', 'Rainfall'], inplace=True)

# Step 3: Data Exploration
print("Available columns:", df.columns)
print(df.head())
print(df.info())
print(df.describe())

# Step 4: Data Visualization (Pair Plot)
sns.pairplot(df[['MinTemp', 'MaxTemp', 'Rainfall']])
plt.show()

# Step 5: Data Analysis (Without 'Date' Column)
# Instead of monthly trends, calculate overall averages
avg_max_temp = df['MaxTemp'].mean()
avg_rainfall = df['Rainfall'].mean()

print(f"Average Max Temperature: {avg_max_temp:.2f}°C")
print(f"Average Rainfall: {avg_rainfall:.2f} mm")

# Step 6: Data Visualization (Trends of Max Temperature & Rainfall)
plt.figure(figsize=(10, 5))
plt.hist(df['MaxTemp'], bins=20, alpha=0.7, label='Max Temperature', color='blue')
plt.hist(df['Rainfall'], bins=20, alpha=0.7, label='Rainfall', color='red')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Distribution of Max Temperature & Rainfall')
plt.legend()
plt.grid(True)
plt.show()

# Step 7: Advanced Analysis (Predict Rainfall)
X = df[['MinTemp', 'MaxTemp']]
y = df['Rainfall']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict & Evaluate Model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error for Rainfall Prediction: {mse:.4f}')

# Step 8: Insights - Identify Highest & Lowest Rainfall Days
highest_rainfall = df['Rainfall'].max()
lowest_rainfall = df['Rainfall'].min()

print(f'Highest Rainfall Recorded: {highest_rainfall:.2f} mm')
print(f'Lowest Rainfall Recorded: {lowest_rainfall:.2f} mm')

# Step 9: Save Results to File
with open("results.txt", "w") as f:
    f.write(f"Average Max Temperature: {avg_max_temp:.2f}°C\n")
    f.write(f"Average Rainfall: {avg_rainfall:.2f} mm\n")
    f.write(f"Mean Squared Error for Rainfall Prediction: {mse:.4f}\n")
    f.write(f"Highest Rainfall Recorded: {highest_rainfall:.2f} mm\n")
    f.write(f"Lowest Rainfall Recorded: {lowest_rainfall:.2f} mm\n")

print("Results saved to results.txt")
