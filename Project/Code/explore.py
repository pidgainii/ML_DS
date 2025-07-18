import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('ev_charging_patterns.csv')

plt.figure(figsize=(8, 6))
plt.scatter(df['Battery Capacity (kWh)'], df['Charging Duration (hours)'], alpha=0.5, color='b')
plt.xlabel('Battery Capacity (kWh)')
plt.ylabel('Charging Duration (hours)')
plt.title('Scatter Plot of Charging Duration vs Battery Capacity')
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(df['Energy Consumed (kWh)'], df['Charging Duration (hours)'], alpha=0.5, color='b')
plt.xlabel('Energy Consumed (kWh)')
plt.ylabel('Charging Duration (hours)')
plt.title('Scatter Plot of Charging Duration vs Energy Consumed')
plt.grid(True)
plt.show()

avg_charging_duration_by_user = df.groupby('Vehicle Model')['Charging Duration (hours)'].mean()

plt.figure(figsize=(8, 6))
avg_charging_duration_by_user.plot(kind='bar', color='g', alpha=0.7)
plt.xlabel('Vehicle Model')
plt.ylabel('Average Charging Duration (hours)')
plt.title('Average Charging Duration by Type of User')
plt.xticks(rotation=35)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


plt.figure(figsize=(8, 6))
plt.scatter(df['State of Charge (End %)'], df['Charging Duration (hours)'], alpha=0.5, color='m')
plt.xlabel('State of Charge (End %)')
plt.ylabel('Charging Duration (hours)')
plt.title('Scatter Plot of Charging Duration vs State of Charge (End %)')
plt.grid(True)
plt.show()
