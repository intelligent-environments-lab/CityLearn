import pandas as pd
import matplotlib.pyplot as plt

# Load CSV files into DataFrames
dfs = [
    pd.read_csv("C:\\Users\\steam\\Desktop\\Bolsa\\Opeva\\EV-11.csv"),
    pd.read_csv("C:\\Users\\steam\\Desktop\\Bolsa\\Opeva\\EV-12.csv"),
]

# List to store the count of destination chargers by hour for each dataset
counts = []

# Process each dataset
for df in dfs:
    df_counts = df.groupby('Time')['State'].apply(lambda x: (x == "Charging").sum())
    counts.append(df_counts)

# Ensure the counts are aligned across all datasets (fill missing hours with 0)
counts = [df_count.reindex(range(23), fill_value=0) for df_count in counts]

# Create the line plot
hours = range(23)

# Plot the lines
plt.plot(hours, counts[0], color='blue', label='Dataset 1', marker='o')
plt.plot(hours, counts[1], color='green', label='Dataset 2', marker='o')

# Fill the area under the lines
plt.fill_between(hours, counts[0], color='blue', alpha=0.3)
plt.fill_between(hours, counts[1], color='green', alpha=0.3)

# Add labels and legends
plt.ylabel('Count of Charges')
plt.xlabel('Hour of the Day')
plt.title('Charging Across Hours')

# Set x-axis ticks to show every 3 hours
plt.xticks(ticks=range(0, 24, 3))  # 0 to 22 in steps of 3

# Move the legend to the top right
plt.legend(loc='upper right')

# Show plot
plt.show()
