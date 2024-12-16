import pandas as pd
import matplotlib.pyplot as plt

# Load CSV files into DataFrames
dfs_1 = [
    pd.read_csv("C:\\Users\\steam\\Desktop\\Bolsa\\Opeva\\EV11.csv"),
    pd.read_csv("C:\\Users\\steam\\Desktop\\Bolsa\\Opeva\\EV12.csv"),
    pd.read_csv("C:\\Users\\steam\\Desktop\\Bolsa\\Opeva\\EV13.csv"),
]

dfs_2 = [
    pd.read_csv("C:\\Users\\steam\\Desktop\\Bolsa\\Opeva\\ev_id_8.csv"),
    pd.read_csv("C:\\Users\\steam\\Desktop\\Bolsa\\Opeva\\ev_id_9.csv"),
    pd.read_csv("C:\\Users\\steam\\Desktop\\Bolsa\\Opeva\\ev_id_10.csv")
]

# Filter the datasets: one specific week (Day Type: 1-5) and one specific weekend (Day Type: 6 or 7)
def filter_one_week(df):
    return df[df['Day Type'].isin([1, 2, 3, 4, 5])].iloc[:5*24]  # Limit to the first week (5 days)

def filter_one_weekend(df):
    return df[df['Day Type'].isin([6, 7])].iloc[:2*24]  # Limit to the first weekend (2 days)

# Create two separate sets for one week and one weekend
dfs_week_1 = [filter_one_week(df) for df in dfs_1]
dfs_week_2 = [filter_one_week(df) for df in dfs_2]

dfs_weekend_1 = [filter_one_weekend(df) for df in dfs_1]
dfs_weekend_2 = [filter_one_weekend(df) for df in dfs_2]

# Function to process and plot data
def plot_charging_data(dfs_1, dfs_2, title):
    counts_1 = []
    counts_2 = []

    for df in dfs_1:
        df_counts = df.groupby('Hour')['Required Soc At Departure'].apply(lambda x: x.notna().sum())
        counts_1.append(df_counts)

    for df in dfs_2:
        df_counts = df.groupby('Hour')['Required Soc At Departure'].apply(lambda x: x.notna().sum())
        counts_2.append(df_counts)

    # Ensure counts are aligned across all datasets (fill missing hours with 0)
    counts_1 = [df_count.reindex(range(23), fill_value=0) for df_count in counts_1]
    counts_2 = [df_count.reindex(range(23), fill_value=0) for df_count in counts_2]

    # Plot data
    hours = range(23)
    blue_shades = ['#1f77b4', '#4a90d9', '#5b9bd5']
    green_shades = ['#2ca02c', '#3dbb55', '#40bf40']

    plt.plot(hours, counts_1[0], color=blue_shades[0], label='Home Car 1', marker='o')
    plt.plot(hours, counts_1[1], color=blue_shades[1], label='Home Car 2', marker='o')
    plt.plot(hours, counts_1[2], color=blue_shades[2], label='Home Car 3', marker='o')

    plt.plot(hours, counts_2[0], color=green_shades[0], label='Office Car 1', marker='o')
    plt.plot(hours, counts_2[1], color=green_shades[1], label='Office Car 2', marker='o')
    plt.plot(hours, counts_2[2], color=green_shades[2], label='Office Car 3', marker='o')

    plt.fill_between(hours, counts_1[0], color=blue_shades[0], alpha=0.3)
    plt.fill_between(hours, counts_1[1], color=blue_shades[1], alpha=0.3)
    plt.fill_between(hours, counts_1[2], color=blue_shades[2], alpha=0.3)

    plt.fill_between(hours, counts_2[0], color=green_shades[0], alpha=0.3)
    plt.fill_between(hours, counts_2[1], color=green_shades[1], alpha=0.3)
    plt.fill_between(hours, counts_2[2], color=green_shades[2], alpha=0.3)

    plt.ylabel('Charging')
    plt.xlabel('Hour of the Day')
    plt.title(title)
    plt.xticks(ticks=range(0, 23, 2))
    plt.legend(loc='upper right')
    plt.show()

# Plot for one week (weekdays)
plot_charging_data(dfs_week_1, dfs_week_2, 'One Week of Charging Routines Office/Home')

# Plot for one weekend
plot_charging_data(dfs_weekend_1, dfs_weekend_2, 'One Weekend of Charging Routines Office/Home')
