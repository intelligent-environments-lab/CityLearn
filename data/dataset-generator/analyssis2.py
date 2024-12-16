import pandas as pd
import matplotlib.pyplot as plt

# Load the datasets
df8 = pd.read_csv("C:\\Users\\steam\\Desktop\\Bolsa\\Opeva\\ev_id_10.csv")  # Office
df9 = pd.read_csv("C:\\Users\\steam\\Desktop\\Bolsa\\Opeva\\ev_id_9.csv")  # Office
df10 = pd.read_csv("C:\\Users\\steam\\Desktop\\Bolsa\\Opeva\\EV12.csv")  # Home
df7 = pd.read_csv("C:\\Users\\steam\\Desktop\\Bolsa\\Opeva\\EV13.csv")    # Home

# Add a 'Car ID' column to each DataFrame
df8['Car ID'] = 'Car 1 Office'
df9['Car ID'] = 'Car 2 Office'
df10['Car ID'] = 'Car 3 Home'
df7['Car ID'] = 'Car 4 Home'

# Function to count consecutive chargers
def count_consecutive_chargers(df):
    charger_counts = []
    current_count = 0
    for index, row in df.iterrows():
        if pd.notna(row[' Estimated SOC at Arrival']) and row[' Estimated SOC at Arrival'] != "":
            current_count += 1
        else:
            if current_count > 0:
                if current_count > 1:
                    current_count -= 1
                charger_counts.append(current_count)
                current_count = 0
    if current_count > 0:
        charger_counts.append(current_count)
    return charger_counts

# Get consecutive charger counts for office DataFrames
counts8_office = count_consecutive_chargers(df8)
counts9_office = count_consecutive_chargers(df9)

# Get consecutive charger counts for home DataFrames
counts10_home = count_consecutive_chargers(df10)
counts7_home = count_consecutive_chargers(df7)

# Convert counts to DataFrames for easy handling
counts_df_office8 = pd.DataFrame(counts8_office, columns=['Consecutive Charger Count'])
counts_df_office9 = pd.DataFrame(counts9_office, columns=['Consecutive Charger Count'])

counts_df_home10 = pd.DataFrame(counts10_home, columns=['Consecutive Charger Count'])
counts_df_home7 = pd.DataFrame(counts7_home, columns=['Consecutive Charger Count'])

# Count the frequency of each consecutive charger count for office
frequency_office8 = counts_df_office8['Consecutive Charger Count'].value_counts().sort_index()
frequency_office9 = counts_df_office9['Consecutive Charger Count'].value_counts().sort_index()

# Count the frequency of each consecutive charger count for home
frequency_home10 = counts_df_home10['Consecutive Charger Count'].value_counts().sort_index()
frequency_home7 = counts_df_home7['Consecutive Charger Count'].value_counts().sort_index()

# Convert frequency to percentage
def to_percentage(frequency):
    total_count = frequency.sum()
    return (frequency / total_count) * 100

# Convert the frequencies to percentages
percentage_office8 = to_percentage(frequency_office8)
percentage_office9 = to_percentage(frequency_office9)
percentage_home10 = to_percentage(frequency_home10)
percentage_home7 = to_percentage(frequency_home7)

# Create a figure for office cars
fig_office, axs_office = plt.subplots(1, 2, figsize=(14, 7))  # Larger figure for office
fig_office.suptitle('Percentage of Consecutive Charger Counts (Office)', fontsize=16)

# Plotting Office Cars (Car 1 and Car 2)
axs_office[0].plot(percentage_office8.index, percentage_office8.values, marker='o', color='skyblue')
axs_office[0].set_title('Car 1 (Office)')
axs_office[0].set_xlabel('Trip Time')
axs_office[0].set_ylabel('Percentage')
axs_office[0].set_xscale('linear')  # Linear x-scale
axs_office[0].grid(True)

axs_office[1].plot(percentage_office9.index, percentage_office9.values, marker='o', color='lightgreen')
axs_office[1].set_title('Car 2 (Office)')
axs_office[1].set_xlabel('Trip Time')
axs_office[1].set_ylabel('Percentage')
axs_office[1].set_xscale('linear')  # Linear x-scale
axs_office[1].grid(True)

# Adjust layout to prevent clipping of labels
fig_office.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for the main title
plt.show()

# Create a figure for home cars
fig_home, axs_home = plt.subplots(1, 2, figsize=(14, 7))  # Larger figure for home
fig_home.suptitle('Percentage of Consecutive Charger Counts (Home)', fontsize=16)

# Plotting Home Cars (Car 3 and Car 4)
axs_home[0].plot(percentage_home10.index, percentage_home10.values, marker='o', color='salmon')
axs_home[0].set_title('Car 3 (Home)')
axs_home[0].set_xlabel('Trip Time')
axs_home[0].set_ylabel('Percentage')
axs_home[0].set_xscale('linear')  # Linear x-scale
axs_home[0].grid(True)

axs_home[1].plot(percentage_home7.index, percentage_home7.values, marker='o', color='violet')
axs_home[1].set_title('Car 4 (Home)')
axs_home[1].set_xlabel('Trip Time')
axs_home[1].set_ylabel('Percentage')
axs_home[1].set_xscale('linear')  # Linear x-scale
axs_home[1].grid(True)

# Adjust layout to prevent clipping of labels
fig_home.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for the main title
plt.show()
