import pandas as pd
import matplotlib.pyplot as plt

# Load the datasets
df8 = pd.read_csv("C:\\Users\\steam\\Desktop\\Bolsa\\Opeva\\ev_id_10.csv")  # Office
df9 = pd.read_csv("C:\\Users\\steam\\Desktop\\Bolsa\\Opeva\\ev_id_9.csv")   # Office
df10 = pd.read_csv("C:\\Users\\steam\\Desktop\\Bolsa\\Opeva\\EV12.csv")      # Home
df7 = pd.read_csv("C:\\Users\\steam\\Desktop\\Bolsa\\Opeva\\EV13.csv")       # Home

# Function to count consecutive charging hours
def count_consecutive_charging_hours(df):
    charging_hours = []
    current_count = 0

    for i in range(len(df)):
        required_soc = df['Required Soc At Departure'].iloc[i]

        if pd.notna(required_soc):  # Check if the value is not empty
            current_count += 1  # Increment the count
        else:
            if current_count > 0:
                charging_hours.append(current_count)  # Store the count when charging ends
            current_count = 0  # Reset the current count

    # After the loop, store any remaining count
    if current_count > 0:
        charging_hours.append(current_count)

    return charging_hours

# Get charging hours from each dataset
charging_hours_office = count_consecutive_charging_hours(pd.concat([df8, df9]))  # Combine office datasets
charging_hours_home = count_consecutive_charging_hours(pd.concat([df10, df7]))    # Combine home datasets

# Create a DataFrame to count the occurrences of each unique charging hour for office
charging_hours_office_df = pd.DataFrame(charging_hours_office, columns=['Consecutive Charging Hours'])
count_summary_office = charging_hours_office_df['Consecutive Charging Hours'].value_counts().sort_index()

# Create a DataFrame to count the occurrences of each unique charging hour for home
charging_hours_home_df = pd.DataFrame(charging_hours_home, columns=['Consecutive Charging Hours'])
count_summary_home = charging_hours_home_df['Consecutive Charging Hours'].value_counts().sort_index()

# Plotting
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Office Plot
ax[0].bar(count_summary_office.index, count_summary_office.values, color='skyblue')
ax[0].set_title('Frequency of Consecutive Charging Hours (Office)')
ax[0].set_xlabel('Consecutive Charging Hours')
ax[0].set_ylabel('Count')
ax[0].grid(axis='y', linestyle='--', alpha=0.7)

# Home Plot
ax[1].bar(count_summary_home.index, count_summary_home.values, color='lightgreen')
ax[1].set_title('Frequency of Consecutive Charging Hours (Home)')
ax[1].set_xlabel('Consecutive Charging Hours')
ax[1].set_ylabel('Count')
ax[1].grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()
