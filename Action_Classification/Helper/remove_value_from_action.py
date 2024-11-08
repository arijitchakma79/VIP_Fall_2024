import pandas as pd

# Load the data from the CSV file
file_path = '../Files/combined.csv'  
output_file_path = '../Files/filtered_output.csv' 

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Remove rows where the 'action' column has a value of 3
df_filtered = df[df['action'] != 3]

# Save the filtered DataFrame to a new CSV file
df_filtered.to_csv(output_file_path, index=False)

print(f"Filtered data saved to {output_file_path}")
