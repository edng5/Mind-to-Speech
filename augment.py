import os
import pandas as pd
import random

def augment_data(data_dir, output_dir, num_files=100):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get all CSV files in the data directory
    csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]

    # Check if there are files to process
    if not csv_files:
        raise ValueError("No valid files found in the data directory.")

    # Generate augmented files
    for i in range(1, num_files + 1):
        augmented_data = []
        for source_file in csv_files:
            file_path = os.path.join(data_dir, source_file)
            df = pd.read_csv(file_path)

            # Skip empty DataFrames
            if df.empty:
                print(f"File {source_file} is empty. Skipping...")
                continue

            num_rows = random.randint(5, 20)  # Random number of rows to select
            sampled_rows = df.sample(n=min(num_rows, len(df)), random_state=random.randint(0, 100))
            augmented_data.append(sampled_rows)

        # Combine sampled rows into a single DataFrame
        if augmented_data:
            augmented_df = pd.concat(augmented_data, ignore_index=True)

            # Save the augmented file
            output_file = os.path.join(output_dir, f"augment_{i}.csv")
            augmented_df.to_csv(output_file, index=False)
            print(f"Generated: {output_file}")
        else:
            print(f"No data sampled for augment_{i}. Skipping...")

if __name__ == "__main__":
    # Paths to the data directory and output directory
    data_dir = os.path.abspath("..\\Mind-to-Speech\\data")
    output_dir = os.path.abspath("..\\Mind-to-Speech\\augmented_data")

    # Generate 100 augmented files
    augment_data(data_dir, output_dir, num_files=100)