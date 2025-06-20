import csv
import os

def calculate_average_accuracy(file_path='accuracies.csv'):
    """
    Reads a CSV file to calculate the average accuracy from the second column.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        float: The calculated average accuracy, or None if an error occurs.
    """
    accuracies = []
    
    # Check if the file exists before trying to open it
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' was not found.")
        print("Please ensure you have saved your data to a file with this name in the same directory as the script.")
        return None

    try:
        # Open the CSV file for reading
        with open(file_path, mode='r', newline='', encoding='utf-8') as csvfile:
            # Create a CSV reader object
            csv_reader = csv.reader(csvfile)
            
            # Optional: Skip header row if your CSV has one
            # next(csv_reader, None) 

            # Iterate over each row in the csv file
            for i, row in enumerate(csv_reader):
                try:
                    # The accuracy is in the second column (index 1)
                    if len(row) > 1:
                        accuracy = float(row[1])
                        accuracies.append(accuracy)
                    else:
                        print(f"Warning: Skipping row {i+1} as it does not have an accuracy column.")
                except (ValueError, IndexError) as e:
                    # Handle cases where conversion to float fails or row is malformed
                    print(f"Warning: Could not process row {i+1}. Error: {e}. Row content: {row}")

        # Calculate the average if any accuracies were found
        if accuracies:
            average = sum(accuracies) / len(accuracies)
            return average
        else:
            print("No valid accuracy data found in the file.")
            return None

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

# --- Main execution part of the script ---
if __name__ == "__main__":
    # Define the name of the input file
    import argparse
    parser = argparse.ArgumentParser(description="Calculate average accuracy from a CSV file.")
    parser.add_argument('--input_file', type=str, default='accuracies.csv', help='Path to the input CSV file containing accuracies.')
    args = parser.parse_args()
    input_file = args.input_file
    
    # Calculate the average
    average_accuracy = calculate_average_accuracy(input_file)
    
    # Print the result if the calculation was successful
    if average_accuracy is not None:
        print(f"Successfully read {input_file}.")
        print(f"The average accuracy is: {average_accuracy}")
        print(f"As a percentage: {average_accuracy:.2%}")

