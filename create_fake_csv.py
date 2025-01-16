import csv
import os

def create_fake_csv(firstname, lastname, recipients):
    # Create the filename using the format "Firstname_Lastname.csv"
    filename = f"{firstname}_{lastname}.csv"
    
    # Define the path where the CSV file will be saved
    csv_directory = 'csv_files'  # This will create the csv_files directory in the current path
    os.makedirs(csv_directory, exist_ok=True)
    filepath = os.path.join(csv_directory, filename)
    
    # Write the recipients to the CSV file
    with open(filepath, mode='w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        
        # Write each recipient name as a new row
        for recipient in recipients:
            csv_writer.writerow([recipient])
    
    print(f"CSV file '{filename}' created with {len(recipients)} recipients.")

# Example usage with fake data
create_fake_csv("John", "Doe", ["Alice Johnson", "Bob Smith", "Charlie Brown", "Diana Prince"])
create_fake_csv("Jane", "Smith", ["Eve Adams", "Frank Martin", "Grace Kelly", "Hank Pym"])