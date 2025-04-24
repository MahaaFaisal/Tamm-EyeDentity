import csv
import os

def read_data(input_file):
    modified_rows = []
    id_to_filename_map = {}
    
    # Read the input CSV file
    with open(input_file, 'r', newline='') as csv_in:
        reader = csv.reader(csv_in)
        
        # Get the header row
        header = next(reader)
        modified_rows.append(header)
        
        # Find the indices of the id and filename columns
        try:
            id_index = header.index('id')
            filename_index = header.index('file_name')
        except ValueError as e:
            print(f"Error: CSV file doesn't have required columns. {e}")
            return {}
        
        # Process each row
        for row in reader:
            if len(row) > max(id_index, filename_index):
                # Get the ID
                file_id = row[id_index]
                
                # Add 'dataset/test/' prefix to the filename
                modified_filename = f"../task3_dataset/test/{row[filename_index]}"
                row[filename_index] = modified_filename
                
                # Add to our ID-to-filename mapping
                id_to_filename_map[file_id] = modified_filename
            modified_rows.append(row)
    return id_to_filename_map

def write_predictions_to_csv(id_to_prediction, output_file):
    try:
        # Create the output directory if it doesn't exist
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Write the data to the CSV file
        with open(output_file, 'w', newline='') as csv_out:
            writer = csv.writer(csv_out)
            
            # Write the header row
            writer.writerow(['id', 'prediction'])
            
            # Write each ID-prediction pair
            for file_id, prediction in id_to_prediction.items():
                writer.writerow([file_id, prediction])
        
        print(f"Predictions successfully written to {output_file}")
        return True
    
    except Exception as e:
        print(f"Error writing predictions to CSV: {e}")
        return False
