# Function to load data from train.csv
def load_data_from_train_file(file_path):
    file_paths = []
    labels = []
    with open(file_path, 'r') as f:
        next(f)
        for line in f:
            # Assuming each line in train.csv is in format: file_path,label
            parts = line.strip().split(',')
            if len(parts) == 2:
                file_paths.append('dataset/train/' + parts[0])
                labels.append(int(parts[1]))
    
    return file_paths, labels

# Function to load data from val.csv
def load_data_from_val_file(file_path):
    file_paths = []
    labels = []
    
    with open(file_path, 'r') as f:
        next(f)
        for line in f:
            # Assuming each line in val.csv is in format: file_path,label
            parts = line.strip().split(',')
            if len(parts) == 2:
                file_paths.append('dataset/val/' + parts[0])
                labels.append(int(parts[1]))
    
    return file_paths, labels

# Function to load data from test.csv
# Note: Test data usually doesn't have labels, so we just load the paths
def load_data_from_test_file(file_path):
    file_paths = []
    
    with open(file_path, 'r') as f:
        next(f)  # Skip header
        for line in f:
            if line:
                file_paths.append('dataset/test/' + line.strip())
    print('Test paths loaded')
    # print(file_paths)
    return file_paths