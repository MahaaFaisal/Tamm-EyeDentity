import threading
import json
from agents import assessor_agent, qaqc_agent
from data_pipeline import read_data, write_predictions_to_csv
def agentic_flow(image_path):
    predictions = []
	## same agent but with diffrent temperature
    for temperature in [0.0, 0.5, 1]:
        prediction = assessor_agent(image_path, temperature)
        if prediction:
            predictions.append(prediction)

    predictions = [f'"{pred}"' for pred in predictions]
    predictions = "[" + ",".join(predictions) + "]"

    final_prediction = qaqc_agent(image_path, prediction)
    return final_prediction

def process_chunk(chunk, id_to_prediction, lock):
    global processed_counter, total_files
    for file_id, filename in chunk:
        prediction = agentic_flow(filename)
        
        with lock:
            id_to_prediction[file_id] = prediction
            processed_counter += 1
            print(f"\rProcessed {processed_counter}/{total_files} files", end='', flush=True)

if __name__ == "__main__":
    input_csv = "../task3_dataset/test.csv"
	# Read the CSV file and get the ID-to-filename mapping
    id_to_prediction = {}
    id_to_filename = read_data(input_csv)

    global processed_counter, total_files
    processed_counter = 0
    total_files = len(id_to_filename)

	# Multi-threading setup
    lock = threading.Lock()
    
    # Convert items to list and split into chunks of 34
    items = list(id_to_filename.items())
    chunks = [items[i*34:(i+1)*34] for i in range(11)]
    total_files = sum(len(chunk) for chunk in chunks)
    print(f"Total files in chunks: {total_files}")
    
    # Start threads
    threads = []
    for chunk in chunks:
        thread = threading.Thread(
            target=process_chunk,
            args=(chunk, id_to_prediction, lock)
        )
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    write_predictions_to_csv(id_to_prediction, "../task3_dataset/predictions.csv")