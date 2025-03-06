import json

def find_best_fall_detection(json_file):
    # Load the JSON data from file
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Find the entry with the maximum fall detection accuracy
    best_entry = max(data, key=lambda x: x.get('fall_accuracy', 0))
    
    # Print the best performing combination in a formatted way
    print("Best combination based on fall detection accuracy:")
    print(json.dumps(best_entry, indent=4))

if __name__ == '__main__':
    # Replace 'gridsearch_results.json' with the path to your JSON file if needed.
    find_best_fall_detection(r"AI_folder(RF)/Model_Result/gridsearch_results.json")
