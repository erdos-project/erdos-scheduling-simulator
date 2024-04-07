import json


def process_json(input_json):
    with open(input_json, 'r') as file:
        data = json.load(file)
    
    processed_data = {}
    
    for key, stages in data.items():
        # Extract query type from the key
        query_type = key.split('_')[1]
        
        # Get max stage number for the query type
        max_stage = max_stage_per_query.get(query_type, 0)
        
        # Sort stages by average_runtime_ms in descending order
        sorted_stages = sorted(stages, key=lambda x: x['average_runtime_ms'], reverse=True)
        
        # Select top k+1 runtimes
        top_stages = sorted_stages[:max_stage+1]
        
        # If the sorted list does not have enough stages, append duplicate stages
        while len(top_stages) < max_stage + 1:
            top_stages.append(top_stages[-1])  # Append the last stage repeatedly
        
        # Create new JSON with stage numbers from 0 to k+1 and corresponding data
        new_json_data = [{"stage_id": i,
                          "stage_name": top_stages[i]['stage_name'],
                          "num_tasks": top_stages[i]['num_tasks'],
                          "average_runtime_ms": top_stages[i]['average_runtime_ms']}
                         for i in range(len(top_stages))]
        
        # Update processed_data with new JSON data
        processed_data[key] = new_json_data
    
    return processed_data


# Define max_stage_per_query mapping
max_stage_per_query = {
    'q1': 2,
    'q2': 16,
    'q3': 4,
    'q4': 4,
    'q5': 12,
    'q6': 1,
    'q7': 11,
    'q8': 16,
    'q9': 12,
    'q10': 7,
    'q11': 6,
    'q12': 4,
    'q13': 5,
    'q14': 2,
    'q15': 5,
    'q16': 7,
    'q17': 3,
    'q18': 5,
    'q19': 2,
    'q20': 8,
    'q21': 13,
    'q22': 4,
}

# Input JSON file path
input_json_file = 'sorted_output.json'

# Process JSON data
processed_json_data = process_json(input_json_file)

# Output JSON file path
output_json_file = 'output_standardized_tpch_profiles.json'

# Write processed JSON data to output file
with open(output_json_file, 'w') as outfile:
    json.dump(processed_json_data, outfile, indent=4)

print(f"Processed data has been written to {output_json_file}")
