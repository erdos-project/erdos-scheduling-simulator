import json


def extract_info(key):
    # Split the key into parts
    parts = key.split('_')

    # Extract query number, dataset size, and max cores
    query_number = int(parts[1][1:])  # Extracting the number after 'q'
    dataset_size = int(parts[2].replace('gb', ''))
    max_cores = int(parts[4])

    return query_number, dataset_size, max_cores


def sort_keys(keys):
    # Sort keys based on extracted information
    return sorted(keys, key=lambda x: extract_info(x))


def sort_json_file(input_file, output_file):
    # Load JSON data from input file
    with open(input_file, 'r') as f:
        data = json.load(f)

    # Sort keys
    sorted_keys = sort_keys(data.keys())

    # Create a new dictionary with sorted keys
    sorted_data = {key: data[key] for key in sorted_keys}

    # Write sorted data to output file
    with open(output_file, 'w') as f:
        json.dump(sorted_data, f, indent=4)


# Example usage
input_file = 'tpch_profiles_updated_4apr24_duplicate.json'
output_file = 'sorted_output.json'
sort_json_file(input_file, output_file)
