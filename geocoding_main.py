import random
import json

def reservoir_sample(file_path, seed, num_lines):
    random.seed(seed)
    sample = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if i < num_lines:
                sample.append(line)
            elif i >= num_lines and random.random() < num_lines / (i+1):
                replace = random.randint(0, len(sample)-1)
                sample[replace] = line
    return sample


def lines_to_json(lines):
    json_array = []
    for line in lines:
        parts = line.split('\t')
        json_obj = {
            'address': " ".join([x for x in [parts[1], parts[13], parts[12], parts[10], parts[8]] if x != '']),
            'coordinates': {
                'latitude': parts[4],
                'longitude': parts[5]
            }
        }
        json_array.append(json_obj)
    return json_array

# Usage
sampled_lines = reservoir_sample('benchmark_dataset/geonames/US.txt', 12345, 10000)
json_array = lines_to_json(sampled_lines)

# Print the JSON array
data = json.dumps(json_array, indent=4)

with open('output.json', 'w') as f:
    f.write(data)

