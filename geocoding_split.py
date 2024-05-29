import random

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


import csv

def lines_to_csv(lines):
    csv_array = []
    for line in lines:
        parts = line.split('\t')
        csv_obj = [
            " ".join([x for x in [parts[1], parts[13], parts[12], parts[10], parts[8]] if x != '']),
            parts[4],
            parts[5]
        ]
        csv_array.append(csv_obj)
    return csv_array

# Usage
sampled_lines = reservoir_sample('benchmark_dataset/geonames/US.txt', 2234567, 1234567)
csv_array = lines_to_csv(sampled_lines)

# Split the CSV array into chunks
chunk_size = 234560  # Define the size of each chunk
chunks = [csv_array[i:i + chunk_size] for i in range(0, len(csv_array), chunk_size)]

for i, chunk in enumerate(chunks):
    with open(f'output_{i}.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['address', 'latitude', 'longitude'])
        writer.writerows(chunk)
