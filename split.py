import os

file_path = "/Users/dy/Documents/dev/Geocoding/benchmark_dataset/geonames/US.txt"

def split_file(file_path, output_paths, n):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        
    chunk_size = len(lines) // n
    for i in range(n):
        with open(output_paths[i], 'w') as file:
            file.writelines(lines[i*chunk_size:(i+1)*chunk_size])

n = 10
output_paths = [f"output_{i}.txt" for i in range(n)]
split_file(file_path, output_paths, n)
