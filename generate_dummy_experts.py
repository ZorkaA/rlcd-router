import os

expert_size = 1409286144
num_experts = 8
num_layers = 56
total_size = expert_size * num_experts * num_layers

dir_path = "/Volumes/SSK SSD/partitioned_experts"
os.makedirs(dir_path, exist_ok=True)
file_path = os.path.join(dir_path, "mixtral_dummy.bin")

print(f"Generating dummy file at {file_path} of size {total_size} bytes...")

chunk_size = 1024 * 1024 * 64
with open(file_path, "wb") as f:
    written = 0
    zero_chunk = bytearray(chunk_size)
    while written < total_size:
        to_write = min(chunk_size, total_size - written)
        f.write(zero_chunk[:to_write])
        written += to_write

print("Dummy file generation complete.")
