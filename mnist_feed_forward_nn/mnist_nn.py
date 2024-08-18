from datasets import load_dataset

ds = load_dataset("ylecun/mnist")
print(ds.size_in_bytes)