import os
from datasets import load_from_disk

test_dataset = load_from_disk("./dataset/test")
print(test_dataset[0])
