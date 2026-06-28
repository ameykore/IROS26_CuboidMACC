import os
import numpy as np

folders = [
    r"C:\Users\biorobotics\Desktop\MACC\cuboidMACC-DARS24\updated_src\data\random",
    r"C:\Users\biorobotics\Desktop\MACC\cuboidMACC-DARS24\updated_src\data\random_10x10x4_15blocks"
]

cube_only = 0
has_cuboid = 0
total = 0

for folder in folders:
    for fn in os.listdir(folder):
        if not fn.endswith(".npy"):
            continue

        state = np.load(os.path.join(folder, fn), allow_pickle=True)
        lengths = [int(b[1]) for b in state]

        total += 1

        if all(l == 1 for l in lengths):
            cube_only += 1
        else:
            has_cuboid += 1

print("Total structures:", total)
print("Cube-only:", cube_only)
print("Contain cuboids:", has_cuboid)