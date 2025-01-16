import numpy as np
import struct

num_points = 13414
points = np.random.uniform(-1, 1, (num_points, 3))  # 隨機生成 (x, y, z)
normals = np.random.uniform(-1, 1, (num_points, 3))  # 隨機生成法向量 (nx, ny, nz)
colors = np.random.randint(0, 256, (num_points, 3), dtype=np.uint8)  # 隨機生成顏色 (red, green, blue)

with open("random_points3D.ply", "wb") as f:
    f.write(b"ply\n")
    f.write(b"format binary_little_endian 1.0\n")
    f.write(f"element vertex {num_points}\n".encode())
    f.write(b"property float x\n")
    f.write(b"property float y\n")
    f.write(b"property float z\n")
    f.write(b"property float nx\n")
    f.write(b"property float ny\n")
    f.write(b"property float nz\n")
    f.write(b"property uchar red\n")
    f.write(b"property uchar green\n")
    f.write(b"property uchar blue\n")
    f.write(b"end_header\n")
    
    # 寫入二進制數據
    for i in range(num_points):
        # x, y, z, nx, ny, nz -> float32
        f.write(struct.pack("<6f", *points[i], *normals[i]))
        # r, g, b -> uint8
        f.write(struct.pack("<3B", *colors[i]))