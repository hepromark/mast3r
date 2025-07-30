from pathlib import Path
from datetime import datetime
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import time
import argparse
import re

def parse_date_from_folder(folder_name):
    if folder_name.startswith("Week-"):
        parts = folder_name.split('-', 2)
        if len(parts) >= 3:
            folder_name = parts[2]
    parts = folder_name.split('-')
    if len(parts) < 3:
        return None
    year_token = parts[-1]
    if not re.fullmatch(r'\d{4}', year_token):
        return None
    year = year_token
    day = parts[-2]
    month = parts[-3]
    date_str = f"{month}-{day}-{year}"
    for fmt in ("%b-%d-%Y", "%B-%d-%Y"):
        try:
            return datetime.strptime(date_str, fmt)
        except Exception:
            continue
    return None

def compute_density(file_path):
    folder_name = Path(file_path).parents[2].name if len(Path(file_path).parents) >= 3 else Path(file_path).parent.name
    print(f'{file_path} started')

    print("Getting points")
    mesh = o3d.io.read_triangle_mesh(file_path)
    tree = o3d.geometry.KDTreeFlann(mesh)
    points = np.asarray(mesh.vertices)
    print(points.shape)

    print("Computing normal and rendering it.")
    mesh.compute_vertex_normals()
    print(np.asarray(mesh.triangle_normals))
    o3d.visualization.draw_geometries([mesh])

    N = points.shape[0]
    radii = [0.01, 0.05, 0.1, 0.5, 1.0]
    r2_thresholds = [r * r for r in radii]
    sum_counts = np.zeros(len(radii), dtype=np.int64)

    print("Computing density")
    for i in range(N):
        [k, idx, dist2] = tree.search_radius_vector_3d(points[i], radii[-1])
        if k <= 0:
            continue
        dist_array = np.asarray(dist2)
        counts = [np.count_nonzero(dist_array <= r2) - 1 for r2 in r2_thresholds]
        sum_counts += counts
    
    print("Computing average")
    avg_counts = (sum_counts.astype(float) / N).tolist()
    return {"folder": folder_name, "densities": avg_counts}

def log(tag, message, logfile=None):
    line = f"[{tag}] {message}"
    tqdm.write(line)
    if logfile:
        logfile.write(line + "\n")
        logfile.flush()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str, help="Path to the point cloud file (.glb)")
    parser.add_argument("--log", action="store_true", help="Log output to log.txt")
    args = parser.parse_args()

    log_file = open("log.txt", "w") if args.log else None

    file_path = Path(args.file)
    log("INFO", f"Processing file: {file_path}", log_file)

    res = compute_density(str(file_path))
    print(res)

    if res.get("error"):
        log("ERROR", f"{res['folder']}: {res['error']}", log_file)
        if log_file:
            log_file.close()
        return

    print("Plotting")

    radii = [0.01, 0.05, 0.1, 0.5, 1.0]
    plt.figure(figsize=(10, 8))
    plt.plot(radii, res["densities"], label=res["folder"])
    plt.xlabel("Radius")
    plt.ylabel("Mean Density")
    plt.title("Point Cloud Density vs Radius")
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig("point_cloud_density.png", dpi=300)
    log("INFO", "Saved plot to point_cloud_density.png", log_file)

if __name__ == "__main__":
    main()

