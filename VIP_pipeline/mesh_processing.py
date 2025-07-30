import trimesh
import numpy as np
import matplotlib.pyplot as plt

def process_mesh(mesh_path):
    # Load the GLB file using trimesh
    scene = trimesh.load(mesh_path)
    
    # Extract all geometries from the scene
    geometries = []
    for name, geometry in scene.geometry.items():
        if isinstance(geometry, trimesh.PointCloud):
            print(f"Found point cloud: {name}")
            print(f"Points: {len(geometry.vertices)}")
            print(f"Colors: {geometry.colors.shape if geometry.colors is not None else 'None'}")
            geometries.append(geometry)
        elif isinstance(geometry, trimesh.Trimesh):
            continue
    
    return geometries

def convert_to_ply(pcd, filename):
    pcd.export(filename)
    print(f"Exported PCL to {filename}")


def compute_density_from_pointcloud(pcd, radii=[0.01, 0.05, 0.1, 0.5, 1.0]):
    """Compute density for a trimesh PointCloud"""
    points = np.array(pcd.vertices)
    N = len(points)
    print(points.shape)
    print(points[0].shape)

    # Build KD-tree for fast nearest neighbor search
    tree = pcd.kdtree
    sum_counts = np.zeros(len(radii), dtype=np.int64)

    for j, r in enumerate(radii):
        print(f"Working on radius {r}")
        sample_indices = np.random.choice(N, size=min(N, 10000), replace=False)
        sum_counts[j] += np.sum(tree.query_ball_point(points[sample_indices], r, workers=-1, return_length=True))
    
    avg_counts = [count / N for count in sum_counts]
    print(f'Avg count is: {avg_counts}')

    density = [avg_counts[i] / radii[i] for i in range(len(radii))]
    print(f'Density is: {density}')

    return density

def main():
    mesh_path = "/home/mark.do/mast3r/reconstruction.glb"
    
    # Extract point clouds from GLB
    geometries = process_mesh(mesh_path)

    convert_to_ply(geometries[0], "pointcloud.ply")
    
    if not geometries:
        print("No point clouds found in the GLB file")
        return
    
    # Compute density for each point cloud
    radii = [0.01, 0.05, 0.1, 0.5, 1.0]
    
    for i, pcd in enumerate(geometries):
        print(f"\nProcessing point cloud {i+1}:")
        densities = compute_density_from_pointcloud(pcd, radii)
        
        
        print(f"Densities: {densities}")

if __name__ == "__main__":
    main()