from pathlib import Path

import numpy as np
import pycolmap

class BinaryFileReader():
    def __init__(self) -> None:
        self.recon = pycolmap.Reconstruction()
    
    def convert_binary_to_text(self, input_dir, output_dir):
        self.recon.read_binary(input_dir)
        self.recon.write_text(output_dir)

    def read_binary_camera_matrices(self, input_dir, output_dir):
        camera_params_path = Path(output_dir) / "cameras.txt"

        if not camera_params_path.exists():
            self.convert_binary_to_text(input_dir, output_dir)

        with open(camera_params_path, "r") as f:
            for line in f:
                if line.startswith("#"): 
                    continue
                parts = line.split()
                fx, fy, cx, cy, k1, k2, k3, k4 = map(float, parts[-8:])
                K = np.array([[fx, 0, cx],
                             [0, fy, cy],
                             [0, 0, 1]])
                dist = np.array([k1, k2, k3, k4])
                return (K, dist)
        
        raise RuntimeError("No correct camera parameters found")
    
