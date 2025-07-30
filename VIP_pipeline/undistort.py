from ntpath import isfile
import cv2
import numpy as np

import shutil
from pathlib import Path

class Undistort:
    def __init__(self, K, dist) -> None:
        self.K = K
        self.dist = dist
        self.undistorted_counter = 0

    def undistort_directory(self, input_dir, output_dir):
        input_directory = Path(input_dir)
        output_directory = Path(output_dir)

        for file in input_directory.iterdir():
            if not file.is_file():
                continue

            # Undistort
            img = cv2.imread(str(file.resolve()))
            undistorted = cv2.undistort(img, self.K, self.dist)

            # Output
            output_full_path = output_directory / f'undistorted_{file.name}'
            cv2.imwrite(str(output_full_path), undistorted)
            self.undistorted_counter += 1
            print(output_full_path)
        
        print(f"Undistortion done: {self.undistorted_counter}")
        self.undistorted_counter = 0

if __name__ == "__main__":
    directory = "./raw"

    K = np.array([[3.34229244e+03, 0.00000000e+00, 2.69925312e+03],
    [0.00000000e+00, 3.14055946e+03, 1.80300491e+03],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],)

    dist = np.array([[-8.61005078e-02,  4.07493066e-02, -5.44964078e-05,  1.98993979e-05,
    -1.87927184e-02]])

    
