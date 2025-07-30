from pathlib import Path
import yaml 

import numpy as np

from bin_read import BinaryFileReader
from reconstruction import MAST3RReconstruction
from undistort import Undistort
from keyframe import KeyFrame

class Pipeline:
    def __init__(self, config_yaml_path : str) -> None:
        with open(config_yaml_path,"r") as f:
            config = yaml.safe_load(f)
        
        print("Running pipline with:")
        print(config)

        self.unique_name = config["input"]["unique_run_name"]
        self.image_dir = config["input"]["image_dir"]

        self.binary_read_bin_path = config["binary_read"]["bin_path"]

        self.undistortion_enabled = config["undistortion"]["enabled"]
        self.undistortion_bypass_path = config["undistortion"]["bypass_path"]

        self.keyframe_selection_enabled = config["keyframe_selection"]["enabled"]
        self.keyframe_selection_keyframes_path = config["keyframe_selection"]["keyframes_path"]
        self.keyframe_selection_bypass_path = config["keyframe_selection"]["bypass_path"]

        self.reconstruction_enabled = config["reconstruction"]["enabled"]
        self.reconstruction_output_dir = config["reconstruction"]["output_dir"]
        self.reconstruction_parameters = config["reconstruction"]["parameters"]

        self.cache_dir = config["reconstruction"]["cache_dir"]

        # Internal pipeline paths
        self._undistortion_input_dir = None
        self._undistortion_output_dir = None
        self._keyframe_input_dir = None
        self._keyframe_output_dir = None
        self._reconstruction_input_dir = None
        self._reconstruction_output_dir = None

        # Interal members
        self._K = None
        self._dist = None

    def run(self):
        # Iterate from undistort -> keyframes -> reconstruction
        # Idea behind bypasses is to use a certain directory as the step's output,
        # instead of actually running that step

        # Undistortion
        if self.undistortion_enabled:
            self.binary_read()

            self._undistortion_input_dir = self.image_dir
            self._undistortion_output_dir = Path(self.image_dir).parent / self.unique_name / "undistorted"
            self._undistortion_output_dir.mkdir(parents=True, exist_ok=True)

            self.undistort()
        else:
            self._undistortion_output_dir = self.undistortion_bypass_path
        
        # Keyframes
        if self.keyframe_selection_enabled:
            self._keyframe_input_dir = self._undistortion_output_dir
            self._keyframe_output_dir = Path(self.image_dir).parent / self.unique_name / "keyframes"
            self._keyframe_output_dir.mkdir(parents=True, exist_ok=True) 

            self.select_keyframes()
        else:
            self._keyframe_output_dir = self.keyframe_selection_bypass_path
        
        # Reconstruction
        print("=== Starting Reconstruction Step ===")
        if self.reconstruction_enabled:
            self._reconstruction_input_dir = self._keyframe_output_dir
            self.reconstruct()
        
        print("Pipeline Execution Complete!")
    
    def binary_read(self):
        binary_reader = BinaryFileReader()
        self._K, self._dist = binary_reader.read_binary_camera_matrices(self.binary_read_bin_path, self.binary_read_bin_path)

    def undistort(self):
        undistort = Undistort(self._K, self._dist)
        undistort.undistort_directory(self._undistortion_input_dir, self._undistortion_output_dir)

    def select_keyframes(self):
        KeyFrame.collect_key_images_from_directory(self._keyframe_input_dir, self._keyframe_output_dir, self.keyframe_selection_keyframes_path)
    
    def reconstruct(self):
        # Initialize the pipeline
        print("=== Starting Reconstruction Step ===")
        reconstructor = MAST3RReconstruction(
            device='cuda',  # or 'cpu' if no GPU
            image_size=512,
            silent=False
        )
        
        # Set default parameters 
        print("Setting default parameters...")
        reconstructor.update_params(
            optim_level='refine+depth',
            min_conf_thr=1.5,
            as_pointcloud=True,
            clean_depth=True,
            cam_size=0.2,
            scenegraph_type='complete',
            shared_intrinsics=True,
        )
        
        # Run reconstruction
        try:
            print("Starting MASt3R reconstruction...")
            scene_state, output_file = reconstructor.reconstruct_from_directory(
                image_dir=self._keyframe_output_dir,
                cache_dir=self.cache_dir,
                output_dir=self.reconstruction_output_dir
            )
            
            print(f"✅ Reconstruction completed successfully!")
            print(f"📁 Output file: {output_file}")
            print(f"📊 Scene state saved for further processing")
            
        except Exception as e:
            print(f"❌ Reconstruction failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    pipeline = Pipeline("VIP_pipeline/configs/full_run.yaml")
    pipeline.run()