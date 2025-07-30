#!/usr/bin/env python3
"""
Clean MASt3R Pipeline - No filelist hijacking required
This pipeline uses the proper parameter structure from the Gradio interface
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import struct
import tempfile
import functools
import copy
import glob
import shutil

from pathlib import Path
from socket import MsgFlag

import numpy as np
import pycolmap
import torch

from mast3r.model import AsymmetricMASt3R
from mast3r.utils.misc import hash_md5
from mast3r.demo import get_reconstructed_scene, SparseGAState
from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
from mast3r.image_pairs import make_pairs
from mast3r.retrieval.processor import Retriever
import mast3r.utils.path_to_dust3r  # noqa
from dust3r.utils.image import load_images
from dust3r.utils.device import to_numpy
import copy
import glob
import shutil

class MAST3RReconstruction:
    """
    MASt3R-sfm 3D reconstruction, inference step
    """
    
    def __init__(self, model_path=None, device='cuda', image_size=512, silent=False):
        """
        Initialize the MASt3R-sfm model for inference
        
        Args:
            model_path: Path to model weights or model name
            device: PyTorch device ('cuda' or 'cpu')
            image_size: Input image size (512 or 224)
            silent: Whether to suppress output
        """
        self.device = device
        self.image_size = image_size
        self.silent = silent
        
        # Load model
        if model_path is None:
            # Default model path
            model_path = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
        
        self.model = AsymmetricMASt3R.from_pretrained(model_path).to(device)
        self.retrieval_model = "/home/mark.do/mast3r/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth"  # TODO add this to yaml
        
        # Default parameters (matching Gradio defaults)
        self.default_params = {
            # Optimization parameters
            'lr1': 0.07,           # Coarse learning rate
            'niter1': 300,          # Coarse iterations
            'lr2': 0.01,            # Fine learning rate  
            'niter2': 300,          # Fine iterations
            'optim_level': 'refine+depth',  # Optimization level
            
            # Matching parameters
            'matching_conf_thr': 0.0,  # Matching confidence threshold
            'shared_intrinsics': False,  # Shared intrinsics
            
            # Scene graph parameters
            'scenegraph_type': 'retrieval',  # Scene graph type
            'winsize': 1,           # Window size
            'win_cyclic': False,    # Cyclic sequence
            'refid': 0,             # Reference ID
            
            # Output parameters
            'min_conf_thr': 1.5,    # Minimum confidence threshold
            'as_pointcloud': True,  # Output as pointcloud
            'mask_sky': False,      # Mask sky
            'clean_depth': True,    # Clean depthmaps
            'transparent_cams': False,  # Transparent cameras
            'cam_size': 0.2,        # Camera size
            'TSDF_thresh': 0.0,    # TSDF threshold
        }
    
    def set_retrieval_model(self, retrieval_model_path):
        """Set retrieval model for similarity-based scene graph"""
        self.retrieval_model = retrieval_model_path
    
    def update_params(self, **kwargs):
        """Update default parameters"""
        self.default_params.update(kwargs)
    
    def reconstruct_scene(self, image_paths, cache_dir, output_dir, **kwargs):
        """
        Reconstruct 3D scene from a list of image paths
        
        Args:
            image_paths: List of image file paths
            output_dir: Output directory (if None, uses temp directory)
            **kwargs: Override default parameters
            
        Returns:
            tuple: (scene_state, output_file_path)
        """
        # Merge parameters
        params = self.default_params.copy()
        params.update(kwargs)
        
        # Setup output directory
        if output_dir is None:
            raise ValueError("Output directory is required")
        os.makedirs(output_dir, exist_ok=True)
        
        # Load images
        if not self.silent:
            print(f"Loading {len(image_paths)} images...")
        
        imgs = load_images(image_paths, size=self.image_size, verbose=not self.silent)
        
        # Handle single image case
        if len(imgs) == 1:
            imgs = [imgs[0], copy.deepcopy(imgs[0])]
            imgs[1]['idx'] = 1
            image_paths = [image_paths[0], image_paths[0] + '_2']
        
        # Build scene graph
        scene_graph_params = [params['scenegraph_type']]
        if params['scenegraph_type'] in ["swin", "logwin"]:
            scene_graph_params.append(str(params['winsize']))
        elif params['scenegraph_type'] == "oneref":
            scene_graph_params.append(str(params['refid']))
        elif params['scenegraph_type'] == "retrieval":
            scene_graph_params.append(str(params['winsize']))
            scene_graph_params.append(str(params['refid']))
        
        if params['scenegraph_type'] in ["swin", "logwin"] and not params['win_cyclic']:
            scene_graph_params.append('noncyclic')
        
        scene_graph = '-'.join(scene_graph_params)
        
        # Handle retrieval similarity matrix
        sim_matrix = None
        print(f"Using retrieval model: {self.retrieval_model}")
        if 'retrieval' in params['scenegraph_type']:
            if self.retrieval_model is None:
                raise ValueError("Retrieval model must be set for retrieval scenegraph type")
            
            retriever = Retriever(self.retrieval_model, backbone=self.model, device=self.device)
            with torch.no_grad():
                sim_matrix = retriever(image_paths)
            
            # Cleanup
            del retriever
            torch.cuda.empty_cache()
        
        # Create image pairs
        pairs = make_pairs(imgs, scene_graph=scene_graph, prefilter=None, 
                          symmetrize=True, sim_mat=sim_matrix)
        
        # Adjust iterations for coarse optimization
        if params['optim_level'] == 'coarse':
            params['niter2'] = 0
        
        # Setup cache directory
        cache_dir = os.path.join(cache_dir)
        os.makedirs(cache_dir, exist_ok=True)
        
        # Run sparse global alignment
        if not self.silent:
            print("Running sparse global alignment...")
        
        scene = sparse_global_alignment(
            image_paths, pairs, cache_dir,
            self.model, 
            lr1=params['lr1'], 
            niter1=params['niter1'], 
            lr2=params['lr2'], 
            niter2=params['niter2'], 
            device=self.device,
            opt_depth='depth' in params['optim_level'], 
            shared_intrinsics=params['shared_intrinsics'],
            matching_conf_thr=params['matching_conf_thr']
        )
        
        # Setup output file
        outfile_name = os.path.join(output_dir, 'reconstruction.glb')
        
        # Create scene state
        scene_state = SparseGAState(scene, False, cache_dir, outfile_name)
        
        # Generate 3D model
        if not self.silent:
            print("Generating 3D model...")
        
        from mast3r.demo import get_3D_model_from_scene
        outfile = get_3D_model_from_scene(
            self.silent, scene_state, 
            params['min_conf_thr'], 
            params['as_pointcloud'], 
            params['mask_sky'],
            params['clean_depth'], 
            params['transparent_cams'], 
            params['cam_size'], 
            params['TSDF_thresh'],
        )
        
        return scene_state, outfile
    
    def reconstruct_from_directory(self, image_dir, cache_dir, output_dir, **kwargs):
        """
        Reconstruct 3D scene from a directory of images
        
        Args:
            image_dir: Directory containing images
            output_dir: Output directory
            **kwargs: Override default parameters
            
        Returns:
            tuple: (scene_state, output_file_path)
        """
        # Get all image files from directory
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_paths = []
        
        for file in os.listdir(image_dir):
            if Path(file).suffix.lower() in image_extensions:
                image_paths.append(os.path.join(image_dir, file))
        
        if not image_paths:
            raise ValueError(f"No image files found in {image_dir}")
        
        return self.reconstruct_scene(image_paths, cache_dir, output_dir, **kwargs)
    
    # Example usage
if __name__ == "__main__":
    pass
    # print(MASt3RPipeline.read_binary_camera_matrices("colmap/cameras.txt"))
    # MASt3RPipeline.collect_key_images_from_directory(
    #     input_dir="/home/mark.do/mast3r/opencv_chessboard_undistort",
    #     output_dir="/home/mark.do/mast3r/key_opencv_chessboard_undistort"
    # )
    # # Initialize pipeline
    # pipeline = MASt3RPipeline(
    #     device='cuda',
    #     image_size=512,
    #     silent=False
    # )
    
    # # Optionally set retrieval model
    # # pipeline.set_retrieval_model("path/to/retrieval/model")
    
    # # Optionally update parameters
    # pipeline.update_params(
    #     optim_level='refine+depth',
    #     min_conf_thr=2.0,
    #     as_pointcloud=True
    # )

    

    # # Reconstruct from directory
    # try:
    #     scene_state, output_file = pipeline.reconstruct_from_directory(
    #         image_dir="./important_imgs",
    #         output_dir="./output",
    #         # Override specific parameters for this run
    #         lr1=0.05,
    #         niter1=200
    #     )
    #     print(f"Reconstruction saved to: {output_file}")
    # except Exception as e:
    #     print(f"Reconstruction failed: {e}") 