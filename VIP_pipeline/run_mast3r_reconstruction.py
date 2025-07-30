#!/usr/bin/env python3
"""
Simple script to run MASt3R reconstruction using the clean pipeline
"""

from mast3r_pipeline import MASt3RPipeline
import os

def main():
    # Setup proper folders
    MASt3RPipeline.collect_key_images_from_directory(
        input_dir="/home/mark.do/mast3r/opencv_chessboard_undistort",
        output_dir="/home/mark.do/mast3r/key_opencv_chessboard_undistort"
    )

    # Initialize the pipeline
    pipeline = MASt3RPipeline(
        device='cuda',  # or 'cpu' if no GPU
        image_size=512,
        silent=False
    )
    
    # Set default parameters 
    pipeline.update_params(
        optim_level='refine+depth',
        min_conf_thr=1.5,
        as_pointcloud=True,
        clean_depth=True,
        cam_size=0.2,
        scenegraph_type='complete',
        shared_intrinsics=True,
    )
    
    # Check if input directory exists
    image_dir = "./key_opencv_chessboard_undistort"
    if not os.path.exists(image_dir):
        print(f"Error: Directory {image_dir} not found!")
        print("Please create the directory and add images.")
        return
    
    # Count images
    image_files = [f for f in os.listdir(image_dir) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'))]
    
    if not image_files:
        print(f"No image files found in {image_dir}")
        return
    
    print(f"Found {len(image_files)} images in {image_dir}")
    
    # Run reconstruction
    try:
        print("Starting MASt3R reconstruction...")
        scene_state, output_file = pipeline.reconstruct_from_directory(
            image_dir=image_dir,
            cache_dir="/pub3/mark.do/mast3r-temp",
            output_dir="./output",
        )
        
        print(f"✅ Reconstruction completed successfully!")
        print(f"📁 Output file: {output_file}")
        print(f"📊 Scene state saved for further processing")
        
    except Exception as e:
        print(f"❌ Reconstruction failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 