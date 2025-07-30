import os
import shutil
import glob

class KeyFrame():

    @staticmethod
    def collect_key_images_from_directory(input_dir, output_dir, key_frames_csv):
        """
        Collect key images from a directory and output it them to a new directory
        """
        filenames = []
        numbers = []
        with open(key_frames_csv) as f:
            keyframes = f.readline().strip().split(",")
            transitions = f.readline().strip().split(",")

            numbers = keyframes + transitions

        for key_image_number in numbers:
            filenames += glob.glob(os.path.join(input_dir, f'*_0{key_image_number}_*.jpg'))
        
        for filename in filenames:
            shutil.copy(filename, os.path.join(output_dir, os.path.basename(filename)))
    