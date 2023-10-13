#!/usr/bin/env python3

"""
Interface to run the observation-generating ML model on saved pano RGB measurements for offline debugging.
"""

import argparse, os, cv2
from cmn_ported import CoarseMapNavDiscrete

class CmnModelRunner:
    """
    Class to init and run the ML model for observation generation.
    """
    cmn:CoarseMapNavDiscrete = CoarseMapNavDiscrete()

    def __init__(self, path_to_model:str):
        """
        Initialize the runner class.
        @param path_to_model - Filepath to trained local occupancy predictor model.
        """
        self.cmn.load_ml_model(path_to_model)


    def run_loop(self, data_dir:str):
        """
        Run the model for every pano rgb image in data_dir.
        """
        # Get all pano RGB image files in this dir.
        files = [f for f in os.listdir(data_dir) if os.path.isfile(f) and "pano_rgb" in f]
        files.sort()
        for f in files:
            pano_rgb = cv2.imread(f)
            local_occ = self.cmn.predict_local_occupancy(pano_rgb)
            # Visualize these.
            self.cmn.visualizer.pano_rgb = pano_rgb
            self.cmn.visualizer.current_predicted_local_map = local_occ
            cmn_viz_img = self.cmn.visualizer.get_updated_img()
            cv2.imshow('cmn viz image', cmn_viz_img)
            cv2.waitKey(0) # Wait forever for keypress before continuing.
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Run the CMN observation model from saved data.")
    parser.add_argument('-m', action="store", dest="model_filepath", type=str, required=True, help="Path to saved local occupancy predictor model.")
    parser.add_argument('-d', action="store", dest="data_dir", type=str, required=True, help="Path to data directory to use.")
    args = parser.parse_args()

    runner = CmnModelRunner(args.model_filepath)
    runner.run_loop(args.data_dir)


if __name__ == "__main__":
    main()
