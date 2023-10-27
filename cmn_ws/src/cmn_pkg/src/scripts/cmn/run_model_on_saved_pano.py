#!/usr/bin/env python3

"""
Interface to run the observation-generating ML model on saved pano RGB measurements for offline debugging.

Example: From cmn_ws directory, run:

python3 src/cmn_pkg/src/scripts/cmn/run_model_on_saved_pano.py -d ~/dev/coarse-map-turtlebot/cmn_ws/src/cmn_pkg/data/20231010-111506 -m ~/dev/coarse-map-turtlebot/cmn_ws/src/cmn_pkg/src/scripts/cmn/model/trained_local_occupancy_predictor_model.pt
"""

import argparse, os, cv2
from cmn_ported import CoarseMapNavDiscrete

class CmnModelRunner:
    """
    Class to init and run the ML model for observation generation.
    """
    cmn:CoarseMapNavDiscrete = CoarseMapNavDiscrete(None, None)

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
        # Get all pano RGB image files in this dir and subdirs.
        files = [os.path.join(dp, f) for dp, dn, fn in os.walk(data_dir) for f in fn if "pano_rgb" in f]
        files.sort()
        print("Found {:} pano_rgb files in data_dir and subdirectories.".format(len(files)))
        for f in files:
            # Exclude data from the d435, before we swapped to the wider-fov d455 realsense.
            if "d435" in f:
                continue
            # Read the pano RGB image.
            pano_rgb = cv2.imread(f)
            # Run the model on this measurement.
            local_occ = self.cmn.predict_local_occupancy(pano_rgb)
            # Visualize these.
            self.cmn.visualizer.pano_rgb = pano_rgb
            self.cmn.visualizer.current_predicted_local_map = local_occ
            cmn_viz_img = self.cmn.visualizer.get_updated_img()
            cv2.imshow('cmn viz image', cmn_viz_img)
            key = cv2.waitKey(0) # Wait forever for keypress before continuing.
            if key == 113: # q for quit
                print("Exiting on user 'Q' press.")
                break
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
