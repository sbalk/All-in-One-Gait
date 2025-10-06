import os
import os.path as osp
import time
import sys
sys.path.append(os.path.abspath('.') + "/demo/libs/")
from track import *
from segment import *
from recognise import *

def main():
    """
    Main function to process gallery and probe videos.
    Automatically detects all probe videos in the input directory.
    """
    output_dir = "./demo/output/OutputVideos/"
    os.makedirs(output_dir, exist_ok=True)
    current_time = time.localtime()
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    video_save_folder = osp.join(output_dir, timestamp)

    save_root = './demo/output/'
    input_dir = "./demo/output/InputVideos/"
    gallery_video_path = osp.join(input_dir, "gallery.mp4")

    # Auto-detect probe videos
    import glob
    probe_video_paths = sorted(glob.glob(osp.join(input_dir, "probe*.mp4")))

    if not probe_video_paths:
        print("No probe videos found in", input_dir)
        return

    print(f"Found {len(probe_video_paths)} probe video(s): {[osp.basename(p) for p in probe_video_paths]}")

    # Tracking
    gallery_track_result = track(gallery_video_path, video_save_folder)
    probe_track_results = [track(path, video_save_folder) for path in probe_video_paths]

    # Check if silhouettes already exist
    gallery_video_name = save_root + '/GaitSilhouette/' + gallery_video_path.split("/")[-1].split(".")[0]
    probe_video_names = [save_root + '/GaitSilhouette/' + path.split("/")[-1].split(".")[0]
                         for path in probe_video_paths]

    exist = os.path.exists(gallery_video_name) and all(os.path.exists(name) for name in probe_video_names)
    print(exist)

    # Segmentation
    if exist:
        gallery_silhouette = getsil(gallery_video_path, save_root + '/GaitSilhouette/')
        probe_silhouettes = [getsil(path, save_root + '/GaitSilhouette/') for path in probe_video_paths]
    else:
        gallery_silhouette = seg(gallery_video_path, gallery_track_result, save_root + '/GaitSilhouette/')
        probe_silhouettes = [seg(path, track_result, save_root + '/GaitSilhouette/')
                            for path, track_result in zip(probe_video_paths, probe_track_results)]

    # Recognition - extract features
    gallery_feat = extract_sil(gallery_silhouette, save_root + '/GaitFeatures/')
    probe_feats = [extract_sil(silhouette, save_root + '/GaitFeatures/') for silhouette in probe_silhouettes]

    # Compare probes with gallery
    probe_results = [compare(feat, gallery_feat) for feat in probe_feats]

    # Write results back to videos
    for result, path in zip(probe_results, probe_video_paths):
        writeresult(result, path, video_save_folder)


if __name__ == "__main__":
    main()
