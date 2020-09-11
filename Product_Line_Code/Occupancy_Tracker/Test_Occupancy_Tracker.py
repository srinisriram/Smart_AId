from human_detector import HumanDetector

# This program will use an input video file taken from a real life work scenario (Temple)

if __name__ == "__main__":
    human_detector_inst = HumanDetector(
        find_humans_from_video_file_name='videos/test_occupancy.mp4',
        use_pi_camera=False, open_display=True)
    human_detector_inst.perform_job()
