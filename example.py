from os.path import basename
import json
import time

from clip_timestamps.scanner import segment_search
from clip_timestamps.model import ClipTimestamp

path = lambda index: f"D:\\Downloads\\Grand Blue Dreaming Season 2\\Grand Blue Dreaming Season 2 0{index}.mp4"
videos = [path(1), path(2), path(3), path(4)]

def on_complete(video_path, timestamps):
    """Callback function to handle results after processing each video."""
    json_output = json.dumps(timestamps, default=ClipTimestamp.serialize, indent=2)
    print(f"{basename(video_path)} -> Repeats: {json_output}\n")

def main():
    """Main execution function to identify recurring segments across multiple videos."""
    # Finds sequences repeated in at least 3 out of 4 videos
    episode_one_intro_start = ClipTimestamp(minutes=1, seconds=0)
    episode_one_intro_end = ClipTimestamp(minutes=2, seconds=30)

    episode_one_outro_start = ClipTimestamp(minutes=22, seconds=15)
    episode_one_outro_end = ClipTimestamp(minutes=23, seconds=45)

    segment_search(videos, first_video_trim_timestamps=[
        (episode_one_intro_start, episode_one_intro_end),
        (episode_one_outro_start, episode_one_outro_end)
    ], on_completed=on_complete)

if __name__ == '__main__':
    start = time.perf_counter()
    main()
    elapsed = f"{(time.perf_counter() - start)/60:.2f}"
    print(f"Total time: {elapsed} minutes")