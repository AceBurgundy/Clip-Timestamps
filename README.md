# 🎬 Clip-Timestamps: Series-Scale Sequence Detector

**Clip-Timestamps** is a high-precision Python module designed to identify specific video sequences—like openings and endings—across multiple files based on a user-provided template. It is optimized for episodic content like anime, K-Dramas, or sit-coms by matching target videos against a "Gold Standard" reference.

## 🚀 Key Features

* 🎯 **Template-Based Matching**: Uses user-defined start and end times from a reference video to create a search DNA consisting of 60 equidistant checkpoints.
* 🧠 **Multi-Feature Signatures**: Extracts Mean Intensity, Variance, and Horizontal Gradient to distinguish structural frame compositions.
* ⚡ **Numba Accelerated**: Critical mathematical operations like signature computation and sequence scoring are JIT-compiled for near-native performance.
* 📏 **Two-Phase Precision**: Combines a global coarse search using keyframe decimation with a local "Sum of Absolute Differences" (SAD) refinement for frame-accurate boundaries.

## 🛠️ How It Works

The system uses a **Two-Phase Search** pipeline:

1.  **Global Coarse Match**: The algorithm extracts 60 equidistant fingerprints from the reference to create a visual sequence. Target videos are scanned using a sliding window to find an approximate match within a ±6-minute range of the original time.
2.  **Local Refinement**: Once a neighborhood is found, the system performs a temporal difference analysis (SAD) by decoding every frame in the window to lock onto the exact frame exhibiting the highest visual change (scene cut).

## 📖 Usage

```python
from os.path import basename
import json
import time

from scan_clip_stamps.scanner import segment_search
from scan_clip_stamps.model import Timestamp

# List of video files to process
videos = ["01.mp4", "02.mp4", "03.mp4", "04.mp4"]

def on_complete(video_path, timestamps):
    """Callback function to handle results after processing each video."""
    json_output = json.dumps(timestamps, default=Timestamp.serialize, indent=2)
    print(f"{basename(video_path)} -> Repeats: {json_output}\n")

def main():
    """Main execution function to identify recurring segments across multiple videos."""
    # Define segments from the first video (reference)
    episode_one_intro_start = Timestamp(minutes=1, seconds=0)
    episode_one_intro_end = Timestamp(minutes=2, seconds=30)

    episode_one_outro_start = Timestamp(minutes=22, seconds=15)
    episode_one_outro_end = Timestamp(minutes=23, seconds=45)

    # Search for these templates in all videos
    segment_search(videos, first_video_trim_timestamps=[
        (episode_one_intro_start, episode_one_intro_end),
        (episode_one_outro_start, episode_one_outro_end)
    ], on_completed=on_complete)

if __name__ == '__main__':
    start = time.perf_counter()
    main()
    elapsed = f"{(time.perf_counter() - start)/60:.2f}"
    print(f"Total time: {elapsed} minutes")
```

## 🔬 Technical Breakdown

### Visual Signature
The algorithm identifies frames by calculating a three-part signature:
* **Mean Intensity**: The overall brightness of the frame.
* **Variance**: The contrast and detail level.
* **Horizontal Gradient**: The edge density and structural composition.

### Sequence Scoring
Matching is performed using a **Weighted Manhattan Distance**. Brightness (Mean) is prioritized with a weight of 1.0, while Variance (0.1) and Gradient (0.2) serve as structural tie-breakers to ensure reliability across different video encodings.

## ⚖️ License & Author

**Author**: AceBurgundy  
**License**: This project is licensed under the **Mozilla Public License 2.0 (MPL)**.

*Built with ❤️ for the community.*