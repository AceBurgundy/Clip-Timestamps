from typing import Callable, Dict, List, Optional, Tuple, Any

from numpy import array, frombuffer, searchsorted, sort, uint8, ndarray, abs, int16, round as np_round
from clip_timestamps.custom_types import ClipBoundaries, ClipTimestamp
from fractions import Fraction

from numba import njit # type: ignore
from dataclasses import dataclass
from av import open as av_open

@njit(fastmath=True, cache=True)
def compute_signature(gray_frame: ndarray) -> Tuple[int, int, int]:
    """
    Generates a three-part perceptual signature for a video frame.

    Concept
    -------
    Multi-Feature Extraction: This function extracts three distinct visual
    properties to create a robust fingerprint.
    1. Mean Intensity: The overall brightness.
    2. Variance: The contrast and detail level.
    3. Horizontal Gradient: The edge density, calculated by summing absolute
       differences between adjacent horizontal pixels.
    Combining these allows the algorithm to distinguish between frames that might 
    have similar brightness but different structural compositions.

    Parameters
    ----------
    gray_frame : ndarray
        A 2D grayscale image array (typically 16x16).

    Returns
    -------
    Tuple[int, int, int]
        A tuple of (mean, variance, normalized_gradient).
    """
    height: int
    width: int
    height, width = gray_frame.shape
    pixel_count: int = height * width
    
    total_sum: int = 0
    total_squared_sum: int = 0
    total_gradient: int = 0

    for row_index in range(height):
        for column_index in range(width):
            pixel_value: int = gray_frame[row_index, column_index]
            total_sum += pixel_value
            total_squared_sum += pixel_value * pixel_value
            
            # Calculate horizontal gradient
            if column_index < width - 1:
                next_pixel_value: int = gray_frame[row_index, column_index + 1]
                total_gradient += abs(int(pixel_value) - int(next_pixel_value))

    mean_intensity: int = total_sum // pixel_count
    variance: int = (total_squared_sum // pixel_count) - mean_intensity * mean_intensity
    normalized_gradient: int = total_gradient // pixel_count
    
    return mean_intensity, variance, normalized_gradient


@njit(fastmath=True, cache=True)
def sequence_score(reference_sequence: ndarray, target_sequence: ndarray) -> float:
    """
    Calculates the weighted error between two sequences of frame signatures.

    Concept
    -------
    Weighted Manhattan Distance: Compares two sequences of fingerprints. 
    It applies different weights to the Mean (1.0), Variance (0.1), and 
    Gradient (0.2). These weights prioritize brightness matching while 
    using structural features as secondary tie-breakers, optimizing for 
    speed and reliability across different video encodings.

    Parameters
    ----------
    reference_sequence : ndarray
        The signatures from the original template.
    target_sequence : ndarray
        The signatures from the candidate window in the target video.

    Returns
    -------
    float
        The total calculated error score (lower is better).
    """
    total_error: float = 0.0
    sequence_length: int = len(reference_sequence)
    
    for index in range(sequence_length):
        # Index 0: Mean, Index 1: Variance, Index 2: Gradient
        mean_diff: float = abs(reference_sequence[index, 0] - target_sequence[index, 0])
        variance_diff: float = abs(reference_sequence[index, 1] - target_sequence[index, 1])
        gradient_diff: float = abs(reference_sequence[index, 2] - target_sequence[index, 2])
        
        total_error += mean_diff + (0.1 * variance_diff) + (0.2 * gradient_diff)
        
    return total_error


@njit(fastmath=True, cache=True)
def nearest_index(sorted_array: ndarray, target_value: float) -> int:
    """
    Finds the index of the value in a sorted array closest to the target.

    Parameters
    ----------
    sorted_array : ndarray
        An array of sorted floating point timestamps.
    target_value : float
        The target timestamp to locate.

    Returns
    -------
    int
        The index of the closest timestamp.
    """
    insertion_index: int = searchsorted(sorted_array, target_value) # type: ignore
    
    if insertion_index == 0:
        return 0
    if insertion_index >= len(sorted_array):
        return len(sorted_array) - 1
    
    # Check if the preceding value is closer to target than the current value
    if abs(sorted_array[insertion_index] - target_value) < abs(sorted_array[insertion_index - 1] - target_value):
        return insertion_index
        
    return insertion_index - 1


def fast_frame_signature(video_frame: Any) -> Tuple[int, int, int]:
    """
    Pre-processes a PyAV frame and computes its signature.

    Parameters
    ----------
    video_frame : Any
        A frame object from the PyAV decoder.

    Returns
    -------
    Tuple[int, int, int]
        The computed visual signature.
    """
    small_grayscale_frame: Any = video_frame.reformat(width=16, height=16, format="gray")
    pixel_data: ndarray = frombuffer(small_grayscale_frame.planes[0], dtype=uint8).reshape(16, 16)
    
    return compute_signature(pixel_data)


def extract_features(
    file_path: str, 
    sample_rate: float = 1.5
) -> Dict[float, Tuple[int, int, int]]:
    """
    Extracts visual fingerprints from a video file at a consistent rate.

    Concept
    -------
    Keyframe Decimation: Speed is achieved by skipping non-keyframes 
    (P-frames/B-frames) in the bitstream. By only calculating signatures 
    for I-frames that meet the minimum temporal gap (min_gap), we reduce 
    computational load by ~90% while maintaining enough data for segment 
    matching.

    Parameters
    ----------
    file_path : str
        Path to the video file.
    sample_rate : float
        Target samples per second.

    Returns
    -------
    Dict[float, Tuple[int, int, int]]
        A dictionary mapping timestamps to frame signatures.
    """
    extracted_features: Dict[float, Tuple[int, int, int]] = {}
    container: Any = av_open(file_path)
    video_stream: Any = container.streams.video[0]
    
    # Speed up by only decoding keyframes
    video_stream.codec_context.skip_frame = "NONKEY"
    
    time_base: Fraction = video_stream.time_base or Fraction(1, 1)
    minimum_pts_gap: int = int(Fraction(1, 1) / Fraction.from_float(sample_rate) / time_base)
    
    last_processed_pts: int = -1
    
    for video_frame in container.decode(video=0):
        current_pts: Optional[int] = video_frame.pts
        
        if current_pts is None or (current_pts - last_processed_pts < minimum_pts_gap):
            continue
            
        extracted_features[float(video_frame.time)] = fast_frame_signature(video_frame)
        last_processed_pts = current_pts
        
    container.close()
    return extracted_features


@dataclass(frozen=True)
class Template:
    """Represents a visual segment template used for searching."""
    duration: float
    checkpoints: ndarray
    interval: float
    start_time: float


def refine_boundary(
    file_path: str, 
    approximate_time: float, 
    search_range: float = 2.0
) -> float:
    """
    Refines a segment boundary by detecting the point of maximum visual change.

    Concept
    -------
    Temporal Difference Analysis: Coarse matching identifies the general area 
    of a segment. This function zooms in on that area and calculates the sum 
    of absolute differences (SAD) between consecutive frames. The frame 
    exhibiting the highest difference is likely a scene cut, which provides 
    a precise frame-accurate boundary for the clip.

    Parameters
    ----------
    file_path : str
        Path to the video file.
    approximate_time : float
        The estimated timestamp found during the coarse search phase.
    search_range : float
        How many seconds to search before and after the approximate time.

    Returns
    -------
    float
        The refined, high-precision timestamp.
    """
    container: Any = av_open(file_path)
    video_stream: Any = container.streams.video[0]
    time_base_float: float = float(video_stream.time_base)
    
    # Seek to start of search range
    seek_pts: int = int((approximate_time - search_range) / time_base_float)
    container.seek(seek_pts, stream=video_stream)
    
    previous_gray_frame: Optional[ndarray] = None
    refined_best_time: float = approximate_time
    maximum_difference_score: int = -1
    
    for video_frame in container.decode(video=0):
        current_frame_time: float = float(video_frame.time)
        
        if current_frame_time > approximate_time + search_range:
            break
            
        # Extract 32x32 for slightly better detail in refinement
        current_gray_frame: ndarray = frombuffer(
            video_frame.reformat(width=32, height=32, format="gray").planes[0], 
            dtype=uint8
        ).reshape(32, 32)
        
        if previous_gray_frame is not None:
            # Calculate Sum of Absolute Differences (SAD)
            frame_difference: int = int(abs(current_gray_frame.astype(int16) - previous_gray_frame.astype(int16)).sum())
            
            if frame_difference > maximum_difference_score:
                maximum_difference_score = frame_difference
                refined_best_time = current_frame_time
                
        previous_gray_frame = current_gray_frame
        
    container.close()
    return refined_best_time


def segment_search(
    file_paths: List[str],
    first_video_trim_timestamps: ClipBoundaries,
    on_completed: Optional[Callable[[str, ClipBoundaries], None]] = None
) -> Dict[str, ClipBoundaries]:
    """
    Coordinates the search for specific video segments across multiple files.

    Concept
    -------
    Two-Phase Search:
    1. Global Coarse Match: Scans the target video at a low sample rate using 
       a sliding window of 60 checkpoints to find the approximate match.
    2. Local Refinement: Uses the SAD boundary refinement to lock onto the 
       exact frame where the segment begins.

    Parameters
    ----------
    file_paths : List[str]
        List of paths to video files. Index 0 is the reference.
    first_video_trim_timestamps : ClipBoundaries
        The start and end times in the reference video to look for.
    on_completed : Optional[Callable[[str, ClipBoundaries], None]]
        Callback function to report progress.

    Returns
    -------
    Dict[str, ClipBoundaries]
        A mapping of file paths to their identified and refined clip boundaries.
    """
    reference_path: str = file_paths[0]
    reference_features: Dict[float, Tuple[int, int, int]] = extract_features(reference_path)
    reference_timestamps: ndarray = sort(array(list(reference_features.keys())))
    
    total_checkpoints: int = 60
    segment_templates: List[Template] = []

    # Phase 1: Build templates from the reference video
    for start_timestamp, end_timestamp in first_video_trim_timestamps:
        reference_start_seconds: float = start_timestamp.to_seconds()
        segment_duration: float = end_timestamp.to_seconds() - reference_start_seconds
        sampling_interval: float = segment_duration / total_checkpoints
        
        signature_sequence: List[Tuple[int, int, int]] = [
            reference_features[reference_timestamps[
                nearest_index(
                    reference_timestamps, reference_start_seconds + (index * sampling_interval)
                )
            ]
        ] 
        for index in range(total_checkpoints)
        ]
        
        segment_templates.append(Template(
            duration=segment_duration,
            checkpoints=array(signature_sequence, dtype=int16),
            interval=sampling_interval,
            start_time=reference_start_seconds
        ))

    search_report: Dict[str, ClipBoundaries] = {reference_path: first_video_trim_timestamps}
    
    if on_completed:
        # Report the reference video results immediately
        on_completed(reference_path, first_video_trim_timestamps)

    # Phase 2: Process target videos
    for target_path in file_paths[1:]:
        target_features: Dict[float, Tuple[int, int, int]] = extract_features(target_path)
        target_timestamps: ndarray = sort(array(list(target_features.keys())))
        
        identified_clips: ClipBoundaries = []

        for template in segment_templates:
            optimal_candidate_start: float = template.start_time
            minimum_error_found: float = 1e12
            
            # Constrain search to +/- 6 minutes around reference time for efficiency
            search_minimum: float = max(0.0, template.start_time - 360.0)
            search_maximum: float = min(target_timestamps[-1], template.start_time + 360.0)
            
            candidate_timestamps: ndarray = target_timestamps[
                (target_timestamps >= search_minimum) & (target_timestamps <= search_maximum)
            ]

            for candidate_start_time in candidate_timestamps:
                # Construct sequence for current sliding window
                current_window_sequence: List[Tuple[int, int, int]] = [
                    target_features[target_timestamps[nearest_index(target_timestamps, candidate_start_time + (index * template.interval))]] 
                    for index in range(total_checkpoints)
                ]
                
                current_error_score: float = sequence_score(
                    template.checkpoints, 
                    array(current_window_sequence, dtype=int16)
                )
                
                if current_error_score < minimum_error_found:
                    minimum_error_found = current_error_score
                    optimal_candidate_start = float(candidate_start_time)

            # Phase 3: Fine-tune the detected boundary
            refined_start_seconds: float = refine_boundary(target_path, optimal_candidate_start)
            refined_end_seconds: float = refined_start_seconds + template.duration
            
            # Format results into final timestamps
            start_result: ClipTimestamp = ClipTimestamp( 
                int(refined_start_seconds // 60), 
                int(np_round(refined_start_seconds % 60))
            )
            end_result: ClipTimestamp = ClipTimestamp(
                int(refined_end_seconds // 60), 
                int(np_round(refined_end_seconds % 60))
            )
            
            identified_clips.append((start_result, end_result))

        search_report[target_path] = identified_clips
        
        if on_completed:
            # Report the identified clips for the current target video
            on_completed(target_path, identified_clips)

    return search_report