from typing import List, Tuple, TypeAlias
from clip_timestamps.model import ClipTimestamp

# A pair of ClipTimestamp objects defining the start and end of a video clip
ClipBoundary: TypeAlias = Tuple[ClipTimestamp, ClipTimestamp]

# A collection of defined video clip boundaries
ClipBoundaries: TypeAlias = List[ClipBoundary]

# A list of strings, typically used for file paths or other string-based data
Strings: TypeAlias = List[str]