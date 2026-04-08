from dataclasses import dataclass

@dataclass(frozen=True)
class ClipTimestamp:
    """Represents a time point in minutes and seconds."""
    minutes: int
    seconds: int

    def to_seconds(self) -> float:
        return float((self.minutes * 60) + self.seconds)

    def __str__(self) -> str:
        return f"{self.minutes:02d}:{self.seconds:02d}"

    def __repr__(self) -> str:
        """Provides a more detailed string representation of the ClipTimestamp object for debugging purposes."""
        return self.__str__()
    
    @staticmethod
    def serialize(timestamp_instance: "ClipTimestamp") -> str:
        """
        Static method to serialize a ClipTimestamp instance into a string format suitable for JSON serialization.

        Parameters
        ----------
        timestamp_instance : ClipTimestamp
            The ClipTimestamp instance to be serialized.

        Returns
        -------
        str
            A string representation of the ClipTimestamp instance in the format "MM:SS".
        """
        if isinstance(timestamp_instance, ClipTimestamp):
            """Converts a ClipTimestamp instance to a string in the format "MM:SS" for JSON serialization."""
            return f"{timestamp_instance.minutes:02d}:{timestamp_instance.seconds:02d}"
        
        raise TypeError(f"Object of type {timestamp_instance.__class__.__name__} is not JSON serializable")