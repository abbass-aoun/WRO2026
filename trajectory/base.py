from abc import ABC, abstractmethod


class TrajectoryBase(ABC):
    """
    Abstract base class for all trajectory types.

    WHAT IS THIS?
    -------------
    Think of this as a "contract" written in code.
    Every type of path we use — Bezier curves, Pure Pursuit centerlines,
    parking maneuver paths — must implement exactly these 4 methods.

    WHY DO WE NEED THIS?
    --------------------
    The steering and driving controllers call these 4 methods.
    They don't care whether the path is a Bezier curve or something else —
    as long as the object they receive can answer these 4 questions:
        1. "What is the closest point on the path to where I am right now?"
        2. "What are the (x, y) coordinates at position s on the path?"
        3. "Which direction is the path pointing at position s?"
        4. "How sharply is the path curving at position s?"

    WHAT IS 's'?
    ------------
    's' is the arc-length parameter — the distance along the path in cm,
    measured from the start point.
        s = 0       → start of the path
        s = 50      → 50 cm along the path
        s = total_length → end of the path

    If Python raises "TypeError: Can't instantiate abstract class ...",
    it means a subclass forgot to implement one of the methods below.
    """

    @abstractmethod
    def find_closest(self, x: float, y: float, near_s: float = None) -> float:
        """
        Find the arc-length 's' of the point on the path closest to (x, y).

        Args:
            x, y  : current position of the car in cm
            near_s: start the search near this arc-length value.
                    Pass the previous result here to avoid scanning the whole
                    path every loop iteration.

        Returns:
            Arc-length s (in cm) of the closest point on the path.
        """

    @abstractmethod
    def get_point(self, s: float) -> tuple:
        """
        Return the (x, y) position on the path at arc-length s.

        Args:
            s: arc-length in cm

        Returns:
            (x, y) tuple in cm
        """

    @abstractmethod
    def get_tangent(self, s: float) -> tuple:
        """
        Return the unit tangent vector (tx, ty) at arc-length s.

        The tangent tells you which direction the path is going at that point.
        It is always a unit vector: tx² + ty² = 1.

        Args:
            s: arc-length in cm

        Returns:
            (tx, ty) unit vector
        """

    @abstractmethod
    def get_curvature(self, s: float) -> float:
        """
        Return the curvature of the path at arc-length s.

        Curvature = 1 / radius_of_curvature.
            0.0  → perfectly straight
            0.1  → gentle curve (radius = 10 cm ... quite tight for a small car)
            Large → very sharp turn

        Args:
            s: arc-length in cm

        Returns:
            Curvature in 1/cm (always >= 0)
        """

    @property
    @abstractmethod
    def total_length(self) -> float:
        """Total arc length of the path in cm."""
