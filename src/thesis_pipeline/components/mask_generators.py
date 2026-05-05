import logging
import numpy as np
import random
from PIL import Image, ImageDraw, ImageFilter
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseMaskGenerator(ABC):
    """Base class for mask generators.

    All generators accept an optional *seed* parameter in :meth:`generate`
    so that each image always receives the same mask regardless of the
    global RNG state or processing order.  When *seed* is ``None`` the
    global RNG is used (legacy behaviour).
    """

    @abstractmethod
    def generate(self, height: int, width: int, *, seed: int | None = None) -> np.ndarray:
        pass

    @staticmethod
    def _local_rng(seed: int | None) -> random.Random:
        """Return a local ``random.Random`` instance seeded per image."""
        return random.Random(seed) if seed is not None else random.Random()

    @staticmethod
    def _local_np_rng(seed: int | None) -> np.random.RandomState:
        return np.random.RandomState(seed % (2**31) if seed is not None else None)

    @staticmethod
    def _organicify(mask_img: Image.Image, rng: random.Random, blur_radius: int = 5) -> np.ndarray:
        """Apply Gaussian blur + re-threshold to soften harsh geometric edges.

        This breaks the perfect 90-degree edges that produce bright spectral
        cross patterns in the Fourier domain.  The slight randomness in the
        threshold introduces organic, non-rectangular boundaries.
        """
        # 1. Gaussian blur to soften edges
        blurred = mask_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        arr = np.array(blurred)

        # 2. Random threshold between 100-160 to create organic boundary
        threshold = rng.randint(100, 160)
        organic = np.where(arr >= threshold, 255, 0).astype(np.uint8)
        return organic


class BoxMaskGenerator(BaseMaskGenerator):
    """Generates rectangular masks with organic (non-perfect) edges.

    The base rectangle is softened via Gaussian blur + re-threshold to
    remove the perfect 90-degree corners that produce spectral cross artifacts.
    """
    def __init__(self, min_ratio=0.2, max_ratio=0.6):
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    def generate(self, height: int, width: int, *, seed: int | None = None) -> np.ndarray:
        rng = self._local_rng(seed)
        img = Image.new('L', (width, height), 0)
        draw = ImageDraw.Draw(img)

        mask_h = rng.randint(int(self.min_ratio * height), int(self.max_ratio * height))
        mask_w = rng.randint(int(self.min_ratio * width), int(self.max_ratio * width))
        top = rng.randint(0, height - mask_h)
        left = rng.randint(0, width - mask_w)

        # Draw rounded rectangle base (PIL ≥ 9.2.0 supports rounded_rectangle)
        try:
            corner_radius = rng.randint(5, max(6, min(mask_h, mask_w) // 4))
            draw.rounded_rectangle(
                [left, top, left + mask_w, top + mask_h],
                radius=corner_radius,
                fill=255,
            )
        except AttributeError:
            # Fallback for older Pillow: plain rectangle
            draw.rectangle([left, top, left + mask_w, top + mask_h], fill=255)

        # Soften edges: Gaussian blur + re-threshold
        blur_radius = max(3, min(mask_h, mask_w) // 15)
        return self._organicify(img, rng, blur_radius=blur_radius)


class IrregularMaskGenerator(BaseMaskGenerator):
    """Generates organic irregular masks using Bezier-like curved strokes.

    Instead of straight lines between random endpoints, each stroke uses
    intermediate control points to create curved, organic-looking paths
    similar to real pottery scratches and chips.
    """
    def __init__(self, min_strokes=3, max_strokes=10, min_width=20, max_width=60):
        self.min_strokes = min_strokes
        self.max_strokes = max_strokes
        self.min_width = min_width
        self.max_width = max_width

    @staticmethod
    def _clamp_range(lower: int, upper: int, floor: int, ceiling: int) -> tuple[int, int]:
        lower = max(floor, min(lower, ceiling))
        upper = max(lower, min(upper, ceiling))
        return lower, upper

    def _random_bezier_points(self, rng, width, height, num_points=5):
        """Generate a sequence of points forming a smooth curve."""
        points = []
        for _ in range(num_points):
            points.append((rng.randint(0, width), rng.randint(0, height)))
        return points

    def generate(self, height: int, width: int, *, seed: int | None = None) -> np.ndarray:
        rng = self._local_rng(seed)
        img = Image.new('L', (width, height), 0)
        draw = ImageDraw.Draw(img)
        min_dim = max(1, min(width, height))

        effective_min_strokes, effective_max_strokes = self._clamp_range(
            self.min_strokes,
            self.max_strokes,
            floor=1,
            ceiling=max(2, min_dim // 32),
        )
        effective_min_width, effective_max_width = self._clamp_range(
            self.min_width,
            self.max_width,
            floor=max(2, min_dim // 40),
            ceiling=max(3, min_dim // 10),
        )
        blob_min_radius, blob_max_radius = self._clamp_range(
            4,
            max(5, min_dim // 8),
            floor=max(3, min_dim // 50),
            ceiling=max(4, min_dim // 12),
        )

        num_strokes = rng.randint(effective_min_strokes, effective_max_strokes)
        for _ in range(num_strokes):
            width_stroke = rng.randint(effective_min_width, effective_max_width)
            # Use multi-segment polyline with 3-6 control points for curves
            num_points = rng.randint(3, 6)
            points = self._random_bezier_points(rng, width, height, num_points)
            draw.line(points, fill=255, width=width_stroke, joint="curve")

        # Add some random elliptical blobs for chip-like damage
        num_blobs = rng.randint(0, max(1, min_dim // 96))
        for _ in range(num_blobs):
            cx = rng.randint(0, width)
            cy = rng.randint(0, height)
            rx = rng.randint(blob_min_radius, blob_max_radius)
            ry = rng.randint(blob_min_radius, blob_max_radius)
            draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=255)

        # Soften edges for organic look
        blur_radius = max(3, min(width, height) // 40)
        return self._organicify(img, rng, blur_radius=blur_radius)


class EdgeMaskGenerator(BaseMaskGenerator):
    """Simulates damage on the rim or base of the pottery with organic edges."""
    MIN_DEPTH_PX = 2  # guard against zero-depth on tiny images

    def __init__(self, depth_ratio=0.25):
        self.depth_ratio = depth_ratio

    def generate(self, height: int, width: int, *, seed: int | None = None) -> np.ndarray:
        rng = self._local_rng(seed)
        np_rng = self._local_np_rng(seed)

        img = Image.new('L', (width, height), 0)
        draw = ImageDraw.Draw(img)

        depth = max(int(height * self.depth_ratio), self.MIN_DEPTH_PX)
        if depth >= height:
            logger.warning(
                f"EdgeMask depth ({depth}) >= image height ({height}); "
                f"clamping to height-1."
            )
            depth = height - 1

        # Create a wavy edge using Perlin-like 1D noise
        # Generate smooth noise by interpolating between random control points
        num_control = max(4, width // 32)
        control_depths = np_rng.randint(depth // 4, depth, size=num_control + 1)
        # Interpolate to full width
        x_control = np.linspace(0, width - 1, num_control + 1)
        x_full = np.arange(width)
        edge_profile = np.interp(x_full, x_control, control_depths).astype(int)

        # Randomly choose Top (Rim) or Bottom (Base)
        if rng.random() > 0.5:
            # Top edge damage
            for c in range(width):
                d = min(edge_profile[c], height - 1)
                draw.line([(c, 0), (c, d)], fill=255, width=1)
        else:
            # Bottom edge damage
            for c in range(width):
                d = min(edge_profile[c], height - 1)
                draw.line([(c, height - 1), (c, height - 1 - d)], fill=255, width=1)

        # Add small semi-circular chips along the edge
        num_chips = rng.randint(1, 4)
        for _ in range(num_chips):
            cx = rng.randint(0, width)
            chip_r = rng.randint(max(3, depth // 6), max(4, depth // 2))
            if rng.random() > 0.5:
                # Chip at top
                draw.ellipse([cx - chip_r, -chip_r, cx + chip_r, chip_r], fill=255)
            else:
                # Chip at bottom
                draw.ellipse([cx - chip_r, height - chip_r, cx + chip_r, height + chip_r], fill=255)

        # Soften edges
        blur_radius = max(2, depth // 8)
        return self._organicify(img, rng, blur_radius=blur_radius)
