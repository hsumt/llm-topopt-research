"""Validated SI input contract; importing this module requires no FEM packages."""

from dataclasses import asdict, dataclass, fields
import math


@dataclass(frozen=True)
class LBracket3DConfig:
    lx_m: float
    ly_m: float
    thickness_m: float
    cut_length_m: float
    load_patch_m: float
    hole_radius_m: float
    hole_centers_m: tuple[tuple[float, float], ...]
    youngs_modulus_pa: float
    poisson_ratio: float
    force_n: tuple[float, float, float]
    volume_fraction: float
    stress_limit_pa: float | None
    nelx: int = 50
    p_norm: float = 6.0
    rho_min: float = 1e-6
    move_limit: float = 0.1
    perturbation: float = 0.15

    def __post_init__(self):
        for name in ("lx_m", "ly_m", "thickness_m", "cut_length_m", "load_patch_m",
                     "hole_radius_m", "youngs_modulus_pa", "volume_fraction", "p_norm",
                     "rho_min", "move_limit", "perturbation"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite positive number")
        if not isinstance(self.nelx, int) or isinstance(self.nelx, bool) or self.nelx < 6:
            raise ValueError("nelx must be an integer of at least 6")
        if not isinstance(self.poisson_ratio, (int, float)) or isinstance(self.poisson_ratio, bool) or not -1 < self.poisson_ratio < .5:
            raise ValueError("poisson_ratio must lie strictly between -1 and 0.5")
        if self.stress_limit_pa is not None and (isinstance(self.stress_limit_pa, bool) or not isinstance(self.stress_limit_pa, (int, float)) or not math.isfinite(self.stress_limit_pa) or self.stress_limit_pa <= 0):
            raise ValueError("stress_limit_pa must be positive or null for a volume-only problem")
        if not math.isclose(self.lx_m, self.ly_m, rel_tol=1e-10):
            raise ValueError("This backend supports equal outer lengths and equal arm widths")
        if not self.cut_length_m < self.lx_m:
            raise ValueError("cut_length_m must be smaller than the outer length")
        arm = self.lx_m - self.cut_length_m
        if self.load_patch_m > arm:
            raise ValueError("load_patch_m cannot extend beyond the horizontal arm")
        if self.volume_fraction > 1 or self.rho_min >= 1:
            raise ValueError("volume_fraction must be <= 1 and rho_min must be < 1")
        if not 2 <= self.p_norm <= 16:
            raise ValueError("p_norm must be between 2 and 16")
        if self.move_limit > 1 or self.perturbation > 1:
            raise ValueError("move_limit and perturbation must be <= 1 cell")
        force = tuple(self.force_n)
        if len(force) != 3 or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in force) or not any(force):
            raise ValueError("force_n must contain three finite components and a nonzero force")
        object.__setattr__(self, "force_n", force)
        centers = tuple(tuple(c) for c in self.hole_centers_m)
        if len(centers) != 5:
            raise ValueError("The simplified_3D_holes template requires five initial holes")
        radius = self.hole_radius_m
        for center in centers:
            if len(center) != 2 or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in center):
                raise ValueError("Each hole center must have two finite coordinates")
            x, y = center
            if min(x, y) < radius or max(x, y) + radius > self.lx_m:
                raise ValueError("Initial holes must lie inside the outer domain")
            if math.hypot(max(arm - x, 0), max(arm - y, 0)) < radius:
                raise ValueError("An initial hole intersects the permanent corner cut")
        object.__setattr__(self, "hole_centers_m", centers)
        nx, ny, nz = self.mesh_shape
        if min(nx, ny, nz) < 6:
            raise ValueError("PyParaLeSTO requires at least 6 cells in every direction")
        h = self.lx_m / nx
        if not math.isclose(self.thickness_m / nz, h, rel_tol=1e-10):
            raise ValueError("The level set requires cubic cells; thickness must align with the mesh")
        for name in ("cut_length_m", "load_patch_m"):
            ratio = getattr(self, name) / h
            if not math.isclose(ratio, round(ratio), abs_tol=1e-8):
                raise ValueError(f"{name} must fall on whole elements")
        if nx * ny * nz > 1_000_000:
            raise ValueError("Mesh exceeds the backend's one-million-cell run bound")

    @property
    def mesh_shape(self) -> tuple[int, int, int]:
        return self.nelx, round(self.nelx * self.ly_m / self.lx_m), round(self.nelx * self.thickness_m / self.lx_m)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "LBracket3DConfig":
        if not isinstance(data, dict):
            raise ValueError("Solver configuration must be a JSON object")
        unexpected = set(data) - {field.name for field in fields(cls)}
        if unexpected:
            raise ValueError(f"Unsupported configuration fields: {', '.join(sorted(unexpected))}")
        try:
            return cls(**data)
        except TypeError as error:
            raise ValueError(f"Invalid or missing solver inputs: {error}") from error


def reference_config() -> LBracket3DConfig:
    """Original physical choices, only for an explicitly selected benchmark."""
    return LBracket3DConfig(
        lx_m=.1, ly_m=.1, thickness_m=.012, cut_length_m=.06,
        load_patch_m=.004, hole_radius_m=.01,
        hole_centers_m=((.02, .02), (.02, .05), (.02, .08), (.05, .02), (.08, .02)),
        youngs_modulus_pa=120e9, poisson_ratio=.36, force_n=(0., -5000., 0.),
        volume_fraction=.75, stress_limit_pa=116e6,
    )
