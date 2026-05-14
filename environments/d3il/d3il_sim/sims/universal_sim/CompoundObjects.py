from typing import Iterable, List, Optional, Sequence, Union

from environments.d3il.d3il_sim.sims.universal_sim.PrimitiveObjects import Cylinder


class CompoundObject:
	"""Base class for compound objects represented as lists of primitive objects."""

	def to_objects(self) -> List:
		raise NotImplementedError


class CylinderCompound(CompoundObject):
	"""Creates many cylinder primitives from a list of 2D/3D centers."""

	def __init__(
		self,
		centers: Iterable[Sequence[float]],
		height: Union[float, Sequence[float]],
		radius: Union[float, Sequence[float]],
		names: Optional[Sequence[str]] = None,
		base_name: str = "cyl",
		z: float = 0.0,
		init_quat: Optional[Sequence[float]] = None,
		rgba: Optional[Sequence[float]] = None,
		mass: float = 0.1,
		static: bool = True,
		visual_only: bool = False,
		solimp: Optional[Sequence[float]] = None,
		solref: Optional[Sequence[float]] = None,
	):
		self.centers = [list(center) for center in centers]
		self.height = height
		self.radius = radius
		self.names = list(names) if names is not None else None
		self.base_name = base_name
		self.z = z
		self.init_quat = list(init_quat) if init_quat is not None else [1, 0, 0, 0]
		self.rgba = list(rgba) if rgba is not None else [1, 0, 0, 1]
		self.mass = mass
		self.static = static
		self.visual_only = visual_only
		self.solimp = list(solimp) if solimp is not None else None
		self.solref = list(solref) if solref is not None else None

		if self.names is not None and len(self.names) != len(self.centers):
			raise ValueError("Length of names must match length of centers.")

	def _value_at(self, value: Union[float, Sequence[float]], i: int, field_name: str) -> float:
		if isinstance(value, (int, float)):
			return float(value)
		if len(value) != len(self.centers):
			raise ValueError(f"Length of {field_name} must match length of centers when a sequence is provided.")
		return float(value[i])

	def _center_to_xyz(self, center: Sequence[float]) -> List[float]:
		if len(center) == 2:
			return [float(center[0]), float(center[1]), float(self.z)]
		if len(center) == 3:
			return [float(center[0]), float(center[1]), float(center[2])]
		raise ValueError("Each center must contain either 2 values (x, y) or 3 values (x, y, z).")

	def to_objects(self) -> List[Cylinder]:
		objects: List[Cylinder] = []
		for i, center in enumerate(self.centers):
			name = self.names[i] if self.names is not None else f"{self.base_name}_{i}"
			radius = self._value_at(self.radius, i, "radius")
			height = self._value_at(self.height, i, "height")
			objects.append(
				Cylinder(
					name=name,
					init_pos=self._center_to_xyz(center),
					init_quat=self.init_quat,
					mass=self.mass,
					size=[radius, height],
					rgba=self.rgba,
					static=self.static,
					visual_only=self.visual_only,
					solimp=self.solimp,
					solref=self.solref,
				)
			)
		return objects
