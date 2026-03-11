import copy

import numpy as np
import astropy.units as u
from astropy import constants as const
from poliastro.bodies import Body
from poliastro.twobody import Orbit
from phringe.core.sources.planet import Planet

from lifesimmc.core.modules.base_module import BaseModule
from lifesimmc.core.resources.base_resource import BaseResource
from lifesimmc.core.resources.data_resource import DataResource


def wrap_to_2pi(angle_rad: float) -> float:
    return float(np.mod(angle_rad, 2.0 * np.pi))


class MultipleDataGenerationModule(BaseModule):
    """Reusable two-epoch data generation aligned with MultipleEpochs.py."""

    def __init__(
        self,
        n_setup_in1: str,
        n_setup_in2: str,
        time_between,
        n_data_out: str,
        offset_mode: str = "from_epoch1_end",
        deterministic_same_noise: bool = False,
    ):
        super().__init__()
        self.n_setup_in1 = n_setup_in1
        self.n_setup_in2 = n_setup_in2
        self.time_between = time_between
        self.n_data_out = n_data_out
        self.offset_mode = offset_mode
        self.deterministic_same_noise = deterministic_same_noise

    @staticmethod
    def _to_days(value) -> float:
        if isinstance(value, str):
            return float(u.Quantity(value).to(u.day).value)
        if hasattr(value, "to"):
            return float(value.to(u.day).value)
        # Keep parity with MultipleEpochs.py where numeric gap is in days.
        return float(value)

    @staticmethod
    def _clone_planet_with_true_anomaly(planet, true_anomaly_rad: float) -> Planet:
        return Planet(
            name=planet.name,
            has_orbital_motion=True,
            mass=f"{planet.mass} kg",
            radius=f"{planet.radius} m",
            temperature=f"{planet.temperature} K",
            semi_major_axis=f"{planet.semi_major_axis} m",
            eccentricity=str(float(planet.eccentricity)),
            inclination=f"{np.rad2deg(float(planet.inclination))} deg",
            raan=f"{np.rad2deg(float(planet.raan))} deg",
            argument_of_periapsis=f"{np.rad2deg(float(planet.argument_of_periapsis))} deg",
            true_anomaly=f"{np.rad2deg(float(true_anomaly_rad))} deg",
            input_spectrum=None,
        )

    def _propagate_orbit(self, planet, star_mass_kg: float, delta_days: float) -> Planet:
        mu = (const.G.value * float(star_mass_kg)) * (u.m**3 / u.s**2)
        star_body = Body(parent=None, k=mu, name="HostStar")
        orbit0 = Orbit.from_classical(
            attractor=star_body,
            a=float(planet.semi_major_axis) * u.m,
            ecc=float(planet.eccentricity) * u.one,
            inc=float(planet.inclination) * u.rad,
            raan=float(planet.raan) * u.rad,
            argp=float(planet.argument_of_periapsis) * u.rad,
            nu=float(planet.true_anomaly) * u.rad,
        )
        orbit1 = orbit0.propagate(float(delta_days) * u.day)
        nu1 = wrap_to_2pi(orbit1.nu.to_value(u.rad))
        return self._clone_planet_with_true_anomaly(planet, nu1)

    @staticmethod
    def _set_seed_if_possible(phringe, seed) -> None:
        if seed is None:
            return
        np.random.seed(int(seed))
        try:
            if hasattr(phringe, "set_seed") and callable(getattr(phringe, "set_seed")):
                phringe.set_seed(int(seed))
            elif hasattr(phringe, "seed"):
                phringe.seed = int(seed)
            elif hasattr(phringe, "_seed"):
                phringe._seed = int(seed)
        except Exception:
            pass

    def apply(self, resources: list[BaseResource]) -> tuple[DataResource, DataResource]:
        print("Generating multi-epoch synthetic data...")

        r_setup1 = self.get_resource_from_name(self.n_setup_in1)
        r_setup2 = self.get_resource_from_name(self.n_setup_in2)

        epoch1_days = self._to_days(getattr(r_setup1.observation, "total_integration_time", 0.0))
        time_between_days = self._to_days(self.time_between)
        phase_offset_days = epoch1_days + time_between_days

        seed_epoch1 = self.seed
        if self.deterministic_same_noise:
            seed_epoch2 = self.seed
        elif self.seed is None:
            seed_epoch2 = None
        else:
            seed_epoch2 = self.seed if abs(time_between_days) < 1e-12 else (self.seed + 1)

        scene_epoch1 = copy.deepcopy(r_setup1.scene)
        scene_epoch2 = copy.deepcopy(r_setup2.scene)

        star_mass_kg = float(scene_epoch1.star.mass)
        n_planets = min(len(scene_epoch1.planets), len(scene_epoch2.planets))
        for i in range(n_planets):
            planet_epoch1 = scene_epoch1.planets[i]
            if bool(getattr(planet_epoch1, "has_orbital_motion", False)):
                scene_epoch2.planets[i] = self._propagate_orbit(
                    planet_epoch1,
                    star_mass_kg,
                    phase_offset_days,
                )

        self._set_seed_if_possible(r_setup1.phringe, seed_epoch1)
        r_setup1.phringe.set(scene_epoch1)
        data_epoch1 = r_setup1.phringe.get_counts(kernels=True)

        self._set_seed_if_possible(r_setup2.phringe, seed_epoch2)
        r_setup2.phringe.set(scene_epoch2)
        data_epoch2 = r_setup2.phringe.get_counts(kernels=True)

        r_data_out_epoch1 = DataResource(name=f"{self.n_data_out}_epoch1")
        r_data_out_epoch2 = DataResource(name=f"{self.n_data_out}_epoch2")
        r_data_out_epoch1.set_data(data_epoch1)
        r_data_out_epoch2.set_data(data_epoch2)

        print("epoch1_days =", epoch1_days)
        print("time_between_days =", time_between_days)
        print("phase_offset_days =", phase_offset_days)
        print("seed_epoch1 =", seed_epoch1)
        print("seed_epoch2 =", seed_epoch2)
        print("Done")
        return r_data_out_epoch1, r_data_out_epoch2
