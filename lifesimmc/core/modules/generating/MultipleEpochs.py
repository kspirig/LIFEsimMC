import numpy as np
from astropy.constants import G, au
import astropy.units as u
from matplotlib import pyplot as plt
from poliastro.bodies import Body
from poliastro.twobody import Orbit

from phringe.core.scene import Scene
from phringe.core.sources.exozodi import Exozodi
from phringe.core.sources.local_zodi import LocalZodi
from phringe.core.sources.planet import Planet
from phringe.core.sources.star import Star
from phringe.lib.beam_combiner import DoubleBracewell
from phringe.util.baseline import OptimalNullingBaseline

from lifesimmc.core.modules.generating.data_generation_module import DataGenerationModule
from lifesimmc.core.modules.loading.setup_module import SetupModule
from lifesimmc.core.pipeline import Pipeline
from lifesimmc.lib.instrument import LIFEReferenceDesign, InstrumentalNoise
from lifesimmc.lib.observation import LIFEReferenceObservation


GPU_INDEX = 6
GRID_SIZE = 60
SEED = 42

PERCENT_EPOCH_1 = 30
PERCENT_EPOCH_2 = 30
ORBIT_DAYS_REFERENCE = 14.91
# Gap between end of epoch 1 and start of epoch 2.
TIME_BETWEEN_EPOCHS_DAYS = 0
SHARED_DETECTOR_INTEGRATION_TIME = "1 h"


def wrap_to_2pi(angle_rad: float) -> float:
    return float(np.mod(angle_rad, 2.0 * np.pi))


def true_to_eccentric_anomaly(true_anomaly: float, eccentricity: float) -> float:
    e = eccentricity
    nu = true_anomaly
    beta = np.sqrt((1.0 - e) / (1.0 + e))
    return 2.0 * np.arctan2(beta * np.sin(nu / 2.0), np.cos(nu / 2.0))


def eccentric_to_true_anomaly(eccentric_anomaly: float, eccentricity: float) -> float:
    e = eccentricity
    E = eccentric_anomaly
    y = np.sqrt(1.0 + e) * np.sin(E / 2.0)
    x = np.sqrt(1.0 - e) * np.cos(E / 2.0)
    return 2.0 * np.arctan2(y, x)


def solve_kepler(mean_anomaly: float, eccentricity: float, tol: float = 1e-12, max_iter: int = 100) -> float:
    e = eccentricity
    M = mean_anomaly
    E = M if e < 0.8 else np.pi
    for _ in range(max_iter):
        f = E - e * np.sin(E) - M
        fp = 1.0 - e * np.cos(E)
        dE = -f / fp
        E += dE
        if abs(dE) < tol:
            break
    return E


def clone_planet_with_true_anomaly(planet: Planet, true_anomaly_rad: float) -> Planet:
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
        true_anomaly=f"{np.rad2deg(true_anomaly_rad)} deg",
        input_spectrum=None,
    )


def PropagateOrbit(PlanetObj: Planet, TimeDifference: float) -> Planet:
    """
    Propagate planet orbital phase by TimeDifference (in days) and return a new Planet object.
    The orbital shape/orientation parameters are unchanged; only true anomaly is advanced.
    """
    a = float(PlanetObj.semi_major_axis)
    e = float(PlanetObj.eccentricity)
    inc = float(PlanetObj.inclination)
    raan = float(PlanetObj.raan)
    argp = float(PlanetObj.argument_of_periapsis)
    nu0 = float(PlanetObj.true_anomaly)

    mu = (G.value * STAR_MASS_KG) * (u.m**3 / u.s**2)
    star_body = Body(parent=None, k=mu, name="HostStar")
    orbit0 = Orbit.from_classical(
        attractor=star_body,
        a=a * u.m,
        ecc=e * u.one,
        inc=inc * u.rad,
        raan=raan * u.rad,
        argp=argp * u.rad,
        nu=nu0 * u.rad,
    )
    orbit1 = orbit0.propagate(float(TimeDifference) * u.day)
    nu1 = wrap_to_2pi(orbit1.nu.to_value(u.rad))

    return clone_planet_with_true_anomaly(PlanetObj, nu1)


def build_observation(percent_of_orbit: float) -> LIFEReferenceObservation:
    total_integration_time_days = percent_of_orbit * ORBIT_DAYS_REFERENCE / 100.0
    return LIFEReferenceObservation(
        total_integration_time=f"{total_integration_time_days} d",
        detector_integration_time=SHARED_DETECTOR_INTEGRATION_TIME,
        nulling_baseline=OptimalNullingBaseline(
            angular_star_separation="habitable-zone",
            wavelength="15 um",
            sep_at_max_mod_eff=DoubleBracewell.sep_at_max_mod_eff[0],
        ),
    )


def build_scene(star: Star, planet: Planet) -> Scene:
    scene = Scene()
    scene.add_source(star)
    scene.add_source(LocalZodi())
    scene.add_source(Exozodi(level=3))
    scene.add_source(planet)
    return scene


def generate_data(scene: Scene, observation: LIFEReferenceObservation, seed: int, tag: str):
    inst = LIFEReferenceDesign(instrumental_noise=InstrumentalNoise.NONE)
    pipeline = Pipeline(gpu_index=GPU_INDEX, seed=seed, grid_size=GRID_SIZE)
    pipeline.add_module(
        SetupModule(
            n_setup_out=f"setup_{tag}",
            n_planet_params_out=f"params_{tag}",
            instrument=inst,
            observation=observation,
            scene=scene,
        )
    )
    setup_name = f"setup_{tag}"
    data_name = f"data_{tag}"
    try:
        pipeline.add_module(
            DataGenerationModule(
                n_setup_in=setup_name,
                n_setup_in2=setup_name,
                n_data_out=data_name,
            )
        )
    except TypeError:
        pipeline.add_module(
            DataGenerationModule(
                n_setup_in=setup_name,
                n_data_out=data_name,
            )
        )
    pipeline.run()
    return pipeline.get_resource(data_name)


def to_numpy(x):
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x)


def extract_2d_map(data_resource):
    arr = to_numpy(data_resource._data)
    # Typical data shape: [n_time, n_lambda] or [n_time, n_pix_y, n_pix_x]
    if arr.ndim == 2:
        return arr
    if arr.ndim >= 3:
        # Use first time slice and first spectral slice if needed.
        first = arr[0]
        if first.ndim == 2:
            return first
        if first.ndim >= 3:
            return first[0]
    raise ValueError(f"Unsupported data shape for plotting: {arr.shape}")


def plot_two_epoch_data(
    data_epoch_1,
    data_epoch_2,
    gap_days: float,
    epoch2_start_days: float,
    duration_epoch_1_days: float,
    duration_epoch_2_days: float,
):
    m1 = extract_2d_map(data_epoch_1)
    m2 = extract_2d_map(data_epoch_2)
    # No resampling: keep each dataset on its native grid and draw both on a common time axis.
    n_rows = min(m1.shape[0], m2.shape[0])
    m1 = m1[:n_rows, :]
    m2 = m2[:n_rows, :]

    vmin = float(np.nanmin([np.nanmin(m1), np.nanmin(m2)]))
    vmax = float(np.nanmax([np.nanmax(m1), np.nanmax(m2)]))
    t_min = 0.0
    t_max = max(duration_epoch_1_days, epoch2_start_days + duration_epoch_2_days)

    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.set_facecolor("white")  # White background where there is no data.

    im1 = ax.imshow(
        m1,
        aspect="auto",
        cmap="viridis",
        origin="lower",
        extent=[0.0, duration_epoch_1_days, 0, n_rows],
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    ax.imshow(
        m2,
        aspect="auto",
        cmap="viridis",
        origin="lower",
        extent=[epoch2_start_days, epoch2_start_days + duration_epoch_2_days, 0, n_rows],
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        alpha=0.95,
    )

    ax.axvline(duration_epoch_1_days, color="k", linestyle="--", linewidth=0.8, alpha=0.7, label="end epoch 1")
    ax.axvline(epoch2_start_days, color="k", linestyle=":", linewidth=0.8, alpha=0.7, label="start epoch 2")
    ax.set_xlim(t_min, t_max)
    plt.title(
        f"Two-Epoch Data Timeline | gap = {gap_days:.2f} d ({gap_days / 365.25:.2f} yr)"
    )
    plt.xlabel("Time [days]")
    plt.ylabel("Spectral bin index")
    plt.colorbar(im1, fraction=0.03, pad=0.02)
    plt.legend(loc="upper right", frameon=True, fontsize=8)
    plt.tight_layout()
    plt.savefig("multiple_epochs_data_timeline.png", dpi=180)
    plt.show()


star = Star(
    name="M star",
    distance="2.5 pc",
    mass="0.123 Msun",
    radius="0.147 Rsun",
    temperature="2992 K",
    right_ascension="10 hourangle",
    declination="45 deg",
)

STAR_MASS_KG = float(star.mass)

sma = float(star._habitable_zone_central_radius)
print(f"SMA = {sma:.6e} m ({sma / au.value:.6f} au)")

planet_epoch_1 = Planet(
    name="Earth Twin",
    has_orbital_motion=True,
    mass="1 Mearth",
    radius="1 Rearth",
    temperature="254 K",
    semi_major_axis=f"{sma} m",
    eccentricity="0.1",
    inclination="30 deg",
    raan="60 deg",
    argument_of_periapsis="40 deg",
    true_anomaly="100 deg",
    input_spectrum=None,
)

obs_epoch_1 = build_observation(PERCENT_EPOCH_1)
obs_epoch_2 = build_observation(PERCENT_EPOCH_2)
duration_epoch_1_days = PERCENT_EPOCH_1 * ORBIT_DAYS_REFERENCE / 100.0
duration_epoch_2_days = PERCENT_EPOCH_2 * ORBIT_DAYS_REFERENCE / 100.0
time_to_epoch2_start_days = duration_epoch_1_days + TIME_BETWEEN_EPOCHS_DAYS
planet_epoch_2 = PropagateOrbit(planet_epoch_1, time_to_epoch2_start_days)
print("Shared detector integration time:", SHARED_DETECTOR_INTEGRATION_TIME)
print("Gap between epochs [days]:", TIME_BETWEEN_EPOCHS_DAYS)
print("Epoch 2 start after epoch 1 t0 [days]:", time_to_epoch2_start_days)

scene_epoch_1 = build_scene(star, planet_epoch_1)
scene_epoch_2 = build_scene(star, planet_epoch_2)

data_epoch_1 = generate_data(scene_epoch_1, obs_epoch_1, seed=SEED, tag="epoch1")
seed_epoch_2 = SEED if abs(TIME_BETWEEN_EPOCHS_DAYS) < 1e-12 else (SEED + 1)
data_epoch_2 = generate_data(scene_epoch_2, obs_epoch_2, seed=seed_epoch_2, tag="epoch2")

print("Epoch 1 true anomaly [deg]:", np.rad2deg(float(planet_epoch_1.true_anomaly)))
print("Epoch 2 true anomaly [deg]:", np.rad2deg(float(planet_epoch_2.true_anomaly)))
print("Data epoch 1 shape:", tuple(data_epoch_1._data.shape))
print("Data epoch 2 shape:", tuple(data_epoch_2._data.shape))

plot_two_epoch_data(
    data_epoch_1,
    data_epoch_2,
    TIME_BETWEEN_EPOCHS_DAYS,
    time_to_epoch2_start_days,
    duration_epoch_1_days,
    duration_epoch_2_days,
)
