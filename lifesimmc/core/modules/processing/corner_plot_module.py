from pathlib import Path
import math

import astropy.constants as const
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from lifesimmc.core.modules.base_module import BaseModule
from lifesimmc.core.resources.base_resource import BaseResource
from lifesimmc.core.resources.plot_resource import PlotResource


COLOR_MAIN = "#0072B2"
COLOR_TRUTH = "red"
PLOT_LABELS = ["a [au]", "e", "$i$ [deg]", "$\\Omega $ [deg]", "$\\omega $ [deg]", "$ \\nu_0 $ [deg]"]

HZ_TIME_THRESHOLD_PERCENT = 99.99
HZ_TIME_CONTOUR_LABEL_PERCENT = 100.0
HZ_CONTOUR_LABEL_POSITION = (1.0, 0.2)
HZ_GRID_SMA_RANGE_AU = (0.05, 5.0)
HZ_GRID_ECC_RANGE = (0.0, 0.99)
HZ_GRID_SIZE_SMA = 1000
HZ_GRID_SIZE_ECC = 1000
HZ_PRIOR_GRID_SIZE = 30000
HZ_PERCENT_VALIDATION_TOL = 1e-8

DEFAULT_LUMINOSITY_SOLAR = 1.0
DEFAULT_STAR_TEMPERATURE_K = 5772.0
DEFAULT_ECCENTRICITY_UPPER = 0.99
DEFAULT_BAYES_FACTOR_EPSILON = 1e-6
DEFAULT_RESAMPLE_SEED_OFFSET = 100_000

L_SUN_FOR_HZ_W = 3.86e26
ECC_BETA_ALPHA = 0.867
ECC_BETA_BETA = 3.03
BETA_CDF_TABLE_SIZE = 131072

_HZ_DEFINITION_CACHE = {}
_BETA_TABLE_CACHE = {}
_HZ_PRIOR_PROBABILITY_CACHE = {}


def _density_per_point_2d(x, y, bins, weights=None):
    H, x_edges, y_edges = np.histogram2d(x, y, bins=bins, weights=weights)
    x_idx = np.searchsorted(x_edges, x, side="right") - 1
    y_idx = np.searchsorted(y_edges, y, side="right") - 1
    x_idx = np.clip(x_idx, 0, H.shape[0] - 1)
    y_idx = np.clip(y_idx, 0, H.shape[1] - 1)
    return H[x_idx, y_idx]


def _overlay_density_colored_points(fig, samples, bins, cmap="plasma", s=2.0, alpha=0.8, weights=None):
    n_dim = samples.shape[1]
    axes = np.asarray(fig.axes[: n_dim * n_dim]).reshape((n_dim, n_dim))

    panel_data = []
    density_min = None
    density_max = None
    for i in range(1, n_dim):
        for j in range(i):
            x = samples[:, j]
            y = samples[:, i]
            density = _density_per_point_2d(x, y, bins=bins, weights=weights)
            panel_data.append((i, j, x, y, density))
            positive_density = density[np.isfinite(density) & (density > 0.0)]
            if positive_density.size == 0:
                continue
            dmin = float(np.min(positive_density))
            dmax = float(np.max(positive_density))
            density_min = dmin if density_min is None else min(density_min, dmin)
            density_max = dmax if density_max is None else max(density_max, dmax)

    if density_min is None:
        density_min = 1.0
        density_max = 1.0
    if density_max <= density_min:
        density_max = density_min * (1.0 + 1e-12)

    norm = LogNorm(vmin=density_min, vmax=density_max)
    for i, j, x, y, density in panel_data:
        axes[i, j].scatter(
            x,
            y,
            c=np.maximum(density, density_min),
            cmap=cmap,
            norm=norm,
            s=s,
            alpha=alpha,
            edgecolors="none",
            rasterized=True,
        )
    return norm


def _union_bbox_of_axes(axes):
    boxes = [ax.get_position() for ax in axes]
    x0 = min(box.x0 for box in boxes)
    y0 = min(box.y0 for box in boxes)
    x1 = max(box.x1 for box in boxes)
    y1 = max(box.y1 for box in boxes)
    return [x0, y0, x1 - x0, y1 - y0]


def _validate_hz_percent_array(hz_percent, label="HZ percent"):
    hz_percent = np.asarray(hz_percent, dtype=float)
    if not np.all(np.isfinite(hz_percent)):
        raise ValueError(f"{label} contains non-finite values.")
    if hz_percent.size == 0:
        raise ValueError(f"{label} is empty.")

    minimum = float(np.min(hz_percent))
    maximum = float(np.max(hz_percent))
    tol = float(HZ_PERCENT_VALIDATION_TOL)
    if minimum < -tol or maximum > 100.0 + tol:
        raise ValueError(f"{label} must lie in [0, 100] percent; got range [{minimum}, {maximum}].")
    return np.clip(hz_percent, 0.0, 100.0)


def _as_float_or_none(value):
    if value is None:
        return None
    if hasattr(value, "si") and hasattr(value.si, "value"):
        value = value.si.value
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(result):
        return None
    return result


def _kopparapu_limits_au(star_temperature_k, luminosity_solar, inner_coefficients, outer_coefficients):
    delta_t = float(star_temperature_k) - 5780.0

    def effective_flux(coefficients):
        return (
            coefficients["s_eff_sun"]
            + coefficients["a"] * delta_t
            + coefficients["b"] * delta_t**2
            + coefficients["c"] * delta_t**3
            + coefficients["d"] * delta_t**4
        )

    luminosity_solar = float(luminosity_solar)
    inner_au = np.sqrt(luminosity_solar / effective_flux(inner_coefficients))
    outer_au = np.sqrt(luminosity_solar / effective_flux(outer_coefficients))
    return float(inner_au), float(outer_au)


def _empirical_hz_limits_au(star_temperature_k, luminosity_solar):
    return _kopparapu_limits_au(
        star_temperature_k=star_temperature_k,
        luminosity_solar=luminosity_solar,
        inner_coefficients={
            "s_eff_sun": 1.776,
            "a": 2.136e-4,
            "b": 2.533e-8,
            "c": -1.332e-11,
            "d": -3.097e-15,
        },
        outer_coefficients={
            "s_eff_sun": 0.320,
            "a": 5.547e-5,
            "b": 1.526e-9,
            "c": -2.874e-12,
            "d": -5.011e-16,
        },
    )


def _conservative_hz_limits_au(star_temperature_k, luminosity_solar):
    return _kopparapu_limits_au(
        star_temperature_k=star_temperature_k,
        luminosity_solar=luminosity_solar,
        inner_coefficients={
            "s_eff_sun": 1.107,
            "a": 1.332e-4,
            "b": 1.580e-8,
            "c": -8.308e-12,
            "d": -1.931e-15,
        },
        outer_coefficients={
            "s_eff_sun": 0.356,
            "a": 6.171e-5,
            "b": 1.698e-9,
            "c": -3.198e-12,
            "d": -5.575e-16,
        },
    )


def _cumulative_orbital_time_fraction_inside_radius(sma_au, ecc, radius_au):
    sma_au, ecc = np.broadcast_arrays(
        np.asarray(sma_au, dtype=float),
        np.asarray(ecc, dtype=float),
    )
    radius_au = float(radius_au)
    if radius_au <= 0.0:
        return np.zeros_like(sma_au, dtype=float)
    if np.any(sma_au <= 0.0):
        raise ValueError("Semi-major axes must be positive when calculating time in HZ.")
    if np.any((ecc < 0.0) | (ecc >= 1.0)):
        raise ValueError("Eccentricities must satisfy 0 <= e < 1 when calculating time in HZ.")

    fraction = np.zeros_like(sma_au, dtype=float)
    circular = ecc <= 0.0
    fraction[circular] = sma_au[circular] <= radius_au

    eccentric = ~circular
    if np.any(eccentric):
        a = sma_au[eccentric]
        e = ecc[eccentric]
        c = (1.0 - radius_au / a) / e

        inside_all = c <= -1.0
        inside_none = c >= 1.0
        partial = ~(inside_all | inside_none)

        eccentric_fraction = np.zeros_like(a, dtype=float)
        eccentric_fraction[inside_all] = 1.0
        if np.any(partial):
            eccentric_anomaly = np.arccos(c[partial])
            mean_anomaly = eccentric_anomaly - e[partial] * np.sin(eccentric_anomaly)
            eccentric_fraction[partial] = mean_anomaly / np.pi
        fraction[eccentric] = eccentric_fraction

    return fraction


def _time_in_hz_percent_for_orbits(sma_au, ecc, inner_au, outer_au):
    outer_fraction = _cumulative_orbital_time_fraction_inside_radius(sma_au, ecc, outer_au)
    inner_fraction = _cumulative_orbital_time_fraction_inside_radius(sma_au, ecc, inner_au)
    return np.clip(100.0 * (outer_fraction - inner_fraction), 0.0, 100.0)


def _generate_time_in_hz_percent_grid(sma_axis_au, ecc_axis, inner_au, outer_au):
    sma_grid, ecc_grid = np.meshgrid(
        np.asarray(sma_axis_au, dtype=float),
        np.asarray(ecc_axis, dtype=float),
        indexing="ij",
    )
    percent = _time_in_hz_percent_for_orbits(
        sma_au=sma_grid,
        ecc=ecc_grid,
        inner_au=inner_au,
        outer_au=outer_au,
    )
    return _validate_hz_percent_array(percent, label="generated HZ percent grid")


def _build_hz_definition(hz_definition, star_temperature_k, grid_size_sma, grid_size_ecc):
    if isinstance(hz_definition, dict):
        return hz_definition

    name = "EmpiricalHZ" if hz_definition is None else str(hz_definition)
    key = (name, round(float(star_temperature_k), 8), int(grid_size_sma), int(grid_size_ecc))
    cached = _HZ_DEFINITION_CACHE.get(key)
    if cached is not None:
        return cached

    if int(grid_size_sma) < 2 or int(grid_size_ecc) < 2:
        raise ValueError("HZ grid sizes must be at least 2.")

    if name == "EmpiricalHZ":
        inner_au, outer_au = _empirical_hz_limits_au(
            star_temperature_k=star_temperature_k,
            luminosity_solar=DEFAULT_LUMINOSITY_SOLAR,
        )
        display_label = "Empirical HZ"
        inner_edge_name = "Recent Venus"
        outer_edge_name = "Early Mars"
    elif name == "ConservativeHZ":
        inner_au, outer_au = _conservative_hz_limits_au(
            star_temperature_k=star_temperature_k,
            luminosity_solar=DEFAULT_LUMINOSITY_SOLAR,
        )
        display_label = "Conservative HZ"
        inner_edge_name = "Runaway Greenhouse"
        outer_edge_name = "Maximum Greenhouse"
    else:
        raise KeyError("Unknown HZ definition {!r}. Valid choices: EmpiricalHZ, ConservativeHZ.".format(name))

    sma_normalized_au = np.geomspace(
        float(HZ_GRID_SMA_RANGE_AU[0]),
        float(HZ_GRID_SMA_RANGE_AU[1]),
        int(grid_size_sma),
    )
    eccentricities = np.linspace(
        float(HZ_GRID_ECC_RANGE[0]),
        float(HZ_GRID_ECC_RANGE[1]),
        int(grid_size_ecc),
    )
    percent_grid = _generate_time_in_hz_percent_grid(
        sma_axis_au=sma_normalized_au,
        ecc_axis=eccentricities,
        inner_au=inner_au,
        outer_au=outer_au,
    )
    definition = {
        "name": name,
        "display_label": display_label,
        "inner_edge_name": inner_edge_name,
        "outer_edge_name": outer_edge_name,
        "sma_normalized_au": sma_normalized_au,
        "eccentricities": eccentricities,
        "percent_grid": percent_grid,
        "circular_inner_normalized_au": float(inner_au),
        "circular_outer_normalized_au": float(outer_au),
    }
    _HZ_DEFINITION_CACHE[key] = definition
    return definition


def _evaluate_hz_percent(sma_au, ecc, luminosity_solar, hz_definition):
    luminosity_solar = float(luminosity_solar)
    if luminosity_solar <= 0.0:
        raise ValueError("luminosity_solar must be positive.")
    scale = np.sqrt(luminosity_solar)
    return _time_in_hz_percent_for_orbits(
        sma_au=np.asarray(sma_au, dtype=float) / scale,
        ecc=np.asarray(ecc, dtype=float),
        inner_au=float(hz_definition["circular_inner_normalized_au"]),
        outer_au=float(hz_definition["circular_outer_normalized_au"]),
    )


def _is_permanently_in_hz(sma_au, ecc, luminosity_solar, hz_definition):
    sma_au, ecc = np.broadcast_arrays(
        np.asarray(sma_au, dtype=float),
        np.asarray(ecc, dtype=float),
    )
    luminosity_solar = float(luminosity_solar)
    if luminosity_solar <= 0.0:
        raise ValueError("luminosity_solar must be positive.")
    if np.any(sma_au <= 0.0):
        raise ValueError("Semi-major axes must be positive for permanent HZ classification.")
    if np.any((ecc < 0.0) | (ecc >= 1.0)):
        raise ValueError("Eccentricities must satisfy 0 <= e < 1 for permanent HZ classification.")

    scale = np.sqrt(luminosity_solar)
    inner_au = float(hz_definition["circular_inner_normalized_au"]) * scale
    outer_au = float(hz_definition["circular_outer_normalized_au"]) * scale
    periastron_au = sma_au * (1.0 - ecc)
    apoastron_au = sma_au * (1.0 + ecc)
    return (periastron_au >= inner_au) & (apoastron_au <= outer_au)


def _beta_table():
    cached = _BETA_TABLE_CACHE.get("default")
    if cached is not None:
        return cached

    e_min = 1e-8
    e_max = 1.0 - 1e-8
    e_grid = np.linspace(e_min, e_max, int(BETA_CDF_TABLE_SIZE), dtype=np.float64)
    log_norm = (
        math.lgamma(ECC_BETA_ALPHA)
        + math.lgamma(ECC_BETA_BETA)
        - math.lgamma(ECC_BETA_ALPHA + ECC_BETA_BETA)
    )
    log_pdf = (
        (ECC_BETA_ALPHA - 1.0) * np.log(e_grid)
        + (ECC_BETA_BETA - 1.0) * np.log1p(-e_grid)
        - log_norm
    )
    pdf = np.exp(log_pdf)
    cdf = np.empty_like(e_grid)
    cdf[0] = 0.0
    cdf[1:] = np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(e_grid))
    cdf /= cdf[-1]
    cdf = np.maximum.accumulate(cdf)

    e_table = np.concatenate(([0.0], e_grid, [1.0]))
    cdf_table = np.concatenate(([0.0], cdf, [1.0]))
    cdf_table = np.maximum.accumulate(cdf_table)
    cached = (e_table, cdf_table)
    _BETA_TABLE_CACHE["default"] = cached
    return cached


def _beta_cdf(eccentricity):
    e_table, cdf_table = _beta_table()
    eccentricity = np.clip(np.asarray(eccentricity, dtype=float), 0.0, 1.0)
    result = np.interp(eccentricity, e_table, cdf_table)
    if result.ndim == 0:
        return float(result)
    return result


def _beta_ppf(probability):
    e_table, cdf_table = _beta_table()
    probability = np.clip(np.asarray(probability, dtype=float), 0.0, 1.0)
    result = np.interp(probability, cdf_table, e_table)
    if result.ndim == 0:
        return float(result)
    return result


def _normalized_or_uniform_weights(weights, n_samples, label="Posterior weights"):
    n_samples = int(n_samples)
    if n_samples <= 0:
        raise ValueError(f"{label} cannot be normalized for zero samples.")

    if weights is None:
        return np.full(n_samples, 1.0 / float(n_samples), dtype=float)

    weights = np.asarray(weights, dtype=float).reshape(-1)
    if weights.shape != (n_samples,):
        raise ValueError(f"{label} must have one entry per posterior point.")
    if not np.all(np.isfinite(weights)):
        raise ValueError(f"{label} contains non-finite values.")
    if np.any(weights < 0.0):
        raise ValueError(f"{label} contains negative values.")

    weight_sum = float(np.sum(weights))
    if weight_sum <= 0.0:
        raise ValueError(f"{label} has a non-positive sum.")
    return weights / weight_sum


def _posterior_hz_probability(samples_phys, weights, luminosity_solar, hz_definition):
    samples = np.asarray(samples_phys, dtype=float)
    if samples.ndim != 2 or samples.shape[1] < 2:
        raise ValueError(f"Expected posterior samples with shape (N, D>=2), got {samples.shape}.")

    normalized_weights = _normalized_or_uniform_weights(weights, samples.shape[0])
    in_hz = _is_permanently_in_hz(
        sma_au=samples[:, 0] / const.au.value,
        ecc=samples[:, 1],
        luminosity_solar=luminosity_solar,
        hz_definition=hz_definition,
    )
    in_hz = np.asarray(in_hz, dtype=bool).reshape(-1)
    if in_hz.shape != (samples.shape[0],):
        raise ValueError("in_hz must have one entry per posterior point.")

    p_in = float(np.sum(normalized_weights[in_hz]))
    p_out = float(np.sum(normalized_weights[~in_hz]))
    total = p_in + p_out
    if total <= 0.0:
        raise ValueError("Posterior HZ and non-HZ probabilities have a non-positive total.")
    return p_in / total, p_out / total


def _posterior_weight_ess(weights, n_samples):
    normalized_weights = _normalized_or_uniform_weights(weights, int(n_samples))
    return float(1.0 / np.sum(normalized_weights * normalized_weights))


def _integrate_hz_prior_probability(
        a_min_au,
        a_max_au,
        e_min,
        e_max,
        luminosity_solar,
        hz_definition,
        n_grid=HZ_PRIOR_GRID_SIZE,
):
    name = str(hz_definition.get("name", "custom"))
    cache_key = (
        name,
        round(float(hz_definition["circular_inner_normalized_au"]), 12),
        round(float(hz_definition["circular_outer_normalized_au"]), 12),
        float(a_min_au),
        float(a_max_au),
        float(e_min),
        float(e_max),
        round(float(luminosity_solar), 12),
        int(n_grid),
        float(HZ_TIME_THRESHOLD_PERCENT),
    )
    cached = _HZ_PRIOR_PROBABILITY_CACHE.get(cache_key)
    if cached is not None:
        return cached

    a_min_au = float(a_min_au)
    a_max_au = float(a_max_au)
    e_min = float(e_min)
    e_max = float(e_max)
    luminosity_solar = float(luminosity_solar)
    n_grid = int(n_grid)

    if not (0.0 < a_min_au < a_max_au):
        raise ValueError(f"Invalid SMA prior bounds: {(a_min_au, a_max_au)}.")
    if not (0.0 <= e_min < e_max < 1.0):
        raise ValueError(f"Invalid eccentricity prior bounds: {(e_min, e_max)}.")
    if luminosity_solar <= 0.0:
        raise ValueError("luminosity_solar must be positive.")
    if n_grid < 2:
        raise ValueError("n_grid must be at least 2 for HZ prior integration.")

    u_min = _beta_cdf(e_min)
    u_max = _beta_cdf(e_max)
    if u_max <= u_min:
        raise ValueError(f"Truncated eccentricity beta prior has zero mass on {(e_min, e_max)}.")

    u_grid = np.linspace(u_min, u_max, n_grid, dtype=float)
    eccentricity = _beta_ppf(u_grid)

    scale = np.sqrt(luminosity_solar)
    inner_au = float(hz_definition["circular_inner_normalized_au"]) * scale
    outer_au = float(hz_definition["circular_outer_normalized_au"]) * scale
    log_a_width = np.log(a_max_au / a_min_au)

    lower = np.maximum(a_min_au, inner_au / (1.0 - eccentricity))
    upper = np.minimum(a_max_au, outer_au / (1.0 + eccentricity))
    prior_fraction_at_e = np.zeros_like(eccentricity, dtype=float)
    valid = upper > lower
    prior_fraction_at_e[valid] = np.log(upper[valid] / lower[valid]) / log_a_width

    pi_in = float(np.trapz(prior_fraction_at_e, u_grid) / (u_max - u_min))
    pi_in = float(np.clip(pi_in, 0.0, 1.0))
    result = (pi_in, 1.0 - pi_in)
    _HZ_PRIOR_PROBABILITY_CACHE[cache_key] = result
    return result


def _compute_bayes_factor_with_ns_residual(p_i, p_j, pi_i, pi_j, epsilon):
    p_i = float(p_i)
    p_j = float(p_j)
    pi_i = float(pi_i)
    pi_j = float(pi_j)
    epsilon = float(epsilon)

    if not all(np.isfinite(value) for value in (p_i, p_j, pi_i, pi_j)):
        raise ValueError(f"Bayes-factor probabilities must be finite, got {p_i}, {p_j}, {pi_i}, {pi_j}.")
    if p_i < 0.0 or p_j < 0.0:
        raise ValueError(f"Posterior probabilities must be non-negative, got {p_i}, {p_j}.")
    if pi_i <= 0.0 or pi_j <= 0.0:
        raise ValueError(f"Prior probabilities must be positive, got {pi_i}, {pi_j}.")
    if not (0.0 < epsilon < 1.0):
        raise ValueError(f"Bayes-factor epsilon must lie inside (0, 1), got {epsilon}.")

    if p_i > 0.0 and p_j > 0.0:
        lnK = np.log(p_i) - np.log(p_j) - np.log(pi_i) + np.log(pi_j)
        return float(lnK), False, "measured"

    if p_i <= 0.0 and p_j <= 0.0:
        return np.nan, True, "unresolved_both"

    if p_j <= 0.0:
        lnK_lower = np.log(p_i) - np.log(epsilon) - np.log(pi_i) + np.log(pi_j)
        return float(lnK_lower), True, "lower_ns_residual"

    lnK_upper = np.log(epsilon) - np.log(p_j) - np.log(pi_i) + np.log(pi_j)
    return float(lnK_upper), True, "upper_ns_residual"


def _compute_hz_bayes_metrics(
        samples_phys,
        weights,
        luminosity_solar,
        prior_sma_min_au,
        prior_sma_max_au,
        prior_ecc_min,
        prior_ecc_max,
        hz_definition,
        bayes_factor_epsilon,
        hz_prior_grid_size,
):
    samples = np.asarray(samples_phys, dtype=float)
    p_in, p_out = _posterior_hz_probability(
        samples_phys=samples,
        weights=weights,
        luminosity_solar=luminosity_solar,
        hz_definition=hz_definition,
    )
    pi_in, pi_out = _integrate_hz_prior_probability(
        a_min_au=prior_sma_min_au,
        a_max_au=prior_sma_max_au,
        e_min=prior_ecc_min,
        e_max=prior_ecc_max,
        luminosity_solar=luminosity_solar,
        hz_definition=hz_definition,
        n_grid=hz_prior_grid_size,
    )
    lnK, is_limit, limit_type = _compute_bayes_factor_with_ns_residual(
        p_i=p_in,
        p_j=p_out,
        pi_i=pi_in,
        pi_j=pi_out,
        epsilon=bayes_factor_epsilon,
    )
    hz_percent = _evaluate_hz_percent(
        sma_au=samples[:, 0] / const.au.value,
        ecc=samples[:, 1],
        luminosity_solar=luminosity_solar,
        hz_definition=hz_definition,
    )
    hz_percent = _validate_hz_percent_array(hz_percent, label="posterior HZ percent").reshape(-1)
    normalized_weights = _normalized_or_uniform_weights(weights, samples.shape[0])
    posterior_f_hz_mean = float(np.sum(normalized_weights * np.clip(hz_percent / 100.0, 0.0, 1.0)))

    return {
        "bayes_factor_lnK": float(lnK),
        "bayes_factor_2lnK": float(2.0 * lnK),
        "bayes_factor_is_limit": int(is_limit),
        "bayes_factor_limit_type": limit_type,
        "posterior_hz_probability": float(p_in),
        "posterior_nonhz_probability": float(p_out),
        "prior_hz_probability": float(pi_in),
        "prior_nonhz_probability": float(pi_out),
        "posterior_weight_ess": _posterior_weight_ess(weights, samples.shape[0]),
        "posterior_f_hz_mean": posterior_f_hz_mean,
        "hz_definition": str(hz_definition.get("name", "custom")),
        "hz_definition_label": str(hz_definition.get("display_label", "HZ")),
    }


def _make_hz_plot(ax, hz_definition):
    heatmap_cmap = "Greys"
    sma_normalized_au = np.asarray(hz_definition["sma_normalized_au"], dtype=float)
    eccentricities = np.asarray(hz_definition["eccentricities"], dtype=float)
    percent_grid = np.asarray(hz_definition["percent_grid"], dtype=float)

    mesh = ax.pcolormesh(
        sma_normalized_au,
        eccentricities,
        percent_grid.T,
        shading="auto",
        cmap=heatmap_cmap,
    )

    if float(np.nanmin(percent_grid)) <= HZ_TIME_THRESHOLD_PERCENT <= float(np.nanmax(percent_grid)):
        sma_mesh, ecc_mesh = np.meshgrid(sma_normalized_au, eccentricities, indexing="xy")
        contour = ax.contour(
            sma_mesh,
            ecc_mesh,
            percent_grid.T,
            levels=[HZ_TIME_THRESHOLD_PERCENT],
            colors="red",
            linewidths=2.0,
        )
        try:
            ax.clabel(
                contour,
                fmt={HZ_TIME_THRESHOLD_PERCENT: rf"${HZ_TIME_CONTOUR_LABEL_PERCENT:g}\%$"},
                inline=True,
                fontsize=16,
                manual=[HZ_CONTOUR_LABEL_POSITION],
            )
        except ValueError:
            ax.clabel(
                contour,
                fmt={HZ_TIME_THRESHOLD_PERCENT: rf"${HZ_TIME_CONTOUR_LABEL_PERCENT:g}\%$"},
                inline=True,
                fontsize=16,
            )

    ax.set_xscale("log")
    ax.set_xlabel(r"$\frac{a}{\sqrt{L_{\star}/L_{\odot}}}\,[\mathrm{au}]$", fontsize=16)
    ax.set_ylabel(r"$e$", fontsize=14)
    ax.set_title(r"$\mathrm{Time\ in\ HZ}$", fontsize=14)
    ax.tick_params(axis="both", which="both", labelsize=10)
    ax.set_ylim(0.0, 0.99)
    ax.grid(True, which="both", alpha=0.25)
    return mesh


def _format_bayes_factor_label(hz_metrics):
    two_lnK = float(hz_metrics["bayes_factor_2lnK"])
    limit_type = str(hz_metrics["bayes_factor_limit_type"])
    if not np.isfinite(two_lnK):
        return r"$2\,\ln K_{\rm HZ/nonHZ}$ unresolved"
    if limit_type in {"lower_ns_residual", "lower_inside"}:
        return rf"$2\,\ln K_{{\rm HZ/nonHZ}} > {two_lnK:.2f}$"
    if limit_type in {"upper_ns_residual", "upper_outside"}:
        return rf"$2\,\ln K_{{\rm HZ/nonHZ}} < {two_lnK:.2f}$"
    return rf"$2\,\ln K_{{\rm HZ/nonHZ}} = {two_lnK:.2f}$"


def _add_hz_overlay(
        fig,
        samples,
        weights,
        truths,
        plot_sma_min,
        plot_sma_max,
        luminosity_solar,
        hz_definition,
        density_norm,
        hz_metrics,
):
    n_dim = samples.shape[1]
    axes = np.asarray(fig.axes[: n_dim * n_dim]).reshape((n_dim, n_dim))
    overlay_axes = []
    for row in range(0, min(3, n_dim)):
        for col in range(max(0, n_dim - 3), n_dim):
            overlay_axes.append(axes[row, col])

    overlay_bbox = _union_bbox_of_axes(overlay_axes)
    pad_x = 0.05
    pad_y = 0.05
    overlay_bbox = [
        overlay_bbox[0] + pad_x,
        overlay_bbox[1] + pad_y,
        max(overlay_bbox[2] - 2.0 * pad_x, 0.05),
        max(overlay_bbox[3] - 2.0 * pad_y, 0.05),
    ]

    for ax in overlay_axes:
        ax.set_visible(False)

    ax_hz = fig.add_axes(overlay_bbox, zorder=20)
    mesh = _make_hz_plot(ax_hz, hz_definition=hz_definition)

    luminosity_solar = float(luminosity_solar)
    scale = np.sqrt(luminosity_solar)
    ax_hz.set_xlim(
        float(plot_sma_min) / scale,
        float(plot_sma_max) / scale,
    )

    if truths is not None and np.all(np.isfinite(truths[:2])):
        ax_hz.scatter(
            float(truths[0]) / scale,
            float(truths[1]),
            marker="*",
            s=180,
            facecolors=COLOR_TRUTH,
            edgecolors="black",
            linewidths=0.8,
            zorder=30,
        )

    x_hz = samples[:, 0] / scale
    y_hz = samples[:, 1]
    density = _density_per_point_2d(x_hz, y_hz, bins=1000, weights=weights)
    ax_hz.scatter(
        x_hz,
        y_hz,
        c=np.maximum(density, density_norm.vmin),
        cmap="plasma",
        norm=density_norm,
        s=2.0,
        alpha=0.8,
        edgecolors="none",
        rasterized=True,
    )

    cbar = fig.colorbar(mesh, ax=ax_hz, pad=0.01, fraction=0.05)
    cbar.set_label(r"$\mathrm{Time\ in\ HZ}\ [\%]$", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    if hz_metrics is not None:
        ax_hz.text(
            0.97,
            0.97,
            _format_bayes_factor_label(hz_metrics),
            transform=ax_hz.transAxes,
            ha="right",
            va="top",
            fontsize=12,
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="black"),
            zorder=30,
        )


class CornerPlotModule(BaseModule):
    """Create a posterior corner plot with a Time-in-HZ inset."""

    def __init__(
            self,
            n_inference_in: str,
            n_plot_out: str,
            out_path: str = "orbit_corner.png",
            n_planets_truth_in: str = None,
            n_setup_in: str = None,
            truths: np.ndarray = None,
            plot_sma_min: float = None,
            plot_sma_max: float = None,
            show: bool = True,
            include_hz_plot: bool = True,
            hz_definition: str = "EmpiricalHZ",
            resample_for_plot: bool = True,
    ):
        super().__init__()
        self.n_inference_in = n_inference_in
        self.n_plot_out = n_plot_out
        self.out_path = out_path
        self.n_planets_truth_in = n_planets_truth_in
        self.n_setup_in = n_setup_in
        self.truths = truths
        self.plot_sma_min = plot_sma_min
        self.plot_sma_max = plot_sma_max
        self.show = show
        self.include_hz_plot = bool(include_hz_plot)
        self.hz_definition = hz_definition
        self.resample_for_plot = bool(resample_for_plot)
        self.last_hz_metrics = None
        self.last_plot_sample_count = None
        self.last_plot_resample_method = None
        self.last_stellar_context = None

    def run(self, pipeline_resources: list[BaseResource]) -> tuple[PlotResource]:
        """Create and save the corner plot."""
        print("Generating corner plot...")

        inference = self.get_resource_from_name(self.n_inference_in)
        samples_phys = self._physical_samples(inference.samples)
        samples = self._plot_samples(samples_phys)
        weights = self._normalized_weights(getattr(inference, "posterior_weights", None), samples.shape[0])
        truths = self._truths_plot() if self.truths is None else self._validated_truths(self.truths)

        hz_definition = None
        hz_metrics = None
        if self.include_hz_plot:
            luminosity_solar, star_temperature_k = self._stellar_context()
            hz_definition = _build_hz_definition(
                hz_definition=self.hz_definition,
                star_temperature_k=star_temperature_k,
                grid_size_sma=HZ_GRID_SIZE_SMA,
                grid_size_ecc=HZ_GRID_SIZE_ECC,
            )
            prior_sma_min_au, prior_sma_max_au, prior_ecc_min, prior_ecc_max = self._prior_bounds(
                inference=inference,
                samples_phys=samples_phys,
            )
            hz_metrics = _compute_hz_bayes_metrics(
                samples_phys=samples_phys,
                weights=weights,
                luminosity_solar=luminosity_solar,
                prior_sma_min_au=prior_sma_min_au,
                prior_sma_max_au=prior_sma_max_au,
                prior_ecc_min=prior_ecc_min,
                prior_ecc_max=prior_ecc_max,
                hz_definition=hz_definition,
                bayes_factor_epsilon=DEFAULT_BAYES_FACTOR_EPSILON,
                hz_prior_grid_size=HZ_PRIOR_GRID_SIZE,
            )
            hz_metrics["luminosity_solar"] = float(luminosity_solar)
            hz_metrics["star_temperature_k"] = float(star_temperature_k)
            hz_metrics["prior_sma_min_au"] = float(prior_sma_min_au)
            hz_metrics["prior_sma_max_au"] = float(prior_sma_max_au)
            hz_metrics["prior_ecc_min"] = float(prior_ecc_min)
            hz_metrics["prior_ecc_max"] = float(prior_ecc_max)
            self.last_hz_metrics = hz_metrics
            print(
                "HZ Bayes factor: "
                f"2 ln K = {hz_metrics['bayes_factor_2lnK']:.6g} "
                f"({hz_metrics['bayes_factor_limit_type']})"
            )
        else:
            luminosity_solar = DEFAULT_LUMINOSITY_SOLAR
            self.last_hz_metrics = None

        plot_samples, plot_weights, resample_method = self._plot_samples_for_visualization(samples, weights)
        self.last_plot_sample_count = int(plot_samples.shape[0])
        self.last_plot_resample_method = resample_method

        fig = self._make_plot(
            samples=plot_samples,
            weights=plot_weights,
            truths=truths,
            plot_sma_min=self._plot_sma_min(inference, samples),
            plot_sma_max=self._plot_sma_max(inference, samples),
            hz_definition=hz_definition,
            luminosity_solar=luminosity_solar,
            hz_metrics=hz_metrics,
        )

        out_path = Path(self.out_path)
        if out_path.parent != Path("."):
            out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        if self.show:
            plt.show()
        plt.close(fig)

        print(f"Saved corner plot: {out_path}")
        print("Done")
        return PlotResource(name=self.n_plot_out, path=str(out_path)),

    @staticmethod
    def _physical_samples(samples):
        samples = np.asarray(samples, dtype=float)
        if samples.ndim != 2 or samples.shape[1] != 6:
            raise ValueError(f"Expected inference samples with shape (N, 6), got {samples.shape}.")
        if samples.shape[0] == 0:
            raise ValueError("Cannot plot an empty posterior sample set.")
        if not np.all(np.isfinite(samples)):
            raise ValueError("Cannot plot posterior samples containing non-finite values.")
        return samples

    @staticmethod
    def _plot_samples(samples):
        samples_plot = np.asarray(samples, dtype=float).copy()
        samples_plot[:, 0] /= const.au.value
        samples_plot[:, 2:] = np.rad2deg(samples_plot[:, 2:])
        return samples_plot

    @staticmethod
    def _normalized_weights(weights, n_samples):
        if weights is None:
            return None
        return _normalized_or_uniform_weights(weights, n_samples)

    @staticmethod
    def _validated_truths(truths):
        truths = np.asarray(truths, dtype=float).reshape(-1)
        if truths.shape != (6,):
            raise ValueError(f"Expected truths with shape (6,), got {truths.shape}.")
        if not np.all(np.isfinite(truths)):
            raise ValueError("truths must contain only finite values.")
        return truths

    def _truths_plot(self):
        if self.n_planets_truth_in is None:
            return None
        planets = self.get_resource_from_name(self.n_planets_truth_in)
        if not planets.collection:
            raise ValueError(f"Planet truth resource '{self.n_planets_truth_in}' is empty.")

        planet = planets.collection[0].planet
        return np.array(
            [
                float(planet.semi_major_axis) / const.au.value,
                float(planet.eccentricity),
                np.rad2deg(float(planet.inclination)),
                np.rad2deg(float(planet.raan)),
                np.rad2deg(float(planet.argument_of_periapsis)),
                np.rad2deg(float(planet.true_anomaly)),
            ],
            dtype=float,
        )

    def _setup_resource(self):
        if self.n_setup_in is not None:
            return self.get_resource_from_name(self.n_setup_in)

        if not isinstance(self.resources, dict):
            return None

        setup = self.resources.get("setup")
        if setup is not None and getattr(setup, "phringe", None) is not None:
            return setup

        for resource in self.resources.values():
            if getattr(resource, "phringe", None) is not None:
                return resource
        return None

    def _stellar_context(self):
        luminosity_solar = None
        star_temperature_k = None

        setup = self._setup_resource()
        phringe = None if setup is None else getattr(setup, "phringe", None)
        if phringe is not None:
            scene = getattr(phringe, "_scene", None)
            star = None if scene is None else getattr(scene, "star", None)
            if star is not None:
                luminosity_w = _as_float_or_none(getattr(star, "luminosity", None))
                if luminosity_w is not None and luminosity_w > 0.0:
                    luminosity_solar = luminosity_w / L_SUN_FOR_HZ_W
                star_temperature_k = _as_float_or_none(getattr(star, "temperature", None))

            observation = getattr(phringe, "_observation", None)
            if observation is not None:
                if star_temperature_k is None:
                    star_temperature_k = _as_float_or_none(getattr(observation, "host_star_temperature", None))
                if luminosity_solar is None:
                    radius_m = _as_float_or_none(getattr(observation, "host_star_radius", None))
                    temperature_k = _as_float_or_none(getattr(observation, "host_star_temperature", None))
                    if radius_m is not None and temperature_k is not None and radius_m > 0.0 and temperature_k > 0.0:
                        luminosity_w = 4.0 * np.pi * radius_m**2 * const.sigma_sb.value * temperature_k**4
                        luminosity_solar = luminosity_w / L_SUN_FOR_HZ_W

        if luminosity_solar is None:
            luminosity_solar = DEFAULT_LUMINOSITY_SOLAR
            print("Warning: using solar luminosity fallback for HZ corner plot.")
        if star_temperature_k is None:
            star_temperature_k = DEFAULT_STAR_TEMPERATURE_K
            print("Warning: using solar temperature fallback for HZ corner plot.")

        if luminosity_solar <= 0.0:
            raise ValueError(f"Stellar luminosity in solar units must be positive, got {luminosity_solar}.")
        if star_temperature_k <= 0.0:
            raise ValueError(f"Stellar temperature must be positive, got {star_temperature_k}.")

        self.last_stellar_context = {
            "luminosity_solar": float(luminosity_solar),
            "star_temperature_k": float(star_temperature_k),
        }
        return float(luminosity_solar), float(star_temperature_k)

    def _prior_bounds(self, inference, samples_phys):
        sampler_info = getattr(inference, "sampler_info", {}) or {}
        samples_sma_au = np.asarray(samples_phys[:, 0], dtype=float) / const.au.value

        if sampler_info.get("sma_lower_m") is not None:
            prior_sma_min_au = float(sampler_info["sma_lower_m"]) / const.au.value
        else:
            prior_sma_min_au = float(np.nanmin(samples_sma_au))
            print("Warning: using posterior minimum as SMA prior lower bound for HZ Bayes factor.")

        if sampler_info.get("sma_upper_m") is not None:
            prior_sma_max_au = float(sampler_info["sma_upper_m"]) / const.au.value
        else:
            prior_sma_max_au = float(np.nanmax(samples_sma_au))
            print("Warning: using posterior maximum as SMA prior upper bound for HZ Bayes factor.")

        prior_ecc_min = 0.0
        if sampler_info.get("eccentricity_upper") is not None:
            prior_ecc_max = float(sampler_info["eccentricity_upper"])
        else:
            prior_ecc_max = float(DEFAULT_ECCENTRICITY_UPPER)

        return prior_sma_min_au, prior_sma_max_au, prior_ecc_min, prior_ecc_max

    def _plot_samples_for_visualization(self, samples, weights):
        if not self.resample_for_plot:
            return samples, weights, "none"

        n_samples = int(samples.shape[0])
        n_plot = n_samples

        if weights is None:
            return samples, None, "already_equal_weight"
        else:
            probabilities = _normalized_or_uniform_weights(weights, n_samples)

        seed_base = 0 if self.seed is None else int(self.seed)
        rng = np.random.default_rng(seed_base + int(DEFAULT_RESAMPLE_SEED_OFFSET))
        indices = rng.choice(
            n_samples,
            size=n_plot,
            replace=True,
            p=probabilities,
        )
        return samples[indices], None, "multinomial"

    def _plot_sma_min(self, inference, samples):
        if self.plot_sma_min is not None:
            return float(self.plot_sma_min)
        sma_lower = getattr(inference, "sampler_info", {}).get("sma_lower_m")
        if sma_lower is not None:
            return float(sma_lower) / const.au.value
        return float(np.nanmin(samples[:, 0]))

    def _plot_sma_max(self, inference, samples):
        if self.plot_sma_max is not None:
            return float(self.plot_sma_max)
        sma_upper = getattr(inference, "sampler_info", {}).get("sma_upper_m")
        if sma_upper is not None:
            return float(sma_upper) / const.au.value
        return float(np.nanmax(samples[:, 0]))

    @staticmethod
    def _make_plot(
            samples,
            weights,
            truths,
            plot_sma_min,
            plot_sma_max,
            hz_definition=None,
            luminosity_solar=DEFAULT_LUMINOSITY_SOLAR,
            hz_metrics=None,
    ):
        try:
            import corner
        except ImportError as exc:
            raise RuntimeError("CornerPlotModule requires the 'corner' package.") from exc

        fig = corner.corner(
            samples,
            weights=weights,
            labels=PLOT_LABELS,
            color=COLOR_MAIN,
            truths=truths,
            truth_color=COLOR_TRUTH,
            bins=60,
            smooth=0.0,
            show_titles=False,
            plot_datapoints=False,
            fill_contours=False,
            plot_contours=False,
            plot_density=False,
            contour_kwargs={"linewidths": 1.2, "alpha": 0.6, "colors": COLOR_MAIN},
            hist_kwargs={"density": True, "linewidth": 1.2, "color": "black", "histtype": "step"},
            data_kwargs={"color": COLOR_MAIN, "alpha": 0.6},
            fig=plt.figure(figsize=(12, 12)),
        )

        density_norm = _overlay_density_colored_points(
            fig,
            samples,
            bins=1000,
            cmap="plasma",
            s=2.0,
            alpha=0.8,
            weights=weights,
        )

        n_dim = samples.shape[1]
        axes = np.asarray(fig.axes[: n_dim * n_dim]).reshape((n_dim, n_dim))
        for i in range(n_dim):
            axes[i, 0].set_xlim(plot_sma_min, plot_sma_max)

        ax = axes[0, 0]
        ax.clear()
        ax.hist(
            samples[:, 0],
            bins=60,
            range=(plot_sma_min, plot_sma_max),
            weights=weights,
            density=True,
            histtype="step",
            color="black",
            linewidth=1.2,
        )
        if truths is not None:
            ax.axvline(float(truths[0]), 0, 1, color=COLOR_TRUTH)
        ax.set_yticks([])
        ax.set_xlim(plot_sma_min, plot_sma_max)

        if hz_definition is not None:
            _add_hz_overlay(
                fig=fig,
                samples=samples,
                weights=weights,
                truths=truths,
                plot_sma_min=plot_sma_min,
                plot_sma_max=plot_sma_max,
                luminosity_solar=luminosity_solar,
                hz_definition=hz_definition,
                density_norm=density_norm,
                hz_metrics=hz_metrics,
            )

        return fig
