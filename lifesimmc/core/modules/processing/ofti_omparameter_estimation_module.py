import json
from datetime import datetime
from multiprocessing.dummy import Pool as ThreadPool
from pathlib import Path

import astropy.units as u
import matplotlib
import numpy as np
import torch
from astropy import constants as const
from phringe.util.spectrum import get_blackbody_spectrum_standard_units

from lifesimmc.core.modules.base_module import BaseModule
from lifesimmc.core.resources.base_resource import BaseResource
from lifesimmc.core.resources.planet_params_resource import PlanetParams, PlanetParamsResource

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class OFTIOMParameterEstimationModule(BaseModule):
    def __init__(
        self,
        n_setup_in: str,
        n_data_in: str,
        n_planet_params_out: str,
        mcmcsteps: int,
        n_transformation_in: str = None,
        n_template_in: str = None,
        n_planet_params_in: str = None,
        bounds: bool = False,
        n_cores: int = None,
    ):
        super().__init__()
        self.n_config_in = n_setup_in
        self.n_data_in = n_data_in
        self.n_template_in = n_template_in
        self.n_transformation_in = n_transformation_in
        self.n_planet_params_out = n_planet_params_out
        self.n_planet_params_in = n_planet_params_in
        self.bounds = bounds
        self.mcmcsteps = mcmcsteps
        self.n_cores = n_cores

    def apply(self, resources: list[BaseResource]) -> PlanetParamsResource:
        print("Performing OFTI with Blackbody and Orbital Motion...")

        r_config_in = self.get_resource_from_name(self.n_config_in)
        r_transformation_in = (
            self.get_resource_from_name(self.n_transformation_in)
            if self.n_transformation_in
            else None
        )
        transf = r_transformation_in.transformation if r_transformation_in else lambda x: x
        mcmcsteps = int(self.mcmcsteps)

        distance = float(r_config_in.scene.star.distance)
        planet0 = r_config_in.scene.planets[0]
        planet_mass_fixed = float(planet0.mass)

        times = r_config_in.phringe.get_time_steps().cpu().numpy()
        wavelengths = r_config_in.phringe.get_wavelength_bin_centers().cpu().numpy()
        wavelength_bin_widths = r_config_in.phringe.get_wavelength_bin_widths().cpu().numpy()
        data_tensor = self.get_resource_from_name(self.n_data_in).get_data()

        data_tensor = data_tensor.permute(0, 2, 1)
        data_tensor = data_tensor.reshape((-1,) + data_tensor.shape[2:])
        data_in = data_tensor.cpu().numpy()
        data_shape = data_in.shape

        hfov_max = r_config_in.phringe.get_field_of_view()[-1].cpu().numpy() / 2
        a_max = distance * np.tan(hfov_max)

        temp_init = 2000.0
        radius_init = 10e6
        semi_major_axis_init = 2 * const.au.value
        inclination_init = np.pi / 2

        param_names = [
            "log_temp",
            "log_radius",
            "log_semi_major_axis",
            "eccentricity",
            "cos_inclination",
            "raan",
            "argument_of_periapsis",
            "true_anomaly",
        ]
        prior_min = np.array(
            [
                np.log10(50.0),
                np.log10(1e5),
                np.log10(0.01 * const.au.value),
                0.0,
                -1.0,
                0.0,
                0.0,
                0.0,
            ],
            dtype=float,
        )
        prior_max = np.array(
            [
                np.log10(4000.0),
                np.log10(2e8),
                np.log10(a_max),
                1.0,
                1.0,
                2.0 * np.pi,
                2.0 * np.pi,
                2.0 * np.pi,
            ],
            dtype=float,
        )
        prior_init = np.array(
            [
                np.log10(temp_init),
                np.log10(radius_init),
                np.log10(semi_major_axis_init),
                0.0,
                np.cos(inclination_init),
                np.pi,
                np.pi,
                np.pi,
            ],
            dtype=float,
        )
        name_to_idx = {name: i for i, name in enumerate(param_names)}

        def _as_float(value, unit=None):
            try:
                if value is None:
                    return np.nan
                if unit is not None and hasattr(value, "to"):
                    return float(value.to(unit).value)
                return float(value)
            except Exception:
                return np.nan

        obs = getattr(r_config_in, "observation", None)
        integration_time_s = _as_float(getattr(obs, "total_integration_time", None), u.s)
        if not np.isfinite(integration_time_s):
            integration_time_s = (
                float((times[-1] - times[0]) + np.median(np.diff(times)))
                if len(times) > 1
                else np.nan
            )

        sma_true_m = _as_float(getattr(planet0, "semi_major_axis", None), u.m)
        star_mass_kg = _as_float(getattr(r_config_in.scene.star, "mass", None), u.kg)
        planet_mass_kg = _as_float(getattr(planet0, "mass", None), u.kg)
        if (
            np.isfinite(sma_true_m)
            and np.isfinite(star_mass_kg)
            and np.isfinite(planet_mass_kg)
            and sma_true_m > 0
            and (star_mass_kg + planet_mass_kg) > 0
        ):
            orbital_period_s = float(
                2.0
                * np.pi
                * np.sqrt(sma_true_m**3 / (const.G.value * (star_mass_kg + planet_mass_kg)))
            )
        else:
            orbital_period_s = np.nan

        integration_fraction = (
            float(integration_time_s / orbital_period_s)
            if np.isfinite(integration_time_s)
            and np.isfinite(orbital_period_s)
            and orbital_period_s > 0
            else np.nan
        )
        integration_percent = (
            float(100.0 * integration_fraction) if np.isfinite(integration_fraction) else np.nan
        )
        integration_percent_tag = (
            f"{integration_percent:.2f}" if np.isfinite(integration_percent) else "unknown"
        )

        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = (
            Path.cwd()
            / "MCMC"
            / "IntegrationPercent"
            / integration_percent_tag
            / f"{self.n_planet_params_out}_{run_timestamp}"
        )
        arrays_dir = run_dir / "arrays"
        arrays_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving OFTI diagnostics to: {run_dir}")

        run_count = int(self.n_cores) if self.n_cores is not None else 1
        run_count = max(1, run_count)
        accepted_per_run = max(200, mcmcsteps)
        max_draws_per_run = max(5000, accepted_per_run * 200)
        print(
            f"OFTI configuration: runs={run_count}, accepted_per_run={accepted_per_run}, "
            f"max_draws_per_run={max_draws_per_run}"
        )

        eval_counter = 0

        def _residual_from_theta(theta: np.ndarray) -> np.ndarray:
            nonlocal eval_counter
            p = {name: float(theta[i]) for i, name in enumerate(param_names)}
            cos_inc = np.clip(p["cos_inclination"], -1.0, 1.0)
            inc = float(np.arccos(cos_inc))
            flux = (
                np.pi
                * get_blackbody_spectrum_standard_units(
                    temperature=10 ** p["log_temp"], wavelengths=wavelengths
                )
                * (10 ** p["log_radius"] / distance) ** 2
            )
            flux = flux.detach().cpu().numpy()
            eval_counter += 1
            if eval_counter % 20 == 0:
                print(
                    f"[Eval {eval_counter}] T={10**p['log_temp']:.6g}, R={10**p['log_radius']:.6g}, "
                    f"a={10**p['log_semi_major_axis']:.6g}, e={p['eccentricity']:.6g}, "
                    f"inc={inc:.6g}, raan={p['raan']:.6g}, argp={p['argument_of_periapsis']:.6g}, "
                    f"nu={p['true_anomaly']:.6g}"
                )
            try:
                model = r_config_in.phringe.get_model_counts(
                    kernels=True,
                    spectral_energy_distribution=flux,
                    semi_major_axis=10 ** p["log_semi_major_axis"],
                    eccentricity=p["eccentricity"],
                    inclination=inc,
                    raan=p["raan"],
                    argument_of_periapsis=p["argument_of_periapsis"],
                    true_anomaly=p["true_anomaly"],
                    host_star_distance=r_config_in.scene.star.distance,
                    host_star_mass=r_config_in.scene.star.mass,
                    planet_mass=planet_mass_fixed,
                )
            except (RuntimeError, ValueError, FloatingPointError) as exc:
                print(f"Orbital model failed; penalty residual used. {exc}")
                return np.full_like(data_in, 1e30, dtype=float)

            model = transf(model)
            model = np.transpose(model, (0, 2, 1)).reshape(data_shape)
            residual = model - data_in
            if not np.all(np.isfinite(residual)):
                return np.full_like(data_in, 1e30, dtype=float)
            return residual

        def _chi2(theta: np.ndarray) -> float:
            residual = _residual_from_theta(theta)
            return float(np.sum(residual * residual))

        def _sample_prior(rng: np.random.Generator, n: int) -> np.ndarray:
            return rng.uniform(prior_min, prior_max, size=(n, len(param_names)))

        seeds = [12345 + i for i in range(run_count)]

        def _run_single_ofti(run_idx: int) -> dict:
            rng = np.random.default_rng(seeds[run_idx])
            accepted = []
            accepted_chi2 = []
            best_theta = None
            best_chi2 = np.inf
            draws = 0

            while draws < max_draws_per_run and len(accepted) < accepted_per_run:
                batch = _sample_prior(rng, min(128, max_draws_per_run - draws))
                for theta in batch:
                    chi2_val = _chi2(theta)
                    draws += 1
                    if chi2_val < best_chi2:
                        best_chi2 = chi2_val
                        best_theta = theta.copy()
                        accepted.append(theta.copy())
                        accepted_chi2.append(chi2_val)
                    else:
                        delta = chi2_val - best_chi2
                        if np.isfinite(delta):
                            accept_prob = np.exp(-0.5 * min(delta, 700.0))
                            if rng.uniform() < accept_prob:
                                accepted.append(theta.copy())
                                accepted_chi2.append(chi2_val)

                    if len(accepted) >= accepted_per_run or draws >= max_draws_per_run:
                        break

            if len(accepted) == 0 and best_theta is not None:
                accepted = [best_theta.copy()]
                accepted_chi2 = [best_chi2]

            accepted = np.asarray(accepted, dtype=float)
            accepted_chi2 = np.asarray(accepted_chi2, dtype=float)
            print(
                f"OFTI run {run_idx + 1}/{run_count}: accepted={accepted.shape[0]}, "
                f"draws={draws}, best_chi2={best_chi2:.6g}"
            )
            return {
                "accepted": accepted,
                "chi2": accepted_chi2,
                "draws": draws,
                "seed": seeds[run_idx],
                "best_chi2": best_chi2,
                "best_theta": best_theta,
            }

        if run_count > 1:
            with ThreadPool(processes=run_count) as pool:
                run_results = pool.map(_run_single_ofti, range(run_count))
        else:
            run_results = [_run_single_ofti(0)]

        accepted_arrays = [r["accepted"] for r in run_results if r["accepted"].size > 0]
        accepted_chi2_arrays = [r["chi2"] for r in run_results if r["chi2"].size > 0]
        if len(accepted_arrays) == 0:
            raise RuntimeError("OFTI failed: no accepted samples.")

        merged_chain = np.vstack(accepted_arrays)
        merged_chi2 = np.concatenate(accepted_chi2_arrays)
        best_idx = int(np.argmin(merged_chi2))
        best_theta = merged_chain[best_idx]
        best_chi2 = float(merged_chi2[best_idx])

        merged_medians = np.median(merged_chain, axis=0)
        merged_std = (
            np.std(merged_chain, axis=0, ddof=1)
            if merged_chain.shape[0] > 1
            else np.full(merged_chain.shape[1], np.nan, dtype=float)
        )
        cov_out = np.cov(merged_chain, rowvar=False) if merged_chain.shape[0] > 1 else None

        print(f"Accepted samples total: {merged_chain.shape[0]}")
        print(f"Best chi2: {best_chi2}")

        np.save(arrays_dir / "posterior_chain_merged.npy", merged_chain)
        np.save(arrays_dir / "posterior_chi2_merged.npy", merged_chi2)
        np.savetxt(
            run_dir / "posterior_chain_merged.csv",
            merged_chain,
            delimiter=",",
            header=",".join(param_names),
            comments="",
        )
        for i, result in enumerate(run_results):
            np.save(arrays_dir / f"run_{i + 1}_accepted.npy", result["accepted"])
            np.save(arrays_dir / f"run_{i + 1}_chi2.npy", result["chi2"])

        plot_samples = merged_chain.copy()
        display_names = []
        is_log_axis = []
        for i, name in enumerate(param_names):
            if name.startswith("log_"):
                plot_samples[:, i] = np.power(10.0, plot_samples[:, i])
                display_names.append(name.replace("log_", "", 1))
                is_log_axis.append(True)
            else:
                display_names.append(name)
                is_log_axis.append(False)

        n_params = len(param_names)
        fig, axes = plt.subplots(n_params, n_params, figsize=(2.6 * n_params, 2.6 * n_params), squeeze=False)
        for row in range(n_params):
            for col in range(n_params):
                ax = axes[row, col]
                if row == col:
                    if is_log_axis[col]:
                        vals = plot_samples[:, col][plot_samples[:, col] > 0]
                        if vals.size > 1 and np.min(vals) < np.max(vals):
                            bins = np.logspace(np.log10(np.min(vals)), np.log10(np.max(vals)), 41)
                            ax.hist(vals, bins=bins, color="tab:blue", alpha=0.85)
                        else:
                            ax.hist(plot_samples[:, col], bins=40, color="tab:blue", alpha=0.85)
                        ax.set_xscale("log")
                    else:
                        ax.hist(plot_samples[:, col], bins=40, color="tab:blue", alpha=0.85)
                else:
                    ax.scatter(
                        plot_samples[:, col],
                        plot_samples[:, row],
                        s=2,
                        alpha=0.20,
                        color="tab:blue",
                        rasterized=True,
                    )
                    if is_log_axis[col]:
                        ax.set_xscale("log")
                    if is_log_axis[row]:
                        ax.set_yscale("log")

                if row == n_params - 1:
                    ax.set_xlabel(display_names[col])
                else:
                    ax.set_xticklabels([])
                if col == 0:
                    ax.set_ylabel(display_names[row])
                else:
                    ax.set_yticklabels([])

        fig.suptitle("OFTI Correlation Plot", y=1.0)
        fig.tight_layout()
        fig.savefig(run_dir / "correlation_plot.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

        def _median(name: str) -> float:
            return float(merged_medians[name_to_idx[name]])

        def _std(name: str) -> float:
            return float(merged_std[name_to_idx[name]])

        log_temp = _median("log_temp")
        log_radius = _median("log_radius")
        log_sma = _median("log_semi_major_axis")
        log_temp_err = _std("log_temp")
        log_radius_err = _std("log_radius")
        log_sma_err = _std("log_semi_major_axis")

        temp = float(10.0**log_temp)
        radius = float(10.0**log_radius)
        semi_major_axis = float(10.0**log_sma)
        temp_err = float(np.log(10.0) * temp * log_temp_err) if np.isfinite(log_temp_err) else np.nan
        radius_err = (
            float(np.log(10.0) * radius * log_radius_err) if np.isfinite(log_radius_err) else np.nan
        )
        semi_major_axis_err = (
            float(np.log(10.0) * semi_major_axis * log_sma_err) if np.isfinite(log_sma_err) else np.nan
        )

        eccentricity = _median("eccentricity")
        eccentricity_err = _std("eccentricity")
        cos_incl = np.clip(_median("cos_inclination"), -1.0, 1.0)
        cos_incl_err = _std("cos_inclination")
        inclination = float(np.arccos(cos_incl))
        if np.isfinite(cos_incl_err):
            inclination_err = float(cos_incl_err / np.sqrt(max(1e-16, 1.0 - cos_incl**2)))
        else:
            inclination_err = np.nan
        raan = _median("raan")
        raan_err = _std("raan")
        argument_of_periapsis = _median("argument_of_periapsis")
        argument_of_periapsis_err = _std("argument_of_periapsis")
        true_anomaly = _median("true_anomaly")
        true_anomaly_err = _std("true_anomaly")

        fluxes = (
            np.pi
            * get_blackbody_spectrum_standard_units(temperature=temp, wavelengths=wavelengths)
            * (radius / distance) ** 2
        )

        def dBdT(rad, t, lam):
            lam_m = np.asarray(lam, dtype=float)
            return (
                (4 * np.pi * const.c.value**2 * const.h.value)
                / (lam_m**5 * const.k_B.value * t)
                / (rad * distance) ** 2
                * np.exp(const.h.value * const.c.value / (const.k_B.value * t * lam_m))
                / (
                    np.expm1(
                        const.h.value * const.c.value / (lam_m * const.k_B.value * t)
                    )
                    ** 2
                )
            )

        def dBdR(rad, t, lam):
            lam_m = np.asarray(lam, dtype=float)
            x = (const.h.value * const.c.value) / (lam_m * const.k_B.value * t)
            denom = np.expm1(x)
            return (2.0 * np.pi * const.c.value / lam_m**4) * 2 * (rad / distance**2) / denom

        if np.isfinite(temp_err) and np.isfinite(radius_err):
            flux_err = np.sqrt(
                dBdT(radius, temp, wavelengths) ** 2 * temp_err**2
                + dBdR(radius, temp, wavelengths) ** 2 * radius_err**2
            )
        else:
            flux_err = np.full_like(fluxes, np.nan, dtype=float)

        np.save(arrays_dir / "times.npy", np.asarray(times))
        np.save(arrays_dir / "wavelengths.npy", np.asarray(wavelengths))
        np.save(arrays_dir / "wavelength_bin_widths.npy", np.asarray(wavelength_bin_widths))
        np.save(arrays_dir / "data_flattened.npy", np.asarray(data_in))
        np.save(arrays_dir / "retrieved_flux.npy", np.asarray(fluxes))
        np.save(arrays_dir / "retrieved_flux_err.npy", np.asarray(flux_err))
        if cov_out is not None:
            np.save(arrays_dir / "covariance.npy", np.asarray(cov_out))

        def _serializable_number(value):
            val = _as_float(value)
            return float(val) if np.isfinite(val) else None

        true_temp = _serializable_number(getattr(planet0, "temperature", getattr(planet0, "temp", None)))
        true_radius = _serializable_number(getattr(planet0, "radius", None))
        true_sma = _serializable_number(getattr(planet0, "semi_major_axis", None))
        true_e = _serializable_number(getattr(planet0, "eccentricity", None))
        true_i = _serializable_number(getattr(planet0, "inclination", None))
        true_raan = _serializable_number(getattr(planet0, "raan", None))
        true_argp = _serializable_number(getattr(planet0, "argument_of_periapsis", None))
        true_nu = _serializable_number(getattr(planet0, "true_anomaly", None))

        def _compare(true_value, fitted_value):
            if true_value is None:
                return {"true": None, "fitted": float(fitted_value), "abs_error": None}
            return {
                "true": float(true_value),
                "fitted": float(fitted_value),
                "abs_error": float(fitted_value - true_value),
            }

        metadata = {
            "module_inputs": {
                "n_setup_in": self.n_config_in,
                "n_data_in": self.n_data_in,
                "n_template_in": self.n_template_in,
                "n_transformation_in": self.n_transformation_in,
                "n_planet_params_in": self.n_planet_params_in,
                "n_planet_params_out": self.n_planet_params_out,
                "bounds": self.bounds,
                "n_cores_input": self.n_cores,
                "mcmcsteps_input": self.mcmcsteps,
            },
            "used_runtime_parameters": {
                "algorithm": "OFTI_rejection_sampling",
                "run_count": run_count,
                "accepted_per_run": accepted_per_run,
                "max_draws_per_run": max_draws_per_run,
                "seeds": seeds,
                "eval_counter": eval_counter,
                "best_chi2": best_chi2,
                "accepted_total": int(merged_chain.shape[0]),
                "integration_time_s": float(integration_time_s) if np.isfinite(integration_time_s) else None,
                "orbital_period_s": float(orbital_period_s) if np.isfinite(orbital_period_s) else None,
                "integration_fraction": float(integration_fraction) if np.isfinite(integration_fraction) else None,
                "integration_percent": float(integration_percent) if np.isfinite(integration_percent) else None,
                "output_integration_percent_tag": integration_percent_tag,
                "data_shape": list(data_shape),
            },
            "input_parameter_definitions": {
                name: {
                    "initial_value": float(prior_init[i]),
                    "min": float(prior_min[i]),
                    "max": float(prior_max[i]),
                }
                for i, name in enumerate(param_names)
            },
            "used_parameter_values": {
                name: {
                    "value": float(merged_medians[i]),
                    "stderr": float(merged_std[i]) if np.isfinite(merged_std[i]) else None,
                }
                for i, name in enumerate(param_names)
            },
            "star_true_parameters": {
                "name": getattr(r_config_in.scene.star, "name", None),
                "distance_m": _serializable_number(getattr(r_config_in.scene.star, "distance", None)),
                "mass_kg": _serializable_number(getattr(r_config_in.scene.star, "mass", None)),
                "radius_m": _serializable_number(getattr(r_config_in.scene.star, "radius", None)),
                "temperature_K": _serializable_number(
                    getattr(r_config_in.scene.star, "temperature", getattr(r_config_in.scene.star, "temp", None))
                ),
            },
            "planet_true_parameters": {
                "name": getattr(planet0, "name", None),
                "mass_kg": _serializable_number(getattr(planet0, "mass", None)),
                "radius_m": true_radius,
                "temperature_K": true_temp,
                "semi_major_axis_m": true_sma,
                "eccentricity": true_e,
                "inclination_rad": true_i,
                "raan_rad": true_raan,
                "argument_of_periapsis_rad": true_argp,
                "true_anomaly_rad": true_nu,
            },
            "true_vs_fitted": {
                "temperature_K": _compare(true_temp, temp),
                "radius_m": _compare(true_radius, radius),
                "semi_major_axis_m": _compare(true_sma, semi_major_axis),
                "eccentricity": _compare(true_e, eccentricity),
                "inclination_rad": _compare(true_i, inclination),
                "raan_rad": _compare(true_raan, raan),
                "argument_of_periapsis_rad": _compare(true_argp, argument_of_periapsis),
                "true_anomaly_rad": _compare(true_nu, true_anomaly),
            },
            "physical_solution": {
                "temperature_K": float(temp),
                "temperature_err_K": float(temp_err) if np.isfinite(temp_err) else None,
                "radius_m": float(radius),
                "radius_err_m": float(radius_err) if np.isfinite(radius_err) else None,
                "semi_major_axis_m": float(semi_major_axis),
                "semi_major_axis_err_m": float(semi_major_axis_err) if np.isfinite(semi_major_axis_err) else None,
                "eccentricity": float(eccentricity),
                "eccentricity_err": float(eccentricity_err) if np.isfinite(eccentricity_err) else None,
                "inclination_rad": float(inclination),
                "inclination_err_rad": float(inclination_err) if np.isfinite(inclination_err) else None,
                "raan_rad": float(raan),
                "raan_err_rad": float(raan_err) if np.isfinite(raan_err) else None,
                "argument_of_periapsis_rad": float(argument_of_periapsis),
                "argument_of_periapsis_err_rad": float(argument_of_periapsis_err)
                if np.isfinite(argument_of_periapsis_err)
                else None,
                "true_anomaly_rad": float(true_anomaly),
                "true_anomaly_err_rad": float(true_anomaly_err) if np.isfinite(true_anomaly_err) else None,
            },
            "best_sample": {
                name: float(best_theta[i]) for i, name in enumerate(param_names)
            },
        }
        with open(run_dir / "run_parameters.json", "w", encoding="utf-8") as fp:
            json.dump(metadata, fp, indent=2)
        print(f"Saved run parameters to: {run_dir / 'run_parameters.json'}")

        r_planet_params_out = PlanetParamsResource(name=self.n_planet_params_out)
        planet_params = PlanetParams(
            name="fitted planet",
            sed_wavelength_bin_centers=r_config_in.phringe.get_wavelength_bin_centers(),
            sed_wavelength_bin_widths=r_config_in.phringe.get_wavelength_bin_widths(),
            sed=torch.tensor(fluxes),
            sed_err_low=torch.tensor(flux_err),
            sed_err_high=torch.tensor(flux_err),
            covariance=cov_out,
            temp=temp,
            temp_err_low=temp_err,
            temp_err_high=temp_err,
            radius=radius,
            radius_err_low=radius_err,
            radius_err_high=radius_err,
            semi_major_axis=semi_major_axis,
            semi_major_axis_err_low=semi_major_axis_err,
            semi_major_axis_err_high=semi_major_axis_err,
            eccentricity=eccentricity,
            eccentricity_err_low=eccentricity_err,
            eccentricity_err_high=eccentricity_err,
            inclination=inclination,
            inclination_err_low=inclination_err,
            inclination_err_high=inclination_err,
            raan=raan,
            raan_err_low=raan_err,
            raan_err_high=raan_err,
            argument_of_periapsis=argument_of_periapsis,
            argument_of_periapsis_err_low=argument_of_periapsis_err,
            argument_of_periapsis_err_high=argument_of_periapsis_err,
            true_anomaly=true_anomaly,
            true_anomaly_err_low=true_anomaly_err,
            true_anomaly_err_high=true_anomaly_err,
        )
        r_planet_params_out.params.append(planet_params)

        print("Fitted Planet Params:")
        print("Inclination ", inclination)
        print("Argument of Periapsis ", argument_of_periapsis)
        print("Eccentricity ", eccentricity)
        print("RAAN ", raan)
        print("SMA ", semi_major_axis)
        print("Temperature ", temp)
        print("Radius ", radius)
        print("True Anomaly", true_anomaly)
        print("Done")
        return r_planet_params_out
