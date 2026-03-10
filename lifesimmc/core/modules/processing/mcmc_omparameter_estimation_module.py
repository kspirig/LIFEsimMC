import numpy as np
import torch
from lmfit import minimize, Parameters
from multiprocessing.dummy import Pool as ThreadPool
from pathlib import Path
from datetime import datetime
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lifesimmc.core.modules.base_module import BaseModule
from lifesimmc.core.resources.base_resource import BaseResource
from lifesimmc.core.resources.planet_params_resource import PlanetParamsResource, PlanetParams
from astropy import constants as const
from phringe.util.spectrum import get_blackbody_spectrum_standard_units
import astropy.units as u

class MCMCOMParameterEstimationModule(BaseModule):
    """Class representation of a module that performs maximum likelihood estimation (MLE) of planet parameters.

    Parameters
    ---------
    n_setup_in : str
        Name of the input configuration resource.
    n_data_in : str
        Name of the input data resource.
    n_planet_params_out : str
        Name of the output planet parameters resource.
    n_transformation_in : str, optional
        Name of the input transformation resource. If None, no transformation is applied.
    n_template_in : str, optional
        Name of the input template resource. If None, no template is used.
    """

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
        """Constructor method.

        Parameters
        ----------
        n_setup_in : str
            Name of the input configuration resource.
        n_data_in : str
            Name of the input data resource.
        n_planet_params_out : str
            Name of the output planet parameters resource.
        n_transformation_in : str, optional
            Name of the input transformation resource. If None, no transformation is applied.
        n_template_in : str, optional
            Name of the input template resource. If None, no template is used. # TODO: handle no templates
        """
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
        print('Performing numerical MCMC with Blackbody and Orbital Motion...')

        r_config_in = self.get_resource_from_name(self.n_config_in)
        r_transformation_in = self.get_resource_from_name(
            self.n_transformation_in) if self.n_transformation_in else None
        transf = r_transformation_in.transformation if r_transformation_in else lambda x: x
        planet_params_in = self.get_resource_from_name(self.n_planet_params_in) if self.n_planet_params_in else None
        mcmcsteps = self.mcmcsteps

        distance = r_config_in.scene.star.distance  # float in meters
        planet_mass_fixed = r_config_in.scene.planets[0].mass


        times = r_config_in.phringe.get_time_steps().cpu().numpy()
        wavelengths = r_config_in.phringe.get_wavelength_bin_centers().cpu().numpy()
        wavelength_bin_widths = r_config_in.phringe.get_wavelength_bin_widths().cpu().numpy()
        data_in = self.get_resource_from_name(self.n_data_in).get_data()

        # Flatten data along differential outputs and times axes
        data_in = data_in.permute(0, 2, 1)
        data_in = data_in.reshape((-1,) + data_in.shape[2:])

        # Set up parameters and initial conditions
        # TODO: implement for multiple planets

        hfov_max = r_config_in.phringe.get_field_of_view()[-1].cpu().numpy() / 2  # TODO: /14 Check this

        a_max = distance * np.tan(hfov_max)

        radius_init = 10 * 1e6  # 1 Earth radius in meters
        temp_init = 2000.0      # Kelvin
        semi_major_axis_init = 2 * const.au.value # Can we evaluate quite good?
        eccentricity_init  = 0.0
        inclination_init = np.pi / 2
        cos_inclination_init = np.cos(inclination_init)
        raan_init = np.pi
        argument_of_periapsis_init = np.pi
        true_anomaly_init = np.pi




        data_in = data_in.cpu().numpy()
        data_shape = data_in.shape

        params = Parameters()

        params.add("log_temp", value=np.log10(temp_init), min=np.log10(50), max=np.log10(4000)) ## kleiner high temperature
        params.add("log_radius", value=np.log10(radius_init), min=np.log10(1e5), max=np.log10(2 * 1e8))
        params.add(
            "log_semi_major_axis",
            value=np.log10(semi_major_axis_init),
            min=np.log10(0.01 * const.au.value),   # Define Minimum correctly
            max=np.log10(a_max),
        )

        params.add('eccentricity',value = eccentricity_init,min = 0,max = 1.0)

        params.add('cos_inclination', value=cos_inclination_init, min=-1.0, max=1.0)
        params.add('raan',value = raan_init,min = 0,max = 2.0 * np.pi)
        params.add('argument_of_periapsis',value = argument_of_periapsis_init,min = 0,max = 2.0 * np.pi)

        params.add('true_anomaly',value = true_anomaly_init,min = 0,max = 2.0 * np.pi)

        params_names = np.array(["log_temp","log_radius","log_semi_major_axis","eccentricity","cos_inclination","raan","argument_of_periapsis","true_anomaly"])

        mcmc_seed = 12345
        ndim = len(params_names)

        # Perform MLE
        eval_counter = 0
        def residual_data(params, target):
            nonlocal eval_counter

            flux = np.pi * get_blackbody_spectrum_standard_units(temperature=10**params['log_temp'].value, wavelengths=wavelengths)  * (10**params['log_radius'].value / distance) ** 2
            flux = flux.detach().cpu().numpy()
            cos_inclination = float(np.clip(params['cos_inclination'].value, -1.0, 1.0))
            inclination = float(np.arccos(cos_inclination))
            eval_counter += 1
            if eval_counter % 20 == 0:
                print(
                    f"[Eval {eval_counter}] "
                    f"T={10**params['log_temp'].value:.6g}, R={10**params['log_radius'].value:.6g}, "
                    f"a={10**params['log_semi_major_axis'].value:.6g}, e={params['eccentricity'].value:.6g}, "
                    f"inc={inclination:.6g}, raan={params['raan'].value:.6g}, "
                    f"argp={params['argument_of_periapsis'].value:.6g}, nu={params['true_anomaly'].value:.6g}"
                )

            try:
                model = r_config_in.phringe.get_model_counts(
                    kernels=True,
                    spectral_energy_distribution=flux,
                    semi_major_axis = 10**params['log_semi_major_axis'].value,
                    eccentricity = params['eccentricity'].value,
                    inclination = inclination ,
                    raan = params['raan'].value,
                    argument_of_periapsis = params['argument_of_periapsis'].value,
                    true_anomaly = params['true_anomaly'].value,
                    host_star_distance=r_config_in.scene.star.distance,
                    host_star_mass=r_config_in.scene.star.mass,
                    planet_mass=planet_mass_fixed
                )
            except (RuntimeError, ValueError, FloatingPointError) as exc:
                print(f"Orbital model failed for trial parameters; continuing with penalty residual. {exc}")
                return np.full_like(target, 1e30, dtype=float)

            model = transf(model)
            model = np.transpose(model, (0, 2, 1))
            model = model.reshape(data_in.shape)

            residual = model - target
            if not np.all(np.isfinite(residual)):
                print("Non-finite residual detected")
            return residual

        run_count = int(self.n_cores) if self.n_cores is not None else 1
        if run_count < 1:
            run_count = 1
        walkers_per_run = 32
        print(f"MCMC configuration: runs={run_count}, walkers_per_run={walkers_per_run}")

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

        planet0 = r_config_in.scene.planets[0]
        sma_m = _as_float(getattr(planet0, "semi_major_axis", None), u.m)
        star_mass_kg = _as_float(getattr(r_config_in.scene.star, "mass", None), u.kg)
        planet_mass_kg = _as_float(getattr(planet0, "mass", None), u.kg)
        if (
            np.isfinite(sma_m)
            and np.isfinite(star_mass_kg)
            and np.isfinite(planet_mass_kg)
            and sma_m > 0
            and (star_mass_kg + planet_mass_kg) > 0
        ):
            orbital_period_s = float(
                2.0 * np.pi * np.sqrt(sma_m ** 3 / (const.G.value * (star_mass_kg + planet_mass_kg)))
            )
        else:
            orbital_period_s = np.nan

        integration_fraction = (
            float(integration_time_s / orbital_period_s)
            if np.isfinite(integration_time_s) and np.isfinite(orbital_period_s) and orbital_period_s > 0
            else np.nan
        )
        integration_percent = (
            float(100.0 * integration_fraction)
            if np.isfinite(integration_fraction)
            else np.nan
        )

        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if np.isfinite(integration_percent):
            integration_percent_tag = f"{integration_percent:.6f}"
        else:
            integration_percent_tag = "unknown"
        print(
            f"Integration/orbit coverage: integration_time_s={integration_time_s}, "
            f"orbital_period_s={orbital_period_s}, percent={integration_percent}"
        )
        output_root = Path.cwd() / "MCMC" / "IntegrationPercent" / integration_percent_tag
        run_dir = output_root / f"{self.n_planet_params_out}_{run_timestamp}"
        arrays_dir = run_dir / "arrays"
        arrays_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving MCMC diagnostics to: {run_dir}")

        run_seeds = [mcmc_seed + i for i in range(run_count)]
        run_initial_positions = [None for _ in range(run_count)]

        def _run_single_mcmc(run_idx):
            run_rng = np.random.default_rng(run_seeds[run_idx])
            run_p0 = np.zeros((walkers_per_run, ndim), dtype=float)
            for k in range(walkers_per_run):
                for j in range(ndim):
                    run_p0[k, j] = run_rng.uniform(
                        params[params_names[j]].min,
                        params[params_names[j]].max
                    )
            run_initial_positions[run_idx] = run_p0.copy()

            print(f"MCMC run {run_idx + 1}/{run_count}: nwalkers={walkers_per_run}")
            return minimize(
                residual_data,
                params.copy(),
                args=(data_in,),
                method='emcee',
                float_behavior='posterior',
                nwalkers=walkers_per_run,
                burn=mcmcsteps // 5,
                steps=mcmcsteps,
                # thin=10,
                pos=run_p0,
                progress=True,
            )

        if run_count > 1:
            print(f"CPU MCMC parallelism: running {run_count} independent MCMC runs.")
            with ThreadPool(processes=run_count) as thread_pool:
                run_results = thread_pool.map(_run_single_mcmc, range(run_count))
        else:
            print("MCMC parallelism: single run.")
            run_results = [_run_single_mcmc(0)]

        out = min(run_results, key=lambda res: res.chisqr if np.isfinite(res.chisqr) else np.inf)

        var_names = list(out.var_names) if out.var_names is not None else list(params_names)
        merged_chain = np.vstack([res.flatchain[var_names].to_numpy() for res in run_results])
        merged_medians = np.median(merged_chain, axis=0)
        combined_cov = np.cov(merged_chain, rowvar=False) if merged_chain.shape[0] > 1 else None

        for i, name in enumerate(var_names):
            if name in out.params:
                out.params[name].value = float(merged_medians[i])
                if combined_cov is not None:
                    out.params[name].stderr = float(np.sqrt(combined_cov[i, i]))

        print(f"Combined posterior samples: {merged_chain.shape[0]}")

        np.save(arrays_dir / "posterior_chain_merged.npy", np.asarray(merged_chain))
        np.savetxt(
            run_dir / "posterior_chain_merged.csv",
            np.asarray(merged_chain),
            delimiter=",",
            header=",".join(var_names),
            comments="",
        )

        for run_idx, run_result in enumerate(run_results):
            run_chain = run_result.flatchain[var_names].to_numpy()
            np.save(arrays_dir / f"run_{run_idx + 1}_flatchain.npy", np.asarray(run_chain))
            if hasattr(run_result, "chain") and run_result.chain is not None:
                np.save(arrays_dir / f"run_{run_idx + 1}_chain.npy", np.asarray(run_result.chain))
            if hasattr(run_result, "lnprob") and run_result.lnprob is not None:
                np.save(arrays_dir / f"run_{run_idx + 1}_lnprob.npy", np.asarray(run_result.lnprob))
        np.save(arrays_dir / "initial_walkers_per_run.npy", np.asarray(run_initial_positions))

        plot_samples = np.asarray(merged_chain, dtype=float).copy()
        display_names = []
        is_log_axis = []
        for i, name in enumerate(var_names):
            if name.startswith("log_"):
                plot_samples[:, i] = np.power(10.0, plot_samples[:, i])
                display_names.append(name.replace("log_", "", 1))
                is_log_axis.append(True)
            else:
                display_names.append(name)
                is_log_axis.append(False)

        try:
            n_params = len(var_names)
            fig, axes = plt.subplots(n_params, n_params, figsize=(2.6 * n_params, 2.6 * n_params), squeeze=False)
            for row in range(n_params):
                for col in range(n_params):
                    ax = axes[row, col]
                    if row == col:
                        ax.hist(plot_samples[:, col], bins=40, color="tab:blue", alpha=0.85)
                        if is_log_axis[col]:
                            ax.set_xscale("log")
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

            fig.suptitle("MCMC Correlation Plot", y=1.0)
            fig.tight_layout()
            corr_plot_path = run_dir / "correlation_plot.png"
            fig.savefig(corr_plot_path, dpi=180, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved correlation plot to: {corr_plot_path}")
        except Exception as exc:
            print(f"Could not generate correlation plot: {exc}")

        print("success:", out.success)
        print("nfev:", out.nfev)
        print("residual evaluations:", eval_counter)
        print("chisqr:", out.chisqr)
        cov_out = combined_cov if combined_cov is not None else out.covar
        cov_err_by_name = {}
        if cov_out is not None and out.var_names is not None:
            try:
                stds = np.sqrt(np.diag(cov_out))
                cov_err_by_name = {
                    name: float(stds[i]) for i, name in enumerate(out.var_names) if i < len(stds)
                }
            except (TypeError, ValueError, IndexError):
                cov_err_by_name = {}


        # fluxes = Blackbody(radius=out.params['radius'].value, temp=out.params['temp'].value, wavelengths=wavelengths)
        fluxes = np.pi * get_blackbody_spectrum_standard_units( temperature=10**out.params['log_temp'].value, wavelengths=wavelengths)  * (10**out.params['log_radius'].value / distance) ** 2

        def dBdT(radius,temp,wavelengths):
            lam_m = np.asarray(wavelengths, dtype=float)
            return (4 * np.pi * const.c.value**2 * const.h.value) / ( lam_m ** 5 * const.k_B.value * temp) / (radius * distance)**2 * np.exp(const.h.value*const.c.value/(const.k_B.value*temp*lam_m)) / (np.expm1(const.h.value * const.c.value / (lam_m*const.k_B.value*temp))**2)

        def dBdR(radius,temp,wavelengths):
            lam_m = np.asarray(wavelengths, dtype=float)
            x = (const.h.value * const.c.value) / (lam_m * const.k_B.value * temp)
            denom = np.expm1(x)
            photon_flux_per_m = (2.0 * np.pi * const.c.value / lam_m ** 4) * 2*(radius / (distance**2)) / denom
            return photon_flux_per_m

        def _get_param_value(name, default=np.nan):
            return out.params[name].value if name in out.params else default

        def _get_param_err(name):
            if name not in out.params:
                return np.nan
            stderr = out.params[name].stderr
            if stderr is not None and np.isfinite(stderr):
                return float(stderr)
            if name in cov_err_by_name:
                return cov_err_by_name[name]
            return np.nan

        def _log10_to_linear_and_err(log_value: float, log_err: float) -> tuple[float, float]:
            linear = float(np.power(10.0, log_value))
            if np.isfinite(log_err):
                linear_err = float(np.log(10.0) * linear * log_err)
            else:
                linear_err = np.nan
            return linear, linear_err

        log_temp = _get_param_value('log_temp')
        log_radius = _get_param_value('log_radius')
        log_semi_major_axis = _get_param_value('log_semi_major_axis')

        log_temp_err = _get_param_err('log_temp')
        log_radius_err = _get_param_err('log_radius')
        log_semi_major_axis_err = _get_param_err('log_semi_major_axis')

        temp, temp_err = _log10_to_linear_and_err(log_temp, log_temp_err)
        radius, radius_err = _log10_to_linear_and_err(log_radius, log_radius_err)
        semi_major_axis, semi_major_axis_err = _log10_to_linear_and_err(log_semi_major_axis, log_semi_major_axis_err)

        eccentricity = _get_param_value('eccentricity')
        cos_inclination = _get_param_value('cos_inclination')
        cos_inclination_clipped = np.clip(cos_inclination, -1.0, 1.0) if np.isfinite(cos_inclination) else np.nan
        inclination = float(np.arccos(cos_inclination_clipped)) if np.isfinite(cos_inclination_clipped) else np.nan
        raan = _get_param_value('raan')
        argument_of_periapsis = _get_param_value('argument_of_periapsis')
        true_anomaly = _get_param_value('true_anomaly')
        eccentricity_err = _get_param_err('eccentricity')
        cos_inclination_err = _get_param_err('cos_inclination')
        if np.isfinite(cos_inclination_err) and np.isfinite(cos_inclination_clipped):
            denom = np.sqrt(max(1e-16, 1.0 - cos_inclination_clipped ** 2))
            inclination_err = float(cos_inclination_err / denom)
        else:
            inclination_err = np.nan
        raan_err = _get_param_err('raan')
        argument_of_periapsis_err = _get_param_err('argument_of_periapsis')
        true_anomaly_err = _get_param_err('true_anomaly')

        if np.isfinite(temp_err) and np.isfinite(radius_err):
            flux_err = np.sqrt(
                dBdT(radius, temp, wavelengths) ** 2 * temp_err ** 2
                + dBdR(radius, temp, wavelengths) ** 2 * radius_err ** 2
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

        input_parameter_definitions = {}
        for name in params_names:
            p = params[name]
            input_parameter_definitions[name] = {
                "initial_value": float(p.value),
                "min": float(p.min) if p.min is not None else None,
                "max": float(p.max) if p.max is not None else None,
            }

        used_parameter_values = {}
        for name in var_names:
            if name in out.params:
                stderr = out.params[name].stderr
                used_parameter_values[name] = {
                    "value": float(out.params[name].value),
                    "stderr": float(stderr) if stderr is not None and np.isfinite(stderr) else None,
                }

        def _serializable_number(value):
            val = _as_float(value)
            return float(val) if np.isfinite(val) else None

        def _compare_true_fitted(true_value, fitted_value):
            if true_value is None:
                return {
                    "true": None,
                    "fitted": float(fitted_value) if np.isfinite(fitted_value) else None,
                    "abs_error": None,
                }
            if np.isfinite(fitted_value):
                return {
                    "true": float(true_value),
                    "fitted": float(fitted_value),
                    "abs_error": float(fitted_value - true_value),
                }
            return {"true": float(true_value), "fitted": None, "abs_error": None}

        star_true_parameters = {
            "name": getattr(r_config_in.scene.star, "name", None),
            "distance_m": _serializable_number(getattr(r_config_in.scene.star, "distance", None)),
            "mass_kg": _serializable_number(getattr(r_config_in.scene.star, "mass", None)),
            "radius_m": _serializable_number(getattr(r_config_in.scene.star, "radius", None)),
            "temperature_K": _serializable_number(
                getattr(
                    r_config_in.scene.star,
                    "temperature",
                    getattr(r_config_in.scene.star, "temp", None),
                )
            ),
        }

        true_temp = _serializable_number(getattr(planet0, "temperature", getattr(planet0, "temp", None)))
        true_radius = _serializable_number(getattr(planet0, "radius", None))
        true_semi_major_axis = _serializable_number(getattr(planet0, "semi_major_axis", None))
        true_eccentricity = _serializable_number(getattr(planet0, "eccentricity", None))
        true_inclination = _serializable_number(getattr(planet0, "inclination", None))
        true_raan = _serializable_number(getattr(planet0, "raan", None))
        true_argument_of_periapsis = _serializable_number(getattr(planet0, "argument_of_periapsis", None))
        true_true_anomaly = _serializable_number(getattr(planet0, "true_anomaly", None))

        planet_true_parameters = {
            "name": getattr(planet0, "name", None),
            "mass_kg": _serializable_number(getattr(planet0, "mass", None)),
            "radius_m": true_radius,
            "temperature_K": true_temp,
            "semi_major_axis_m": true_semi_major_axis,
            "eccentricity": true_eccentricity,
            "inclination_rad": true_inclination,
            "raan_rad": true_raan,
            "argument_of_periapsis_rad": true_argument_of_periapsis,
            "true_anomaly_rad": true_true_anomaly,
        }

        true_vs_fitted = {
            "temperature_K": _compare_true_fitted(true_temp, temp),
            "radius_m": _compare_true_fitted(true_radius, radius),
            "semi_major_axis_m": _compare_true_fitted(true_semi_major_axis, semi_major_axis),
            "eccentricity": _compare_true_fitted(true_eccentricity, eccentricity),
            "inclination_rad": _compare_true_fitted(true_inclination, inclination),
            "raan_rad": _compare_true_fitted(true_raan, raan),
            "argument_of_periapsis_rad": _compare_true_fitted(true_argument_of_periapsis, argument_of_periapsis),
            "true_anomaly_rad": _compare_true_fitted(true_true_anomaly, true_anomaly),
        }

        reproducibility_metadata = {
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
                "run_count": run_count,
                "walkers_per_run": walkers_per_run,
                "burn": mcmcsteps // 5,
                "steps": mcmcsteps,
                "seed_base": mcmc_seed,
                "run_seeds": run_seeds,
                "ndim": ndim,
                "eval_counter": eval_counter,
                "distance_m": float(distance),
                "planet_mass_kg": float(planet_mass_fixed),
                "a_max_m": float(a_max),
                "data_shape": list(data_shape),
                "integration_time_s": float(integration_time_s) if np.isfinite(integration_time_s) else None,
                "orbital_period_s": float(orbital_period_s) if np.isfinite(orbital_period_s) else None,
                "integration_fraction": float(integration_fraction) if np.isfinite(integration_fraction) else None,
                "integration_percent": float(integration_percent) if np.isfinite(integration_percent) else None,
                "output_integration_percent_tag": integration_percent_tag,
            },
            "input_parameter_definitions": input_parameter_definitions,
            "used_parameter_values": used_parameter_values,
            "star_true_parameters": star_true_parameters,
            "planet_true_parameters": planet_true_parameters,
            "true_vs_fitted": true_vs_fitted,
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
                "argument_of_periapsis_err_rad": float(argument_of_periapsis_err) if np.isfinite(argument_of_periapsis_err) else None,
                "true_anomaly_rad": float(true_anomaly),
                "true_anomaly_err_rad": float(true_anomaly_err) if np.isfinite(true_anomaly_err) else None,
            },
        }
        with open(run_dir / "run_parameters.json", "w", encoding="utf-8") as fp:
            json.dump(reproducibility_metadata, fp, indent=2)
        print(f"Saved run parameters to: {run_dir / 'run_parameters.json'}")

        # TODO: Implement multi-planet signal extraction
        r_planet_params_out = PlanetParamsResource(
            name=self.n_planet_params_out,
        )
        planet_params = PlanetParams(
            name='fitted planet',
            sed_wavelength_bin_centers=r_config_in.phringe.get_wavelength_bin_centers(),
            sed_wavelength_bin_widths=r_config_in.phringe.get_wavelength_bin_widths(),
            sed=torch.tensor(fluxes),
            sed_err_low=torch.tensor(flux_err),
            sed_err_high=torch.tensor(flux_err),
            covariance=cov_out,
            temp = temp,
            temp_err_low = temp_err,
            temp_err_high = temp_err,
            radius = radius,
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
        print("Argument of Periapsis ", out.params["argument_of_periapsis"].value)
        print("Eccentricity ",out.params["eccentricity"].value)
        print("RAAN ",out.params["raan"].value)
        print("SMA ",10**out.params["log_semi_major_axis"].value)

        print("Temperature ", 10**out.params['log_temp'].value)
        print("Radius ", 10**out.params['log_radius'].value)
        print("True Anomaly",out.params['true_anomaly'].value)
        print('Done')
        return r_planet_params_out
