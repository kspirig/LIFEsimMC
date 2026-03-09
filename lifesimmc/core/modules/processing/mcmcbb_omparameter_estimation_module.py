import numpy as np
import torch
from lmfit import minimize, Parameters
from pathlib import Path
from typing import Optional
from typing import Any, Callable
import json
import platform
import sys
from datetime import datetime
from uuid import uuid4
from dataclasses import dataclass
from multiprocessing.dummy import Pool as ThreadPool

from lifesimmc.core.modules.base_module import BaseModule
from lifesimmc.core.resources.base_resource import BaseResource
from lifesimmc.core.resources.planet_params_resource import PlanetParamsResource, PlanetParams
from astropy import constants as const
from phringe.util.spectrum import get_blackbody_spectrum_standard_units

import astropy.units as u


@dataclass
class _MCMCResidualContext:
    r_config_in: Any
    wavelengths: np.ndarray
    distance: float
    transf: Callable[[Any], Any]
    data_shape: tuple
    model_transpose_axes: tuple
    planet_mass_fixed: float
    sigma_in: Optional[np.ndarray]
    use_poisson_likelihood: bool


_MCMC_RESIDUAL_CONTEXT: Optional[_MCMCResidualContext] = None
_MCMC_EVAL_COUNTER = 0
_MCMC_ORBITAL_FAILURE_PRINTED = False
_MCMC_LOG_EVERY = 200





def _pow10(x: float) -> float:
    return float(np.power(10.0, x))


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def _snapshot_resource(resource: Any) -> dict:
    snapshot = {}
    if resource is None:
        return snapshot
    try:
        resource_items = vars(resource).items()
    except TypeError:
        return {"type": str(type(resource)), "repr": str(resource)}

    for key, value in resource_items:
        if key.startswith("_") or callable(value):
            continue
        if isinstance(value, (str, int, float, bool, type(None))):
            snapshot[key] = value
        elif isinstance(value, np.ndarray):
            snapshot[f"{key}_shape"] = list(value.shape)
            snapshot[f"{key}_dtype"] = str(value.dtype)
        elif isinstance(value, torch.Tensor):
            snapshot[f"{key}_shape"] = list(value.shape)
            snapshot[f"{key}_dtype"] = str(value.dtype)
        elif isinstance(value, (list, tuple)) and len(value) <= 20:
            if all(isinstance(item, (str, int, float, bool, type(None))) for item in value):
                snapshot[key] = list(value)
            else:
                snapshot[key] = f"{type(value).__name__}(len={len(value)})"
        elif isinstance(value, dict):
            small = {}
            for k, v in value.items():
                if isinstance(v, (str, int, float, bool, type(None))):
                    small[str(k)] = v
                else:
                    small[str(k)] = str(type(v))
            snapshot[key] = small
        else:
            snapshot[key] = str(type(value))
    return snapshot


def _residual_data_emcee(params, target):
    global _MCMC_EVAL_COUNTER, _MCMC_ORBITAL_FAILURE_PRINTED
    ctx = _MCMC_RESIDUAL_CONTEXT
    if ctx is None:
        raise RuntimeError("MCMC residual context not set.")

    _MCMC_EVAL_COUNTER += 1

    two_pi = 2.0 * np.pi
    temp = _pow10(params["log_temp"].value)
    radius = _pow10(params["log_radius"].value)
    semi_major_axis = _pow10(params["log_semi_major_axis"].value)
    raan = float(np.mod(params["raan_raw"].value, two_pi))
    argument_of_periapsis = float(np.mod(params["argument_of_periapsis_raw"].value, two_pi))
    true_anomaly = float(np.mod(params["true_anomaly_raw"].value, two_pi))

    if _MCMC_LOG_EVERY > 0 and _MCMC_EVAL_COUNTER % _MCMC_LOG_EVERY == 0:
        print(
            f"[Eval {_MCMC_EVAL_COUNTER}] "
            f"T={temp:.6g}, R={radius:.6g}, "
            f"a={semi_major_axis:.6g}, e={params['eccentricity'].value:.6g}, "
            f"inc={params['inclination'].value:.6g}, raan={raan:.6g}, "
            f"argp={argument_of_periapsis:.6g}, nu={true_anomaly:.6g}"
        )

    flux = np.pi * get_blackbody_spectrum_standard_units(
        temperature=temp,
        wavelengths=ctx.wavelengths,
    ) * (radius / ctx.distance) ** 2
    flux = flux.detach().cpu().numpy()

    try:
        model = ctx.r_config_in.phringe.get_model_counts(
            kernels=True,
            spectral_energy_distribution=flux,
            semi_major_axis=semi_major_axis,
            eccentricity=params["eccentricity"].value,
            inclination=params["inclination"].value,
            raan=raan,
            argument_of_periapsis=argument_of_periapsis,
            true_anomaly=true_anomaly,
            host_star_distance=ctx.r_config_in.scene.star.distance,
            host_star_mass=ctx.r_config_in.scene.star.mass,
            planet_mass=ctx.planet_mass_fixed,
        )
    except (RuntimeError, ValueError, FloatingPointError) as exc:
        if not _MCMC_ORBITAL_FAILURE_PRINTED:
            print(f"Orbital model failed for trial parameters; continuing with penalty residual. {exc}")
            _MCMC_ORBITAL_FAILURE_PRINTED = True
        if ctx.use_poisson_likelihood:
            return -1e100
        return np.full_like(target, 1e30, dtype=float)

    model = ctx.transf(model)
    if isinstance(model, torch.Tensor):
        model = model.detach().cpu().numpy()
    else:
        model = np.asarray(model)
    model = np.transpose(model, ctx.model_transpose_axes).reshape(ctx.data_shape)

    if ctx.use_poisson_likelihood:
        model_safe = np.clip(model, 1e-12, None)
        nll = np.sum(model_safe - target * np.log(model_safe))
        return -float(nll)

    if ctx.sigma_in is not None:
        residual = (model - target) / ctx.sigma_in
    else:
        residual = model - target
    if not np.all(np.isfinite(residual)):
        print("Non-finite residual detected")
    return residual

class MCMCBBOMParameterEstimationModule(BaseModule):
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
            n_transformation_in: str = None,
            n_template_in: str = None,
            n_planet_params_in: str = None,
            bounds: bool = False,
            n_cores: int = None
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
        self.n_cores = n_cores

    def apply(self, resources: list[BaseResource]) -> PlanetParamsResource:
        print('Performing numerical MLE with Blackbody and Orbital Motion...')
        run_started_utc = datetime.utcnow()
        plot_run_id = f"{datetime.utcnow().strftime('%Y%m%dT%H%M%S%fZ')}_{uuid4().hex[:8]}"
        output_root = Path.cwd() / "mcmcbb_om_runs"
        run_dir = output_root / f"{self.n_planet_params_out}_{plot_run_id}"
        plots_dir = run_dir / "plots"
        arrays_dir = run_dir / "arrays"
        tables_dir = run_dir / "tables"
        for path in (run_dir, plots_dir, arrays_dir, tables_dir):
            path.mkdir(parents=True, exist_ok=True)
        print(f"Saving MCMC artifacts to: {run_dir}")

        r_config_in = self.get_resource_from_name(self.n_config_in)
        r_transformation_in = self.get_resource_from_name(
            self.n_transformation_in) if self.n_transformation_in else None
        transf = r_transformation_in.transformation if r_transformation_in else None
        planet_params_in = self.get_resource_from_name(self.n_planet_params_in) if self.n_planet_params_in else None


        distance = r_config_in.scene.star.distance  # float in meters



        times_tensor = r_config_in.phringe.get_time_steps()
        wavelengths_tensor = r_config_in.phringe.get_wavelength_bin_centers()
        wavelength_bin_widths_tensor = r_config_in.phringe.get_wavelength_bin_widths()
        times = times_tensor.cpu().numpy()
        wavelengths = wavelengths_tensor.cpu().numpy()
        wavelength_bin_widths = wavelength_bin_widths_tensor.cpu().numpy()
        print("Wavelength min/max:", np.min(wavelengths), np.max(wavelengths))
        print("Wavelength width min/max:", np.min(wavelength_bin_widths), np.max(wavelength_bin_widths))
        data_resource = self.get_resource_from_name(self.n_data_in)
        data_in = data_resource.get_data()
        cuda_active = (
            (isinstance(times_tensor, torch.Tensor) and times_tensor.is_cuda)
            or (isinstance(wavelengths_tensor, torch.Tensor) and wavelengths_tensor.is_cuda)
            or (isinstance(wavelength_bin_widths_tensor, torch.Tensor) and wavelength_bin_widths_tensor.is_cuda)
            or (isinstance(data_in, torch.Tensor) and data_in.is_cuda)
        )

        # Flatten data along differential outputs and times axes
        data_in = data_in.permute(0, 2, 1)
        data_in = data_in.reshape((-1,) + data_in.shape[2:])

        # Set up parameters and initial conditions
        # TODO: implement for multiple planets

        radius_init = 6 * 1e6  # 1 Earth radius in meters
        temp_init = 300.0  # Kelvin
        semi_major_axis_init = 1 * const.au.value  # Can we evaluate quite good?
        eccentricity_init = 0.0
        planet_mass_fixed = 1 * const.M_earth.value
        inclination_init = np.pi / 2
        raan_init = np.pi
        argument_of_periapsis_init = np.pi
        true_anomaly_init = 0



        data_in = data_in.detach().cpu().numpy()
        data_shape = data_in.shape
        model_transpose_axes = (0, 2, 1)

        sigma_in: Optional[np.ndarray] = None

        def _extract_sigma(resource) -> Optional[np.ndarray]:
            for attr in (
                "sigma",
                "uncertainty",
                "uncertainties",
                "std",
                "noise_std",
                "_sigma",
                "_uncertainty",
            ):
                if hasattr(resource, attr):
                    value = getattr(resource, attr)
                    if value is not None:
                        if isinstance(value, torch.Tensor):
                            return value.detach().cpu().numpy()
                        return np.asarray(value)
            for method in ("get_sigma", "get_uncertainty", "get_uncertainties", "get_std"):
                if hasattr(resource, method):
                    fn = getattr(resource, method)
                    if callable(fn):
                        value = fn()
                        if value is not None:
                            if isinstance(value, torch.Tensor):
                                return value.detach().cpu().numpy()
                            return np.asarray(value)
            return None

        sigma_in = _extract_sigma(data_resource)
        if sigma_in is not None:
            if sigma_in.shape != data_shape:
                try:
                    sigma_in = np.reshape(sigma_in, data_shape)
                except ValueError:
                    print("Sigma shape mismatch; ignoring provided uncertainties.")
                    sigma_in = None

        use_poisson_likelihood = sigma_in is None and np.all(np.isfinite(data_in)) and np.all(data_in >= 0)
        if use_poisson_likelihood:
            print("Likelihood mode: Poisson (count data).")
        elif sigma_in is not None:
            sigma_in = np.maximum(sigma_in, 1e-12)
            print("Likelihood mode: Gaussian with provided sigma.")
        else:
            print("Likelihood mode: Gaussian with unit sigma.")

        params = Parameters()
        two_pi = 2.0 * np.pi
        log10 = np.log10
        print("Sampling in log10-space for temp, radius, and semi_major_axis.")

        params.add("log_temp", value=log10(temp_init), min=log10(50), max=log10(4000))
        params.add("log_radius", value=log10(radius_init), min=log10(1e5), max=log10(2 * 1e8))

        params.add(
            "log_semi_major_axis",
            value=log10(semi_major_axis_init),
            min=log10(0.01 * const.au.value),
            max=log10(20 * const.au.value),
        )
        params.add('eccentricity',value = eccentricity_init,min = 0,max = 0.5)

        params.add('inclination',value = inclination_init, min = 0 ,max = np.pi)
        # Keep angle parameters bounded so walker initialization can be uniform in [0, 2pi).
        params.add('raan_raw', value=raan_init, min=0.0, max=two_pi)
        params.add('argument_of_periapsis_raw', value=argument_of_periapsis_init, min=0.0, max=two_pi)
        params.add('true_anomaly_raw', value=true_anomaly_init, min=0.0, max=two_pi)

        vary_names = [name for name, par in params.items() if par.vary]
        display_name_map = {
            "log_temp": "log10(temp)",
            "log_radius": "log10(radius)",
            "log_semi_major_axis": "log10(semi_major_axis)",
            "raan_raw": "raan",
            "argument_of_periapsis_raw": "argument_of_periapsis",
            "true_anomaly_raw": "true_anomaly",
        }
        varying_params = [par for _, par in params.items() if par.vary]
        ndim = len(varying_params)
        if ndim == 0:
            raise ValueError("No varying parameters configured for emcee.")

        print("Initializing MCMC walkers uniformly within parameter bounds.")
        mcmc_seed = 12345
        rng = np.random.default_rng(mcmc_seed)
        nwalkers = max(40, 2 * ndim + 8)
        p0 = np.zeros((nwalkers, ndim), dtype=float)
        for j, par in enumerate(varying_params):
            lo = par.min if par.min is not None else -np.inf
            hi = par.max if par.max is not None else np.inf
            if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
                raise ValueError(
                    f"Parameter '{vary_names[j]}' requires finite min/max bounds for uniform walker initialization."
                )
            eps = 1e-12 * max(1.0, abs(lo), abs(hi))
            lo_safe = lo + eps
            hi_safe = hi - eps
            if hi_safe <= lo_safe:
                lo_safe, hi_safe = lo, hi
            p0[:, j] = rng.uniform(lo_safe, hi_safe, size=nwalkers)

        eval_counter = 0

        def _wrap_angle(x: float) -> float:
            return float(np.mod(x, two_pi))

        def _evaluate_model_numpy(
            temp: float,
            radius: float,
            semi_major_axis: float,
            eccentricity: float,
            inclination: float,
            raan: float,
            argument_of_periapsis: float,
            true_anomaly: float,
        ) -> tuple[np.ndarray, np.ndarray]:
            flux = np.pi * get_blackbody_spectrum_standard_units(
                temperature=temp,
                wavelengths=wavelengths,
            ) * (radius / distance) ** 2
            flux = flux.detach().cpu().numpy()

            model = r_config_in.phringe.get_model_counts(
                kernels=True,
                spectral_energy_distribution=flux,
                semi_major_axis=semi_major_axis,
                eccentricity=eccentricity,
                inclination=inclination,
                raan=raan,
                argument_of_periapsis=argument_of_periapsis,
                true_anomaly=true_anomaly,
                host_star_distance=r_config_in.scene.star.distance,
                host_star_mass=r_config_in.scene.star.mass,
                planet_mass=planet_mass_fixed,
            )
            model = transf(model)
            model = np.transpose(model, (0, 2, 1))
            model = model.reshape(data_in.shape)

            return model, flux

        global _MCMC_RESIDUAL_CONTEXT, _MCMC_EVAL_COUNTER, _MCMC_ORBITAL_FAILURE_PRINTED
        _MCMC_RESIDUAL_CONTEXT = _MCMCResidualContext(
            r_config_in=r_config_in,
            wavelengths=wavelengths,
            distance=distance,
            transf=transf,
            data_shape=data_shape,
            model_transpose_axes=model_transpose_axes,
            planet_mass_fixed=planet_mass_fixed,
            sigma_in=sigma_in,
            use_poisson_likelihood=use_poisson_likelihood,
        )
        _MCMC_EVAL_COUNTER = 0
        _MCMC_ORBITAL_FAILURE_PRINTED = False
        requested_workers = int(self.n_cores) if self.n_cores is not None else 1
        if requested_workers < 1:
            requested_workers = 1
        max_useful_workers = max(1, nwalkers // 2)
        worker_count = min(requested_workers, max_useful_workers)
        mcmc_thin = 10
        mcmc_burn = 600
        mcmc_steps = 4800

        try:
            # emcee stretch moves update approximately half the walkers at a time.
            if requested_workers > max_useful_workers:
                print(
                    f"MCMC parallelism: requested n_cores={requested_workers}, "
                    f"capped to {worker_count} (nwalkers={nwalkers})."
                )

            if cuda_active and worker_count > 1:
                print(
                    "MCMC parallelism: CUDA detected; disabling thread pool "
                    "because a shared GPU context is typically slower with Python threads."
                )
                worker_count = 1

            if worker_count > 1:
                mcmc_burn = 300
                mcmc_steps = 2400
                print(
                    f"CPU MCMC parallelism: running {worker_count} thread workers "
                    f"(requested {requested_workers})."
                )
                with ThreadPool(processes=worker_count) as thread_pool:
                    out = minimize(
                        _residual_data_emcee,
                        params,
                        args=(data_in,),
                        method='emcee',
                        float_behavior='posterior',
                        nwalkers=nwalkers,
                        burn=mcmc_burn,
                        steps=mcmc_steps,
                        thin=mcmc_thin,
                        pos=p0,
                        progress=True,
                        workers=thread_pool,
                    )
            else:
                print(
                    f"MCMC parallelism: disabled (effective workers={worker_count}, "
                    f"requested n_cores={requested_workers}, cuda_active={cuda_active})."
                )
                out = minimize(
                    _residual_data_emcee,
                    params,
                    args=(data_in,),
                    method='emcee',
                    float_behavior='posterior',
                    nwalkers=nwalkers,
                    burn=mcmc_burn,
                    steps=mcmc_steps,
                    thin=mcmc_thin,
                    pos=p0,
                    progress=True,
                )
        finally:
            eval_counter = _MCMC_EVAL_COUNTER
            _MCMC_RESIDUAL_CONTEXT = None

        acceptance_fraction_mean = np.nan
        chain_normalized = None
        split_rhat_by_param = {}
        autocorr_tau_by_param = {}
        ess_by_param = {}

        print("success:", out.success)
        print("message:", getattr(out, "message", "n/a"))
        print("nfev:", getattr(out, "nfev", "n/a"))
        print("residual evaluations:", eval_counter)
        if hasattr(out, "acceptance_fraction"):
            acceptance_fraction_mean = float(np.mean(out.acceptance_fraction))
            print("acceptance fraction:", acceptance_fraction_mean)
        print("chisqr:", getattr(out, "chisqr", "n/a"))

        # Basic convergence diagnostics for emcee results.
        # These are heuristic checks and should be interpreted together.
        try:
            chain = getattr(out, "chain", None)
            if chain is not None:
                chain = np.asarray(chain)
                # Normalize chain shape to (nwalkers, nsteps, npar).
                if chain.ndim == 3:
                    if chain.shape[0] == len(vary_names):
                        chain = np.transpose(chain, (1, 2, 0))
                    elif chain.shape[2] == len(vary_names):
                        pass
                    else:
                        # Fallback assumption: (nsteps, nwalkers, npar)
                        chain = np.transpose(chain, (1, 0, 2))

                    nwalkers_c, nsteps_c, npar_c = chain.shape
                    chain_normalized = chain
                    print(f"chain shape: walkers={nwalkers_c}, steps={nsteps_c}, params={npar_c}")

                    # Split-Rhat per parameter (split each walker chain in half).
                    if nsteps_c >= 4:
                        half = nsteps_c // 2
                        split = np.concatenate([chain[:, :half, :], chain[:, half:2 * half, :]], axis=0)
                        m = split.shape[0]
                        n = split.shape[1]
                        chain_means = np.mean(split, axis=1)
                        chain_vars = np.var(split, axis=1, ddof=1)
                        W = np.mean(chain_vars, axis=0)
                        B = n * np.var(chain_means, axis=0, ddof=1)
                        var_hat = ((n - 1) / n) * W + (1 / n) * B
                        rhat = np.sqrt(np.maximum(var_hat / np.maximum(W, 1e-30), 0.0))
                        print("split R_hat by parameter:")
                        for name, val in zip(vary_names, rhat):
                            label = display_name_map.get(name, name)
                            split_rhat_by_param[label] = float(val)
                            print(f"  {label}: {val:.4f}")
                    else:
                        print("split R_hat skipped: too few steps.")

                    # Integrated autocorrelation time and rough ESS per parameter.
                    try:
                        from emcee.autocorr import integrated_time
                        print("autocorr/ESS by parameter:")
                        for j, name in enumerate(vary_names):
                            # integrated_time expects shape (nsteps, nwalkers)
                            param_chain = np.transpose(chain[:, :, j], (1, 0))
                            tau = float(integrated_time(param_chain, quiet=True))
                            ess = (nwalkers_c * nsteps_c) / max(2.0 * tau, 1e-30)
                            label = display_name_map.get(name, name)
                            autocorr_tau_by_param[label] = float(tau)
                            ess_by_param[label] = float(ess)
                            print(f"  {label}: tau={tau:.2f}, ESS~{ess:.1f}, steps/(50*tau)={nsteps_c / max(50.0 * tau, 1e-30):.3f}")
                    except Exception as exc:
                        print(f"autocorr/ESS skipped: {exc}")
                else:
                    print("chain diagnostics skipped: unexpected chain dimensionality.")
            else:
                print("chain diagnostics skipped: no chain available on result.")
        except Exception as exc:
            print(f"chain diagnostics failed: {exc}")

        cov_out = getattr(out, "covar", None)
        posterior_df = None
        if hasattr(out, "flatchain"):
            try:
                posterior_df = out.flatchain.copy()
                if "log_temp" in posterior_df:
                    posterior_df["temp"] = np.power(10.0, posterior_df["log_temp"].to_numpy(dtype=float))
                if "log_radius" in posterior_df:
                    posterior_df["radius"] = np.power(10.0, posterior_df["log_radius"].to_numpy(dtype=float))
                if "log_semi_major_axis" in posterior_df:
                    posterior_df["semi_major_axis"] = np.power(
                        10.0, posterior_df["log_semi_major_axis"].to_numpy(dtype=float)
                    )
                if "raan_raw" in posterior_df:
                    posterior_df["raan"] = np.mod(posterior_df["raan_raw"].to_numpy(), two_pi)
                if "argument_of_periapsis_raw" in posterior_df:
                    posterior_df["argument_of_periapsis"] = np.mod(
                        posterior_df["argument_of_periapsis_raw"].to_numpy(), two_pi
                    )
                if "true_anomaly_raw" in posterior_df:
                    posterior_df["true_anomaly"] = np.mod(posterior_df["true_anomaly_raw"].to_numpy(), two_pi)
            except Exception:
                posterior_df = None

        if posterior_df is not None:
            try:
                import matplotlib.pyplot as plt
                from pandas.plotting import scatter_matrix

                posterior_df.to_csv(tables_dir / "posterior_samples.csv", index=False)

                corr_cols = [
                    "temp",
                    "radius",
                    "semi_major_axis",
                    "eccentricity",
                    "inclination",
                    "raan",
                    "argument_of_periapsis",
                    "true_anomaly",
                ]
                corr_cols = [c for c in corr_cols if c in posterior_df.columns]
                if len(corr_cols) >= 2 and len(posterior_df) > 1:
                    fig_axes = scatter_matrix(
                        posterior_df[corr_cols],
                        diagonal="kde",
                        alpha=0.2,
                        figsize=(2.4 * len(corr_cols), 2.4 * len(corr_cols)),
                    )
                    fig = fig_axes[0, 0].figure
                    fig.suptitle("MCMC Parameter Correlations", y=1.0)
                    fig.tight_layout()
                    plot_path = plots_dir / "parameter_correlations.png"
                    fig.savefig(plot_path, dpi=180, bbox_inches="tight")
                    plt.close(fig)
                    print(f"Saved MCMC correlation plot to: {plot_path}")

                if chain_normalized is not None and chain_normalized.ndim == 3:
                    nwalkers_c, nsteps_c, npar_c = chain_normalized.shape
                    fig, axes = plt.subplots(
                        npar_c,
                        1,
                        figsize=(12, max(2.0 * npar_c, 4.0)),
                        sharex=True,
                        squeeze=False,
                    )
                    steps_axis = np.arange(nsteps_c)
                    for j in range(npar_c):
                        ax = axes[j, 0]
                        label = display_name_map.get(vary_names[j], vary_names[j]) if j < len(vary_names) else f"param_{j}"
                        ax.plot(steps_axis, chain_normalized[:, :, j].T, alpha=0.25, lw=0.7, color="tab:blue")
                        ax.set_ylabel(label)
                    axes[-1, 0].set_xlabel("MCMC step")
                    fig.suptitle(f"MCMC Trace Plot ({nwalkers_c} walkers)")
                    fig.tight_layout()
                    trace_plot_path = plots_dir / "trace_plot.png"
                    fig.savefig(trace_plot_path, dpi=180, bbox_inches="tight")
                    plt.close(fig)
                    print(f"Saved MCMC trace plot to: {trace_plot_path}")
            except Exception as exc:
                print(f"Could not generate MCMC posterior plots/tables: {exc}")

        def _linear_summary(samples: np.ndarray) -> tuple[float, float, float]:
            q16, q50, q84 = np.percentile(samples, [16, 50, 84])
            return float(q50), float(q50 - q16), float(q84 - q50)

        def _circular_summary(samples: np.ndarray) -> tuple[float, float, float]:
            wrapped = np.mod(samples, two_pi)
            mu = np.arctan2(np.mean(np.sin(wrapped)), np.mean(np.cos(wrapped)))
            mu = float(np.mod(mu, two_pi))
            shifted = (wrapped - mu + np.pi) % two_pi - np.pi
            q16, q50, q84 = np.percentile(shifted, [16, 50, 84])
            med = float(np.mod(mu + q50, two_pi))
            return med, float(q50 - q16), float(q84 - q50)

        def _posterior_summary(name: str, circular: bool, fallback: float) -> tuple[float, float, float]:
            if posterior_df is not None and name in posterior_df.columns and len(posterior_df[name]) > 1:
                samples = posterior_df[name].to_numpy(dtype=float)
                if circular:
                    return _circular_summary(samples)
                return _linear_summary(samples)
            return float(fallback), np.nan, np.nan

        fallback_raan = _wrap_angle(out.params["raan_raw"].value) if "raan_raw" in out.params else np.nan
        fallback_argp = _wrap_angle(out.params["argument_of_periapsis_raw"].value) if "argument_of_periapsis_raw" in out.params else np.nan
        fallback_nu = _wrap_angle(out.params["true_anomaly_raw"].value) if "true_anomaly_raw" in out.params else np.nan
        fallback_temp = _pow10(out.params["log_temp"].value) if "log_temp" in out.params else np.nan
        fallback_radius = _pow10(out.params["log_radius"].value) if "log_radius" in out.params else np.nan
        fallback_sma = (
            _pow10(out.params["log_semi_major_axis"].value)
            if "log_semi_major_axis" in out.params
            else np.nan
        )

        temp, temp_err_low, temp_err_high = _posterior_summary("temp", circular=False, fallback=fallback_temp)
        radius, radius_err_low, radius_err_high = _posterior_summary("radius", circular=False, fallback=fallback_radius)
        semi_major_axis, semi_major_axis_err_low, semi_major_axis_err_high = _posterior_summary(
            "semi_major_axis", circular=False, fallback=fallback_sma
        )
        eccentricity, eccentricity_err_low, eccentricity_err_high = _posterior_summary(
            "eccentricity", circular=False, fallback=out.params["eccentricity"].value
        )
        inclination, inclination_err_low, inclination_err_high = _posterior_summary(
            "inclination", circular=False, fallback=out.params["inclination"].value
        )
        raan, raan_err_low, raan_err_high = _posterior_summary("raan", circular=True, fallback=fallback_raan)
        argument_of_periapsis, argument_of_periapsis_err_low, argument_of_periapsis_err_high = _posterior_summary(
            "argument_of_periapsis", circular=True, fallback=fallback_argp
        )
        true_anomaly, true_anomaly_err_low, true_anomaly_err_high = _posterior_summary(
            "true_anomaly", circular=True, fallback=fallback_nu
        )

        fluxes = np.pi * get_blackbody_spectrum_standard_units(
            temperature=temp,
            wavelengths=wavelengths,
        ) * (radius / distance) ** 2
        fluxes = fluxes.detach().cpu().numpy()
        best_sample_info = {}
        best_model_flat = None
        best_data_flat = None
        best_residual_flat = None
        lnprob_flat = None

        if posterior_df is not None and hasattr(out, "lnprob"):
            try:
                import matplotlib.pyplot as plt

                lnprob_flat = np.asarray(out.lnprob).reshape(-1)
                if len(lnprob_flat) == len(posterior_df):
                    i_best = int(np.argmax(lnprob_flat))
                    best = posterior_df.iloc[i_best]
                    best_sample_info = {
                        "flat_index": int(i_best),
                        "lnprob": float(lnprob_flat[i_best]),
                        "temp": float(best["temp"]),
                        "radius": float(best["radius"]),
                        "semi_major_axis": float(best["semi_major_axis"]),
                        "eccentricity": float(best["eccentricity"]),
                        "inclination": float(best["inclination"]),
                        "raan": float(best["raan"]),
                        "argument_of_periapsis": float(best["argument_of_periapsis"]),
                        "true_anomaly": float(best["true_anomaly"]),
                    }
                    model_best, _ = _evaluate_model_numpy(
                        temp=float(best["temp"]),
                        radius=float(best["radius"]),
                        semi_major_axis=float(best["semi_major_axis"]),
                        eccentricity=float(best["eccentricity"]),
                        inclination=float(best["inclination"]),
                        raan=float(best["raan"]),
                        argument_of_periapsis=float(best["argument_of_periapsis"]),
                        true_anomaly=float(best["true_anomaly"]),
                    )
                    y_data = data_in.ravel().astype(float)
                    y_model = model_best.ravel().astype(float)
                    y_res = y_data - y_model
                    best_data_flat = y_data
                    best_model_flat = y_model
                    best_residual_flat = y_res
                    x = np.arange(y_data.size)

                    fig, axes = plt.subplots(
                        3,
                        1,
                        figsize=(12, 8),
                        gridspec_kw={"height_ratios": [3, 1.4, 1.2]},
                        sharex=True,
                    )
                    axes[0].plot(x, y_data, lw=1.0, label="data")
                    axes[0].plot(x, y_model, lw=1.0, label="ML model")
                    axes[0].legend(loc="best")
                    axes[0].set_ylabel("counts")
                    axes[0].set_title("Maximum-Likelihood Sample Fit Check")
                    axes[1].plot(x, y_res, lw=0.8, color="tab:green")
                    axes[1].axhline(0.0, color="k", lw=0.8, ls="--")
                    axes[1].set_ylabel("residual")
                    if np.isfinite(np.nanstd(y_data)) and np.nanstd(y_data) > 0:
                        normalized_res = y_res / np.nanstd(y_data)
                    else:
                        normalized_res = y_res
                    axes[2].plot(x, normalized_res, lw=0.8, color="tab:red")
                    axes[2].axhline(0.0, color="k", lw=0.8, ls="--")
                    axes[2].set_ylabel("res/std(data)")
                    axes[2].set_xlabel("flattened data index")
                    fig.tight_layout()
                    sanity_plot_path = plots_dir / "fitcheck_best_sample.png"
                    fig.savefig(sanity_plot_path, dpi=180, bbox_inches="tight")
                    plt.close(fig)
                    print(f"Saved MCMC fit sanity plot to: {sanity_plot_path}")

                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.hist(y_res, bins=80, density=True, alpha=0.8, color="tab:green")
                    ax.set_title("Residual Distribution (Best Posterior Sample)")
                    ax.set_xlabel("residual")
                    ax.set_ylabel("density")
                    fig.tight_layout()
                    residual_hist_path = plots_dir / "residual_histogram.png"
                    fig.savefig(residual_hist_path, dpi=180, bbox_inches="tight")
                    plt.close(fig)
                    print(f"Saved residual histogram to: {residual_hist_path}")
            except Exception as exc:
                print(f"Could not generate MCMC fit sanity plot: {exc}")

        mass = planet_mass_fixed

        def dBdT(radius,temp,wavelengths):
            lam_m = np.asarray(wavelengths, dtype=float)
            return (4 * np.pi * const.c.value**2 * const.h.value) / ( lam_m ** 5 * const.k_B.value * temp) / (radius * distance)**2 * np.exp(const.h.value*const.c.value/(const.k_B.value*temp*lam_m)) / (np.expm1(const.h.value * const.c.value / (lam_m*const.k_B.value*temp))**2)

        def dBdR(radius,temp,wavelengths):
            lam_m = np.asarray(wavelengths, dtype=float)
            x = (const.h.value * const.c.value) / (lam_m * const.k_B.value * temp)
            denom = np.expm1(x)
            photon_flux_per_m = (2.0 * np.pi * const.c.value / lam_m ** 4) * 2*(radius / (distance**2)) / denom
            return photon_flux_per_m

        temp_err = 0.5 * (temp_err_low + temp_err_high) if np.isfinite(temp_err_low) and np.isfinite(temp_err_high) else np.nan
        radius_err = 0.5 * (radius_err_low + radius_err_high) if np.isfinite(radius_err_low) and np.isfinite(radius_err_high) else np.nan
        semi_major_axis_err = (
            0.5 * (semi_major_axis_err_low + semi_major_axis_err_high)
            if np.isfinite(semi_major_axis_err_low) and np.isfinite(semi_major_axis_err_high)
            else np.nan
        )
        eccentricity_err = (
            0.5 * (eccentricity_err_low + eccentricity_err_high)
            if np.isfinite(eccentricity_err_low) and np.isfinite(eccentricity_err_high)
            else np.nan
        )
        inclination_err = (
            0.5 * (inclination_err_low + inclination_err_high)
            if np.isfinite(inclination_err_low) and np.isfinite(inclination_err_high)
            else np.nan
        )
        raan_err = 0.5 * (raan_err_low + raan_err_high) if np.isfinite(raan_err_low) and np.isfinite(raan_err_high) else np.nan
        argument_of_periapsis_err = (
            0.5 * (argument_of_periapsis_err_low + argument_of_periapsis_err_high)
            if np.isfinite(argument_of_periapsis_err_low) and np.isfinite(argument_of_periapsis_err_high)
            else np.nan
        )
        true_anomaly_err = (
            0.5 * (true_anomaly_err_low + true_anomaly_err_high)
            if np.isfinite(true_anomaly_err_low) and np.isfinite(true_anomaly_err_high)
            else np.nan
        )
        mass_err = np.nan

        if np.isfinite(temp_err) and np.isfinite(radius_err):
            flux_err = np.sqrt(
                dBdT(radius, temp, wavelengths) ** 2 * temp_err ** 2
                + dBdR(radius, temp, wavelengths) ** 2 * radius_err ** 2
            )
        else:
            flux_err = np.full_like(fluxes, np.nan, dtype=float)

        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.plot(wavelengths, fluxes, lw=1.6, color="tab:blue", label="retrieved SED")
            if np.any(np.isfinite(flux_err)):
                low = np.asarray(fluxes - flux_err, dtype=float)
                high = np.asarray(fluxes + flux_err, dtype=float)
                ax.fill_between(wavelengths, low, high, color="tab:blue", alpha=0.25, label="1-sigma")
            ax.set_xlabel("wavelength [m]")
            ax.set_ylabel("photon flux density")
            ax.set_title("Retrieved Planet SED")
            ax.legend(loc="best")
            fig.tight_layout()
            sed_plot_path = plots_dir / "retrieved_sed.png"
            fig.savefig(sed_plot_path, dpi=180, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved retrieved SED plot to: {sed_plot_path}")
        except Exception as exc:
            print(f"Could not generate retrieved SED plot: {exc}")

        param_setup = {}
        for name, par in params.items():
            init_value = par.value
            if hasattr(par, "init_value") and par.init_value is not None:
                init_value = par.init_value
            param_setup[name] = {
                "initial_value": init_value,
                "min": par.min,
                "max": par.max,
                "vary": bool(par.vary),
            }

        final_parameter_summary = {
            "temp": {"value": temp, "err_low": temp_err_low, "err_high": temp_err_high},
            "radius": {"value": radius, "err_low": radius_err_low, "err_high": radius_err_high},
            "semi_major_axis": {
                "value": semi_major_axis,
                "err_low": semi_major_axis_err_low,
                "err_high": semi_major_axis_err_high,
            },
            "eccentricity": {"value": eccentricity, "err_low": eccentricity_err_low, "err_high": eccentricity_err_high},
            "inclination": {"value": inclination, "err_low": inclination_err_low, "err_high": inclination_err_high},
            "raan": {"value": raan, "err_low": raan_err_low, "err_high": raan_err_high},
            "argument_of_periapsis": {
                "value": argument_of_periapsis,
                "err_low": argument_of_periapsis_err_low,
                "err_high": argument_of_periapsis_err_high,
            },
            "true_anomaly": {"value": true_anomaly, "err_low": true_anomaly_err_low, "err_high": true_anomaly_err_high},
            "mass": {"value": mass, "err_low": mass_err, "err_high": mass_err},
        }

        np.save(arrays_dir / "times.npy", np.asarray(times))
        np.save(arrays_dir / "wavelengths.npy", np.asarray(wavelengths))
        np.save(arrays_dir / "wavelength_bin_widths.npy", np.asarray(wavelength_bin_widths))
        np.save(arrays_dir / "data_flattened.npy", np.asarray(data_in))
        np.save(arrays_dir / "initial_walkers_p0.npy", np.asarray(p0))
        np.save(arrays_dir / "retrieved_flux.npy", np.asarray(fluxes))
        np.save(arrays_dir / "retrieved_flux_err.npy", np.asarray(flux_err))
        if sigma_in is not None:
            np.save(arrays_dir / "sigma_flattened.npy", np.asarray(sigma_in))
        if cov_out is not None:
            np.save(arrays_dir / "covariance.npy", np.asarray(cov_out))
        chain_raw = getattr(out, "chain", None)
        if chain_raw is not None:
            np.save(arrays_dir / "chain_raw.npy", np.asarray(chain_raw))
        if chain_normalized is not None:
            np.save(arrays_dir / "chain_normalized.npy", np.asarray(chain_normalized))
        if hasattr(out, "lnprob"):
            np.save(arrays_dir / "lnprob.npy", np.asarray(out.lnprob))
        if lnprob_flat is not None:
            np.save(arrays_dir / "lnprob_flat.npy", np.asarray(lnprob_flat))
        if best_data_flat is not None:
            np.save(arrays_dir / "best_data_flat.npy", np.asarray(best_data_flat))
        if best_model_flat is not None:
            np.save(arrays_dir / "best_model_flat.npy", np.asarray(best_model_flat))
        if best_residual_flat is not None:
            np.save(arrays_dir / "best_residual_flat.npy", np.asarray(best_residual_flat))

        if posterior_df is not None:
            try:
                posterior_df.describe(percentiles=[0.16, 0.5, 0.84]).to_csv(tables_dir / "posterior_describe.csv")
            except Exception as exc:
                print(f"Could not save posterior summary table: {exc}")

        run_finished_utc = datetime.utcnow()
        repro_metadata = {
            "run_id": plot_run_id,
            "started_utc": run_started_utc.isoformat() + "Z",
            "finished_utc": run_finished_utc.isoformat() + "Z",
            "duration_seconds": (run_finished_utc - run_started_utc).total_seconds(),
            "module": self.__class__.__name__,
            "module_file": str(Path(__file__).resolve()),
            "python_version": sys.version,
            "platform": platform.platform(),
            "numpy_version": np.__version__,
            "torch_version": torch.__version__,
            "resource_names": {
                "config_in": self.n_config_in,
                "data_in": self.n_data_in,
                "transformation_in": self.n_transformation_in,
                "template_in": self.n_template_in,
                "planet_params_in": self.n_planet_params_in,
                "planet_params_out": self.n_planet_params_out,
            },
            "resource_snapshots": {
                "config": _snapshot_resource(r_config_in),
                "data": _snapshot_resource(data_resource),
                "transformation": _snapshot_resource(r_transformation_in),
                "planet_params_in": _snapshot_resource(planet_params_in),
            },
            "star_context": {
                "distance_m": r_config_in.scene.star.distance,
                "mass_kg": r_config_in.scene.star.mass,
            },
            "likelihood": {
                "mode": "poisson" if use_poisson_likelihood else ("gaussian_with_sigma" if sigma_in is not None else "gaussian_unit_sigma"),
                "sigma_present": bool(sigma_in is not None),
            },
            "sampling_config": {
                "method": "lmfit.emcee",
                "seed": mcmc_seed,
                "nwalkers": nwalkers,
                "ndim": ndim,
                "varying_parameters": vary_names,
                "thin": mcmc_thin,
                "burn": mcmc_burn,
                "steps": mcmc_steps,
                "requested_workers": requested_workers,
                "effective_workers": worker_count,
                "cuda_active": bool(cuda_active),
            },
            "parameter_setup": param_setup,
            "diagnostics": {
                "success": bool(getattr(out, "success", False)),
                "message": str(getattr(out, "message", "")),
                "nfev": int(getattr(out, "nfev", -1)) if getattr(out, "nfev", None) is not None else None,
                "residual_evaluations": int(eval_counter),
                "chisqr": getattr(out, "chisqr", None),
                "acceptance_fraction_mean": acceptance_fraction_mean,
                "split_rhat": split_rhat_by_param,
                "autocorr_tau": autocorr_tau_by_param,
                "ess": ess_by_param,
            },
            "best_sample": best_sample_info,
            "final_parameters": final_parameter_summary,
            "units": {
                "wavelength": "m",
                "distance": "m",
                "mass": "kg",
                "angles": "rad",
                "temperature": "K",
                "radius": "m",
                "semi_major_axis": "m",
            },
        }
        _write_json(run_dir / "reproducibility_metadata.json", repro_metadata)
        _write_json(run_dir / "final_parameter_summary.json", final_parameter_summary)

        readme_text = (
            "MCMCBB OM run artifacts\n"
            "=======================\n\n"
            "This folder contains enough saved outputs to inspect posterior distributions,\n"
            "recreate plots, and analyze the fitted result without rerunning MCMC.\n\n"
            "Key files:\n"
            "- reproducibility_metadata.json: full run settings, diagnostics, and input snapshots\n"
            "- final_parameter_summary.json: fitted parameters with uncertainty bounds\n"
            "- tables/posterior_samples.csv: flattened posterior samples\n"
            "- arrays/chain_raw.npy and arrays/lnprob.npy: raw emcee outputs\n"
            "- arrays/data_flattened.npy and arrays/best_*_flat.npy: fit comparison vectors\n"
            "- plots/*.png: correlation, trace, fit-check, residual, and SED plots\n"
        )
        (run_dir / "README.txt").write_text(readme_text, encoding="utf-8")
        print(f"Saved reproducibility metadata to: {run_dir / 'reproducibility_metadata.json'}")

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
            temp_err_low = temp_err_low,
            temp_err_high = temp_err_high,
            radius = radius,
            radius_err_low=radius_err_low,
            radius_err_high=radius_err_high,
            semi_major_axis=semi_major_axis,
            semi_major_axis_err_low=semi_major_axis_err_low,
            semi_major_axis_err_high=semi_major_axis_err_high,
            eccentricity=eccentricity,
            eccentricity_err_low=eccentricity_err_low,
            eccentricity_err_high=eccentricity_err_high,
            inclination=inclination,
            inclination_err_low=inclination_err_low,
            inclination_err_high=inclination_err_high,
            raan=raan,
            raan_err_low=raan_err_low,
            raan_err_high=raan_err_high,
            argument_of_periapsis=argument_of_periapsis,
            argument_of_periapsis_err_low=argument_of_periapsis_err_low,
            argument_of_periapsis_err_high=argument_of_periapsis_err_high,
            true_anomaly=true_anomaly,
            true_anomaly_err_low=true_anomaly_err_low,
            true_anomaly_err_high=true_anomaly_err_high,
            mass=mass,
            mass_err_low=mass_err,
            mass_err_high=mass_err,

        )
        r_planet_params_out.params.append(planet_params)
        print("Fitted Planet Params:")
        print(f"Inclination {inclination:.6g} -{inclination_err_low:.3g}/+{inclination_err_high:.3g}")
        print(
            f"Argument of Periapsis {argument_of_periapsis:.6g} "
            f"-{argument_of_periapsis_err_low:.3g}/+{argument_of_periapsis_err_high:.3g}"
        )
        print(f"Eccentricity {eccentricity:.6g} -{eccentricity_err_low:.3g}/+{eccentricity_err_high:.3g}")
        print(f"RAAN {raan:.6g} -{raan_err_low:.3g}/+{raan_err_high:.3g}")
        print(f"SMA {semi_major_axis:.6g} -{semi_major_axis_err_low:.3g}/+{semi_major_axis_err_high:.3g}")
        print(f"Temperature {temp:.6g} -{temp_err_low:.3g}/+{temp_err_high:.3g}")
        print(f"Radius {radius:.6g} -{radius_err_low:.3g}/+{radius_err_high:.3g}")
        print(f"Run outputs saved under: {run_dir}")
        print('Done')
        return r_planet_params_out
