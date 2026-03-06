import numpy as np
import torch
from lmfit import minimize, Parameters
from pathlib import Path
from typing import Optional
from datetime import datetime
from uuid import uuid4

from lifesimmc.core.modules.base_module import BaseModule
from lifesimmc.core.resources.base_resource import BaseResource
from lifesimmc.core.resources.planet_params_resource import PlanetParamsResource, PlanetParams
from astropy import constants as const
from phringe.util.spectrum import get_blackbody_spectrum_standard_units

import astropy.units as u

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
            bounds: bool = False
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

    def apply(self, resources: list[BaseResource]) -> PlanetParamsResource:
        print('Performing numerical MLE with Blackbody and Orbital Motion...')
        plot_run_id = f"{datetime.utcnow().strftime('%Y%m%dT%H%M%S%fZ')}_{uuid4().hex[:8]}"

        r_config_in = self.get_resource_from_name(self.n_config_in)
        r_transformation_in = self.get_resource_from_name(
            self.n_transformation_in) if self.n_transformation_in else None
        transf = r_transformation_in.transformation if r_transformation_in else lambda x: x
        planet_params_in = self.get_resource_from_name(self.n_planet_params_in) if self.n_planet_params_in else None


        distance = r_config_in.scene.star.distance  # float in meters



        times = r_config_in.phringe.get_time_steps().cpu().numpy()
        wavelengths = r_config_in.phringe.get_wavelength_bin_centers().cpu().numpy()
        wavelength_bin_widths = r_config_in.phringe.get_wavelength_bin_widths().cpu().numpy()
        print("Wavelength min/max:", np.min(wavelengths), np.max(wavelengths))
        print("Wavelength width min/max:", np.min(wavelength_bin_widths), np.max(wavelength_bin_widths))
        data_resource = self.get_resource_from_name(self.n_data_in)
        data_in = data_resource.get_data()

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




        params.add("temp",value = temp_init,min = 50,max = 4000)
        params.add("radius",value = radius_init,min = 1e5,max = 2*1e8)

        params.add('semi_major_axis',value = semi_major_axis_init,min = 0.01 * const.au.value,max = 20*const.au.value)
        params.add('eccentricity',value = eccentricity_init,min = 0,max = 0.5)

        params.add('inclination',value = inclination_init, min = 0 ,max = np.pi)
        # Sample unconstrained angle variables and wrap to [0, 2pi) in the model.
        params.add('raan_raw', value=raan_init)
        params.add('argument_of_periapsis_raw', value=argument_of_periapsis_init)
        params.add('true_anomaly_raw', value=true_anomaly_init)

        vary_names = [name for name, par in params.items() if par.vary]
        display_name_map = {
            "raan_raw": "raan",
            "argument_of_periapsis_raw": "argument_of_periapsis",
            "true_anomaly_raw": "true_anomaly",
        }
        varying_params = [par for _, par in params.items() if par.vary]
        ndim = len(varying_params)
        if ndim == 0:
            raise ValueError("No varying parameters configured for emcee.")

        # Initialize walkers near the initial values for faster convergence.
        rng = np.random.default_rng(12345)
        nwalkers = max(40, 2 * ndim + 8)
        p0 = np.zeros((nwalkers, ndim), dtype=float)
        for j, par in enumerate(varying_params):
            lo = par.min if par.min is not None else -np.inf
            hi = par.max if par.max is not None else np.inf
            center = float(par.value)
            cloud = center * (1.0 + 1e-2 * rng.normal(size=nwalkers))
            if center == 0.0:
                scale = 1e-6
                if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                    scale = max(scale, 1e-3 * (hi - lo))
                cloud = center + scale * rng.normal(size=nwalkers)
            p0[:, j] = cloud
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                eps = 1e-12 * max(1.0, abs(lo), abs(hi))
                p0[:, j] = np.clip(p0[:, j], lo + eps, hi - eps)

        # Final perturbation to avoid accidental duplicate walker rows.
        p0 += rng.normal(0.0, 1e-10, size=p0.shape)



        orbital_failure_printed = False
        eval_counter = 0

        two_pi = 2.0 * np.pi

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
            if isinstance(model, torch.Tensor):
                model = model.detach().cpu().numpy()
            else:
                model = np.asarray(model)
            model = np.transpose(model, model_transpose_axes).reshape(data_shape)
            return model, flux

        # Perform MLE/MCMC objective evaluations
        def residual_data(params, target):
            nonlocal orbital_failure_printed, eval_counter
            eval_counter += 1

            raan = _wrap_angle(params['raan_raw'].value)
            argument_of_periapsis = _wrap_angle(params['argument_of_periapsis_raw'].value)
            true_anomaly = _wrap_angle(params['true_anomaly_raw'].value)

            if eval_counter % 20 == 0:
                print(
                    f"[Eval {eval_counter}] "
                    f"T={params['temp'].value:.6g}, R={params['radius'].value:.6g}, "
                    f"a={params['semi_major_axis'].value:.6g}, e={params['eccentricity'].value:.6g}, "
                    f"inc={params['inclination'].value:.6g}, raan={raan:.6g}, "
                    f"argp={argument_of_periapsis:.6g}, nu={true_anomaly:.6g}"
                )

            try:
                model, _ = _evaluate_model_numpy(
                    temp=params['temp'].value,
                    radius=params['radius'].value,
                    semi_major_axis=params['semi_major_axis'].value,
                    eccentricity=params['eccentricity'].value,
                    inclination=params['inclination'].value,
                    raan=raan,
                    argument_of_periapsis=argument_of_periapsis,
                    true_anomaly=true_anomaly,
                )
            except (RuntimeError, ValueError, FloatingPointError) as exc:
                if not orbital_failure_printed:
                    print(f"Orbital model failed for trial parameters; continuing with penalty residual. {exc}")
                    orbital_failure_printed = True
                if use_poisson_likelihood:
                    return -1e100
                return np.full_like(target, 1e30, dtype=float)

            if use_poisson_likelihood:
                # emcee expects log-posterior for scalar objective values.
                model_safe = np.clip(model, 1e-12, None)
                nll = np.sum(model_safe - target * np.log(model_safe))
                return -float(nll)

            if sigma_in is not None:
                residual = (model - target) / sigma_in
            else:
                residual = model - target
            if not np.all(np.isfinite(residual)):
                print("Non-finite residual detected")
            return residual

        out = minimize(
            residual_data,
            params,
            args=(data_in,),
            method='emcee',
            float_behavior='posterior',
            nwalkers=nwalkers,
            burn=300,
            steps=2400,
            thin=10,
            pos=p0,
            progress=True,
        )

        print("success:", out.success)
        print("message:", getattr(out, "message", "n/a"))
        print("nfev:", getattr(out, "nfev", "n/a"))
        print("residual evaluations:", eval_counter)
        if hasattr(out, "acceptance_fraction"):
            print("acceptance fraction:", float(np.mean(out.acceptance_fraction)))
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
                    plot_path = Path.cwd() / f"mcmcbb_om_correlations_{self.n_planet_params_out}_{plot_run_id}.png"
                    fig.savefig(plot_path, dpi=180, bbox_inches="tight")
                    plt.close(fig)
                    print(f"Saved MCMC correlation plot to: {plot_path}")
            except Exception as exc:
                print(f"Could not generate MCMC correlation plot: {exc}")

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

        temp, temp_err_low, temp_err_high = _posterior_summary("temp", circular=False, fallback=out.params["temp"].value)
        radius, radius_err_low, radius_err_high = _posterior_summary("radius", circular=False, fallback=out.params["radius"].value)
        semi_major_axis, semi_major_axis_err_low, semi_major_axis_err_high = _posterior_summary(
            "semi_major_axis", circular=False, fallback=out.params["semi_major_axis"].value
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

        if posterior_df is not None and hasattr(out, "lnprob"):
            try:
                import matplotlib.pyplot as plt

                lnprob = np.asarray(out.lnprob).reshape(-1)
                if len(lnprob) == len(posterior_df):
                    i_best = int(np.argmax(lnprob))
                    best = posterior_df.iloc[i_best]
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
                    y_data = data_in.ravel()
                    y_model = model_best.ravel()
                    y_res = y_data - y_model
                    x = np.arange(y_data.size)

                    fig, axes = plt.subplots(
                        2,
                        1,
                        figsize=(12, 6),
                        gridspec_kw={"height_ratios": [3, 1]},
                        sharex=True,
                    )
                    axes[0].plot(x, y_data, lw=1.0, label="data")
                    axes[0].plot(x, y_model, lw=1.0, label="ML model")
                    axes[0].legend(loc="best")
                    axes[0].set_ylabel("counts")
                    axes[0].set_title("Maximum-Likelihood Sample Fit Check")
                    axes[1].plot(x, y_res, lw=0.8)
                    axes[1].axhline(0.0, color="k", lw=0.8, ls="--")
                    axes[1].set_ylabel("residual")
                    axes[1].set_xlabel("flattened data index")
                    fig.tight_layout()
                    sanity_plot_path = Path.cwd() / f"mcmcbb_om_fitcheck_{self.n_planet_params_out}_{plot_run_id}.png"
                    fig.savefig(sanity_plot_path, dpi=180, bbox_inches="tight")
                    plt.close(fig)
                    print(f"Saved MCMC fit sanity plot to: {sanity_plot_path}")
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
        print('Done')
        return r_planet_params_out
