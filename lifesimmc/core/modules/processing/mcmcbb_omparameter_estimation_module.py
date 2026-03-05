import numpy as np
import torch
from lmfit import minimize, Parameters
from pathlib import Path

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
        data_in = self.get_resource_from_name(self.n_data_in).get_data()

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



        data_in = data_in.cpu().numpy()

        params = Parameters()




        params.add("temp",value = temp_init,min = 50,max = 4000)
        params.add("radius",value = radius_init,min = 1e5,max = 2*1e8)

        params.add('semi_major_axis',value = semi_major_axis_init,min = 0.01 * const.au.value,max = 20*const.au.value)
        params.add('eccentricity',value = eccentricity_init,min = 0,max = 0.5)

        params.add('inclination',value =inclination_init, min = 0 ,max = np.pi)
        params.add('raan',value = raan_init,min = 0,max = 2.0 * np.pi)
        params.add('argument_of_periapsis',value = argument_of_periapsis_init,min = 0,max = 2.0 * np.pi)

        params.add('true_anomaly',value = true_anomaly_init,min = 0,max = 2.0 * np.pi)

        vary_names = [name for name, par in params.items() if par.vary]
        varying_params = [par for _, par in params.items() if par.vary]
        ndim = len(varying_params)
        if ndim == 0:
            raise ValueError("No varying parameters configured for emcee.")

        # Build a linearly independent initial walker cloud.
        rng = np.random.default_rng(12345)
        nwalkers = max(40, 2 * ndim + 8)
        p0 = np.zeros((nwalkers, ndim), dtype=float)
        for j, par in enumerate(varying_params):
            lo = par.min if par.min is not None else -np.inf
            hi = par.max if par.max is not None else np.inf
            center = float(par.value)
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                p0[:, j] = rng.uniform(lo, hi, size=nwalkers)
                eps = 1e-12 * max(1.0, abs(lo), abs(hi))
                p0[:, j] = np.clip(p0[:, j], lo + eps, hi - eps)
            else:
                scale = max(1e-6, abs(center) * 1e-2)
                p0[:, j] = center + rng.normal(0.0, scale, size=nwalkers)

        # Final perturbation to avoid accidental duplicate walker rows.
        p0 += rng.normal(0.0, 1e-10, size=p0.shape)



        orbital_failure_printed = False
        eval_counter = 0

        # Perform MLE
        def residual_data(params, target):
            nonlocal orbital_failure_printed, eval_counter
            eval_counter += 1

            flux = np.pi * get_blackbody_spectrum_standard_units(temperature=params['temp'].value,
                                                                 wavelengths=wavelengths) * (
                               params['radius'].value / distance) ** 2
            flux = flux.detach().cpu().numpy()

            if eval_counter % 20 == 0:
                print(
                    f"[Eval {eval_counter}] "
                    f"T={params['temp'].value:.6g}, R={params['radius'].value:.6g}, "
                    f"a={params['semi_major_axis'].value:.6g}, e={params['eccentricity'].value:.6g}, "
                    f"inc={params['inclination'].value:.6g}, raan={params['raan'].value:.6g}, "
                    f"argp={params['argument_of_periapsis'].value:.6g}, nu={params['true_anomaly'].value:.6g}"
                )

            try:
                model = r_config_in.phringe.get_model_counts(
                    kernels=True,
                    spectral_energy_distribution=flux,
                    # x_position=3e-7,  # in radians
                    # y_position=0e-7, # in radians
                    semi_major_axis = params['semi_major_axis'].value,
                    eccentricity = params['eccentricity'].value,
                    inclination = params['inclination'].value ,
                    raan = params['raan'].value,
                    argument_of_periapsis = params['argument_of_periapsis'].value,
                    true_anomaly = params['true_anomaly'].value ,

                    host_star_distance=r_config_in.scene.star.distance,
                    host_star_mass=r_config_in.scene.star.mass,
                    planet_mass=planet_mass_fixed,
                    # radius=1 * u.Rearth,
                    # temperature=254 * u.K,
                    # input_spectrum=None,
                )
            except (RuntimeError, ValueError, FloatingPointError) as exc:
                if not orbital_failure_printed:
                    print(f"Orbital model failed for trial parameters; continuing with penalty residual. {exc}")
                    orbital_failure_printed = True
                return np.full_like(target, 1e30, dtype=float)


            model = transf(model)
            model = np.transpose(model, (0, 2, 1))
            model = model.reshape(data_in.shape)


            residual = model - target
            if not np.all(np.isfinite(residual)):
                print("Non-finite residual detected")
            return residual

        out = minimize(
            residual_data,
            params,
            args=(data_in,),
            method='emcee',
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
                            print(f"  {name}: {val:.4f}")
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
                            print(f"  {name}: tau={tau:.2f}, ESS~{ess:.1f}, steps/(50*tau)={nsteps_c / max(50.0 * tau, 1e-30):.3f}")
                    except Exception as exc:
                        print(f"autocorr/ESS skipped: {exc}")
                else:
                    print("chain diagnostics skipped: unexpected chain dimensionality.")
            else:
                print("chain diagnostics skipped: no chain available on result.")
        except Exception as exc:
            print(f"chain diagnostics failed: {exc}")

        cov_out = getattr(out, "covar", None)
        if hasattr(out, "flatchain"):
            try:
                import matplotlib.pyplot as plt
                from pandas.plotting import scatter_matrix

                flatchain = out.flatchain
                cols = [c for c in flatchain.columns if c in vary_names]
                if len(cols) >= 2 and len(flatchain) > 1:
                    fig_axes = scatter_matrix(
                        flatchain[cols],
                        diagonal="kde",
                        alpha=0.2,
                        figsize=(2.4 * len(cols), 2.4 * len(cols)),
                    )
                    # scatter_matrix returns an array of axes; get the figure from the first one.
                    fig = fig_axes[0, 0].figure
                    fig.suptitle("MCMC Parameter Correlations", y=1.0)
                    fig.tight_layout()
                    plot_path = Path.cwd() / f"mcmcbb_om_correlations_{self.n_planet_params_out}.png"
                    fig.savefig(plot_path, dpi=180, bbox_inches="tight")
                    plt.close(fig)
                    print(f"Saved MCMC correlation plot to: {plot_path}")
            except Exception as exc:
                print(f"Could not generate MCMC correlation plot: {exc}")

        fluxes = np.pi * get_blackbody_spectrum_standard_units(temperature=out.params['temp'].value,
                                                             wavelengths=wavelengths) * (
                       out.params['radius'].value / distance) ** 2
        fluxes = fluxes.detach().cpu().numpy()

        def _get_param_value(name, default=np.nan):
            return out.params[name].value if name in out.params else default

        def _get_param_err(name):
            if name not in out.params:
                return np.nan
            stderr = out.params[name].stderr
            if stderr is not None and np.isfinite(stderr):
                return float(stderr)
            if hasattr(out, "flatchain") and name in out.flatchain:
                try:
                    return float(np.std(out.flatchain[name].to_numpy(), ddof=1))
                except Exception:
                    pass
            if cov_out is not None and out.var_names is not None and name in out.var_names:
                try:
                    idx = out.var_names.index(name)
                    return float(np.sqrt(np.diag(cov_out))[idx])
                except (TypeError, ValueError, IndexError):
                    return np.nan
            return np.nan


        temp = _get_param_value('temp')
        radius = _get_param_value('radius')
        semi_major_axis = _get_param_value('semi_major_axis')
        eccentricity = _get_param_value('eccentricity')
        inclination = _get_param_value('inclination')
        raan = _get_param_value('raan')
        argument_of_periapsis = _get_param_value('argument_of_periapsis')
        true_anomaly = _get_param_value('true_anomaly')
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

        temp_err = _get_param_err('temp')
        radius_err = _get_param_err('radius')
        semi_major_axis_err = _get_param_err('semi_major_axis')
        eccentricity_err = _get_param_err('eccentricity')
        inclination_err = _get_param_err('inclination')
        raan_err = _get_param_err('raan')
        argument_of_periapsis_err = _get_param_err('argument_of_periapsis')
        true_anomaly_err = _get_param_err('true_anomaly')
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
            mass=mass,
            mass_err_low=mass_err,
            mass_err_high=mass_err,

        )
        r_planet_params_out.params.append(planet_params)
        print("Fitted Planet Params:")
        print("Inclination ",out.params["inclination"].value)
        print("Argument of Periapsis ", out.params["argument_of_periapsis"].value)
        print("Eccentricity ",out.params["eccentricity"].value)
        print("RAAN ",out.params["raan"].value)
        print("SMA ",out.params["semi_major_axis"].value)

        print("Temperature ", out.params['temp'].value)
        print("Radius ", out.params['radius'].value)
        print('Done')
        return r_planet_params_out
