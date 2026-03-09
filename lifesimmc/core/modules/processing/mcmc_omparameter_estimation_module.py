import numpy as np
import torch
from lmfit import minimize, Parameters
from multiprocessing.dummy import Pool as ThreadPool

from lifesimmc.core.modules.base_module import BaseModule
from lifesimmc.core.resources.base_resource import BaseResource
from lifesimmc.core.resources.planet_params_resource import PlanetParamsResource, PlanetParams
from astropy import constants as const
from phringe.util.spectrum import get_blackbody_spectrum_standard_units
import astropy.units as u

_TORCH_THREAD_BUDGET = 8
_TORCH_THREADS_SINGLE_WORKER = 8

class MLOMParameterEstimationModule(BaseModule):
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
        raan_init = np.pi
        argument_of_periapsis_init = np.pi
        true_anomaly_init = np.pi




        data_in = data_in.cpu().numpy()
        data_shape = data_in.shape

        params = Parameters()

        params.add("log_temp", value=np.log10(temp_init), min=np.log10(50), max=np.log10(4000))
        params.add("log_radius", value=np.log10(radius_init), min=np.log10(1e5), max=np.log10(2 * 1e8))
        params.add(
            "log_semi_major_axis",
            value=np.log10(semi_major_axis_init),
            min=np.log10(0.01 * const.au.value),
            max=np.log10(a_max),
        )

        params.add('eccentricity',value = eccentricity_init,min = 0,max = 0.5)

        params.add('inclination',value =inclination_init, min = 0 ,max = np.pi)
        params.add('raan',value = raan_init,min = 0,max = 2.0 * np.pi)
        params.add('argument_of_periapsis',value = argument_of_periapsis_init,min = 0,max = 2.0 * np.pi)

        params.add('true_anomaly',value = true_anomaly_init,min = 0,max = 2.0 * np.pi)

        params_names = np.array(["log_temp","log_radius","log_semi_major_axis","eccentricity","inclination","raan","argument_of_periapsis","true_anomaly"])

        mcmc_seed = 12345
        rng = np.random.default_rng(mcmc_seed)
        nwalkers = 40
        ndim = len(params_names)
        p0 = np.zeros((nwalkers, ndim), dtype=float)

        for k in range(0,nwalkers):
            for j in range(0,ndim):
                p0[k, j] = rng.uniform(params[params_names[j]].min, params[params_names[j]].max)
            # print("p0 ", k, " = ", p0[k])

        # Perform MLE
        eval_counter = 0
        def residual_data(params, target):
            nonlocal eval_counter

            flux = np.pi * get_blackbody_spectrum_standard_units(temperature=10**params['log_temp'].value, wavelengths=wavelengths)  * (10**params['log_radius'].value / distance) ** 2
            flux = flux.detach().cpu().numpy()
            eval_counter += 1
            if eval_counter % 20 == 0:
                print(
                    f"[Eval {eval_counter}] "
                    f"T={10**params['log_temp'].value:.6g}, R={10**params['log_radius'].value:.6g}, "
                    f"a={10**params['log_semi_major_axis'].value:.6g}, e={params['eccentricity'].value:.6g}, "
                    f"inc={params['inclination'].value:.6g}, raan={params['raan'].value:.6g}, "
                    f"argp={params['argument_of_periapsis'].value:.6g}, nu={params['true_anomaly'].value:.6g}"
                )

            try:
                model = r_config_in.phringe.get_model_counts(
                    kernels=True,
                    spectral_energy_distribution=flux,
                    semi_major_axis = 10**params['log_semi_major_axis'].value,
                    eccentricity = params['eccentricity'].value,
                    inclination = params['inclination'].value ,
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

        requested_workers = int(self.n_cores) if self.n_cores is not None else 1
        if requested_workers < 1:
            requested_workers = 1

        max_useful_workers = max(1, nwalkers // 2)
        worker_count = min(requested_workers, max_useful_workers)
        if requested_workers > max_useful_workers:
            print(
                f"MCMC parallelism: requested n_cores={requested_workers}, "
                f"capped to {worker_count} (nwalkers={nwalkers})."
            )

        # Avoid nested parallelism: scale torch intra-op threads with emcee workers.
        if worker_count > 1:
            torch_threads = max(1, _TORCH_THREAD_BUDGET // worker_count)
            interop_threads = 1
        else:
            torch_threads = _TORCH_THREADS_SINGLE_WORKER
            interop_threads = 1
        try:
            torch.set_num_threads(torch_threads)
        except RuntimeError as exc:
            print(f"Could not set torch intra-op threads to {torch_threads}: {exc}")
        try:
            torch.set_num_interop_threads(interop_threads)
        except RuntimeError:
            # set_num_interop_threads can only be called once in a process.
            pass
        print(
            f"Torch threading: intra-op={torch.get_num_threads()}, "
            f"interop={interop_threads}, worker_count={worker_count}"
        )

        if worker_count > 1:
            print(f"CPU MCMC parallelism: running {worker_count} thread workers.")
            with ThreadPool(processes=worker_count) as thread_pool:
                out = minimize(
                    residual_data,
                    params,
                    args=(data_in,),
                    method='emcee',
                    float_behavior='posterior',
                    nwalkers=nwalkers,
                    burn=mcmcsteps // 5,
                    steps=mcmcsteps,
                    # thin=10,
                    pos=p0,
                    progress=True,
                    workers=thread_pool,
                )
        else:
            print(f"MCMC parallelism: disabled (effective workers={worker_count}).")
            out = minimize(
                residual_data,
                params,
                args=(data_in,),
                method='emcee',
                float_behavior='posterior',
                nwalkers=nwalkers,
                burn=mcmcsteps // 5,
                steps=mcmcsteps,
                # thin=10,
                pos=p0,
                progress=True,
            )

        print("success:", out.success)
        print("message:", out.message)
        print("nfev:", out.nfev)
        print("residual evaluations:", eval_counter)
        print("chisqr:", out.chisqr)
        cov_out = out.covar
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
        inclination = _get_param_value('inclination')
        raan = _get_param_value('raan')
        argument_of_periapsis = _get_param_value('argument_of_periapsis')
        true_anomaly = _get_param_value('true_anomaly')
        eccentricity_err = _get_param_err('eccentricity')
        inclination_err = _get_param_err('inclination')
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
        print("Inclination ",out.params["inclination"].value)
        print("Argument of Periapsis ", out.params["argument_of_periapsis"].value)
        print("Eccentricity ",out.params["eccentricity"].value)
        print("RAAN ",out.params["raan"].value)
        print("SMA ",10**out.params["log_semi_major_axis"].value)

        print("Temperature ", 10**out.params['log_temp'].value)
        print("Radius ", 10**out.params['log_radius'].value)
        print("True Anomaly",out.params['true_anomaly'].value)
        print('Done')
        return r_planet_params_out
