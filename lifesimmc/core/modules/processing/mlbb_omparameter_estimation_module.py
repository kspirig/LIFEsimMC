import numpy as np
import torch
from lmfit import minimize, Parameters
from astropy import units as u

from lifesimmc.core.modules.base_module import BaseModule
from lifesimmc.core.resources.base_resource import BaseResource
from lifesimmc.core.resources.planet_params_resource import PlanetParamsResource, PlanetParams
from astropy import constants as const


class MLBBOMParameterEstimationModule(BaseModule):
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

    def _get_analytical_initial_guess(self, data, template_data, grid_coordinates):
        # Normalize template data with their variance along axis 2
        # template_data = template_data / torch.var(template_data, axis=2, keepdim=True) ** 0.5

        # Calculate matrix C according to equation B.2
        data_variance = torch.var(data, axis=0)
        sum = torch.sum(torch.einsum('ij, ijkl->ijkl', data, template_data), axis=0)
        vector_c = torch.einsum('ijk, i->ijk', sum, 1 / data_variance)

        # Calculate matrix B according to equation B.3
        sum = torch.sum(template_data ** 2, axis=0)
        vector_b = torch.nan_to_num(torch.einsum('ijk, i->ijk', sum, 1 / data_variance), 1)

        # Create diagonal matrix B
        b_shape = vector_b.shape
        eye = torch.eye(b_shape[0], device=self.device).unsqueeze(-1).unsqueeze(-1)
        x_diag = vector_b.unsqueeze(1)
        matrix_b = eye * x_diag

        # Calculate the optimum flux according to equation B.6 and set positivity constraint
        b_perm = matrix_b.permute(2, 3, 0, 1)
        b_inv = torch.linalg.inv(b_perm).permute(2, 3, 0, 1)

        optimum_flux = torch.einsum('ijkl, jkl->ikl', b_inv, vector_c)

        # Calculate the cost function according to equation B.8
        optimum_flux = torch.where(optimum_flux >= 0, optimum_flux, 0)
        cost_function = optimum_flux * vector_c
        cost_function = torch.sum(torch.nan_to_num(cost_function, 0), axis=0)

        # plt.imshow(cost_function.cpu().numpy(), cmap='magma')
        # plt.colorbar()
        # plt.show()

        # Get the optimum flux at the position of the maximum of the cost function
        flat_idx = cost_function.argmax()  # scalar
        row = flat_idx // cost_function.shape[1]
        col = flat_idx % cost_function.shape[1]
        optimum_flux_at_maximum = optimum_flux[:, row.item(), col.item()].cpu().numpy()  # TODO: fix this scaling

        # plt.plot(optimum_flux_at_maximum)
        # plt.show()

        # Get the coordinates of the maximum
        x_coord = grid_coordinates[0][row.item(), col.item()].cpu().numpy()
        y_coord = grid_coordinates[1][row.item(), col.item()].cpu().numpy()

        return optimum_flux_at_maximum, x_coord, y_coord

    def apply(self, resources: list[BaseResource]) -> PlanetParamsResource:
        print('Performing numerical MLE with Blackbody and Orbital Motion...')

        r_config_in = self.get_resource_from_name(self.n_config_in)
        r_templates_in = self.get_resource_from_name(self.n_template_in)
        r_transformation_in = self.get_resource_from_name(
            self.n_transformation_in) if self.n_transformation_in else None
        transf = r_transformation_in.transformation if r_transformation_in else lambda x: x
        planet_params_in = self.get_resource_from_name(self.n_planet_params_in) if self.n_planet_params_in else None

        print("Planet params in: ", planet_params_in.params if planet_params_in is not None else None)

        distance = r_config_in.scene.star.distance  # float in meters
        print("Distance: ", distance)



        times = r_config_in.phringe.get_time_steps().cpu().numpy()
        wavelengths = r_config_in.phringe.get_wavelength_bin_centers().cpu().numpy()
        wavelength_bin_widths = r_config_in.phringe.get_wavelength_bin_widths().cpu().numpy()
        print("Wavelength min/max:", np.min(wavelengths), np.max(wavelengths))
        print("Wavelength width min/max:", np.min(wavelength_bin_widths), np.max(wavelength_bin_widths))
        data_in = self.get_resource_from_name(self.n_data_in).get_data()
        template_data = r_templates_in.get_data()
        grid_coordinates = r_templates_in.grid_coordinates

        # Flatten data along differential outputs and times axes
        data_in = data_in.permute(0, 2, 1)
        data_in = data_in.reshape((-1,) + data_in.shape[2:])
        template_data = template_data.permute(0, 2, 1, 3, 4)
        template_data = template_data.reshape((-1,) + template_data.shape[2:])

        # Set up parameters and initial conditions
        if planet_params_in is None:
            #ToDO: Change the analytical aswell
            flux_init, posx_init, posy_init = self._get_analytical_initial_guess(
                data_in,
                template_data,
                grid_coordinates
            )
        # If planet_params_in is provided, use its values as initial conditions
        else:
            # TODO: implement for multiple planets

            print(r_config_in.scene)
            if len(r_config_in.scene.planets) > 0:
                radius_init = r_config_in.scene.planets[0].radius
                temp_init = r_config_in.scene.planets[0].temperature
                semi_major_axis_init = r_config_in.scene.planets[0].semi_major_axis
                eccentricity_init = r_config_in.scene.planets[0].eccentricity
                planet_mass_init = r_config_in.scene.planets[0].mass
                inclination_init = r_config_in.scene.planets[0].inclination
                raan_init = r_config_in.scene.planets[0].raan
                argument_of_periapsis_init = r_config_in.planets[0].argument_of_periapsis
                true_anomaly_init = r_config_in.planets[0].true_anomaly

            else:
                radius_init = 6 * 1e6  # 1 Earth radius in meters
                temp_init = 300.0      # Kelvin
                semi_major_axis_init = 1 * const.au.value
                eccentricity_init  = 0.0
                planet_mass_init = 1 * const.M_earth.value
                inclination_init = 0
                raan_init = 0
                argument_of_periapsis_init = 0
                true_anomaly_init = 0



            posx_init = planet_params_in.params[0].pos_x
            posy_init = planet_params_in.params[0].pos_y

        data_in = data_in.cpu().numpy()
        hfov_max = r_config_in.phringe.get_field_of_view()[-1].cpu().numpy() / 2  # TODO: /14 Check this

        params = Parameters()

        #ToDO: Add all needed parameters

        params.add('pos_x', value=posx_init, min=-hfov_max, max=hfov_max)
        params.add('pos_y', value=posy_init, min=-hfov_max, max=hfov_max)
        params.add("temp",value = temp_init,min = 50,max = 4000)
        params.add("radius",value = radius_init,min = 1e5,max = 2*1e8)

        params.add('semi_major_axis',value = semi_major_axis_init,min = 0.01 * const.au.value,max = 20*const.au.value)
        params.add('eccentricity',value = eccentricity_init,min = 0,max = 0.99)
        params.add('planet_mass',value = planet_mass_init,min = 1e-3 * const.M_earth.value,max = 1000*const.M_earth.value)

        params.add('inclination',value = inclination_init, min = 0 ,max = 180)
        params.add('raan',value = raan_init,min = 0,max = 360)
        params.add('argument_of_periapsis',value = argument_of_periapsis_init,min = 0,max = 360)

        params.add('true_anomaly',value = true_anomaly_init,min = 0,max = 360)

        _bb_debug_printed = False

        def Blackbody(radius,temp,wavelengths):
            lam_m = np.asarray(wavelengths, dtype=float)
            x = (const.h.value * const.c.value) / (lam_m * const.k_B.value * temp)
            denom = np.expm1(x)
            photon_flux_per_m = (2.0 * np.pi * const.c.value / lam_m ** 4) * (radius / distance) ** 2 / denom
            return photon_flux_per_m


        # Perform MLE
        def residual_data(params, target):
            posx = params['pos_x'].value
            posy = params['pos_y'].value


            #ToDo: Inlcude fitting Orbital Motion

            flux = Blackbody(radius=params['radius'].value, temp=params['temp'].value, wavelengths=wavelengths)

            model = r_config_in.phringe.get_model_counts(
                kernels=True,
                spectral_energy_distribution=flux,
                # x_position=3e-7,  # in radians
                # y_position=0e-7, # in radians
                semi_major_axis = params['semi_major_axis'].value,
                eccentricity = params['eccentricity'].value,
                inclination = params['inclination'].value,
                raan = params['raan'].value,
                argument_of_periapsis = params['argument_of_periapsis'].value,

                true_anomaly = params['true_anomaly'].value,
                host_star_distance=r_config_in.scene.star.distance,

                host_star_mass= r_config_in.scene.star.mass,

                planet_mass=params['planet_mass'].value,
                # radius=1 * u.Rearth,
                # temperature=254 * u.K,
                # input_spectrum=None,
            )


            model = transf(model)
            model = np.transpose(model, (0, 2, 1))
            model = model.reshape(data_in.shape)

            residual = model - target
            if not np.all(np.isfinite(residual)):
                print("Non-finite residual detected")
            return residual

        out = minimize(residual_data, params, args=(data_in,), method='leastsq')
        print("success:", out.success)
        print("message:", out.message)
        print("nfev:", out.nfev)
        print("chisqr:", out.chisqr)
        cov_out = out.covar

        fluxes = Blackbody(radius=out.params['radius'].value, temp=out.params['temp'].value, wavelengths=wavelengths)

        def _get_param_value(name, default=np.nan):
            return out.params[name].value if name in out.params else default

        def _get_param_err(name):
            if name not in out.params:
                return np.nan
            stderr = out.params[name].stderr
            if stderr is not None and np.isfinite(stderr):
                return float(stderr)
            if cov_out is not None and out.var_names is not None and name in out.var_names:
                try:
                    idx = out.var_names.index(name)
                    return float(np.sqrt(np.diag(cov_out))[idx])
                except (TypeError, ValueError, IndexError):
                    return np.nan
            return np.nan

        posx = _get_param_value('pos_x')
        posy = _get_param_value('pos_y')
        temp = _get_param_value('temp')
        radius = _get_param_value('radius')
        semi_major_axis = _get_param_value('semi_major_axis')
        eccentricity = _get_param_value('eccentricity')
        inclination = _get_param_value('inclination')
        raan = _get_param_value('raan')
        argument_of_periapsis = _get_param_value('argument_of_periapsis')
        true_anomaly = _get_param_value('true_anomaly')
        mass = _get_param_value('planet_mass', _get_param_value('mass'))

        def dBdT(radius,temp,wavelengths):
            lam_m = np.asarray(wavelengths, dtype=float)
            return (4 * np.pi * const.c.value**2 * const.h.value) / ( lam_m ** 5 * const.k_B.value * temp) / (radius * distance)**2 * np.exp(const.h.value*const.c.value/(const.k_B.value*temp*lam_m)) / (np.expm1(const.h.value * const.c.value / (lam_m*const.k_B.value*temp))**2)

        def dBdR(radius,temp,wavelengths):
            lam_m = np.asarray(wavelengths, dtype=float)
            x = (const.h.value * const.c.value) / (lam_m * const.k_B.value * temp)
            denom = np.expm1(x)
            photon_flux_per_m = (2.0 * np.pi * const.c.value / lam_m ** 4) * 2*(radius / (distance**2)) / denom
            return photon_flux_per_m

        posx_err = _get_param_err('pos_x')
        posy_err = _get_param_err('pos_y')
        temp_err = _get_param_err('temp')
        radius_err = _get_param_err('radius')
        semi_major_axis_err = _get_param_err('semi_major_axis')
        eccentricity_err = _get_param_err('eccentricity')
        inclination_err = _get_param_err('inclination')
        raan_err = _get_param_err('raan')
        argument_of_periapsis_err = _get_param_err('argument_of_periapsis')
        true_anomaly_err = _get_param_err('true_anomaly')
        mass_err = _get_param_err('planet_mass')
        if np.isnan(mass_err):
            mass_err = _get_param_err('mass')

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
            pos_x=posx,
            pos_y=posy,
            pos_x_err_low=posx_err,
            pos_x_err_high=posx_err,
            pos_y_err_low=posy_err,
            pos_y_err_high=posy_err,
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
        print("Fitted Temperature", out.params['temp'].value)
        print("Fitted Radius", out.params['radius'].value)
        print('Done')
        return r_planet_params_out
