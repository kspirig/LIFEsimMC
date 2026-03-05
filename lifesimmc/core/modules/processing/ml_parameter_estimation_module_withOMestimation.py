import numpy as np
import torch
from lmfit import minimize, Parameters

from lifesimmc.core.modules.base_module import BaseModule
from lifesimmc.core.resources.base_resource import BaseResource
from lifesimmc.core.resources.planet_params_resource import PlanetParamsResource, PlanetParams
from astropy import constants as const


class MLParameterEstimationModulewithOMestimation(BaseModule):
    """Class representation of a module that performs maximum likelihood estimation (MLE) of planet parameters.

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
        print('Performing numerical MLE with known orbital parameters...')

        r_config_in = self.get_resource_from_name(self.n_config_in)
        r_templates_in = self.get_resource_from_name(self.n_template_in)
        r_transformation_in = self.get_resource_from_name(
            self.n_transformation_in) if self.n_transformation_in else None
        transf = r_transformation_in.transformation if r_transformation_in else lambda x: x
        planet_params_in = self.get_resource_from_name(self.n_planet_params_in) if self.n_planet_params_in else None

        times = r_config_in.phringe.get_time_steps().cpu().numpy()
        wavelengths = r_config_in.phringe.get_wavelength_bin_centers().cpu().numpy()
        wavelength_bin_widths = r_config_in.phringe.get_wavelength_bin_widths().cpu().numpy()
        data_in = self.get_resource_from_name(self.n_data_in).get_data()
        template_data = r_templates_in.get_data()
        grid_coordinates = r_templates_in.grid_coordinates

        # Flatten data along differential outputs and times axes
        data_in = data_in.permute(0, 2, 1)
        data_in = data_in.reshape((-1,) + data_in.shape[2:])
        template_data = template_data.permute(0, 2, 1, 3, 4)
        template_data = template_data.reshape((-1,) + template_data.shape[2:])

        # Set up parameters and initial conditions
        if planet_params_in is not None and len(planet_params_in.params) > 0:


            semi_major_axis_init = planet_params_in.params[0].semi_major_axis
            eccentricity_init = planet_params_in.params[0].eccentricity
            planet_mass_fixed = planet_params_in.params[0].mass
            inclination_init = planet_params_in.params[0].inclination
            raan_init = planet_params_in.params[0].raan
            argument_of_periapsis_init = planet_params_in.params[0].argument_of_periapsis
            true_anomaly_init = planet_params_in.params[0].true_anomaly
            flux_init, _ , _ = self._get_analytical_initial_guess(
                data_in,
                template_data,
                grid_coordinates
            )
        else:
            radius_init = 6 * 1e6  # 1 Earth radius in meters
            temp_init = 300.0  # Kelvin
            semi_major_axis_init = 1 * const.au.value  # Can we evaluate quite good?
            eccentricity_init = 0.0
            planet_mass_fixed = 1 * const.M_earth.value
            inclination_init = np.pi / 2
            raan_init = np.pi
            argument_of_periapsis_init = np.pi
            true_anomaly_init = 0
            flux_init = planet_params_in.params[0].sed.cpu().numpy()


        data_in = data_in.cpu().numpy()
        hfov_max = r_config_in.phringe.get_field_of_view()[-1].cpu().numpy() / 2  # TODO: /14 Check this

        params = Parameters()

        for j in range(len(flux_init)):
            if self.bounds:
                params.add(f'flux_{j}', value=flux_init[j], min=0)
            else:
                params.add(f'flux_{j}', value=flux_init[j])

        # Perform MLE
        def residual_data(params, target):

            flux = np.array([params[f'flux_{z}'].value for z in range(len(flux_init))])

            model = r_config_in.phringe.get_model_counts(
                kernels=True,
                spectral_energy_distribution=flux,
                semi_major_axis=semi_major_axis_init,
                eccentricity=eccentricity_init,
                inclination=inclination_init,
                raan=raan_init,
                argument_of_periapsis=argument_of_periapsis_init,
                true_anomaly=true_anomaly_init,

                host_star_distance=r_config_in.scene.star.distance,
                host_star_mass=r_config_in.scene.star.mass,
                planet_mass=planet_mass_fixed
            )
            model = transf(model)
            model = np.transpose(model, (0, 2, 1))
            model = model.reshape(data_in.shape)

            return model - target

        out = minimize(residual_data, params, args=(data_in,), method='leastsq')
        cov_out = out.covar

        fluxes = np.array([out.params[f'flux_{k}'].value for k in range(len(flux_init))])


        try:
            stds = np.sqrt(np.diag(cov_out))
            flux_err = stds

        except ValueError:
            flux_err = np.full_like(fluxes, np.nan)


        # TODO: Implement multi-planet signal extraction
        r_planet_params_out = PlanetParamsResource(
            name=self.n_planet_params_out,
        )
        planet_params = PlanetParams(
            name='',
            sed_wavelength_bin_centers=r_config_in.phringe.get_wavelength_bin_centers(),
            sed_wavelength_bin_widths=r_config_in.phringe.get_wavelength_bin_widths(),
            sed=torch.tensor(fluxes),
            sed_err_low=torch.tensor(flux_err),
            sed_err_high=torch.tensor(flux_err),
            covariance=cov_out
        )
        r_planet_params_out.params.append(planet_params)

        print('Done')
        return r_planet_params_out
