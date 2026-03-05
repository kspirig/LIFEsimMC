import numpy as np
import torch
from lmfit import minimize, Parameters
from astropy import units as u
from astropy import constants as const
from pathlib import Path
from datetime import datetime

from lifesimmc.core.modules.base_module import BaseModule
from lifesimmc.core.resources.base_resource import BaseResource
from lifesimmc.core.resources.planet_params_resource import PlanetParamsResource, PlanetParams
from phringe.util.spectrum import get_blackbody_spectrum_standard_units
import matplotlib.pyplot as plt

class MLBBParameterEstimationModule(BaseModule):
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
        print('Performing numerical MLE with Blackbody...')

        r_config_in = self.get_resource_from_name(self.n_config_in)
        r_templates_in = self.get_resource_from_name(self.n_template_in)
        r_transformation_in = self.get_resource_from_name(
            self.n_transformation_in) if self.n_transformation_in else None
        transf = r_transformation_in.transformation if r_transformation_in else lambda x: x
        planet_params_in = self.get_resource_from_name(self.n_planet_params_in) if self.n_planet_params_in else None


        distance = r_config_in.scene.star.distance  # float in meters



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
            radius_init = planet_params_in.params[0].radius
            temp_init = planet_params_in.params[0].temp
        else:
            radius_init = 6 * 1e6  # 1 Earth radius in meters
            temp_init = 300.0      # Kelvin


        _, posx_init, posy_init = self._get_analytical_initial_guess(
                data_in,
                template_data,
                grid_coordinates
            )


        print("Radius init", radius_init)
        print("Temperature init:", temp_init)

        data_in = data_in.cpu().numpy()
        hfov_max = r_config_in.phringe.get_field_of_view()[-1].cpu().numpy() / 2  # TODO: /14 Check this

        params = Parameters()


        params.add('pos_x', value=posx_init, min=-hfov_max, max=hfov_max)
        params.add('pos_y', value=posy_init, min=-hfov_max, max=hfov_max)
        params.add("temp",value = temp_init,min = 50,max = 1e4)
        params.add("radius",value = radius_init,min = 5*1e4,max = 1e9)



        OUTPUT_DIR = Path('bb_resultsGridSize40PPHRINGE')
        plots_dir = OUTPUT_DIR / "mlbb_parameter_estimation_plots"


        eval_counter = 0


        # Perform MLE
        def residual_data(params, target):
            nonlocal eval_counter
            eval_counter += 1
            posx = params['pos_x'].value
            posy = params['pos_y'].value
            temp = params['temp'].value

            #ToDo: Inlcude fitting Orbital Motion

            flux = np.pi * get_blackbody_spectrum_standard_units(temperature=temp, wavelengths=wavelengths) * (params['radius'].value / distance) ** 2



            if torch.is_tensor(flux):
                flux = flux.detach().cpu().numpy()


            model = r_config_in.phringe.get_model_counts(
                spectral_energy_distribution=flux,
                x_position=posx,
                y_position=posy,
                kernels=True
            )



            #
            #
            # # # Save loop diagnostics with unique names to avoid overwriting.
            # try:
            #     eval_name = f"eval_{eval_counter:06d}.png"
            #     target_name = f"target_eval_{eval_counter:06d}.png"
            #     flux_name = f"flux_eval_{eval_counter:06d}.png"
            #
            #
            #     plt.figure(figsize=(7, 4.5))
            #     plt.imshow(model[0], cmap='Greys', aspect='auto')
            #     plt.title(f'Model (channel 0), eval={eval_counter}')
            #     plt.ylabel('Wavelength Channel')
            #     plt.xlabel('Time Step')
            #     plt.colorbar()
            #     plt.tight_layout()
            #     plt.savefig(run_plots_dir / eval_name, dpi=150)
            #     plt.close()
            #
            #
            #
            #     plt.figure(figsize=(7, 4.5))
            #     plt.imshow(target.T, cmap='Greys', aspect='auto')
            #     plt.title(f'Target (channel 0), eval={eval_counter}')
            #     plt.ylabel('Wavelength Channel')
            #     plt.xlabel('Time Step')
            #     plt.colorbar()
            #     plt.tight_layout()
            #     plt.savefig(run_plots_dir / target_name, dpi=150)
            #     plt.close()
            #
            #     print(temp, " ", params["radius"].value)
            #
            #     plt.figure(figsize=(7, 4.5))
            #     plt.plot(
            #         wavelengths,
            #         flux,
            #         label=f"T={temp:.3f} K, R={params['radius'].value:.3e} m"
            #     )
            #     plt.title(f'Flux, eval={eval_counter}')
            #     plt.xlabel('Wavelength [m]')
            #     plt.ylabel('Flux')
            #     plt.legend()
            #     plt.tight_layout()
            #     plt.savefig(run_plots_dir / flux_name, dpi=150)
            #     plt.close()
            #
            #
            #
            # except Exception as exc:
            #     print(f"Could not save loop plots at eval {eval_counter}")

            model = transf(model)
            model = np.transpose(model, (0, 2, 1))
            model = model.reshape(data_in.shape)

            return model - target


        # out = minimize(
        #     residual_data,
        #     params,
        #     args=(data_in,),
        #     method="least_squares",
        #     max_nfev=20000,
        #     ftol=1e-10,
        #     xtol=None,
        #     gtol=1e-10,
        #     loss="soft_l1",
        #     f_scale=1.0,
        #     x_scale="jac",
        #     nan_policy="omit",
        # )

        out = minimize(residual_data, params, args=(data_in,), method='leastsq')

        print("success:", out.success)
        print("message:", out.message)
        print("nfev:", out.nfev)
        print("chisqr:", out.chisqr)
        cov_out = out.covar

        fluxes = np.pi* get_blackbody_spectrum_standard_units(
            temperature=out.params['temp'].value,
            wavelengths=wavelengths
        ) * (out.params['radius'].value / distance) ** 2

        if torch.is_tensor(fluxes):
            fluxes = fluxes.detach().cpu().numpy()

        posx = out.params['pos_x'].value
        posy = out.params['pos_y'].value

        temp = out.params['temp'].value
        radius = out.params['radius'].value

        def dBdT(radius,temp,wavelengths):
            lam_m = np.asarray(wavelengths, dtype=float)
            return (4 * np.pi * const.c.value**2 * const.h.value) / ( lam_m ** 5 * const.k_B.value * temp) / (radius * distance)**2 * np.exp(const.h.value*const.c.value/(const.k_B.value*temp*lam_m)) / (np.expm1(const.h.value * const.c.value / (lam_m*const.k_B.value*temp))**2)

        def dBdR(radius,temp,wavelengths):
            lam_m = np.asarray(wavelengths, dtype=float)
            x = (const.h.value * const.c.value) / (lam_m * const.k_B.value * temp)
            denom = np.expm1(x)
            photon_flux_per_m = (2.0 * np.pi * const.c.value / lam_m ** 4) * 2*(radius / (distance**2)) / denom
            return photon_flux_per_m

        # Extract parameter errors by name so covariance ordering cannot corrupt uncertainties.
        try:
            stds = np.sqrt(np.diag(cov_out))
            var_names = out.var_names if out.var_names is not None else []
            idx_map = {name: i for i, name in enumerate(var_names)}

            temp_err = stds[idx_map['temp']] if 'temp' in idx_map else np.nan
            radius_err = stds[idx_map['radius']] if 'radius' in idx_map else np.nan
            posx_err = stds[idx_map['pos_x']] if 'pos_x' in idx_map else np.nan
            posy_err = stds[idx_map['pos_y']] if 'pos_y' in idx_map else np.nan

            flux_err = np.sqrt(dBdT(radius,temp,wavelengths) ** 2  * temp_err**2 + dBdR(radius,temp,wavelengths) ** 2  * radius_err**2)
        except (TypeError, ValueError, KeyError, IndexError):
            flux_err = np.full_like(fluxes, np.nan, dtype=float)
            posx_err = np.nan
            posy_err = np.nan
            temp_err = np.nan
            radius_err = np.nan

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

        )
        r_planet_params_out.params.append(planet_params)
        print("Fitted Temperature", temp)
        print("Fitted Radius", out.params['radius'].value)
        print('Done')
        return r_planet_params_out
