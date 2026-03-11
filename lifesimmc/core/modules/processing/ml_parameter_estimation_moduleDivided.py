import numpy as np
import torch
from lmfit import minimize, Parameters
from astropy.table import Table

from lifesimmc.core.modules.base_module import BaseModule
from lifesimmc.core.resources.base_resource import BaseResource
from lifesimmc.core.resources.planet_params_resource import PlanetParamsResource, PlanetParams
from astropy import constants as const


class MLParameterEstimationModuleDivided(BaseModule):
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

        # Get the coordinates of the da
        x_coord = grid_coordinates[0][row.item(), col.item()].cpu().numpy()
        y_coord = grid_coordinates[1][row.item(), col.item()].cpu().numpy()

        return optimum_flux_at_maximum, x_coord, y_coord

    def apply(self, resources: list[BaseResource]) -> PlanetParamsResource:
        print('Performing numerical MLE...')

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

        # Reorder to (differential_output, time, wavelength) so residual chunks can be strictly time-based.
        data_in = data_in.permute(0, 2, 1)
        data_for_guess = data_in.reshape((-1,) + data_in.shape[2:])
        template_data = template_data.permute(0, 2, 1, 3, 4)
        template_data = template_data.reshape((-1,) + template_data.shape[2:])

        # Set up parameters and initial conditions:
        # prefer true input parameters; fall back to analytical only if unavailable.
        flux_analytical, posx_analytical, posy_analytical = self._get_analytical_initial_guess(
            data_for_guess,
            template_data,
            grid_coordinates
        )
        if planet_params_in is not None and len(planet_params_in.params) > 0:
            if planet_params_in.params[0].sed is not None:
                flux_init = planet_params_in.params[0].sed.cpu().numpy()
            else:
                flux_init = flux_analytical
            posx_init = planet_params_in.params[0].pos_x if planet_params_in.params[0].pos_x is not None else posx_analytical
            posy_init = planet_params_in.params[0].pos_y if planet_params_in.params[0].pos_y is not None else posy_analytical
        else:
            flux_init = flux_analytical
            posx_init = posx_analytical
            posy_init = posy_analytical
        posx_init *= 1.1
        posy_init *= 1.1

        data_in = data_in.cpu().numpy()
        hfov_max = r_config_in.phringe.get_field_of_view()[-1].cpu().numpy() / 2  # TODO: /14 Check this

        params = Parameters()

        params.add('pos_x', value=posx_init, min=-hfov_max, max=hfov_max)
        params.add('pos_y', value=posy_init, min=-hfov_max, max=hfov_max)

        # Ground-truth sky tracks for all injected planets.
        gt_tracks = []
        for gt_planet in r_config_in.scene.planets:
            gt_sky_brightness_distribution = gt_planet._sky_brightness_distribution
            gt_sky_coordinates = gt_planet._sky_coordinates
            x_track = []
            y_track = []
            if gt_planet.has_orbital_motion:
                n_time_steps = gt_sky_brightness_distribution.shape[0]
                for t in range(n_time_steps):
                    gt_non_zero_indices = torch.nonzero(gt_sky_brightness_distribution[t, 1])
                    if gt_non_zero_indices.numel() == 0:
                        continue
                    row = gt_non_zero_indices[0][0]
                    col = gt_non_zero_indices[0][1]
                    x_track.append(gt_sky_coordinates[0, t, row, col].item())
                    y_track.append(gt_sky_coordinates[1, t, row, col].item())
            else:
                gt_non_zero_indices = torch.nonzero(gt_sky_brightness_distribution[1])
                if gt_non_zero_indices.numel() != 0:
                    row = gt_non_zero_indices[0][0]
                    col = gt_non_zero_indices[0][1]
                    x_track.append(gt_sky_coordinates[0, row, col].item())
                    y_track.append(gt_sky_coordinates[1, row, col].item())
            if len(x_track) > 0:
                gt_tracks.append(
                    {
                        "name": gt_planet.name,
                        "x": np.asarray(x_track),
                        "y": np.asarray(y_track),
                    }
                )

        # Perform MLE

        N = 20


        total_time = int(data_in.shape[1])
        N = max(1, min(N, total_time))
        chunk_size = int(np.ceil(total_time / N))
        import matplotlib.pyplot as plt
        fitted_posx = []
        fitted_posy = []
        for n in range(N):

            def residual_data(params, target):
                posx = params['pos_x'].value
                posy = params['pos_y'].value
                print("Try pos x",posx)
                print("Try pos y", posy)

                model = r_config_in.phringe.get_model_counts(
                    spectral_energy_distribution=flux_init,
                    x_position=posx,
                    y_position=posy,
                    kernels=True
                )

                model = transf(model)
                model = np.transpose(model, (0, 2, 1))
                model = model.reshape(data_in.shape)

                start = n * chunk_size
                stop = min((n + 1) * chunk_size, total_time)
                residual_chunk = model[:, start:stop, :] - target[:, start:stop, :]
                return residual_chunk.reshape(-1)


            out = minimize(residual_data, params, args=(data_in,), method='leastsq',ftol = 1e-12)
            cov_out = out.covar
            # Use previous chunk result as start for the next chunk.
            params = out.params

            posx = out.params['pos_x'].value
            posy = out.params['pos_y'].value

            print("pos x",posx)
            print("pos y", posy)
            fitted_posx.append(posx)
            fitted_posy.append(posy)

        # Fit orbit with orbitize (OFTI sampler) on the fitted astrometric points.
        orbit_fit_curve_x = None
        orbit_fit_curve_y = None
        orbit_fit_ok = False
        posx_obs = np.asarray(fitted_posx, dtype=float)
        posy_obs = np.asarray(fitted_posy, dtype=float)
        n_obs = len(posx_obs)
        if n_obs >= 3:
            try:
                import orbitize.kepler
                import orbitize.sampler
                import orbitize.system

                rad_to_mas = (180.0 / np.pi) * 3600.0 * 1000.0
                mas_to_rad = 1.0 / rad_to_mas

                sep_mas = np.sqrt(posx_obs ** 2 + posy_obs ** 2) * rad_to_mas
                # PA convention for orbitize (East of North): atan2(RA_offset, Dec_offset)
                pa_deg = np.mod(np.degrees(np.arctan2(posx_obs, posy_obs)), 360.0)

                sep_err_mas = np.maximum(1e-3, 0.05 * np.maximum(sep_mas, 1e-3))
                pa_err_deg = np.full_like(pa_deg, 1.0)

                if len(times) >= n_obs:
                    epoch_mjd = 59000.0 + times[:n_obs] / 86400.0
                else:
                    epoch_mjd = 59000.0 + np.arange(n_obs, dtype=float)

                table_rows = []
                for i_obs in range(n_obs):
                    table_rows.append(
                        (
                            float(epoch_mjd[i_obs]),
                            1,
                            "seppa",
                            float(sep_mas[i_obs]),
                            float(sep_err_mas[i_obs]),
                            float(pa_deg[i_obs]),
                            float(pa_err_deg[i_obs]),
                            np.nan,
                            "LIFEsim",
                        )
                    )
                data_table = Table(
                    rows=table_rows,
                    names=[
                        "epoch",
                        "object",
                        "quant_type",
                        "quant1",
                        "quant1_err",
                        "quant2",
                        "quant2_err",
                        "quant12_corr",
                        "instrument",
                    ],
                )

                stellar_mass_msun = float(r_config_in.scene.star.mass / const.M_sun.value)
                distance_pc = float(r_config_in.scene.star.distance / const.pc.value)
                plx_mas = 1000.0 / max(distance_pc, 1e-6)

                orb_system = orbitize.system.System(
                    1,
                    data_table,
                    stellar_mass_msun,
                    plx_mas,
                )
                ofti_sampler = orbitize.sampler.OFTI(orb_system)
                # Small default for runtime; increase for smoother posterior.
                ofti_samples = ofti_sampler.run_sampler(total_orbits=2000)

                median_sample = np.median(ofti_samples, axis=0)
                sma_au = float(median_sample[0])
                ecc = float(median_sample[1])
                inc = float(median_sample[2])
                aop = float(median_sample[3])
                pan = float(median_sample[4])
                tau = float(median_sample[5])
                plx_fit = float(median_sample[6])
                mtot_fit = float(median_sample[7])

                epoch_grid = np.linspace(np.min(epoch_mjd), np.max(epoch_mjd), 720)
                raoff_mas, deoff_mas, _ = orbitize.kepler.calc_orbit(
                    epoch_grid,
                    sma_au,
                    ecc,
                    inc,
                    aop,
                    pan,
                    tau,
                    plx_fit,
                    mtot_fit,
                )
                orbit_fit_curve_x = raoff_mas * mas_to_rad
                orbit_fit_curve_y = deoff_mas * mas_to_rad
                orbit_fit_ok = True

                print("orbitize OFTI fit successful:")
                print("sma [au]:", sma_au)
                print("ecc:", ecc)
                print("inc [deg]:", np.degrees(inc))
                print("aop [deg]:", np.degrees(aop))
                print("pan [deg]:", np.degrees(pan))
            except Exception as exc:
                print(f"orbitize OFTI fit skipped: {exc}")

        plt.figure(figsize=(6, 6))
        for i, gt_track in enumerate(gt_tracks):
            plt.plot(
                gt_track["x"],
                gt_track["y"],
                linewidth=1.4,
                alpha=0.9,
                label=f"GT orbit {i}: {gt_track['name']}"
            )
            plt.scatter(
                gt_track["x"],
                gt_track["y"],
                c='limegreen',
                marker='x',
                s=25,
                alpha=0.7,
                label=f"GT points {i}"
            )
        if orbit_fit_ok and orbit_fit_curve_x is not None and orbit_fit_curve_y is not None:
            plt.plot(
                orbit_fit_curve_x,
                orbit_fit_curve_y,
                c='royalblue',
                linewidth=2.0,
                linestyle='--',
                label='Fitted Kepler orbit'
            )
        if len(fitted_posx) > 0:
            plt.scatter(fitted_posx, fitted_posy, c='crimson', marker='o', s=28, label='Fitted points')
            for i, (x_fit, y_fit) in enumerate(zip(fitted_posx, fitted_posy), start=1):
                plt.annotate(str(i), (x_fit, y_fit), textcoords="offset points", xytext=(4, 4), fontsize=8, color='crimson')


        plt.xlim(-hfov_max/2, hfov_max/2)
        plt.ylim(-hfov_max/2, hfov_max/2)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlabel('x position [rad]')
        plt.ylabel('y position [rad]')
        plt.title('Orbit, ground truth, and fitted points')
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig("PositionComparison")
        plt.close()
        try:
            stds = np.sqrt(np.diag(cov_out))
            posx_err = stds[-2]
            posy_err = stds[-1]
        except ValueError:
            posx_err = np.nan
            posy_err = np.nan

        # TODO: Implement multi-planet signal extraction
        r_planet_params_out = PlanetParamsResource(
            name=self.n_planet_params_out,
        )
        planet_params = PlanetParams(
            name='',
            sed_wavelength_bin_centers=r_config_in.phringe.get_wavelength_bin_centers(),
            sed_wavelength_bin_widths=r_config_in.phringe.get_wavelength_bin_widths(),
            sed=torch.tensor(flux_init),
            sed_err_low=torch.tensor(np.ones_like(flux_init)),
            sed_err_high=torch.tensor((np.ones_like(flux_init))),
            pos_x=posx,
            pos_y=posy,
            pos_x_err_low=posx_err,
            pos_x_err_high=posx_err,
            pos_y_err_low=posy_err,
            pos_y_err_high=posy_err,
            covariance=cov_out
        )
        r_planet_params_out.params.append(planet_params)

        print('Done')
        return r_planet_params_out
