import numpy as np
import torch
from lmfit import minimize, Parameters

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

        # Fit geometric Keplerian orbit (no timing) with per-point anomaly nuisance parameters.
        orbit_fit_curve_x = None
        orbit_fit_curve_y = None
        orbit_fit_ok = False
        posx_obs = np.asarray(fitted_posx, dtype=float)
        posy_obs = np.asarray(fitted_posy, dtype=float)
        n_obs = len(posx_obs)
        if n_obs >= 3:
            try:
                from scipy.optimize import least_squares

                def _orbit_xy(log_a_val, h_val, k_val, cos_i_val, omega_node_val, nus):
                    e_val = np.sqrt(h_val ** 2 + k_val ** 2)
                    if e_val >= 1.0:
                        raise ValueError("Eccentricity must be < 1")
                    omega_arg_val = np.arctan2(k_val, h_val)
                    inc_val = np.arccos(np.clip(cos_i_val, -1.0, 1.0))
                    a_val = 10.0 ** log_a_val  # angular semi-major axis [rad]
                    nus = np.asarray(nus, dtype=float)

                    # Kepler geometry in orbital plane: r = a(1-e^2)/(1+e cos nu)
                    r = a_val * (1.0 - e_val ** 2) / (1.0 + e_val * np.cos(nus))
                    u_arg = omega_arg_val + nus

                    cos_O = np.cos(omega_node_val)
                    sin_O = np.sin(omega_node_val)
                    cos_u = np.cos(u_arg)
                    sin_u = np.sin(u_arg)
                    cos_i = np.cos(inc_val)

                    # Projection to sky-plane offsets (x, y), in radians.
                    x_vals = r * (cos_O * cos_u - sin_O * sin_u * cos_i)
                    y_vals = r * (sin_O * cos_u + cos_O * sin_u * cos_i)
                    return x_vals, y_vals

                def _residual_orbit(theta):
                    log_a_val = theta[0]
                    h_val = theta[1]
                    k_val = theta[2]
                    cos_i_val = theta[3]
                    omega_node_val = theta[4]
                    nus = theta[5:]
                    e_val = np.sqrt(h_val ** 2 + k_val ** 2)
                    if e_val >= 0.999 or np.abs(cos_i_val) > 1.0:
                        return np.full(2 * n_obs, 1e6, dtype=float)
                    try:
                        x_model, y_model = _orbit_xy(log_a_val, h_val, k_val, cos_i_val, omega_node_val, nus)
                    except Exception:
                        return np.full(2 * n_obs, 1e6, dtype=float)
                    return np.concatenate((posx_obs - x_model, posy_obs - y_model))

                pa_guess = np.mod(np.arctan2(posy_obs, posx_obs), 2.0 * np.pi)
                a_guess = np.maximum(np.median(np.sqrt(posx_obs ** 2 + posy_obs ** 2)), 1e-12)
                log_a0 = np.log10(a_guess)
                log_a_min = np.log10(max(1e-12, a_guess * 0.01))
                log_a_max = np.log10(max(1e-11, a_guess * 100.0))
                lower = np.concatenate(([log_a_min, -0.99, -0.99, -1.0, 0.0], np.zeros(n_obs)))
                upper = np.concatenate(([log_a_max, 0.99, 0.99, 1.0, 2.0 * np.pi], np.full(n_obs, 2.0 * np.pi)))

                best_result = None
                rng = np.random.default_rng(123)
                n_restarts = 24
                initial_x0 = np.concatenate((
                    [log_a0, 0.1, 0.0, np.cos(np.deg2rad(60.0)), 0.0],
                    pa_guess
                ))
                initial_x0 = np.clip(initial_x0, lower + 1e-12, upper - 1e-12)
                for r in range(n_restarts):
                    if r == 0:
                        x0 = initial_x0.copy()
                    else:
                        h0 = rng.uniform(-0.5, 0.5)
                        k0 = rng.uniform(-0.5, 0.5)
                        if np.sqrt(h0 ** 2 + k0 ** 2) >= 0.99:
                            h0, k0 = 0.1, 0.0
                        x0 = np.concatenate((
                            [rng.uniform(log_a_min, log_a_max), h0, k0, rng.uniform(-1.0, 1.0), rng.uniform(0.0, 2.0 * np.pi)],
                            np.mod(pa_guess + rng.normal(0.0, 0.35, size=n_obs), 2.0 * np.pi)
                        ))
                        if r % 3 == 0:
                            # occasional wider restart to escape local minima
                            x0[5:] = rng.uniform(0.0, 2.0 * np.pi, size=n_obs)
                    x0 = np.clip(x0, lower + 1e-12, upper - 1e-12)

                    result = least_squares(
                        _residual_orbit,
                        x0,
                        bounds=(lower, upper),
                        method="trf",
                        max_nfev=12000,
                        ftol=1e-10,
                        xtol=1e-10,
                        gtol=1e-10,
                    )
                    if best_result is None or result.cost < best_result.cost:
                        best_result = result

                if best_result is not None and best_result.success:
                    theta_best = best_result.x
                    log_a_fit = theta_best[0]
                    h_fit = theta_best[1]
                    k_fit = theta_best[2]
                    cos_i_fit = theta_best[3]
                    omega_node_fit = theta_best[4]
                    nus_fit = theta_best[5:]
                    e_fit = np.sqrt(h_fit ** 2 + k_fit ** 2)
                    omega_arg_fit = np.arctan2(k_fit, h_fit)
                    inc_fit = np.arccos(np.clip(cos_i_fit, -1.0, 1.0))
                    a_fit = 10.0 ** log_a_fit

                    print("Geometric orbit fit successful:")
                    print("initial cost:", 0.5 * np.sum(_residual_orbit(initial_x0) ** 2))
                    print("final cost:", best_result.cost)
                    print("cost ratio final/initial:", best_result.cost / (0.5 * np.sum(_residual_orbit(initial_x0) ** 2) + 1e-30))
                    print("a (angular, rad):", a_fit)
                    print("log_a:", log_a_fit)
                    print("e:", e_fit)
                    print("h = e cos(omega):", h_fit)
                    print("k = e sin(omega):", k_fit)
                    print("i (rad):", inc_fit)
                    print("i (deg):", np.rad2deg(inc_fit))
                    print("Omega (rad):", omega_node_fit)
                    print("Omega (deg):", np.rad2deg(omega_node_fit))
                    print("omega (rad):", omega_arg_fit)
                    print("omega (deg):", np.rad2deg(omega_arg_fit))
                    print("nu_i (rad):", nus_fit)
                    print("nu_i (deg):", np.rad2deg(nus_fit))
                    nu_grid = np.linspace(0.0, 2.0 * np.pi, 720, endpoint=False)
                    orbit_fit_curve_x, orbit_fit_curve_y = _orbit_xy(
                        log_a_fit,
                        h_fit,
                        k_fit,
                        cos_i_fit,
                        omega_node_fit,
                        nu_grid
                    )
                    orbit_fit_ok = True
                else:
                    print("Geometric orbit fit failed to converge.")
            except Exception as exc:
                print(f"Geometric orbit fit skipped: {exc}")

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
