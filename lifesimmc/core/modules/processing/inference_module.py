from dataclasses import dataclass
from dataclasses import field
import math
import sys
import time
from typing import Callable

import astropy.constants as const
import numpy as np
from phringe.lib.array_configuration import XArrayConfiguration
from phringe.lib.beam_combiner import DoubleBracewell
import torch
from rich.console import Console
try:
    from torchns import NestedSampler
except ImportError:
    NestedSampler = object
    TORCHNS_BACKEND = "local_jittered_nested_sampler"
else:
    TORCHNS_BACKEND = "torchns_jittered_nested_sampler"
try:
    from tqdm.auto import tqdm
except ImportError:

    def tqdm(iterable):
        return iterable

from lifesimmc.core.modules.base_module import BaseModule
from lifesimmc.core.resources.base_resource import BaseResource
from lifesimmc.core.resources.resource_collection import ResourceCollection


ECC_BETA_ALPHA = 0.867
ECC_BETA_BETA = 3.03
ECC_MAX = 0.99
BETA_ICDF_TABLE_SIZE = 131072
BETA_ICDF_U_MIN = 1e-7
BETA_ICDF_U_MAX = 1.0 - 1e-7
BETA_ICDF_Z_MIN = float(np.log(BETA_ICDF_U_MIN / (1.0 - BETA_ICDF_U_MIN)))
BETA_ICDF_Z_MAX = float(np.log(BETA_ICDF_U_MAX / (1.0 - BETA_ICDF_U_MAX)))
POSTERIOR_TRANSFORM_CHUNK_SIZE = 262144
FAST_PHRINGE_XARRAY_Q = 6.0

SAMPLE_LABELS = ["sma_m", "ecc", "inc_rad", "raan_rad", "argp_rad", "nu0_rad"]
SAMPLE_UNITS = ["m", "1", "rad", "rad", "rad", "rad"]

LIKELIHOOD_CONVENTION = (
    "Internal sampler log_likelihood = -0.5 * sum((data - model)^2) + likelihood_offset "
    "in the supplied data space, without the Gaussian normalization constant. "
    "The likelihood_offset is the Runner-style 0.5 * truth_residual_chi2 shift. "
    "The planet spectrum is profiled for each candidate orbit with second-difference "
    "Tikhonov regularization; it is not jointly sampled."
)
NU0_EPOCH_DESCRIPTION = (
    "nu0_rad is the true anomaly at the PHRINGE simulation time origin t=0 s. "
    "Candidate positions are propagated to every detector time returned by "
    "setup.phringe.get_time_steps()."
)
PRIOR_DESCRIPTION = (
    "Unit-cube prior transform: semimajor axis is log-uniform between sma_lower "
    "and sma_upper, in meters; eccentricity uses the implemented Kipping/Guimond "
    f"Beta({ECC_BETA_ALPHA}, {ECC_BETA_BETA}) inverse-CDF transform and is clipped "
    "to [0, eccentricity_upper]; inclination is isotropic by sampling cos(i) "
    "uniformly on [-1, 1]; raan and argument of periapsis are uniform on [0, 2*pi]; "
    "mean anomaly at t=0 is uniform on [0, 2*pi] and then converted to nu0_rad."
)
SCIENTIFIC_LIMITATIONS = (
    "Single planet and single continuous observing interval only.",
    "The fitted spectrum is profiled/regularized rather than sampled jointly, so "
    "evidence values do not marginalize over spectral uncertainty.",
    "The eccentricity prior is clipped at eccentricity_upper, not renormalized as "
    "an explicitly truncated beta distribution.",
)

_BETA_ICDF_TABLE_CACHE = {}
_CUDA_LINALG_CPU_FALLBACK = False
_CUDA_LINALG_CPU_FALLBACK_WARNED = False
_CUDA_SOLVE_CUSTOM_FALLBACK = False
_CUDA_SOLVE_CUSTOM_FALLBACK_WARNED = False
_CUDA_NO_KERNEL_IMAGE_MESSAGE = "no kernel image is available for execution on the device"


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


class _SingleLineProgress:
    """Minimal carriage-return progress line for consoles where tqdm is noisy."""

    def __init__(self, enabled: bool):
        self.enabled = bool(enabled)
        self._last_length = 0

    def update(self, text: str):
        if not self.enabled:
            return
        padding = " " * max(0, self._last_length - len(text))
        sys.stdout.write("\r" + text + padding)
        sys.stdout.flush()
        self._last_length = len(text)

    def close(self):
        if self.enabled:
            sys.stdout.write("\n")
            sys.stdout.flush()


def _is_cuda_no_kernel_image_error(error: BaseException) -> bool:
    return _CUDA_NO_KERNEL_IMAGE_MESSAGE in str(error).lower()


def _enable_cuda_linalg_cpu_fallback(operation: str):
    global _CUDA_LINALG_CPU_FALLBACK, _CUDA_LINALG_CPU_FALLBACK_WARNED
    _CUDA_LINALG_CPU_FALLBACK = True
    if not _CUDA_LINALG_CPU_FALLBACK_WARNED:
        print(
            f"CUDA linear algebra kernel {operation!r} is not available for this GPU. "
            "Falling back to CPU for small torch.linalg operations."
        )
        _CUDA_LINALG_CPU_FALLBACK_WARNED = True


def _enable_cuda_solve_custom_fallback():
    global _CUDA_SOLVE_CUSTOM_FALLBACK, _CUDA_SOLVE_CUSTOM_FALLBACK_WARNED
    _CUDA_SOLVE_CUSTOM_FALLBACK = True
    if not _CUDA_SOLVE_CUSTOM_FALLBACK_WARNED:
        print(
            "CUDA torch.linalg.solve is not available for this GPU/build. "
            "Using an exact CUDA tensor Cholesky solve for the profiled flux systems."
        )
        _CUDA_SOLVE_CUSTOM_FALLBACK_WARNED = True


def _prefer_custom_cuda_solve(device: torch.device) -> bool:
    if device.type != "cuda":
        return False
    cuda_version = torch.version.cuda or ""
    if not cuda_version.startswith("13"):
        return False
    try:
        major, minor = torch.cuda.get_device_capability(device)
    except Exception:
        return False
    return (major, minor) <= (7, 5)


def _cholesky_with_cuda_linalg_fallback(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.device.type == "cuda" and _CUDA_LINALG_CPU_FALLBACK:
        return torch.linalg.cholesky(matrix.cpu()).to(device=matrix.device, dtype=matrix.dtype)

    try:
        return torch.linalg.cholesky(matrix)
    except Exception as error:
        if matrix.device.type != "cuda" or not _is_cuda_no_kernel_image_error(error):
            raise
        _enable_cuda_linalg_cpu_fallback("cholesky")
        return torch.linalg.cholesky(matrix.cpu()).to(device=matrix.device, dtype=matrix.dtype)


def _eigvalsh_with_cuda_linalg_fallback(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.device.type == "cuda" and _CUDA_LINALG_CPU_FALLBACK:
        return torch.linalg.eigvalsh(matrix.cpu()).to(device=matrix.device, dtype=matrix.dtype)

    try:
        return torch.linalg.eigvalsh(matrix)
    except Exception as error:
        if matrix.device.type != "cuda" or not _is_cuda_no_kernel_image_error(error):
            raise
        _enable_cuda_linalg_cpu_fallback("eigvalsh")
        return torch.linalg.eigvalsh(matrix.cpu()).to(device=matrix.device, dtype=matrix.dtype)


@dataclass
class InferenceResultResource(BaseResource):
    """Resource containing weighted posterior orbital samples.

    Attributes
    ----------
    samples : np.ndarray
        Physical orbital samples with shape (n_samples, 6), ordered as
        sma_m, ecc, inc_rad, raan_rad, argp_rad, nu0_rad.
    posterior_weights : np.ndarray
        Normalized posterior weights for the returned samples.
    log_likelihoods : np.ndarray
        Runner-style shifted log-likelihood values for each returned sample.
    sampler_info : dict
        Basic sampler completion and evidence information.
    """

    samples: np.ndarray = None
    posterior_weights: np.ndarray = None
    log_likelihoods: np.ndarray = None
    log_likelihoods_unshifted: np.ndarray = None
    parameter_labels: list[str] = field(default_factory=lambda: list(SAMPLE_LABELS))
    parameter_units: list[str] = field(default_factory=lambda: list(SAMPLE_UNITS))
    unit_cube_samples: np.ndarray = None
    log_weights: np.ndarray = None
    sampler_info: dict = field(default_factory=dict)
    likelihood_convention: str = LIKELIHOOD_CONVENTION
    prior_description: str = PRIOR_DESCRIPTION
    nu0_epoch_description: str = NU0_EPOCH_DESCRIPTION
    scientific_limitations: tuple[str, ...] = SCIENTIFIC_LIMITATIONS


class JitteredNestedSampler(NestedSampler):
    """NestedSampler with conservative adaptive diagonal jitter in Cholesky."""

    def __init__(self, X_init, L_init=None, bound="unitcube", progress_update_interval=1, show_progress=False):
        self.X_init = X_init
        self.X_live = X_init * 1.0
        self.L_live = L_init if L_init is not None else None
        self.device = X_init.device
        self.progress_update_interval = max(1, int(progress_update_interval))
        self.show_progress = bool(show_progress)
        self.num_outer_iterations = 0
        self.num_replacements = 0
        self.stopped_at_iteration_limit = False
        self.stop_reason = "not_started"
        self.final_log_prior_volume = 0.0

        if bound == "unitcube":
            self._inbound = self._inbound_unitcube
        elif callable(bound):
            self._inbound = bound
        else:
            raise KeyError("Bound unknown")
        if not bool(torch.all(self._inbound(X_init)).detach().cpu().item()):
            raise AssertionError("X_init not within specified bounds")

    def _inbound_unitcube(self, X):
        return ((X <= 1.0) & (X >= 0.0)).all(dim=-1)

    def _get_directions(self, B, D):
        t = torch.randn(B, D, device=self.device)
        return t / torch.linalg.vector_norm(t, dim=-1, keepdim=True)

    def _get_slice_sample_points(self, B, S):
        dtype = self.X_live.dtype
        current_bounds = torch.empty((B, 2), dtype=dtype, device=self.device)
        current_bounds[:, 0] = -1.0
        current_bounds[:, 1] = 1.0
        L = torch.empty((B, S), dtype=dtype, device=self.device)
        for i in range(S):
            x = (
                    torch.rand(B, device=self.device)
                    * (current_bounds[:, 1] - current_bounds[:, 0])
                    + current_bounds[:, 0]
            )
            L[:, i] = x
            neg = x < 0
            pos = x > 0
            current_bounds[neg, 0] = x[neg]
            current_bounds[pos, 1] = x[pos]
        return L

    def _gen_new_samples(
            self,
            X_seeds,
            logl_fn,
            logl_th,
            num_steps=3,
            max_step_size=1.0,
            samples_per_slice=5,
            Lchol=None,
    ):
        """Generate constrained samples, avoiding the upstream Python loop over B."""
        if Lchol is None:
            Lchol = self._calc_Lchol(X_seeds)
        B, D = X_seeds.shape
        row_idx = torch.arange(B, device=self.device)
        C = torch.zeros(B, dtype=torch.int16, device=self.device)
        X = X_seeds.clone()
        logl = torch.full((B,), -np.inf, dtype=X.dtype, device=self.device)
        Lchol_t = Lchol.T if Lchol.device == self.device else Lchol.T.to(self.device)

        for _ in range(num_steps):
            N = self._get_directions(B, D)
            L = self._get_slice_sample_points(B, S=samples_per_slice) * max_step_size
            dX_uniform = N.unsqueeze(-2) * L.unsqueeze(-1)
            dX = torch.matmul(dX_uniform, Lchol_t)
            pX = X.unsqueeze(-2) + dX
            pX2 = pX.flatten(0, -2)
            logl_prop = logl_fn(pX2).view(B, samples_per_slice)
            inbound = self._inbound(pX2).view(B, samples_per_slice)
            accept_matrix = (logl_prop > logl_th) & inbound
            idx = torch.argmax(accept_matrix.to(torch.int32), dim=1)
            accept_any = accept_matrix.any(dim=-1)

            nX = pX[row_idx, idx]
            logl_selected = logl_prop[row_idx, idx]
            X[accept_any] = nX[accept_any]
            logl[accept_any] = logl_selected[accept_any]
            C[accept_any] += 1

        return X[C == num_steps], logl[C == num_steps]

    def nested_sampling(
            self,
            logl_fn,
            logl_th_max=np.inf,
            max_steps=100000,
            num_batch_samples=200,
            epsilon=1e-6,
            max_step_size=1.0,
            samples_per_slice=10,
            num_steps=5,
            progress_update_interval=None,
    ):
        """Run nested sampling and retain dead plus final live-point weights."""
        X_init = self.X_live
        NLP, _ = X_init.shape
        X_live = X_init.clone()
        if self.L_live is None:
            L_live = logl_fn(X_live)
        else:
            L_live = self.L_live

        B = min(int(num_batch_samples), int(NLP))
        log_shrink = np.log1p(-1.0 / float(NLP))
        log_prior_weight = -np.log(float(NLP))
        logV = 0.0
        logZ = -np.inf
        logZ_rest_bound = np.inf
        logl_th = torch.tensor(-np.inf, dtype=L_live.dtype, device=self.device)
        progress_every = (
            self.progress_update_interval
            if progress_update_interval is None
            else max(1, int(progress_update_interval))
        )

        samples_X_chunks = []
        samples_logl_chunks = []
        samples_logv = []
        samples_logwt = []
        num_dead = 0
        completed_steps = 0
        stop_reason = "iteration_limit"

        progress = _SingleLineProgress(self.show_progress)
        progress_t0 = time.time()
        for step in range(max_steps):
            completed_steps = step + 1
            if self.show_progress and step % progress_every == 0:
                elapsed = time.time() - progress_t0
                seconds_per_iteration = elapsed / max(completed_steps, 1)
                remaining_seconds = seconds_per_iteration * max(int(max_steps) - completed_steps, 0)
                progress.update(
                    "torchns "
                    f"step={completed_steps}/{int(max_steps)} "
                    f"dead={num_dead} "
                    f"logZ={logZ:.2f} "
                    f"logZ_rest<={logZ_rest_bound:.2f} "
                    f"logl_min={float(logl_th.detach().cpu().item()):.2f} "
                    f"elapsed={_format_duration(elapsed)} "
                    f"eta={_format_duration(remaining_seconds)} "
                    f"{seconds_per_iteration:.2f}s/it"
                )

            idx_batch = torch.randint(int(NLP), (B,), device=self.device)
            X_batch = X_live[idx_batch]
            logl_th = torch.min(L_live)
            if np.isfinite(logl_th_max) and bool((logl_th > logl_th_max).detach().cpu().item()):
                stop_reason = "logl_threshold"
                break

            Lchol = self._calc_Lchol(X_live)
            X_new, L_new = self._gen_new_samples(
                X_batch,
                logl_fn,
                logl_th,
                num_steps=num_steps,
                Lchol=Lchol,
                max_step_size=max_step_size,
                samples_per_slice=samples_per_slice,
            )

            dead_X_step = []
            dead_logl_step = []
            for new_idx in range(int(X_new.shape[0])):
                idx_min = torch.argmin(L_live)
                Lmin_t = L_live[idx_min].detach().clone()
                if not bool((L_new[new_idx] > Lmin_t).detach().cpu().item()):
                    break

                Lmin = float(Lmin_t.cpu().item())
                dead_X_step.append(X_live[idx_min].detach().clone())
                dead_logl_step.append(Lmin_t)
                samples_logv.append(logV)

                logwt = Lmin + logV + log_prior_weight
                samples_logwt.append(logwt)
                logZ = np.logaddexp(logZ, logwt)
                num_dead += 1

                L_live[idx_min] = L_new[new_idx]
                X_live[idx_min] = X_new[new_idx]
                logV += log_shrink
                logZ_rest_bound = logV + float(torch.max(L_live).detach().cpu().item())

            if dead_X_step:
                samples_X_chunks.append(torch.stack(dead_X_step).detach().cpu().float())
                samples_logl_chunks.append(torch.stack(dead_logl_step).detach().cpu().float())

            if np.isfinite(logZ) and np.isfinite(logZ_rest_bound) and logZ_rest_bound < logZ + np.log(float(epsilon)):
                stop_reason = "epsilon"
                break

        progress.close()

        if samples_X_chunks:
            samples_X = torch.cat(samples_X_chunks, dim=0)
            samples_logl = torch.cat(samples_logl_chunks, dim=0)
            samples_logv = torch.tensor(samples_logv, dtype=torch.float32)
            samples_logwt = torch.tensor(samples_logwt, dtype=torch.float32)
        else:
            samples_X = torch.empty((0, X_init.shape[1]), dtype=torch.float32)
            samples_logl = torch.empty((0,), dtype=torch.float32)
            samples_logv = torch.empty((0,), dtype=torch.float32)
            samples_logwt = torch.empty((0,), dtype=torch.float32)

        live_logwt = L_live.detach().cpu().float() + float(logV + log_prior_weight)
        self.X_live = X_live
        self.L_live = L_live
        self.samples_X = samples_X
        self.samples_logv = samples_logv
        self.samples_logl = samples_logl
        self.samples_logwt = samples_logwt
        self.live_X = X_live.detach().cpu().float()
        self.live_logl = L_live.detach().cpu().float()
        self.live_logwt = live_logwt
        self.num_outer_iterations = completed_steps
        self.num_replacements = int(samples_logl.numel())
        self.stopped_at_iteration_limit = stop_reason == "iteration_limit"
        self.stop_reason = stop_reason
        self.final_log_prior_volume = float(logV)

    def _calc_Lchol(self, X):
        if not hasattr(self, "_chol_fallback_eig_count"):
            self._chol_fallback_eig_count = 0
        if not hasattr(self, "_chol_fallback_jitter_count"):
            self._chol_fallback_jitter_count = 0

        cov = torch.cov(X.T)
        d = cov.shape[0]
        eye = torch.eye(d, device=cov.device, dtype=cov.dtype)

        try:
            return _cholesky_with_cuda_linalg_fallback(cov)
        except torch.linalg.LinAlgError:
            pass

        try:
            eigvals = _eigvalsh_with_cuda_linalg_fallback(cov)
            min_eig = torch.min(eigvals)
            eps_floor = cov.new_tensor(1e-12)
            shift = torch.clamp(-min_eig + eps_floor, min=eps_floor)
            L = _cholesky_with_cuda_linalg_fallback(cov + shift * eye)
            self._chol_fallback_eig_count += 1
            print(
                f"[torchns][chol] used eig fallback: count={self._chol_fallback_eig_count}, "
                f"shift={float(shift.item()):.3e}"
            )
            return L
        except torch.linalg.LinAlgError:
            pass

        trace = torch.trace(cov)
        scale = torch.clamp(
            torch.abs(trace) / max(d, 1),
            min=torch.tensor(1.0, device=cov.device, dtype=cov.dtype),
        )
        jitter = cov.new_tensor(1e-12) * scale

        for _ in range(8):
            try:
                L = _cholesky_with_cuda_linalg_fallback(cov + jitter * eye)
                self._chol_fallback_jitter_count += 1
                print(
                    f"[torchns][chol] used jitter fallback: count={self._chol_fallback_jitter_count}, "
                    f"jitter={float(jitter.item()):.3e}"
                )
                return L
            except torch.linalg.LinAlgError:
                jitter = jitter * cov.new_tensor(10.0)

        return _cholesky_with_cuda_linalg_fallback(cov)


def make_second_difference_matrix(n_wave: int, device=None, dtype=torch.float32) -> torch.Tensor:
    D = torch.zeros((n_wave, n_wave), device=device, dtype=dtype)
    if n_wave == 1:
        D[0, 0] = 0.0
        return D
    D[0, 0] = -1.0
    D[0, 1] = 1.0
    for i in range(1, n_wave - 1):
        D[i, i - 1] = 1.0
        D[i, i] = -2.0
        D[i, i + 1] = 1.0
    D[n_wave - 1, n_wave - 2] = 1.0
    D[n_wave - 1, n_wave - 1] = -1.0
    return D


def _batched_spd_solve_tensor_cholesky(A: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """Batched SPD solve using basic tensor ops, avoiding torch.linalg kernels."""
    if rhs.ndim == 3 and int(rhs.shape[-1]) == 1:
        rhs = rhs.squeeze(-1)
    if rhs.ndim != 2:
        raise ValueError(f"Expected rhs with shape (B, N) or (B, N, 1), got {tuple(rhs.shape)}")

    batch, n, _ = A.shape
    L = torch.zeros_like(A)
    eps = torch.finfo(A.dtype).eps

    for j in range(n):
        if j == 0:
            diag_correction = A.new_zeros((batch,))
        else:
            diag_correction = torch.sum(L[:, j, :j] * L[:, j, :j], dim=1)
        diag = torch.sqrt(torch.clamp(A[:, j, j] - diag_correction, min=eps))
        L[:, j, j] = diag

        if j + 1 < n:
            if j == 0:
                offdiag_correction = A.new_zeros((batch, n - j - 1))
            else:
                offdiag_correction = torch.sum(
                    L[:, j + 1:, :j] * L[:, j, :j].unsqueeze(1),
                    dim=2,
                )
            L[:, j + 1:, j] = (A[:, j + 1:, j] - offdiag_correction) / diag[:, None]

    y = torch.empty_like(rhs)
    for i in range(n):
        if i == 0:
            correction = rhs.new_zeros((batch,))
        else:
            correction = torch.sum(L[:, i, :i] * y[:, :i], dim=1)
        y[:, i] = (rhs[:, i] - correction) / L[:, i, i]

    x = torch.empty_like(rhs)
    for i in range(n - 1, -1, -1):
        if i + 1 == n:
            correction = rhs.new_zeros((batch,))
        else:
            correction = torch.sum(L[:, i + 1:, i] * x[:, i + 1:], dim=1)
        x[:, i] = (y[:, i] - correction) / L[:, i, i]

    return x


def solve_regularized_flux(weighted_num: torch.Tensor, weighted_den: torch.Tensor, mu: float) -> torch.Tensor:
    _, n_wave = weighted_num.shape
    device = weighted_num.device
    dtype = weighted_num.dtype
    D = make_second_difference_matrix(n_wave, device=device, dtype=dtype)
    DtD = D.T @ D
    eye = torch.eye(n_wave, device=device, dtype=dtype)
    A = torch.diag_embed(weighted_den) + mu * DtD.unsqueeze(0) + 1e-12 * eye.unsqueeze(0)

    if device.type == "cuda" and (_CUDA_SOLVE_CUSTOM_FALLBACK or _prefer_custom_cuda_solve(device)):
        if not _CUDA_SOLVE_CUSTOM_FALLBACK:
            _enable_cuda_solve_custom_fallback()
        flux = _batched_spd_solve_tensor_cholesky(A, weighted_num)
    else:
        try:
            flux = torch.linalg.solve(A, weighted_num.unsqueeze(-1)).squeeze(-1)
        except Exception as error:
            if device.type != "cuda" or not _is_cuda_no_kernel_image_error(error):
                raise
            _enable_cuda_solve_custom_fallback()
            flux = _batched_spd_solve_tensor_cholesky(A, weighted_num)
    return torch.clamp(flux, min=0.0)


def _sympy_matrix_to_complex_tensor(matrix, device):
    rows = matrix.tolist() if hasattr(matrix, "tolist") else matrix
    values = []
    for row in rows:
        values.append([complex(value.evalf()) if hasattr(value, "evalf") else complex(value) for value in row])
    return torch.as_tensor(np.asarray(values, dtype=np.complex64), device=device)


def _sympy_matrix_to_float_tensor(matrix, device):
    rows = matrix.tolist() if hasattr(matrix, "tolist") else matrix
    values = []
    for row in rows:
        values.append([float(value.evalf()) if hasattr(value, "evalf") else float(value) for value in row])
    return torch.as_tensor(np.asarray(values, dtype=np.float32), device=device)


def _matrix_matches_reference(matrix, reference) -> bool:
    if matrix is None or getattr(matrix, "shape", None) != getattr(reference, "shape", None):
        return False

    try:
        same = matrix == reference
    except Exception:
        same = False
    if isinstance(same, bool) and same:
        return True

    equals = getattr(matrix, "equals", None)
    if callable(equals):
        try:
            return equals(reference) is True
        except Exception:
            return False
    return False


def _is_fast_phringe_double_bracewell_instrument(instrument) -> bool:
    return (
            int(getattr(instrument, "number_of_inputs", 0)) == 4
            and int(getattr(instrument, "number_of_outputs", 0)) == 4
            and _matrix_matches_reference(
                getattr(instrument, "array_configuration_matrix", None),
                XArrayConfiguration.acm,
            )
            and _matrix_matches_reference(
                getattr(instrument, "complex_amplitude_transfer_matrix", None),
                DoubleBracewell.catm,
            )
            and _matrix_matches_reference(
                getattr(instrument, "kernels", None),
                DoubleBracewell.kernels,
            )
    )


def make_fast_phringe_double_bracewell_context(setup_resource, device, dtype=torch.float32):
    """Build the cached tensors needed for the direct-torch LIFE Double-Bracewell model."""
    phringe = getattr(setup_resource, "phringe", None)
    if phringe is None:
        return None

    instrument = getattr(phringe, "_instrument", None)
    observation = getattr(phringe, "_observation", None)
    if instrument is None or observation is None:
        return None
    if not _is_fast_phringe_double_bracewell_instrument(instrument):
        return None

    try:
        simulation_times = torch.as_tensor(phringe.simulation_time_steps, dtype=dtype, device=device)
        detector_times = torch.as_tensor(phringe.get_time_steps(), dtype=dtype, device=device)
        wavelengths = torch.as_tensor(phringe.get_wavelength_bin_centers(), dtype=dtype, device=device)
        wavelength_widths = torch.as_tensor(phringe.get_wavelength_bin_widths(), dtype=dtype, device=device)
        field_of_view = torch.as_tensor(phringe.get_field_of_view(), dtype=dtype, device=device)
        nulling_baseline = torch.as_tensor(float(phringe.get_nulling_baseline()), dtype=dtype, device=device)
        modulation_period = torch.as_tensor(float(observation.modulation_period), dtype=dtype, device=device)
        detector_integration_time = torch.as_tensor(
            float(observation.detector_integration_time),
            dtype=dtype,
            device=device,
        )
        catm = _sympy_matrix_to_complex_tensor(instrument.complex_amplitude_transfer_matrix, device=device)
        kernels_matrix = _sympy_matrix_to_float_tensor(instrument.kernels, device=device)
    except Exception:
        return None

    if (
            simulation_times.ndim != 1
            or detector_times.ndim != 1
            or int(simulation_times.shape[0]) != int(detector_times.shape[0])
            or not torch.allclose(simulation_times, detector_times, rtol=0.0, atol=1e-6)
    ):
        return None
    if wavelengths.ndim != 1 or wavelength_widths.shape != wavelengths.shape or field_of_view.shape != wavelengths.shape:
        return None
    if catm.shape != (4, 4) or kernels_matrix.shape[-1] != 4:
        return None

    aperture = torch.as_tensor(instrument.aperture_diameter, dtype=dtype, device=device)
    throughput_qe = torch.as_tensor(
        float(instrument.throughput) * float(instrument.quantum_efficiency),
        dtype=dtype,
        device=device,
    )
    field_amplitude = aperture / 2.0 * torch.sqrt(throughput_qe) * torch.sqrt(
        torch.as_tensor(np.pi, dtype=dtype, device=device)
    )

    angle = 2.0 * np.pi / modulation_period * simulation_times
    cos_angle = torch.cos(angle)
    sin_angle = torch.sin(angle)
    base_x = torch.as_tensor(
        [FAST_PHRINGE_XARRAY_Q, FAST_PHRINGE_XARRAY_Q, -FAST_PHRINGE_XARRAY_Q, -FAST_PHRINGE_XARRAY_Q],
        dtype=dtype,
        device=device,
    )
    base_y = torch.as_tensor([1.0, -1.0, -1.0, 1.0], dtype=dtype, device=device)
    half_baseline = nulling_baseline / 2.0
    array_x = half_baseline * (cos_angle[:, None] * base_x[None, :] - sin_angle[:, None] * base_y[None, :])
    array_y = half_baseline * (sin_angle[:, None] * base_x[None, :] + cos_angle[:, None] * base_y[None, :])

    return {
        "times": simulation_times,
        "wavelengths": wavelengths,
        "wavelength_widths": wavelength_widths,
        "field_of_view": field_of_view,
        "aperture": aperture,
        "field_amplitude": field_amplitude.to(torch.complex64),
        "detector_integration_time": detector_integration_time,
        "catm": catm,
        "kernels": kernels_matrix,
        "array_x": array_x,
        "array_y": array_y,
        "fov_mask_mode": "per_time",
    }


def get_model_counts_fast_double_bracewell(
        fast_ctx,
        x_positions,
        y_positions,
        spectral_energy_distribution,
        kernels=True,
):
    """Vectorized direct-torch planet count model for the LIFE Double-Bracewell design."""
    x_positions = torch.as_tensor(x_positions, dtype=torch.float32, device=fast_ctx["times"].device)
    y_positions = torch.as_tensor(y_positions, dtype=torch.float32, device=fast_ctx["times"].device)
    if x_positions.ndim != 2 or y_positions.ndim != 2 or x_positions.shape != y_positions.shape:
        raise ValueError("x_positions and y_positions must have matching shape (batch, n_time_steps).")
    if int(x_positions.shape[1]) != int(fast_ctx["times"].shape[0]):
        raise ValueError("x_positions and y_positions must match the fast PHRINGE time grid length.")

    wavelengths = fast_ctx["wavelengths"]
    phase = (
            2.0
            * np.pi
            / wavelengths[None, :, None, None]
            * (
                    x_positions[:, None, :, None] * fast_ctx["array_x"][None, None, :, :]
                    + y_positions[:, None, :, None] * fast_ctx["array_y"][None, None, :, :]
            )
    )
    input_fields = fast_ctx["field_amplitude"] * torch.complex(torch.cos(phase), torch.sin(phase))
    output_fields = torch.einsum("ok,bwtk->bowt", fast_ctx["catm"], input_fields)
    outputs = (output_fields.real * output_fields.real + output_fields.imag * output_fields.imag).to(torch.float32)

    taper = torch.exp(
        -(
                np.pi ** 2
                * (x_positions * x_positions + y_positions * y_positions)[:, None, :]
                * fast_ctx["aperture"] ** 2
                / (wavelengths[None, :, None] * wavelengths[None, :, None])
        )
    )
    outputs = outputs * taper[:, None, :, :]

    if kernels:
        outputs = torch.einsum("ko,bowt->bkwt", fast_ctx["kernels"], outputs)

    sed = torch.as_tensor(spectral_energy_distribution, dtype=torch.float32, device=x_positions.device)
    if sed.ndim == 1:
        sed_scale = sed[None, None, :, None]
    elif sed.ndim == 2:
        if int(sed.shape[0]) != int(x_positions.shape[0]):
            raise ValueError("Batched spectral_energy_distribution must match x_positions batch size.")
        sed_scale = sed[:, None, :, None]
    else:
        raise ValueError("spectral_energy_distribution must have shape (n_wavelengths,) or (batch, n_wavelengths).")
    if int(sed_scale.shape[-2]) != int(wavelengths.shape[0]):
        raise ValueError("spectral_energy_distribution must match the number of wavelength bins.")

    outputs = outputs * sed_scale
    outputs = outputs * fast_ctx["detector_integration_time"]
    outputs = outputs * fast_ctx["wavelength_widths"][None, None, :, None]

    fov_mode = fast_ctx.get("fov_mask_mode", "per_time")
    if fov_mode == "per_time":
        in_fov = (
                (torch.abs(x_positions)[:, None, :] <= fast_ctx["field_of_view"][None, :, None] / 2.0)
                & (torch.abs(y_positions)[:, None, :] <= fast_ctx["field_of_view"][None, :, None] / 2.0)
        )
        return outputs * in_fov[:, None, :, :].to(outputs.dtype)
    if fov_mode == "trajectory_max":
        max_abs_position = torch.maximum(
            torch.amax(torch.abs(x_positions), dim=1),
            torch.amax(torch.abs(y_positions), dim=1),
        )
        in_fov = max_abs_position[:, None] <= fast_ctx["field_of_view"][None, :] / 2.0
        return outputs * in_fov[:, None, :, None].to(outputs.dtype)
    raise ValueError(f"Unsupported fast PHRINGE FOV mask mode: {fov_mode}")


def _mean_to_true_anomaly_torch(mean_anomaly: torch.Tensor, eccentricity: torch.Tensor) -> torch.Tensor:
    mean_anomaly = torch.remainder(mean_anomaly, 2.0 * np.pi)
    eccentricity = torch.clamp(eccentricity, 0.0, 0.999999)
    eccentric_anomaly = torch.where(
        eccentricity >= 0.8,
        torch.full_like(mean_anomaly, np.pi),
        mean_anomaly.clone(),
    )
    for _ in range(12):
        residual = eccentric_anomaly - eccentricity * torch.sin(eccentric_anomaly) - mean_anomaly
        jacobian = 1.0 - eccentricity * torch.cos(eccentric_anomaly)
        eccentric_anomaly = eccentric_anomaly - residual / torch.clamp(jacobian, min=1e-12)

    true_anomaly = 2.0 * torch.atan2(
        torch.sqrt(1.0 + eccentricity) * torch.sin(eccentric_anomaly / 2.0),
        torch.sqrt(1.0 - eccentricity) * torch.cos(eccentric_anomaly / 2.0),
    )
    return torch.remainder(true_anomaly, 2.0 * np.pi)


def _get_beta_icdf_table(device, dtype):
    device = torch.device(device)
    key = (str(device), dtype)
    table = _BETA_ICDF_TABLE_CACHE.get(key)
    if table is not None:
        return table

    z_grid_np = np.linspace(
        BETA_ICDF_Z_MIN,
        BETA_ICDF_Z_MAX,
        int(BETA_ICDF_TABLE_SIZE),
        dtype=np.float64,
    )
    u_grid_np = 1.0 / (1.0 + np.exp(-z_grid_np))
    e_grid_np = _beta_icdf_numpy(u_grid_np)
    e_grid_np = np.nan_to_num(e_grid_np, nan=0.0, posinf=1.0, neginf=0.0)
    e_grid_np = np.clip(e_grid_np, 0.0, 1.0)

    e_grid = torch.as_tensor(e_grid_np, dtype=dtype, device=device)
    inv_dz = torch.as_tensor(
        (int(BETA_ICDF_TABLE_SIZE) - 1) / (BETA_ICDF_Z_MAX - BETA_ICDF_Z_MIN),
        dtype=dtype,
        device=device,
    )
    z_min = torch.as_tensor(BETA_ICDF_Z_MIN, dtype=dtype, device=device)
    table = (e_grid, inv_dz, z_min)
    _BETA_ICDF_TABLE_CACHE[key] = table
    return table


def _beta_icdf_numpy(u_grid: np.ndarray) -> np.ndarray:
    # try:
    #     from scipy.special import betaincinv
    #
    #     return np.asarray(
    #         betaincinv(ECC_BETA_ALPHA, ECC_BETA_BETA, u_grid),
    #         dtype=np.float64,
    #     )
    # except ImportError:
    #     pass

    e_min = 1e-8
    e_max = 1.0 - 1e-8
    e_pdf_grid = np.linspace(e_min, e_max, int(BETA_ICDF_TABLE_SIZE), dtype=np.float64)
    log_norm = math.lgamma(ECC_BETA_ALPHA) + math.lgamma(ECC_BETA_BETA) - math.lgamma(
        ECC_BETA_ALPHA + ECC_BETA_BETA
    )
    log_pdf = (
            (ECC_BETA_ALPHA - 1.0) * np.log(e_pdf_grid)
            + (ECC_BETA_BETA - 1.0) * np.log1p(-e_pdf_grid)
            - log_norm
    )
    pdf = np.exp(log_pdf)
    cdf = np.empty_like(e_pdf_grid)
    cdf[0] = 0.0
    cdf[1:] = np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(e_pdf_grid))
    cdf /= cdf[-1]
    cdf = np.maximum.accumulate(cdf)
    return np.interp(u_grid, cdf, e_pdf_grid)


def _beta_icdf_torch_interpolated(u: torch.Tensor) -> torch.Tensor:
    u = torch.as_tensor(u)
    e_grid, inv_dz, z_min = _get_beta_icdf_table(u.device, u.dtype)
    u_clamped = torch.clamp(u, BETA_ICDF_U_MIN, BETA_ICDF_U_MAX)
    z = torch.log(u_clamped) - torch.log1p(-u_clamped)
    pos = (z - z_min) * inv_dz
    idx0 = torch.floor(pos).to(torch.long)
    idx0 = torch.clamp(idx0, 0, int(BETA_ICDF_TABLE_SIZE) - 2)
    frac = pos - idx0.to(dtype=u.dtype)
    return e_grid[idx0] + frac * (e_grid[idx0 + 1] - e_grid[idx0])


def unitcube_to_physical(X_u: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """Transform unit-cube samples to SI orbital elements."""
    X_u = torch.as_tensor(X_u, dtype=torch.float32)
    lower_t = torch.as_tensor(lower, dtype=torch.float32, device=X_u.device)
    upper_t = torch.as_tensor(upper, dtype=torch.float32, device=X_u.device)

    X_phys = lower_t[None, :] + X_u * (upper_t - lower_t)[None, :]

    log_lo = torch.log(lower_t[0])
    log_hi = torch.log(upper_t[0])
    X_phys[:, 0] = torch.exp(log_lo + X_u[:, 0] * (log_hi - log_lo))

    u_e = torch.clamp(X_u[:, 1], BETA_ICDF_U_MIN, BETA_ICDF_U_MAX)
    e_t = _beta_icdf_torch_interpolated(u_e)
    X_phys[:, 1] = torch.clamp(e_t, min=lower_t[1], max=upper_t[1])

    cos_i_lo = torch.cos(upper_t[2])
    cos_i_hi = torch.cos(lower_t[2])
    cos_i = cos_i_lo + X_u[:, 2] * (cos_i_hi - cos_i_lo)
    X_phys[:, 2] = torch.acos(torch.clamp(cos_i, -1.0, 1.0))

    m0 = lower_t[5] + X_u[:, 5] * (upper_t[5] - lower_t[5])
    X_phys[:, 5] = _mean_to_true_anomaly_torch(m0, X_phys[:, 1])
    return X_phys


def unitcube_to_physical_numpy_cpu(
        X_u: torch.Tensor,
        lower: np.ndarray,
        upper: np.ndarray,
        chunk_size: int = POSTERIOR_TRANSFORM_CHUNK_SIZE,
) -> np.ndarray:
    X_u = torch.as_tensor(X_u, dtype=torch.float32, device="cpu")
    chunks = []
    lower_t = torch.as_tensor(lower, dtype=torch.float32, device="cpu")
    upper_t = torch.as_tensor(upper, dtype=torch.float32, device="cpu")
    with torch.no_grad():
        for start in range(0, int(X_u.shape[0]), int(chunk_size)):
            end = min(start + int(chunk_size), int(X_u.shape[0]))
            X_phys_chunk = unitcube_to_physical(X_u[start:end], lower=lower_t, upper=upper_t)
            chunks.append(X_phys_chunk.cpu().numpy().astype(np.float64, copy=False))
    if chunks:
        return np.concatenate(chunks, axis=0)
    return np.empty((0, X_u.shape[1]), dtype=np.float64)


def _get_model_counts_batched_torch_no_perturbations(
        phringe,
        x_positions: torch.Tensor,
        y_positions: torch.Tensor,
        spectral_energy_distribution: torch.Tensor,
        kernels: bool = True,
):
    try:
        return phringe.get_model_counts_batched_torch(
            x_positions=x_positions,
            y_positions=y_positions,
            spectral_energy_distribution=spectral_energy_distribution,
            kernels=kernels,
            perturbations=False,
        )
    except TypeError as exc:
        if "perturbations" not in str(exc):
            raise
        return phringe.get_model_counts_batched_torch(
            x_positions=x_positions,
            y_positions=y_positions,
            spectral_energy_distribution=spectral_energy_distribution,
            kernels=kernels,
        )


def _transform_signal_batch(
        transformation: Callable[[torch.Tensor], torch.Tensor],
        signal_batch: torch.Tensor,
        device: torch.device,
) -> torch.Tensor:
    transformed_ok = False
    try:
        transformed = transformation(signal_batch.clone())
        transformed = torch.as_tensor(transformed, dtype=torch.float32, device=device)
        if transformed.shape == signal_batch.shape:
            signal_batch = transformed
            transformed_ok = True
    except Exception:
        transformed_ok = False

    if transformed_ok:
        return signal_batch

    per_item = torch.empty_like(signal_batch)
    for b in range(signal_batch.shape[0]):
        transformed = transformation(signal_batch[b].clone())
        transformed = torch.as_tensor(transformed, dtype=torch.float32, device=device)
        if transformed.shape != signal_batch[b].shape:
            raise ValueError(
                "Transformation returned candidate signal with shape "
                f"{tuple(transformed.shape)}, expected {tuple(signal_batch[b].shape)}."
            )
        per_item[b] = transformed
    return per_item





def _normalize_log_weights(log_weights: np.ndarray) -> tuple[np.ndarray, float]:

    max_logw = float(np.max(log_weights))
    weights = np.exp(log_weights - max_logw)
    weights_sum = float(np.sum(weights))
    log_evidence = max_logw + np.log(weights_sum)
    return weights / weights_sum, float(log_evidence)


class _ProfiledSpectrumOrbitLogLikelihood:
    """Torch likelihood for one setup and one continuous observation."""

    def __init__(
            self,
            setup_resource: BaseResource,
            data: torch.Tensor,
            transformation: Callable[[torch.Tensor], torch.Tensor],
            flux_regularization: float,
            device: torch.device,
            eval_count_print_every: int,
            use_fast_phringe_double_bracewell: bool,
    ):
        self.setup_resource = setup_resource
        self.phringe = setup_resource.phringe
        self.observation = getattr(self.phringe, "_observation", None)
        self.scene = getattr(setup_resource, "scene", None)
        if self.scene is None:
            self.scene = getattr(self.phringe, "_scene", None)
        if self.scene is None:
            raise ValueError("Setup resource does not contain a scene.")
        if getattr(self.scene, "star", None) is None:
            raise ValueError("Setup scene does not contain a host star.")
        self.star = self.scene.star
        self.data = torch.as_tensor(data, dtype=torch.float32, device=device)
        self.transformation = transformation
        self.flux_regularization = float(flux_regularization)
        self.device = self.data.device
        self.forward_model_evaluations = 0
        self.eval_count_print_every = int(eval_count_print_every)
        self.next_eval_print = self.eval_count_print_every if self.eval_count_print_every > 0 else None
        self.fast_phringe_context = None


        planets = getattr(self.scene, "planets", [])
        if not planets:
            raise ValueError("Setup scene does not contain any planets.")

        self.planet = planets[0]
        self.planet_mass = float(self.planet.mass)

        self.time_grid = torch.as_tensor(self.phringe.get_time_steps(), dtype=torch.float32, device=self.device)
        if self.time_grid.ndim != 1 or int(self.time_grid.shape[0]) != int(self.data.shape[2]):
            raise ValueError(
                "Observation time grid from setup.phringe.get_time_steps() must be one-dimensional "
                "and match the data time axis."
            )

        self.wavelengths = torch.as_tensor(
            self.phringe.get_wavelength_bin_centers(),
            dtype=torch.float32,
            device=self.device,
        )

        data_by_wave = self.data.permute(1, 0, 2).reshape(self.data.shape[1], -1)
        data_variance = torch.var(data_by_wave, dim=1)
        self.inv_data_variance = 1.0 / torch.clamp(data_variance, min=1e-30)

        self.star_mass = float(self.star.mass)
        self.host_distance = float(self.star.distance)
        self.propagation_mu = torch.tensor(
            float(const.G.value * self.star_mass),
            dtype=torch.float32,
            device=self.device,
        )
        if use_fast_phringe_double_bracewell:
            self.fast_phringe_context = make_fast_phringe_double_bracewell_context(
                setup_resource=self.setup_resource,
                device=self.device,
                dtype=torch.float32,
            )
            if self.fast_phringe_context is None:
                print("[fast-phringe] unsupported setup; using original PHRINGE path")
            else:
                print("[fast-phringe] enabled Double-Bracewell direct torch model")
        self.chi2_truth = self._compute_truth_residual_chi2()
        self.logl_shift = 0.5 * self.chi2_truth

    def _propagate_orbits(self, theta_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sma = theta_batch[:, 0]
        ecc = theta_batch[:, 1]
        inc = theta_batch[:, 2]
        raan = theta_batch[:, 3]
        argp = theta_batch[:, 4]
        nu0 = theta_batch[:, 5]

        nu_half = nu0 / 2.0
        beta = torch.sqrt(torch.clamp((1.0 - ecc) / (1.0 + ecc), min=0.0))
        eccentric_anomaly_0 = 2.0 * torch.atan2(beta * torch.sin(nu_half), torch.cos(nu_half))
        mean_motion = torch.sqrt(self.propagation_mu / torch.clamp(sma ** 3, min=1e-18))
        mean_anomaly_0 = eccentric_anomaly_0 - ecc * torch.sin(eccentric_anomaly_0)
        mean_anomaly_t = torch.remainder(
            mean_anomaly_0[:, None] + mean_motion[:, None] * self.time_grid[None, :],
            2.0 * np.pi,
        )
        ecc_t = ecc[:, None].expand(-1, int(self.time_grid.shape[0]))

        eccentric_anomaly_t = torch.where(
            ecc_t >= 0.8,
            torch.full_like(mean_anomaly_t, np.pi),
            mean_anomaly_t.clone(),
        )
        for _ in range(12):
            residual = eccentric_anomaly_t - ecc_t * torch.sin(eccentric_anomaly_t) - mean_anomaly_t
            jacobian = 1.0 - ecc_t * torch.cos(eccentric_anomaly_t)
            eccentric_anomaly_t = eccentric_anomaly_t - residual / torch.clamp(jacobian, min=1e-12)

        true_anomaly_t = 2.0 * torch.atan2(
            torch.sqrt(1.0 + ecc_t) * torch.sin(eccentric_anomaly_t / 2.0),
            torch.sqrt(1.0 - ecc_t) * torch.cos(eccentric_anomaly_t / 2.0),
        )
        radius_t = sma[:, None] * (1.0 - ecc_t * torch.cos(eccentric_anomaly_t))
        phase = argp[:, None] + true_anomaly_t
        cos_phase = torch.cos(phase)
        sin_phase = torch.sin(phase)
        cos_raan = torch.cos(raan)[:, None]
        sin_raan = torch.sin(raan)[:, None]
        cos_inc = torch.cos(inc)[:, None]

        x_positions = radius_t * (cos_raan * cos_phase - sin_raan * sin_phase * cos_inc) / self.host_distance
        y_positions = radius_t * (sin_raan * cos_phase + cos_raan * sin_phase * cos_inc) / self.host_distance
        return x_positions, y_positions

    def _build_candidate_signals(self, theta_batch: torch.Tensor) -> torch.Tensor:
        sed = torch.ones(int(self.wavelengths.shape[0]), dtype=torch.float32, device=self.device)
        x_positions, y_positions = self._propagate_orbits(theta_batch)
        if self.fast_phringe_context is not None:
            signal_batch = get_model_counts_fast_double_bracewell(
                fast_ctx=self.fast_phringe_context,
                x_positions=x_positions,
                y_positions=y_positions,
                spectral_energy_distribution=sed,
                kernels=True,
            )
        elif hasattr(self.phringe, "get_model_counts_batched_torch"):
            signal_batch = _get_model_counts_batched_torch_no_perturbations(
                self.phringe,
                x_positions=x_positions,
                y_positions=y_positions,
                spectral_energy_distribution=sed,
                kernels=True,
            )
        else:
            signal_batch = self._build_candidate_signals_with_numpy_api(theta_batch, sed)
        signal_batch = torch.as_tensor(signal_batch, dtype=torch.float32, device=self.device)
        expected_shape = (
            int(theta_batch.shape[0]),
            int(self.data.shape[0]),
            int(self.data.shape[1]),
            int(self.data.shape[2]),
        )
        if tuple(signal_batch.shape) != expected_shape:
            raise ValueError(
                "PHRINGE candidate signal shape "
                f"{tuple(signal_batch.shape)} does not match expected shape {expected_shape}."
            )
        return _transform_signal_batch(self.transformation, signal_batch, self.device)

    def _build_candidate_signals_with_numpy_api(
            self,
            theta_batch: torch.Tensor,
            sed: torch.Tensor,
    ) -> torch.Tensor:
        """Use PHRINGE's scalar orbital forward model when no batched torch API exists."""
        sed_np = sed.detach().cpu().numpy().astype(np.float32, copy=False)
        theta_np = theta_batch.detach().cpu().numpy().astype(np.float64, copy=False)
        signals = []
        for theta in theta_np:
            signal = self.phringe.get_model_counts(
                spectral_energy_distribution=sed_np,
                kernels=True,
                semi_major_axis=float(theta[0]),
                eccentricity=float(theta[1]),
                inclination=float(theta[2]),
                raan=float(theta[3]),
                argument_of_periapsis=float(theta[4]),
                true_anomaly=float(theta[5]),
                host_star_distance=self.host_distance,
                host_star_mass=self.star_mass,
                planet_mass=self.planet_mass,
            )
            signal = np.asarray(signal, dtype=np.float32)
            expected_shape = (
                int(self.data.shape[0]),
                int(self.data.shape[1]),
                int(self.data.shape[2]),
            )
            if tuple(signal.shape) != expected_shape:
                raise ValueError(
                    "PHRINGE candidate signal shape "
                    f"{tuple(signal.shape)} does not match expected shape {expected_shape}."
                )
            signals.append(torch.as_tensor(signal, dtype=torch.float32, device=self.device))

        if not signals:
            return torch.empty((0, *self.data.shape), dtype=torch.float32, device=self.device)
        return torch.stack(signals, dim=0)

    @torch.no_grad()
    def _compute_truth_residual_chi2(self) -> float:
        theta_truth = torch.tensor(
            [[
                float(self.planet.semi_major_axis),
                float(self.planet.eccentricity),
                float(self.planet.inclination),
                float(self.planet.raan),
                float(self.planet.argument_of_periapsis),
                float(self.planet.true_anomaly),
            ]],
            dtype=torch.float32,
            device=self.device,
        )
        signal_batch = self._build_candidate_signals(theta_truth)
        weighted_num = torch.einsum("kwt,bkwt,w->bw", self.data, signal_batch, self.inv_data_variance)
        weighted_den = torch.einsum("bkwt,bkwt,w->bw", signal_batch, signal_batch, self.inv_data_variance)
        den_flux = torch.clamp(torch.nan_to_num(weighted_den, nan=1.0, posinf=1.0, neginf=1.0), min=1e-30)
        flux = solve_regularized_flux(weighted_num, den_flux, mu=self.flux_regularization)
        template_scaled = signal_batch * flux[:, None, :, None]
        residual = self.data.unsqueeze(0) - template_scaled
        chi2_truth = float(torch.nansum(residual ** 2, dim=(1, 2, 3))[0].item())
        return chi2_truth

    @torch.no_grad()
    def __call__(self, theta_batch: torch.Tensor) -> torch.Tensor:
        theta_batch = torch.as_tensor(theta_batch, dtype=torch.float32, device=self.device)
        if theta_batch.ndim != 2 or int(theta_batch.shape[1]) != 6:
            raise ValueError(f"Expected physical theta batch shape (B, 6), got {tuple(theta_batch.shape)}")

        bsz = int(theta_batch.shape[0])
        if bsz == 0:
            return torch.empty((0,), dtype=torch.float32, device=self.device)

        self.forward_model_evaluations += bsz
        if self.next_eval_print is not None:
            while self.forward_model_evaluations >= self.next_eval_print:
                print(f"[torchns] forward model evaluations: {self.forward_model_evaluations}")
                self.next_eval_print += self.eval_count_print_every

        signal_batch = self._build_candidate_signals(theta_batch)
        weighted_num = torch.einsum("kwt,bkwt,w->bw", self.data, signal_batch, self.inv_data_variance)
        weighted_den = torch.einsum("bkwt,bkwt,w->bw", signal_batch, signal_batch, self.inv_data_variance)
        den_flux = torch.clamp(torch.nan_to_num(weighted_den, nan=1.0, posinf=1.0, neginf=1.0), min=1e-30)
        flux = solve_regularized_flux(weighted_num, den_flux, mu=self.flux_regularization)
        model = signal_batch * flux[:, None, :, None]
        residual = self.data.unsqueeze(0) - model
        chi2 = torch.nansum(residual ** 2, dim=(1, 2, 3))
        return -0.5 * chi2 + self.logl_shift


class BayesianInferenceModule(BaseModule):
    """Bayesian orbital inference for one planet in one continuous observation.

    This module samples six orbital parameters, ordered as semimajor axis in
    meters, eccentricity, inclination in radians, longitude of ascending node in
    radians, argument of periapsis in radians, and initial true anomaly in
    radians. Candidate signals are generated with PHRINGE for the full
    observation time grid and the planet spectrum is fitted as a profiled
    nuisance parameter for each candidate orbit.

    Parameters
    ----------
    n_setup_in : str
        Name of the setup resource.
    n_data_in : str
        Name of the data resource. If n_transformation_in is supplied this data
        must already be in that transformed space.
    n_inference_out : str
        Name of the output inference result resource.
    sma_lower : float
        Lower semimajor-axis prior bound in meters. Must be strictly positive.
    sma_upper : float
        Upper semimajor-axis prior bound in meters. Must exceed sma_lower.
    n_transformation_in : str, optional
        Name of the transformation resource to apply exactly once to each raw
        candidate signal before comparison with n_data_in.
    eccentricity_upper : float
        Upper eccentricity clip for the implemented beta prior.
    flux_regularization : float
        Second-difference Tikhonov regularization strength for the profiled
        spectrum solve.
    num_live : int
        Number of live points.
    max_steps : int
        Maximum number of nested-sampling outer iterations.
    num_batch_samples : int
        Number of live points used as replacement seeds per outer iteration.
    samples_per_slice : int
        Number of slice proposals per constrained sampling step.
    num_steps : int
        Number of successful slice steps required for a replacement proposal.
    epsilon : float
        Evidence-rest stopping tolerance.
    likelihood_batch_size : int
        Maximum likelihood batch size. Use 0 or None to disable chunking.
    use_fast_phringe_double_bracewell : bool
        If True, try the direct torch LIFE Double-Bracewell count model and
        fall back to the generic PHRINGE path when the setup is incompatible.
    progress_update_interval : int
        Progress update cadence in nested-sampling outer iterations.
    eval_count_print_every : int
        Print forward-model evaluation count every N evaluations. Defaults to
        0, which disables those extra lines.
    show_progress : bool
        If True, show the nested-sampling tqdm progress bar.
    """

    def __init__(
            self,
            n_setup_in: str,
            n_data_in: str,
            n_inference_out: str,
            sma_lower: float,
            sma_upper: float,
            n_transformation_in: str = None,
            eccentricity_upper: float = ECC_MAX,
            flux_regularization: float = 1e-6,
            num_live: int = 128,
            max_steps: int = 1000,
            num_batch_samples: int = 256,
            samples_per_slice: int = 1024,
            num_steps: int = 1000,
            epsilon: float = 1e-6,
            likelihood_batch_size: int = 16384,
            use_fast_phringe_double_bracewell: bool = True,
            progress_update_interval: int = 1,
            eval_count_print_every: int = 0,
            show_progress: bool = True,
    ):
        """Constructor method."""
        super().__init__()
        self.n_setup_in = n_setup_in
        self.n_data_in = n_data_in
        self.n_transformation_in = n_transformation_in
        self.n_inference_out = n_inference_out
        self.sma_lower = float(sma_lower)
        self.sma_upper = float(sma_upper)
        self.eccentricity_upper = float(eccentricity_upper)
        self.flux_regularization = float(flux_regularization)
        self.num_live = int(num_live)
        self.max_steps = int(max_steps)
        self.num_batch_samples = int(num_batch_samples)
        self.samples_per_slice = int(samples_per_slice)
        self.num_steps = int(num_steps)
        self.epsilon = float(epsilon)
        self.likelihood_batch_size = 0 if likelihood_batch_size is None else int(likelihood_batch_size)
        self.use_fast_phringe_double_bracewell = bool(use_fast_phringe_double_bracewell)
        self.progress_update_interval = int(progress_update_interval)
        self.eval_count_print_every = int(eval_count_print_every)
        self.show_progress = bool(show_progress)

        self._validate_constructor_args()

    def _validate_constructor_args(self):
        if self.sma_lower <= 0.0:
            raise ValueError("sma_lower must be strictly positive and in meters.")
        if self.sma_upper <= self.sma_lower:
            raise ValueError("sma_upper must be greater than sma_lower.")
        if not (0.0 < self.eccentricity_upper < 1.0):
            raise ValueError("eccentricity_upper must be in the open interval (0, 1).")
        if self.flux_regularization < 0.0:
            raise ValueError("flux_regularization must be non-negative.")
        if self.num_live < 8:
            raise ValueError("num_live must be at least 8 for the six-dimensional sampler.")
        if self.max_steps < 0:
            raise ValueError("max_steps must be non-negative.")
        if self.num_batch_samples <= 0:
            raise ValueError("num_batch_samples must be strictly positive.")
        if self.samples_per_slice <= 0:
            raise ValueError("samples_per_slice must be strictly positive.")
        if self.num_steps <= 0:
            raise ValueError("num_steps must be strictly positive.")
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be strictly positive.")
        if self.likelihood_batch_size < 0:
            raise ValueError("likelihood_batch_size must be non-negative.")
        if self.progress_update_interval <= 0:
            raise ValueError("progress_update_interval must be strictly positive.")

    def _device(self):
        if torch.cuda.is_available() and self.gpu_index is not None:
            return int(self.gpu_index)
        if self.device is not None:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _prior_bounds(self, device: torch.device) -> tuple[np.ndarray, np.ndarray, torch.Tensor, torch.Tensor]:
        lower = np.array(
            [
                self.sma_lower,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            dtype=np.float32,
        )
        upper = np.array(
            [
                self.sma_upper,
                self.eccentricity_upper,
                np.pi,
                2.0 * np.pi,
                2.0 * np.pi,
                2.0 * np.pi,
            ],
            dtype=np.float32,
        )
        return (
            lower,
            upper,
            torch.as_tensor(lower, dtype=torch.float32, device=device),
            torch.as_tensor(upper, dtype=torch.float32, device=device),
        )

    def run(self, pipeline_resources: list[BaseResource | ResourceCollection]) -> tuple[InferenceResultResource]:
        """Run orbital nested sampling and return a weighted posterior resource."""
        device = self._device()
        if self.seed is not None:
            torch.manual_seed(int(self.seed))
            np.random.seed(int(self.seed))
        if torch.cuda.is_available() and not (isinstance(device, torch.device) and device.type == "cpu"):
            torch.cuda.manual_seed_all(int(self.seed)) if self.seed is not None else None

        console = Console()
        with console.status("Running Bayesian orbital inference...", spinner="dots"):
            r_setup_in = self.get_resource_from_name(self.n_setup_in)
            data_in = self.get_resource_from_name(self.n_data_in).get_data()
            r_transformation_in = (
                self.get_resource_from_name(self.n_transformation_in)
                if self.n_transformation_in
                else None
            )
            transformation = r_transformation_in.transformation if r_transformation_in else lambda x: x
            data_torch = torch.as_tensor(data_in, dtype=torch.float32, device=device)

            loglike_phys = _ProfiledSpectrumOrbitLogLikelihood(
                setup_resource=r_setup_in,
                data=data_torch,
                transformation=transformation,
                flux_regularization=self.flux_regularization,
                device=device,
                eval_count_print_every=self.eval_count_print_every,
                use_fast_phringe_double_bracewell=self.use_fast_phringe_double_bracewell,
            )
            print(f"torch version: {torch.__version__}")
            print(f"torch cuda available: {torch.cuda.is_available()}")
            print(f"torch cuda device count: {torch.cuda.device_count()}")
            print(f"torch cuda runtime version: {torch.version.cuda}")
            if torch.cuda.is_available():
                print(f"torch selected cuda device: {device}")
                print(f"torch selected cuda name: {torch.cuda.get_device_name(device)}")
            else:
                print(f"Using torch device: {device}")
            proposal_batch_per_slice = int(self.num_batch_samples) * int(self.samples_per_slice)
            proposals_per_outer_iteration = proposal_batch_per_slice * int(self.num_steps)
            print(f"torchns proposal batch per slice step = {proposal_batch_per_slice}")
            print(f"torchns proposal evaluations per outer iteration = {proposals_per_outer_iteration}")
            print(
                "torchns likelihood internal chunk size = "
                f"{self.likelihood_batch_size if self.likelihood_batch_size > 0 else 'disabled'}"
            )
            lower_np, upper_np, lower_t, upper_t = self._prior_bounds(device)
            X_init_u = torch.rand((self.num_live, 6), dtype=torch.float32, device=device)

            def logl_u(X_u):
                X_u = torch.as_tensor(X_u, dtype=torch.float32, device=device)
                if int(X_u.shape[0]) == 0:
                    return torch.empty((0,), dtype=torch.float32, device=device)
                if self.likelihood_batch_size > 0 and int(X_u.shape[0]) > self.likelihood_batch_size:
                    out = torch.empty((int(X_u.shape[0]),), dtype=torch.float32, device=device)
                    for start in range(0, int(X_u.shape[0]), self.likelihood_batch_size):
                        end = min(start + self.likelihood_batch_size, int(X_u.shape[0]))
                        X_phys_chunk = unitcube_to_physical(X_u[start:end], lower=lower_t, upper=upper_t)
                        out[start:end] = loglike_phys(X_phys_chunk)
                    return out
                X_phys = unitcube_to_physical(X_u, lower=lower_t, upper=upper_t)
                return loglike_phys(X_phys)

            sampler = JitteredNestedSampler(
                X_init=X_init_u,
                bound="unitcube",
                progress_update_interval=self.progress_update_interval,
                show_progress=self.show_progress,
            )

            t0 = time.time()
            sampler.nested_sampling(
                logl_fn=logl_u,
                max_steps=self.max_steps,
                num_batch_samples=self.num_batch_samples,
                epsilon=self.epsilon,
                samples_per_slice=self.samples_per_slice,
                num_steps=self.num_steps,
                progress_update_interval=self.progress_update_interval,
            )
            elapsed = time.time() - t0

            sample_u_t = torch.cat((sampler.samples_X, sampler.live_X), dim=0)
            sample_logl_t = torch.cat((sampler.samples_logl, sampler.live_logl), dim=0)
            sample_logwt_t = torch.cat((sampler.samples_logwt, sampler.live_logwt), dim=0)
            sample_u = sample_u_t.numpy().astype(np.float32, copy=False)
            log_likelihoods = sample_logl_t.double().numpy().astype(np.float64, copy=False)
            log_likelihoods_unshifted = log_likelihoods - float(loglike_phys.logl_shift)
            log_weights = sample_logwt_t.double().numpy().astype(np.float64, copy=False)
            posterior_weights, log_evidence_shifted = _normalize_log_weights(log_weights)
            log_evidence = (
                log_evidence_shifted - float(loglike_phys.logl_shift)
                if np.isfinite(log_evidence_shifted)
                else float("nan")
            )
            posterior_weight_ess = (
                float(1.0 / np.sum(posterior_weights * posterior_weights))
                if posterior_weights.size > 0
                else 0.0
            )
            samples_phys = unitcube_to_physical_numpy_cpu(sample_u_t, lower=lower_np, upper=upper_np)

            sampler_info = {
                "n_live": self.num_live,
                "sampler_backend": TORCHNS_BACKEND,
                "max_steps": self.max_steps,
                "num_outer_iterations": int(sampler.num_outer_iterations),
                "num_dead_points": int(sampler.num_replacements),
                "num_live_points": int(sampler.live_X.shape[0]),
                "stopped_at_iteration_limit": bool(sampler.stopped_at_iteration_limit),
                "stop_reason": sampler.stop_reason,
                "log_evidence": log_evidence,
                "log_evidence_shifted": log_evidence_shifted,
                "log_evidence_likelihood_convention": LIKELIHOOD_CONVENTION,
                "likelihood_offset": float(loglike_phys.logl_shift),
                "truth_residual_chi2": float(loglike_phys.chi2_truth),
                "posterior_weight_ess": posterior_weight_ess,
                "forward_model_evaluations": int(loglike_phys.forward_model_evaluations),
                "elapsed_seconds": float(elapsed),
                "likelihood_batch_size": self.likelihood_batch_size,
                "fast_phringe_double_bracewell_requested": self.use_fast_phringe_double_bracewell,
                "fast_phringe_double_bracewell_enabled": loglike_phys.fast_phringe_context is not None,
                "show_progress": self.show_progress,
                "flux_regularization": self.flux_regularization,
                "sma_lower_m": self.sma_lower,
                "sma_upper_m": self.sma_upper,
                "eccentricity_upper": self.eccentricity_upper,
                "n_time_steps": int(loglike_phys.time_grid.shape[0]),
                "nu0_epoch": NU0_EPOCH_DESCRIPTION,
            }

            r_inference_out = InferenceResultResource(
                name=self.n_inference_out,
                samples=samples_phys,
                posterior_weights=posterior_weights,
                log_likelihoods=log_likelihoods,
                log_likelihoods_unshifted=log_likelihoods_unshifted,
                unit_cube_samples=sample_u,
                log_weights=log_weights,
                sampler_info=sampler_info,
            )

        print("Done")
        return r_inference_out,


BayesianInference = BayesianInferenceModule
