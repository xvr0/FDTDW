import warp as wp
import numpy as np
from typing import Callable, Tuple, Union, Any
from abc import ABC, abstractmethod

from fdtdw.kernels.structs32 import TEMStates, TEMStates_full
from ._base_simulation import BaseSimulation
from .. import kernels as kn
from .. import postprocessing as pp
import functools
from ._graph_tape import GraphTape


def use_device(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        with wp.ScopedDevice(self._device):
            return func(self, *args, **kwargs)

    return wrapper


def ensure_synced(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        msg = f"--> [Auto-Sync] Syncing data to VRAM before {func.__name__}..."
        self._sync(msg=msg)
        return func(self, *args, **kwargs)

    return wrapper


class AdjointSimulation(BaseSimulation):

    def __init__(
        self,
        STEPS: int = 1000,
        NX: int = 128,
        NY: int = 128,
        NZ: int = 128,
        S: float = 0.5,
        PML_THICKNESS: int = 20,
        dx: float = 1.0,
        eta: float = 1.0,
        DEVICE: str = "cuda:0",
        boundaries: dict = {
            "xmin": "PML",
            "xmax": "PML",
            "ymin": "PML",
            "ymax": "PML",
            "zmin": "PML",
            "zmax": "PML",
        },
        kernel: str = "warp",
    ):

        super().__init__(
            STEPS=STEPS, 
            NX=NX, 
            NY=NY, 
            NZ=NZ, 
            S=S, 
            PML_THICKNESS=PML_THICKNESS, 
            dx=dx, 
            eta=eta, 
            DEVICE=DEVICE, 
            boundaries=boundaries,
            kernel=kernel
        )

    @abstractmethod
    def _allocate_specific(self):
        pass

    @use_device
    def _allocate_common(self):
        super()._allocate_common()

        self._state_adj = kn.EMState()
        self._state_adj.Ex = wp.zeros(self._shape_grid, dtype=wp.float32)
        self._state_adj.Ey = wp.zeros(self._shape_grid, dtype=wp.float32)
        self._state_adj.Ez = wp.zeros(self._shape_grid, dtype=wp.float32)
        self._state_adj.Hx = wp.zeros(self._shape_grid, dtype=wp.float32)
        self._state_adj.Hy = wp.zeros(self._shape_grid, dtype=wp.float32)
        self._state_adj.Hz = wp.zeros(self._shape_grid, dtype=wp.float32)

        self._state_adj.psi_ex_y = wp.zeros(self._shape_grid, dtype=wp.float32)
        self._state_adj.psi_ex_z = wp.zeros(self._shape_grid, dtype=wp.float32)
        self._state_adj.psi_ey_x = wp.zeros(self._shape_grid, dtype=wp.float32)
        self._state_adj.psi_ey_z = wp.zeros(self._shape_grid, dtype=wp.float32)
        self._state_adj.psi_ez_x = wp.zeros(self._shape_grid, dtype=wp.float32)
        self._state_adj.psi_ez_y = wp.zeros(self._shape_grid, dtype=wp.float32)
        self._state_adj.psi_hx_y = wp.zeros(self._shape_grid, dtype=wp.float32)
        self._state_adj.psi_hx_z = wp.zeros(self._shape_grid, dtype=wp.float32)
        self._state_adj.psi_hy_x = wp.zeros(self._shape_grid, dtype=wp.float32)
        self._state_adj.psi_hy_z = wp.zeros(self._shape_grid, dtype=wp.float32)
        self._state_adj.psi_hz_x = wp.zeros(self._shape_grid, dtype=wp.float32)
        self._state_adj.psi_hz_y = wp.zeros(self._shape_grid, dtype=wp.float32)

        self._grads = kn.Gradients()

        self._grads.grad_CE_x = wp.zeros(self._shape_grid, dtype=wp.float32)
        self._grads.grad_CE_y = wp.zeros(self._shape_grid, dtype=wp.float32)
        self._grads.grad_CE_z = wp.zeros(self._shape_grid, dtype=wp.float32)

        self._grads.grad_CH_x = wp.zeros(self._shape_grid, dtype=wp.float32)
        self._grads.grad_CH_y = wp.zeros(self._shape_grid, dtype=wp.float32)
        self._grads.grad_CH_z = wp.zeros(self._shape_grid, dtype=wp.float32)

    @property
    def grads(self) -> dict[str, np.ndarray]:
        return {
            "grad_CE_x": self._grads.grad_CE_x.numpy(),
            "grad_CE_y": self._grads.grad_CE_y.numpy(),
            "grad_CE_z": self._grads.grad_CE_z.numpy(),
            "grad_CH_x": self._grads.grad_CH_x.numpy(),
            "grad_CH_y": self._grads.grad_CH_y.numpy(),
            "grad_CH_z": self._grads.grad_CH_z.numpy(),
        }

    @property
    @use_device
    def source_adj(self) -> list[dict[str, Any]]:
        return [
            (
                {
                    "Eu": wrapper.source_adj.Eu.numpy(),
                    "Ev": wrapper.source_adj.Ev.numpy(),
                    "Hu": wrapper.source_adj.Hu.numpy(),
                    "Hv": wrapper.source_adj.Hv.numpy(),
                    "OFFSETX": wrapper.source_adj.OFFSETX,
                    "OFFSETY": wrapper.source_adj.OFFSETY,
                    "OFFSETZ": wrapper.source_adj.OFFSETZ,
                    "plane": wrapper.plane,
                    "shape": wrapper.shape,
                }
                if not wrapper.full
                else {
                    "Eu": wrapper.source_adj.Eu.numpy(),
                    "Ev": wrapper.source_adj.Ev.numpy(),
                    "Hu": wrapper.source_adj.Hu.numpy(),
                    "Hv": wrapper.source_adj.Hv.numpy(),
                    "Hu_n": wrapper.source_adj.Hu_n.numpy(),
                    "Hv_n": wrapper.source_adj.Hv_n.numpy(),
                    "OFFSETX": wrapper.source_adj.OFFSETX,
                    "OFFSETY": wrapper.source_adj.OFFSETY,
                    "OFFSETZ": wrapper.source_adj.OFFSETZ,
                    "plane": wrapper.plane,
                    "shape": wrapper.shape,
                }
            )
            for wrapper in self._detectors
        ]

    @source_adj.setter
    @use_device
    def source_adj(self, data: list[dict[str, Any]]):

        if len(data) != len(self._detectors):
            raise ValueError(f"Detectors must be mapped bijective to Adjoint Sources.")

        for wrapper, item in zip(self._detectors, data):

            wp.copy(
                wrapper.source_adj.Eu,
                wp.array(
                    item["Eu"],
                    dtype=wp.float32,
                ),
            )
            wp.copy(
                wrapper.source_adj.Ev,
                wp.array(
                    item["Ev"],
                    dtype=wp.float32,
                ),
            )
            wp.copy(
                wrapper.source_adj.Hu,
                wp.array(
                    item["Hu"],
                    dtype=wp.float32,
                ),
            )
            wp.copy(
                wrapper.source_adj.Hv,
                wp.array(
                    item["Hv"],
                    dtype=wp.float32,
                ),
            )

            wrapper.source_adj.OFFSETX = item["OFFSETX"]
            wrapper.source_adj.OFFSETY = item["OFFSETY"]
            wrapper.source_adj.OFFSETZ = item["OFFSETZ"]

    @use_device
    def init_detector(
        self,
        shape: Tuple[int, int],
        OFFSETX: int,
        OFFSETY: int,
        OFFSETZ: int,
        plane: str,
        full: bool = False,
    ):
        super().init_detector(
            shape=shape,
            OFFSETX=OFFSETX,
            OFFSETY=OFFSETY,
            OFFSETZ=OFFSETZ,
            plane=plane,
            full=full,
        )
        wrapper = self._detectors[-1]
        if full:
            source_adj = TEMStates_full()
        else:
            source_adj = TEMStates()
        source_adj.Eu = wp.zeros(wrapper.shape_record, dtype=wp.float32)
        source_adj.Ev = wp.zeros(wrapper.shape_record, dtype=wp.float32)
        source_adj.Hu = wp.zeros(wrapper.shape_record, dtype=wp.float32)
        source_adj.Hv = wp.zeros(wrapper.shape_record, dtype=wp.float32)
        if full:
            source_adj.Hu_n = wp.zeros(wrapper.shape_record, dtype=wp.float32)
            source_adj.Hv_n = wp.zeros(wrapper.shape_record, dtype=wp.float32)
        source_adj.OFFSETX, source_adj.OFFSETY, source_adj.OFFSETZ = (
            OFFSETX,
            OFFSETY,
            OFFSETZ,
        )
        wrapper.source_adj = source_adj

    @use_device
    def init_adj_source(
        self,
        eufunc: Callable[[np.ndarray], np.ndarray],
        euprofile: np.ndarray,
        evfunc: Callable[[np.ndarray], np.ndarray],
        evprofile: np.ndarray,
        hufunc: Callable[[np.ndarray], np.ndarray],
        huprofile: np.ndarray,
        hvfunc: Callable[[np.ndarray], np.ndarray],
        hvprofile: np.ndarray,
        idx=0,
    ):
        detector_wrapper = self._detectors[idx]
        NU, NV = euprofile.shape
        shape_source = (NU, NV)
        if shape_source != detector_wrapper.shape:
            print("error dims")

        t = np.arange(self._STEPS)

        def get_analytic_signal(x):
            """(x + j*Hilbert(x))"""
            N = len(x)
            Xf = np.fft.fft(x)
            h = np.zeros(N)
            if N % 2 == 0:
                h[0] = h[N // 2] = 1
                h[1 : N // 2] = 2
            else:
                h[0] = 1
                h[1 : (N + 1) // 2] = 2
            return np.fft.ifft(Xf * h)

        def process_field(func, profile):
            sig_real = func(t)

            if np.iscomplexobj(sig_real):
                sig_complex = sig_real
            else:
                sig_complex = get_analytic_signal(sig_real)
            import matplotlib.pyplot as plt

            plt.plot(abs(sig_complex))
            mixed_complex = sig_complex[:, None, None] * profile[None, :, :]

            return wp.array(mixed_complex.real.astype(np.float32))

        source = detector_wrapper.source_adj
        source.Eu = process_field(eufunc, euprofile)
        source.Ev = process_field(evfunc, evprofile)
        source.Hu = process_field(hufunc, huprofile)
        source.Hv = process_field(hvfunc, hvprofile)

    def record_graphs(self):
        super().record_graphs()
        self._adjoint_graph_tape = self._record_adjoint()

    def _set_kernels(self):
        super()._set_kernels()
        for wrapper in self._detectors:
            match wrapper.plane:
                case "xy":
                    wrapper.inject_esource_ad = kn.inject_esources_xy
                    wrapper.inject_hsource_ad = kn.inject_hsources_xy
                    # wrapper.map = kn.map_xy
                case "xz":
                    wrapper.inject_esource_ad = kn.inject_esources_xz
                    wrapper.inject_hsource_ad = kn.inject_hsources_xz
                    # wrapper.map = kn.map_xz
                case "yz":
                    wrapper.inject_esource_ad = kn.inject_esources_yz
                    wrapper.inject_hsource_ad = kn.inject_hsources_yz
                    # wrapper.map = kn.map_yz

    @abstractmethod
    def _record_forward(self) -> Any:
        pass

    @use_device
    def render_adj_source_video(
        self,
        idx=0,
        field: str = "Eu",
        filename: str = "adj_source.mp4",
        limit: float = 0.3,
        scale: int = 4,
        fps: int = 30,
    ) -> None:
        match field:
            case "Eu":
                field_arr = self._detectors[idx].source_adj.Eu
            case "Ev":
                field_arr = self._detectors[idx].source_adj.Ev
            case "Hu":
                field_arr = self._detectors[idx].source_adj.Hu
            case "Hv":
                field_arr = self._detectors[idx].source_adj.Hv
            case _:
                field_arr = self._detectors[idx].source_adj.Ev

        pp.render_array(field_arr, filename=filename, limit=limit, scale=scale, fps=fps)

    @abstractmethod
    def _record_adjoint(self) -> GraphTape:
        pass

    @abstractmethod
    def generate_adjoint_source(self) -> None:
        pass

    @ensure_synced
    def launch_adjoint(self) -> None:
        self._adjoint_graph_tape()

    @ensure_synced
    def export_gradients(
        self,
        filename: str = "grads.vti",
        roi: Any = np.s_[:, :, :],
    ) -> None:
        pp.export_vti(
            filename=filename,
            fields={
                "Gradient_CE": (
                    self._grads.grad_CE_x,
                    self._grads.grad_CE_y,
                    self._grads.grad_CE_z,
                ),
                "Gradient_CH": (
                    self._grads.grad_CH_x,
                    self._grads.grad_CH_y,
                    self._grads.grad_CH_z,
                ),
            },
            roi=roi,
        )

    @abstractmethod
    def __str__(self) -> str:
        pass
