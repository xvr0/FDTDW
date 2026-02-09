import warp as wp
import numpy as np
from typing import Callable, Any, Optional

from ._graph_tape import GraphTape

from .. import kernels as kn
from .. import postprocessing as pp
from ._adjoint_simulation import AdjointSimulation
import functools


def use_device(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        with wp.ScopedDevice(self._device):
            return func(self, *args, **kwargs)

    return wrapper


class AdjointDftSimulation(AdjointSimulation):

    def __init__(
        self,
        NX: int = 128,
        NY: int = 128,
        NZ: int = 128,
        S: float = 0.5,
        STEPS: int = 500,
        FREQUENCIES: np.ndarray = np.arange(0.025, 0.026, 0.001),
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

        self._freqs = wp.array(FREQUENCIES, dtype=wp.float32, device=DEVICE)
        self._freqs_shape = FREQUENCIES.shape
        self._w_freqs = wp.ones(self._freqs_shape)
        super().__init__(
            STEPS, NX, NY, NZ, S, PML_THICKNESS, dx, eta, DEVICE, boundaries, kernel=kernel
        )

    @use_device
    def _allocate_specific(self):

        self._shape_dfts = (self._freqs_shape[0], self._NX, self._NY, self._NZ)
        self._DFTs = kn.DFTs()
        self._DFTs.Ex = wp.zeros(self._shape_dfts, dtype=wp.vec2)
        self._DFTs.Ey = wp.zeros(self._shape_dfts, dtype=wp.vec2)
        self._DFTs.Ez = wp.zeros(self._shape_dfts, dtype=wp.vec2)
        self._DFTs.Hx = wp.zeros(self._shape_dfts, dtype=wp.vec2)
        self._DFTs.Hy = wp.zeros(self._shape_dfts, dtype=wp.vec2)
        self._DFTs.Hz = wp.zeros(self._shape_dfts, dtype=wp.vec2)

        self._DFTs_adj = kn.DFTs()
        self._DFTs_adj.Ex = wp.zeros(self._shape_dfts, dtype=wp.vec2)
        self._DFTs_adj.Ey = wp.zeros(self._shape_dfts, dtype=wp.vec2)
        self._DFTs_adj.Ez = wp.zeros(self._shape_dfts, dtype=wp.vec2)
        self._DFTs_adj.Hx = wp.zeros(self._shape_dfts, dtype=wp.vec2)
        self._DFTs_adj.Hy = wp.zeros(self._shape_dfts, dtype=wp.vec2)
        self._DFTs_adj.Hz = wp.zeros(self._shape_dfts, dtype=wp.vec2)

        self._shape_cdft = (self._STEPS, self._freqs_shape[0])
        self._cdft = wp.empty(self._shape_cdft, dtype=wp.vec2)

        wp.launch(
            kn.compute_dft_coeffs,
            dim=self._shape_cdft,
            inputs=[self._freqs, self._S, self._cdft],
        )

    @property
    def DFT(self) -> dict[str, np.ndarray]:
        return {
            "Ex": self._DFTs.Ex.numpy(),
            "Ey": self._DFTs.Ey.numpy(),
            "Ez": self._DFTs.Ez.numpy(),
            "Hx": self._DFTs.Hx.numpy(),
            "Hy": self._DFTs.Hy.numpy(),
            "Hz": self._DFTs.Hz.numpy(),
        }

    @use_device
    def _record_forward(self) -> Any:

        tape = GraphTape(self._device, max_nodes=500)
        with tape:
            wp.launch(kn.reset_integer, dim=1, inputs=[self._idx_time])
            for c in range(self._STEPS):

                for wrapper in self._detectors:
                    wp.launch(
                        wrapper.save_detector,
                        dim=wrapper.shape,
                        inputs=[self._state, wrapper.detector, self._idx_time, 0],
                    )
                for wrapper in self._sources:
                    wp.launch(
                        wrapper.inject_esource,
                        dim=wrapper.shape,
                        inputs=[self._state, wrapper.source, self._idx_time],
                    )

                wp.launch(
                    self.update_e,
                    dim=self._shape_grid,
                    inputs=[self._state, self._properties],
                )
                for wrapper in self._sources:
                    wp.launch(
                        wrapper.inject_hsource,
                        dim=wrapper.shape,
                        inputs=[self._state, wrapper.source, self._idx_time],
                    )
                wp.launch(
                    self.update_h,
                    dim=self._shape_grid,
                    inputs=[self._state, self._properties],
                )
                wp.launch(
                    kn.accumulate_dft,
                    dim=self._shape_grid,
                    inputs=[self._cdft, self._state, self._DFTs, self._idx_time],
                )
                wp.launch(kn.inc_integer, dim=1, inputs=[self._idx_time, 1])

        return tape

    @use_device
    def _record_adjoint(self) -> Any:

        wp.synchronize()

        tape = GraphTape(self._device, max_nodes=500)
        with tape:
            wp.launch(kn.reset_integer, dim=1, inputs=[self._idx_time])
            for c in range(self._STEPS):

                for wrapper in self._detectors:
                    wp.launch(
                        wrapper.inject_esource_ad,
                        dim=wrapper.shape,
                        inputs=[self._state_adj, wrapper.source_adj, self._idx_time],
                    )
                wp.launch(
                    self.update_e,
                    dim=self._shape_grid,
                    inputs=[self._state_adj, self._properties],
                )
                for wrapper in self._detectors:
                    wp.launch(
                        wrapper.inject_hsource_ad,
                        dim=wrapper.shape,
                        inputs=[self._state_adj, wrapper.source_adj, self._idx_time],
                    )
                wp.launch(
                    self.update_h,
                    dim=self._shape_grid,
                    inputs=[self._state_adj, self._properties],
                )
                wp.launch(
                    kn.accumulate_dft,
                    dim=self._shape_grid,
                    inputs=[
                        self._cdft,
                        self._state_adj,
                        self._DFTs_adj,
                        self._idx_time,
                    ],
                )
                wp.launch(kn.inc_integer, dim=1, inputs=[self._idx_time, 1])

            wp.launch(
                kn.compute_gradients,
                dim=self._shape_dfts,
                inputs=[self._DFTs, self._DFTs_adj, self._w_freqs, self._grads],
            )
        return tape

    @use_device
    def recompute_gradients(self, w_freqs: Optional[np.ndarray] = None):
        if w_freqs is None:
            w_freqs_wp = self._w_freqs
        w_freqs_wp = wp.array(w_freqs)
        wp.launch(
            kn.compute_gradients,
            dim=self._shape_dfts,
            inputs=[self._DFTs, self._DFTs_adj, w_freqs_wp, self._grads],
        )

    @use_device
    def compute_pdf(self, signal: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        t = np.arange(self._STEPS)
        signal_np = signal(t)
        signal_wp = wp.array(signal_np, dtype=wp.float32)
        pdf = wp.empty(self._freqs_shape)
        wp.launch(
            kn.compute_dft_pdf,
            dim=self._freqs_shape,
            inputs=[signal_wp, self._cdft, pdf],
        )
        return pdf.numpy()

    @property
    @use_device
    def dft_weights(self):
        return self._w_freqs.numpy()

    @dft_weights.setter
    @use_device
    def dft_weights(self, arr: np.ndarray):
        wp.copy(
            self._w_freqs,
            wp.array(
                arr,
                dtype=wp.float32,
            ),
        )

    @use_device
    def generate_adjoint_source(self) -> None:
        for i in range(len(self._detectors)):
            wp.launch(
                kn.flip_axis_h,
                self._detectors[i].shape_record,
                inputs=[self._sources[i].source, self._detectors[i].source_adj],
            )

        wp.synchronize()

    def export_dft_to_vti(self, pos: int, filename: str = "dft_field.vti") -> None:

        wp.launch(
            kn.load_dft_magnitude,
            dim=self._shape_grid,
            inputs=[self._DFTs, self._state, pos],
            device=self._device,
        )
        pp.export_vti(
            filename="fw" + filename,
            fields={
                "E": (self._state.Ex, self._state.Ey, self._state.Ez),
                "H": (self._state.Hx, self._state.Hy, self._state.Hz),
            },
        )

        wp.launch(
            kn.load_dft_magnitude,
            dim=self._shape_grid,
            inputs=[self._DFTs_adj, self._state, pos],
            device=self._device,
        )
        pp.export_vti(
            filename="adj" + filename,
            fields={
                "E": (self._state.Ex, self._state.Ey, self._state.Ez),
                "H": (self._state.Hx, self._state.Hy, self._state.Hz),
            },
        )

    def __str__(self) -> str:
        return super().__str__()
