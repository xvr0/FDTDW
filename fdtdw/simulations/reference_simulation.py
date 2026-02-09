import warp as wp
import numpy as np
from typing import Callable, Tuple, Union, Any

from ._graph_tape import GraphTape
from .. import kernels as kn
from ._base_simulation import BaseSimulation
import functools


def use_device(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        with wp.ScopedDevice(self._device):
            return func(self, *args, **kwargs)

    return wrapper


class ReferenceSimulation(BaseSimulation):

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
        R_0: float = 10e-8,
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
            STEPS, NX, NY, NZ, S, PML_THICKNESS, dx, eta, R_0, DEVICE, boundaries,kernel=kernel
        )

    def _allocate_specific(self):
        pass

    @use_device
    def _record_forward(self) -> Any:

        tape = GraphTape(self._device, max_nodes=500)
        with tape:
            wp.launch(kn.reset_integer, dim=1, inputs=[self._idx_time])
            for c in range(self._STEPS):

                wp.launch(
                    self.update_e,
                    dim=self._shape_grid,
                    inputs=[self._state, self._properties],
                )
                for wrapper in self._sources:
                    wp.launch(
                        wrapper.inject_esource,
                        dim=wrapper.shape,
                        inputs=[self._state, wrapper.source, self._idx_time],
                    )
                wp.launch(
                    self.update_h,
                    dim=self._shape_grid,
                    inputs=[self._state, self._properties],
                )
                for wrapper in self._sources:
                    wp.launch(
                        wrapper.inject_hsource,
                        dim=wrapper.shape,
                        inputs=[self._state, wrapper.source, self._idx_time],
                    )
                for wrapper in self._detectors:
                    wp.launch(
                        wrapper.save_detector,
                        dim=wrapper.shape,
                        inputs=[self._state, wrapper.detector, self._idx_time, 0],
                    )
                wp.launch(kn.inc_integer, dim=1, inputs=[self._idx_time, 1])
        return tape

    def __str__(self) -> str:
        return "todo"
