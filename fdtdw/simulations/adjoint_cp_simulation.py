import warp as wp
import numpy as np
from typing import Any
from ._graph_tape import GraphTape
from .. import kernels as kn
from .. import postprocessing as pp
import functools
from ._adjoint_simulation import AdjointSimulation


def use_device(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        with wp.ScopedDevice(self._device):
            return func(self, *args, **kwargs)

    return wrapper


class AdjointCpSimulation(AdjointSimulation):

    def __init__(
        self,
        NX: int = 128,
        NY: int = 128,
        NZ: int = 128,
        S: float = 0.5,
        STEPS: int = -1,
        BUFFERSIZE: int = -1,
        CHECKPOINTS: int = -1,
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
        if STEPS == -1:
            STEPS = BUFFERSIZE * CHECKPOINTS
        if CHECKPOINTS == -1 and BUFFERSIZE == -1:
            BUFFERSIZE = int((3 * STEPS) ** 0.5)
            CHECKPOINTS = (STEPS + BUFFERSIZE - 1) // BUFFERSIZE
        if BUFFERSIZE == -1:
            BUFFERSIZE = (STEPS + CHECKPOINTS - 1) // CHECKPOINTS
        if CHECKPOINTS == -1:
            CHECKPOINTS = (STEPS + BUFFERSIZE - 1) // BUFFERSIZE

        self._BUFFERSIZE = BUFFERSIZE
        self._CHECKPOINTS = CHECKPOINTS
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

    @use_device
    def _allocate_specific(self):

        self._shape_cp = (self._CHECKPOINTS, self._NX, self._NY, self._NZ)

        self._states = kn.EMStates()
        self._states.Ex = wp.empty(self._shape_cp, dtype=float)
        self._states.Ey = wp.empty(self._shape_cp, dtype=float)
        self._states.Ez = wp.empty(self._shape_cp, dtype=float)
        self._states.Hx = wp.empty(self._shape_cp, dtype=float)
        self._states.Hy = wp.empty(self._shape_cp, dtype=float)
        self._states.Hz = wp.empty(self._shape_cp, dtype=float)

        self._states.psi_ex_y = wp.empty(self._shape_cp, dtype=float)
        self._states.psi_ex_z = wp.empty(self._shape_cp, dtype=float)
        self._states.psi_ey_x = wp.empty(self._shape_cp, dtype=float)
        self._states.psi_ey_z = wp.empty(self._shape_cp, dtype=float)
        self._states.psi_ez_x = wp.empty(self._shape_cp, dtype=float)
        self._states.psi_ez_y = wp.empty(self._shape_cp, dtype=float)
        self._states.psi_hx_y = wp.empty(self._shape_cp, dtype=float)
        self._states.psi_hx_z = wp.empty(self._shape_cp, dtype=float)
        self._states.psi_hy_x = wp.empty(self._shape_cp, dtype=float)
        self._states.psi_hy_z = wp.empty(self._shape_cp, dtype=float)
        self._states.psi_hz_x = wp.empty(self._shape_cp, dtype=float)
        self._states.psi_hz_y = wp.empty(self._shape_cp, dtype=float)

        self._buf_shape = (self._BUFFERSIZE, self._NX, self._NY, self._NZ)

        self._buffer = kn.FieldsBuffer()
        self._buffer.Ex = wp.empty(self._buf_shape, dtype=float)
        self._buffer.Ey = wp.empty(self._buf_shape, dtype=float)
        self._buffer.Ez = wp.empty(self._buf_shape, dtype=float)
        self._buffer.Hx = wp.empty(self._buf_shape, dtype=float)
        self._buffer.Hy = wp.empty(self._buf_shape, dtype=float)
        self._buffer.Hz = wp.empty(self._buf_shape, dtype=float)

        self._idx_ckpt = wp.empty(1, dtype=int)
        self._idx_buffer = wp.zeros(1, dtype=int)

    @property
    def states(self) -> dict[str, np.ndarray]:
        return {
            "Ex": self._states.Ex.numpy(),
            "Ey": self._states.Ey.numpy(),
            "Ez": self._states.Ez.numpy(),
            "Hx": self._states.Hx.numpy(),
            "Hy": self._states.Hy.numpy(),
            "Hz": self._states.Hz.numpy(),
            "psi_ex_y": self._states.psi_ex_y.numpy(),
            "psi_ex_z": self._states.psi_ex_z.numpy(),
            "psi_ey_x": self._states.psi_ey_x.numpy(),
            "psi_ey_z": self._states.psi_ey_z.numpy(),
            "psi_ez_x": self._states.psi_ez_x.numpy(),
            "psi_ez_y": self._states.psi_ez_y.numpy(),
            "psi_hx_y": self._states.psi_hx_y.numpy(),
            "psi_hx_z": self._states.psi_hx_z.numpy(),
            "psi_hy_x": self._states.psi_hy_x.numpy(),
            "psi_hy_z": self._states.psi_hy_z.numpy(),
            "psi_hz_x": self._states.psi_hz_x.numpy(),
            "psi_hz_y": self._states.psi_hz_y.numpy(),
        }

    @use_device
    def _record_forward(self) -> Any:

        tape = GraphTape(self._device, max_nodes=500)

        with tape:
            wp.launch(kn.reset_integer, dim=1, inputs=[self._idx_time])
            wp.launch(kn.reset_integer, dim=1, inputs=[self._idx_ckpt])
            for c in range(self._CHECKPOINTS):

                wp.launch(
                    kn.save_checkpoint,
                    dim=self._shape_grid,
                    inputs=[self._state, self._states, self._idx_ckpt],
                )
                for b in range(self._BUFFERSIZE):
                    for wrapper in self._detectors:
                        wp.launch(
                            wrapper.save_detector,
                            dim=wrapper.shape,
                            inputs=[self._state, wrapper.detector, self._idx_time, 0],
                        )

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
                    wp.launch(kn.inc_integer, dim=1, inputs=[self._idx_time, 1])

                wp.launch(kn.inc_integer, dim=1, inputs=[self._idx_ckpt, 1])

        return tape

    @use_device
    def render_checkpoint_video(
        self,
        filename: str = "checkpoints",
        limit: float = 0.3,
        field="Ey",
        slice_idx: tuple = (64, 64, 64),
        scale: int = 4,
        fps: int = 30,
    ) -> None:
        match field:
            case "Ex":
                field_arr = self._states.Ex
            case "Ey":
                field_arr = self._states.Ey
            case "Ez":
                field_arr = self._states.Ez
            case "Hx":
                field_arr = self._states.Hx
            case "Hy":
                field_arr = self._states.Hy
            case "Hz":
                field_arr = self._states.Hz
            case _:
                field_arr = self._states.Ey

        slice_x_array = wp.empty((self._CHECKPOINTS, self._NY, self._NZ))
        slice_y_array = wp.empty((self._CHECKPOINTS, self._NX, self._NZ))
        slice_z_array = wp.empty((self._CHECKPOINTS, self._NX, self._NY))

        wp.launch(
            pp.slice,
            dim=(
                self._CHECKPOINTS,
                self._NY,
                self._NZ,
            ),
            inputs=[field_arr, slice_x_array, slice_idx[0], 1],
        )
        wp.launch(
            pp.slice,
            dim=(
                self._CHECKPOINTS,
                self._NX,
                self._NZ,
            ),
            inputs=[field_arr, slice_y_array, slice_idx[1], 2],
        )
        wp.launch(
            pp.slice,
            dim=(
                self._CHECKPOINTS,
                self._NX,
                self._NY,
            ),
            inputs=[field_arr, slice_z_array, slice_idx[2], 3],
        )
        pp.render_array(
            slice_x_array, filename="x_" + filename, limit=limit, scale=scale, fps=fps
        )
        pp.render_array(
            slice_y_array, filename="y_" + filename, limit=limit, scale=scale, fps=fps
        )
        pp.render_array(
            slice_z_array, filename="z_" + filename, limit=limit, scale=scale, fps=fps
        )

    @use_device
    def _record_adjoint(self) -> Any:

        wp.synchronize()

        tape = GraphTape(self._device, max_nodes=500)
        with tape:

            for b in range(self._CHECKPOINTS):
                wp.launch(kn.dec_integer, dim=1, inputs=[self._idx_ckpt, 1])
                wp.launch(
                    kn.dec_integer, dim=1, inputs=[self._idx_time, self._BUFFERSIZE]
                )

                wp.launch(
                    kn.load_checkpoint,
                    dim=self._shape_grid,
                    inputs=[self._state, self._states, self._idx_ckpt],
                )

                wp.launch(kn.reset_integer, dim=1, inputs=[self._idx_buffer])
                for c in range(self._BUFFERSIZE):

                    wp.launch(
                        kn.save_field,
                        dim=self._shape_grid,
                        inputs=[self._state, self._buffer, self._idx_buffer],
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

                    wp.launch(kn.inc_integer, dim=1, inputs=[self._idx_time, 1])
                    wp.launch(kn.inc_integer, dim=1, inputs=[self._idx_buffer, 1])

                for c in range(self._BUFFERSIZE):
                    wp.launch(kn.dec_integer, dim=1, inputs=[self._idx_time, 1])
                    wp.launch(kn.dec_integer, dim=1, inputs=[self._idx_buffer, 1])

                    for wrapper in self._detectors:
                        wp.launch(
                            wrapper.inject_esource_ad,
                            dim=wrapper.shape,
                            inputs=[
                                self._state_adj,
                                wrapper.source_adj,
                                self._idx_time,
                            ],
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
                            inputs=[
                                self._state_adj,
                                wrapper.source_adj,
                                self._idx_time,
                            ],
                        )
                    wp.launch(
                        self.update_h,
                        dim=self._shape_grid,
                        inputs=[self._state_adj, self._properties],
                    )
                    wp.launch(
                        kn.calc_grad,
                        dim=self._shape_grid,
                        inputs=[
                            self._grads,
                            self._buffer,
                            self._state_adj,
                            self._idx_buffer,
                        ],
                    )

        return tape

    @use_device
    def generate_adjoint_source(self) -> None:

        print("default FOM: total throughput (no mode)")
        for wrapper in self._detectors:
            wp.launch(
                kn.flip_h,
                wrapper.shape_record,
                inputs=[wrapper.detector, wrapper.source_adj],
            )
        wp.synchronize()

    def export_checkpoint_to_vti(
        self, filename: str = "checkp.vti", pos: int = 0
    ) -> None:

        self._idx_buffer = wp.zeros(1, dtype=int, device=self._device)
        wp.launch(kn.inc_integer, dim=1, inputs=[self._idx_buffer, pos])
        wp.launch(
            kn.load_checkpoint,
            dim=self._shape_grid,
            inputs=[self._state, self._states, self._idx_buffer],
            device=self._device,
        )
        pp.export_vti(
            filename=filename,
            fields={
                "E": (self._state.Ex, self._state.Ey, self._state.Ez),
                "H": (self._state.Hx, self._state.Hy, self._state.Hz),
            },
        )

    def __str__(self) -> str:
        return (
            f"Grid: {self._NX}x{self._NY}x{self._NZ}, pml {self._PML_THICKNESS} \n"
            + f"Courant: {self._S}, {self._CHECKPOINTS} checkpoints on {self._CHECKPOINTS*self._BUFFERSIZE} steps total \n"
            + f"statepointer at: {self._idx_time.numpy()[0]}"
        )
