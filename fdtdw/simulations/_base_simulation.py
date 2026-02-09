import warp as wp
import numpy as np
from typing import Callable, Tuple, Union, Any, Optional
from abc import ABC, abstractmethod

from fdtdw.kernels.sources32 import inject_esources_xy, inject_hsources_xy
from fdtdw.kernels.structs32 import TEMStates, TEMStates_full, TEMDFT

from ._graph_tape import GraphTape
from .. import kernels as kn
from .. import postprocessing as pp
import functools
from dataclasses import dataclass


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


@dataclass
class detector_wrapper:
    detector: Any
    source_adj: Any
    plane: str
    shape: tuple
    shape_record: tuple
    save_detector: Any = None
    inject_hsource_ad: Any = None
    inject_esource_ad: Any = None
    map: Any = None
    full: bool = False


@dataclass
class source_wrapper:
    source: Any
    plane: str
    shape: tuple
    shape_shedule: tuple
    inject_esource: Any = None
    inject_hsource: Any = None


def to_wp(arr):
    return wp.array(arr, dtype=wp.float32)


class BaseSimulation(ABC):

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
        kernel: str = "warp"
    ):
        self._STEPS = STEPS
        self._NX = NX
        self._NY = NY
        self._NZ = NZ
        self._S = S
        self._dx = dx
        self._eta = eta
        self._PML_THICKNESS = PML_THICKNESS
        self._device = DEVICE
        self._boundaries = boundaries
        self._R_0 = R_0
        self.kernel = kernel
        self._save_detector: Any = None
        self._forward_graph_tape: Any = None

        self._sources: list[source_wrapper] = []
        self._detectors: list[detector_wrapper] = []

        wp.init()
        self._allocate_common()
        self._allocate_specific()

    @abstractmethod
    def _allocate_specific(self):
        pass

    @use_device
    def _allocate_common(self):
        self._state = kn.EMState()
        self._shape_grid = (self._NX, self._NY, self._NZ)
        self._state.Ex = wp.zeros(self._shape_grid, dtype=wp.float32)
        self._state.Ey = wp.zeros(self._shape_grid, dtype=wp.float32)
        self._state.Ez = wp.zeros(self._shape_grid, dtype=wp.float32)
        self._state.Hx = wp.zeros(self._shape_grid, dtype=wp.float32)
        self._state.Hy = wp.zeros(self._shape_grid, dtype=wp.float32)
        self._state.Hz = wp.zeros(self._shape_grid, dtype=wp.float32)

        self._state.psi_ex_y = wp.zeros(self._shape_grid, dtype=wp.float32)
        self._state.psi_ex_z = wp.zeros(self._shape_grid, dtype=wp.float32)
        self._state.psi_ey_x = wp.zeros(self._shape_grid, dtype=wp.float32)
        self._state.psi_ey_z = wp.zeros(self._shape_grid, dtype=wp.float32)
        self._state.psi_ez_x = wp.zeros(self._shape_grid, dtype=wp.float32)
        self._state.psi_ez_y = wp.zeros(self._shape_grid, dtype=wp.float32)
        self._state.psi_hx_y = wp.zeros(self._shape_grid, dtype=wp.float32)
        self._state.psi_hx_z = wp.zeros(self._shape_grid, dtype=wp.float32)
        self._state.psi_hy_x = wp.zeros(self._shape_grid, dtype=wp.float32)
        self._state.psi_hy_z = wp.zeros(self._shape_grid, dtype=wp.float32)
        self._state.psi_hz_x = wp.zeros(self._shape_grid, dtype=wp.float32)
        self._state.psi_hz_y = wp.zeros(self._shape_grid, dtype=wp.float32)

        self._properties = kn.Properties()

        self._set_pml()

        self._ceb_np = np.full(self._shape_grid, self._S, dtype=np.float32)
        self._chb_np = np.full(self._shape_grid, self._S, dtype=np.float32)
        self._cea_np = np.full(self._shape_grid, 1.0, dtype=np.float32)
        self._cha_np = np.full(self._shape_grid, 1.0, dtype=np.float32)
        self._PEC_np = np.zeros(self._shape_grid, dtype=np.int32)
        if self._boundaries["xmin"] == "PEC":
            self._PEC_np[0, :, :] = 1
        if self._boundaries["xmax"] == "PEC":
            self._PEC_np[-1, :, :] = 1
        if self._boundaries["ymin"] == "PEC":
            self._PEC_np[:, 0, :] = 1
        if self._boundaries["ymax"] == "PEC":
            self._PEC_np[:, -1, :] = 1
        if self._boundaries["zmin"] == "PEC":
            self._PEC_np[:, :, 0] = 1
        if self._boundaries["zmax"] == "PEC":
            self._PEC_np[:, :, -1] = 1
        self._PMC_np = np.zeros(self._shape_grid, dtype=np.int32)
        if self._boundaries["xmin"] == "PMC":
            self._PMC_np[0, :, :] = 1
        if self._boundaries["xmax"] == "PMC":
            self._PMC_np[-1, :, :] = 1
        if self._boundaries["ymin"] == "PMC":
            self._PMC_np[:, 0, :] = 1
        if self._boundaries["ymax"] == "PMC":
            self._PMC_np[:, -1, :] = 1
        if self._boundaries["zmin"] == "PMC":
            self._PMC_np[:, :, 0] = 1
        if self._boundaries["zmax"] == "PMC":
            self._PMC_np[:, :, -1] = 1
        self._dirty = False
        is_pml_np = np.zeros((self._NX, self._NY, self._NZ), dtype=bool)

        t = self._PML_THICKNESS

        if self._boundaries.get("xmin") == "PML":
            is_pml_np[:t, :, :] = True
        if self._boundaries.get("xmax") == "PML":
            is_pml_np[-t:, :, :] = True

        if self._boundaries.get("ymin") == "PML":
            is_pml_np[:, :t, :] = True
        if self._boundaries.get("ymax") == "PML":
            is_pml_np[:, -t:, :] = True

        if self._boundaries.get("zmin") == "PML":
            is_pml_np[:, :, :t] = True
        if self._boundaries.get("zmax") == "PML":
            is_pml_np[:, :, -t:] = True

        self._properties.is_PML = wp.from_numpy(is_pml_np, dtype=bool)
        self._properties.CEA_X = wp.empty(self._shape_grid, dtype=float)
        self._properties.CEA_Y = wp.empty(self._shape_grid, dtype=float)
        self._properties.CEA_Z = wp.empty(self._shape_grid, dtype=float)
        self._properties.CHA_X = wp.empty(self._shape_grid, dtype=float)
        self._properties.CHA_Y = wp.empty(self._shape_grid, dtype=float)
        self._properties.CHA_Z = wp.empty(self._shape_grid, dtype=float)

        self._properties.CEB_X = wp.empty(self._shape_grid, dtype=float)
        self._properties.CEB_Y = wp.empty(self._shape_grid, dtype=float)
        self._properties.CEB_Z = wp.empty(self._shape_grid, dtype=float)
        self._properties.CHB_X = wp.empty(self._shape_grid, dtype=float)
        self._properties.CHB_Y = wp.empty(self._shape_grid, dtype=float)
        self._properties.CHB_Z = wp.empty(self._shape_grid, dtype=float)

        self._idx_time = wp.empty(1, dtype=int)

    @use_device
    def _sync(self, msg: str):

        if self._dirty == False:
            return

        print(msg)

        def get_fill_factors_edges_hard(mask):
            grid = (mask > 0).astype(np.float32)
            padded = np.pad(grid, ((1, 0), (1, 0), (1, 0)), mode="constant")

            s_c = slice(1, None)
            s_p = slice(0, -1)

            stack_x = np.stack(
                [
                    padded[s_c, s_c, s_c],
                    padded[s_c, s_p, s_c],
                    padded[s_c, s_c, s_p],
                    padded[s_c, s_p, s_p],
                ],
                axis=0,
            )
            fill_x = np.max(stack_x, axis=0)

            stack_y = np.stack(
                [
                    padded[s_c, s_c, s_c],
                    padded[s_p, s_c, s_c],
                    padded[s_c, s_c, s_p],
                    padded[s_p, s_c, s_p],
                ],
                axis=0,
            )
            fill_y = np.max(stack_y, axis=0)

            stack_z = np.stack(
                [
                    padded[s_c, s_c, s_c],
                    padded[s_p, s_c, s_c],
                    padded[s_c, s_p, s_c],
                    padded[s_p, s_p, s_c],
                ],
                axis=0,
            )
            fill_z = np.max(stack_z, axis=0)

            return fill_x, fill_y, fill_z

        import numpy as np

        def get_fill_factors_planes_hard(mask: np.ndarray):

            grid = (mask > 0).astype(np.float32)

            padded = np.pad(
                grid, ((1, 0), (1, 0), (1, 0)), mode="constant", constant_values=0
            )

            s_c = slice(1, None)
            s_p = slice(0, -1)

            fill_x = np.minimum(padded[s_c, s_c, s_c], padded[s_p, s_c, s_c])

            fill_y = np.minimum(padded[s_c, s_c, s_c], padded[s_c, s_p, s_c])

            fill_z = np.minimum(padded[s_c, s_c, s_c], padded[s_c, s_c, s_p])

            return fill_x, fill_y, fill_z

        fex_np, fey_np, fez_np = get_fill_factors_edges_hard(self._PEC_np)
        fhx_np, fhy_np, fhz_np = get_fill_factors_planes_hard(self._PMC_np)

        def to_wp(arr):
            return wp.array(arr, dtype=wp.float32)

        fex = to_wp(fex_np)
        fey = to_wp(fey_np)
        fez = to_wp(fez_np)
        fhx = to_wp(fhx_np)
        fhy = to_wp(fhy_np)
        fhz = to_wp(fhz_np)
        CEA = to_wp(self._cea_np)
        CEB = to_wp(self._ceb_np)
        CHA = to_wp(self._cha_np)
        CHB = to_wp(self._chb_np)

        wp.launch(
            kernel=kn.set_material_properties,
            dim=self._shape_grid,
            inputs=[CEB, CEA, CHB, CHA, fex, fey, fez, fhx, fhy, fhz, self._properties],
            device=self._device,
        )

        self._dirty = False

    @property
    @ensure_synced
    def ceb(self) -> np.ndarray:
        return self._ceb_np

    @property
    @ensure_synced
    def chb(self) -> np.ndarray:
        return self._chb_np

    @ceb.setter
    def ceb(self, arr: np.ndarray):
        self._ceb_np = arr
        self._dirty = True

    @chb.setter
    def chb(self, arr: np.ndarray):
        self._chb_np = arr
        self._dirty = True

    @property
    @ensure_synced
    def pec(self) -> np.ndarray:
        return self._PEC_np

    @property
    @ensure_synced
    def pmc(self) -> np.ndarray:
        return self._PMC_np

    @pec.setter
    def pec(self, arr: np.ndarray):
        self._PEC_np = arr
        self._dirty = True

    @pmc.setter
    def pmc(self, arr: np.ndarray):
        self._PMC_np = arr
        self._dirty = True

    @property
    @ensure_synced
    def cea(self) -> np.ndarray:
        return self._cea_np

    @property
    @ensure_synced
    def cha(self) -> np.ndarray:
        return self._cha_np

    @cea.setter
    def cea(self, arr: np.ndarray):
        self._cea_np = arr
        self._dirty = True

    @cha.setter
    def cha(self, arr: np.ndarray):
        self._cha_np = arr
        self._dirty = True

    @property
    def state(self) -> dict[str, np.ndarray]:
        return {
            "Ex": self._state.Ex.numpy(),
            "Ey": self._state.Ey.numpy(),
            "Ez": self._state.Ez.numpy(),
            "Hx": self._state.Hx.numpy(),
            "Hy": self._state.Hy.numpy(),
            "Hz": self._state.Hz.numpy(),
            "psi_ex_y": self._state.psi_ex_y.numpy(),
            "psi_ex_z": self._state.psi_ex_z.numpy(),
            "psi_ey_x": self._state.psi_ey_x.numpy(),
            "psi_ey_z": self._state.psi_ey_z.numpy(),
            "psi_ez_x": self._state.psi_ez_x.numpy(),
            "psi_ez_y": self._state.psi_ez_y.numpy(),
            "psi_hx_y": self._state.psi_hx_y.numpy(),
            "psi_hx_z": self._state.psi_hx_z.numpy(),
            "psi_hy_x": self._state.psi_hy_x.numpy(),
            "psi_hy_z": self._state.psi_hy_z.numpy(),
            "psi_hz_x": self._state.psi_hz_x.numpy(),
            "psi_hz_y": self._state.psi_hz_y.numpy(),
        }

    @state.setter
    @use_device
    def state(self, new_state_dict: dict[str, np.ndarray]):
        def set_field(target_arr, src_data):
            wp.copy(target_arr, wp.array(src_data, dtype=wp.float32))

        set_field(self._state.Ex, new_state_dict["Ex"])
        set_field(self._state.Ey, new_state_dict["Ey"])
        set_field(self._state.Ez, new_state_dict["Ez"])
        set_field(self._state.Hx, new_state_dict["Hx"])
        set_field(self._state.Hy, new_state_dict["Hy"])
        set_field(self._state.Hz, new_state_dict["Hz"])

        set_field(self._state.psi_ex_y, new_state_dict["psi_ex_y"])
        set_field(self._state.psi_ex_z, new_state_dict["psi_ex_z"])
        set_field(self._state.psi_ey_x, new_state_dict["psi_ey_x"])
        set_field(self._state.psi_ey_z, new_state_dict["psi_ey_z"])
        set_field(self._state.psi_ez_x, new_state_dict["psi_ez_x"])
        set_field(self._state.psi_ez_y, new_state_dict["psi_ez_y"])

        set_field(self._state.psi_hx_y, new_state_dict["psi_hx_y"])
        set_field(self._state.psi_hx_z, new_state_dict["psi_hx_z"])
        set_field(self._state.psi_hy_x, new_state_dict["psi_hy_x"])
        set_field(self._state.psi_hy_z, new_state_dict["psi_hy_z"])
        set_field(self._state.psi_hz_x, new_state_dict["psi_hz_x"])
        set_field(self._state.psi_hz_y, new_state_dict["psi_hz_y"])

    @property
    @use_device
    def detectors(self) -> list[dict[str, Any]]:
        return [
            (
                {
                    "Eu": wrapper.detector.Eu.numpy(),
                    "Ev": wrapper.detector.Ev.numpy(),
                    "Hu": wrapper.detector.Hu.numpy(),
                    "Hv": wrapper.detector.Hv.numpy(),
                    "OFFSETX": wrapper.detector.OFFSETX,
                    "OFFSETY": wrapper.detector.OFFSETY,
                    "OFFSETZ": wrapper.detector.OFFSETZ,
                    "plane": wrapper.plane,
                    "shape": wrapper.shape,
                }
                if not wrapper.full
                else {
                    "Eu": wrapper.detector.Eu.numpy(),
                    "Ev": wrapper.detector.Ev.numpy(),
                    "Hu": wrapper.detector.Hu.numpy(),
                    "Hv": wrapper.detector.Hv.numpy(),
                    "Hu_n": wrapper.detector.Hu_n.numpy(),
                    "Hv_n": wrapper.detector.Hv_n.numpy(),
                    "OFFSETX": wrapper.detector.OFFSETX,
                    "OFFSETY": wrapper.detector.OFFSETY,
                    "OFFSETZ": wrapper.detector.OFFSETZ,
                    "plane": wrapper.plane,
                    "shape": wrapper.shape,
                }
            )
            for wrapper in self._detectors
        ]

    @property
    @use_device
    def sources(self) -> list[dict[str, Any]]:
        return [
            {
                "Eu": wrapper.source.Eu.numpy(),
                "Ev": wrapper.source.Ev.numpy(),
                "Hu": wrapper.source.Hu.numpy(),
                "Hv": wrapper.source.Hv.numpy(),
                "OFFSETX": wrapper.source.OFFSETX,
                "OFFSETY": wrapper.source.OFFSETY,
                "OFFSETZ": wrapper.source.OFFSETZ,
                "plane": wrapper.plane,
            }
            for wrapper in self._sources
        ]

    @sources.setter
    @use_device
    def sources(self, data: list[dict[str, Any]]):

        for i, item in enumerate(data):

            if i >= len(self._sources):

                new_source_struct = TEMStates()

                self._sources.append(
                    source_wrapper(
                        source=new_source_struct,
                        plane="",
                        shape=(1, 1),
                        shape_shedule=(1, 1, 1),
                    )
                )

            eu_arr = item["Eu"]
            wrapper = self._sources[i]
            wrapper.shape_shedule = eu_arr.shape
            wrapper.shape = eu_arr.shape[1:]

            wrapper.source.Eu = to_wp(item["Eu"])
            wrapper.source.Ev = to_wp(item["Ev"])
            wrapper.source.Hu = to_wp(item["Hu"])
            wrapper.source.Hv = to_wp(item["Hv"])

            wrapper.source.OFFSETX = item["OFFSETX"]
            wrapper.source.OFFSETY = item["OFFSETY"]
            wrapper.source.OFFSETZ = item["OFFSETZ"]

            wrapper.plane = item["plane"]

    @use_device
    def init_source(
        self,
        eufunc: Callable[[np.ndarray], np.ndarray],
        euprofile: np.ndarray,
        evfunc: Callable[[np.ndarray], np.ndarray],
        evprofile: np.ndarray,
        hufunc: Callable[[np.ndarray], np.ndarray],
        huprofile: np.ndarray,
        hvfunc: Callable[[np.ndarray], np.ndarray],
        hvprofile: np.ndarray,
        OFFSETX: int,
        OFFSETY: int,
        OFFSETZ: int,
        plane: str,
    ):

        source_plane = plane
        NU, NV = euprofile.shape
        shape_source = (NU, NV)
        shape_schedule = (self._STEPS, NU, NV)

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

            mixed_complex = sig_complex[:, None, None] * profile[None, :, :]

            return wp.array(mixed_complex.real.astype(np.float32))

        source = TEMStates()
        source.Eu = process_field(eufunc, euprofile)
        source.Ev = process_field(evfunc, evprofile)
        source.Hu = process_field(hufunc, huprofile)
        source.Hv = process_field(hvfunc, hvprofile)

        source.OFFSETX, source.OFFSETY, source.OFFSETZ = (
            OFFSETX,
            OFFSETY,
            OFFSETZ,
        )
        self._sources.append(
            source_wrapper(
                source=source,
                plane=source_plane,
                shape=shape_source,
                shape_shedule=shape_schedule,
            )
        )

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
        detector_plane = plane
        NU, NV = shape
        shape_detector = (NU, NV)
        shape_record = (self._STEPS, NU, NV)
        if full:
            detector = TEMStates_full()
        else:
            detector = TEMStates()
        detector.Eu = wp.zeros(shape_record, dtype=wp.float32)
        detector.Ev = wp.zeros(shape_record, dtype=wp.float32)
        detector.Hu = wp.zeros(shape_record, dtype=wp.float32)
        detector.Hv = wp.zeros(shape_record, dtype=wp.float32)
        if full:
            detector.Hu_n = wp.zeros(shape_record, dtype=wp.float32)
            detector.Hv_n = wp.zeros(shape_record, dtype=wp.float32)
        detector.OFFSETX, detector.OFFSETY, detector.OFFSETZ = (
            OFFSETX,
            OFFSETY,
            OFFSETZ,
        )
        self._detectors.append(
            detector_wrapper(
                detector=detector,
                source_adj=None,
                plane=detector_plane,
                shape=shape_detector,
                shape_record=shape_record,
                full=full,
            )
        )

    @use_device
    def get_dft_detector(
        self,
        freqs: np.ndarray,
        idx=0,
        collocated=False,
        save: bool = False,
        base_filename: str = "mode_profile",
    ) -> dict:
        if self._detectors[idx].full is False:
            collocated = False

        freqs_wp = wp.array(freqs, dtype=float)

        nu = self._detectors[idx].detector.Eu.shape[1]
        nv = self._detectors[idx].detector.Eu.shape[2]

        shape_cdft = (self._STEPS, len(freqs))
        cdft = wp.empty(shape_cdft, dtype=wp.vec2)

        wp.launch(
            kernel=kn.compute_dft_coeffs,
            dim=shape_cdft,
            inputs=[freqs_wp, self._S, cdft],
        )

        dfts = kn.TEMDFTs()
        shape_out = (len(freqs), nu, nv)

        dfts.Eu = wp.empty(shape_out, dtype=wp.vec2)
        dfts.Ev = wp.empty(shape_out, dtype=wp.vec2)
        dfts.Hu = wp.empty(shape_out, dtype=wp.vec2)
        dfts.Hv = wp.empty(shape_out, dtype=wp.vec2)

        if collocated:
            wp.launch(
                kernel=kn.compute_collocated_tem_dft,
                dim=(nu, nv),
                inputs=[cdft, self._detectors[idx].detector, dfts],
            )
        else:
            wp.launch(
                kernel=kn.compute_tem_dft,
                dim=(nu, nv),
                inputs=[cdft, self._detectors[idx].detector, dfts],
            )

        results = {
            "Eu": dfts.Eu.numpy(),
            "Ev": dfts.Ev.numpy(),
            "Hu": dfts.Hu.numpy(),
            "Hv": dfts.Hv.numpy(),
            "freqs": freqs,
        }

        if save:
            print(f"Saving {len(freqs)} mode profiles...")

            def to_complex(arr_slice):
                if arr_slice.shape[-1] == 2:
                    return arr_slice.view(np.complex64).squeeze(-1)
                return arr_slice

            for i, omega in enumerate(freqs):
                filename = f"{base_filename}_w{omega:.4f}.npz"

                save_data = {
                    "Eu": to_complex(results["Eu"][i]),
                    "Ev": to_complex(results["Ev"][i]),
                    "Hu": to_complex(results["Hu"][i]),
                    "Hv": to_complex(results["Hv"][i]),
                    "omega": omega,
                }

                np.savez_compressed(filename, **save_data)
                print(f"  -> Saved: {filename}")

        return results

    def record_graphs(self):
        self._set_kernels()
        self._forward_graph_tape = self._record_forward()

    def _set_kernels(self):
        
        match self.kernel:
            case "warp":
                self.update_e = kn.update_e
                self.update_h = kn.update_h
            case "warp_iso":
                self.update_e = kn.updateiso_e
                self.update_h = kn.updateiso_h
            case "yee":
                self.update_e = kn.update_yee_e
                self.update_h = kn.update_yee_h
            case "pml":
                self.update_e = kn.update_pml_e
                self.update_h = kn.update_pml_h


        for wrapper in self._sources:

            match wrapper.plane:
                case "xy":
                    wrapper.inject_esource = kn.inject_esources_xy
                    wrapper.inject_hsource = kn.inject_hsources_xy
                case "xz":
                    wrapper.inject_esource = kn.inject_esources_xz
                    wrapper.inject_hsource = kn.inject_hsources_xz
                case "yz":
                    wrapper.inject_esource = kn.inject_esources_yz
                    wrapper.inject_hsource = kn.inject_hsources_yz

        for wrapper in self._detectors:
            match wrapper.plane:
                case "xy":
                    wrapper.save_detector = kn.save_detector_xy
                    if wrapper.full is True:
                        wrapper.save_detector = kn.save_detector_full_xy
                case "xz":
                    wrapper.save_detector = kn.save_detector_xz
                    if wrapper.full is True:
                        wrapper.save_detector = kn.save_detector_full_xz
                case "yz":
                    wrapper.save_detector = kn.save_detector_yz
                    if wrapper.full is True:
                        wrapper.save_detector = kn.save_detector_full_yz

    @use_device
    def _set_pml(self) -> bool:

        m = 4
        eta = float(self._eta)
        dx = float(self._dx)
        c_light = 1.0
        eps_0 = 1.0
        dt = (self._S * dx) / c_light

        sigma_max = (
            -(m + 1) * np.log(self._R_0) / (2 * eta * (self._PML_THICKNESS * dx))
        )

        def sigma_fn(x, d):
            return sigma_max * (x / d) ** m

        def create_profiles(
            dim_size: int, bc_min: str, bc_max: str
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            sigma_e_p = np.zeros(dim_size, dtype=np.float32)
            sigma_h_p = np.zeros(dim_size, dtype=np.float32)
            kappa_p = np.ones(dim_size, dtype=np.float32)
            alpha_p = np.zeros(dim_size, dtype=np.float32)

            kappa_max = 10

            alpha_min = 0.0
            alpha_max = 0.05

            if bc_min == "PML":
                d = self._PML_THICKNESS

                dist_e = np.arange(d, 0, -1.0)
                sigma_e_p[:d] = sigma_fn(dist_e, d)

                dist_h = np.arange(d - 0.5, -0.5, -1.0)
                sigma_h_p[:d] = sigma_fn(dist_h, d)

                r_min = dist_e / d
                kappa_p[:d] = 1.0 + (kappa_max - 1.0) * (r_min**m)

                alpha_p[:d] = alpha_max - (alpha_max - alpha_min) * r_min

            if bc_max == "PML":
                d = self._PML_THICKNESS

                dist_e = np.arange(0, d + 0, 1.0)
                sigma_e_p[-d:] = sigma_fn(dist_e, d)

                dist_h = np.arange(0.5, d + 0.5, 1.0)
                sigma_h_p[-d:] = sigma_fn(dist_h, d)

                r_max = dist_e / d
                kappa_p[-d:] = 1.0 + (kappa_max - 1.0) * (r_max**m)

                alpha_p[-d:] = alpha_max - (alpha_max - alpha_min) * r_max

            return (
                np.asarray(sigma_e_p, dtype=np.float32),
                np.asarray(sigma_h_p, dtype=np.float32),
                np.asarray(kappa_p, dtype=np.float32),
                np.asarray(alpha_p, dtype=np.float32),
            )

        sigma_ex, sigma_hx, kappa_x, alpha_x = create_profiles(
            self._NX, self._boundaries["xmin"], self._boundaries["xmax"]
        )
        sigma_ey, sigma_hy, kappa_y, alpha_y = create_profiles(
            self._NY, self._boundaries["ymin"], self._boundaries["ymax"]
        )
        sigma_ez, sigma_hz, kappa_z, alpha_z = create_profiles(
            self._NZ, self._boundaries["zmin"], self._boundaries["zmax"]
        )

        def get_coeffs(
            sigma: np.ndarray, kappa: np.ndarray, alpha: np.ndarray
        ) -> Tuple[np.ndarray, np.ndarray]:

            term = (sigma / (kappa * eps_0)) + (alpha / eps_0)
            b = np.exp(-term * dt)

            denom = (sigma * kappa) + (kappa**2 * alpha)
            c = (b - 1.0) * sigma

            mask = denom != 0
            c[mask] /= denom[mask]
            c[~mask] = 0.0

            return b, c

        b_e_x, c_e_x = get_coeffs(sigma_ex, kappa_x, alpha_x)
        b_e_y, c_e_y = get_coeffs(sigma_ey, kappa_y, alpha_y)
        b_e_z, c_e_z = get_coeffs(sigma_ez, kappa_z, alpha_z)

        b_h_x, c_h_x = get_coeffs(sigma_hx, kappa_x, alpha_x)
        b_h_y, c_h_y = get_coeffs(sigma_hy, kappa_y, alpha_y)
        b_h_z, c_h_z = get_coeffs(sigma_hz, kappa_z, alpha_z)

        def to_wp(arr: np.ndarray) -> wp.array:
            return wp.array(arr, dtype=wp.float32)

        self._properties.B_E_X = to_wp(b_e_x)
        self._properties.C_E_X = to_wp(c_e_x)
        self._properties.B_E_Y = to_wp(b_e_y)
        self._properties.C_E_Y = to_wp(c_e_y)
        self._properties.B_E_Z = to_wp(b_e_z)
        self._properties.C_E_Z = to_wp(c_e_z)

        self._properties.B_H_X = to_wp(b_h_x)
        self._properties.C_H_X = to_wp(c_h_x)
        self._properties.B_H_Y = to_wp(b_h_y)
        self._properties.C_H_Y = to_wp(c_h_y)
        self._properties.B_H_Z = to_wp(b_h_z)
        self._properties.C_H_Z = to_wp(c_h_z)

        self._properties.INV_K_X = to_wp(1 / kappa_x)
        self._properties.INV_K_Y = to_wp(1 / kappa_y)
        self._properties.INV_K_Z = to_wp(1 / kappa_z)

        return True

    @abstractmethod
    def _record_forward(self) -> GraphTape:
        pass

    @ensure_synced
    def launch_forward(self) -> None:
        self._forward_graph_tape()
        wp.synchronize()

    @use_device
    def render_source_video(
        self,
        idx=0,
        field: str = "Eu",
        filename: str = "source.mp4",
        limit: float = 0.3,
        scale: int = 4,
        fps: int = 30,
    ) -> None:
        match field:
            case "Eu":
                field_arr = self._sources[idx].source.Eu
            case "Ev":
                field_arr = self._sources[idx].source.Ev
            case "Hu":
                field_arr = self._sources[idx].source.Hu
            case "Hv":
                field_arr = self._sources[idx].source.Hv
            case _:
                field_arr = self._sources[idx].source.Ev

        pp.render_array(field_arr, filename=filename, limit=limit, scale=scale, fps=fps)

    @use_device
    def render_detector_video(
        self,
        idx=0,
        field: str = "Eu",
        filename: str = "detector.mp4",
        limit: float = 0.3,
        scale: int = 4,
        fps: int = 30,
    ) -> None:
        match field:
            case "Eu":
                field_arr = self._detectors[idx].detector.Eu
            case "Ev":
                field_arr = self._detectors[idx].detector.Ev
            case "Hu":
                field_arr = self._detectors[idx].detector.Hu
            case "Hv":
                field_arr = self._detectors[idx].detector.Hv
            case _:
                field_arr = self._detectors[idx].detector.Ev

        pp.render_array(field_arr, filename=filename, limit=limit, scale=scale, fps=fps)

    @ensure_synced
    def export_geometrie(
        self,
        filename: str = "layout.vti",
        roi: Any = np.s_[:, :, :],
    ) -> None:
        pp.export_vti(
            filename=filename,
            fields={
                "PEC": self._PEC_np,
                "PMC": self._PMC_np,
                "CEA": self._cea_np,
                "CEB": self._ceb_np,
                "CHA": self._cha_np,
                "CHB": self._chb_np,
            },
            roi=roi,
        )

    @use_device
    def detector_flux(self, idx=0) -> Optional[np.ndarray]:
        if self._detectors[idx].full is False:
            return None
        match self._detectors[idx].plane:
            case "xy":
                calc_flux = kn.calc_flux_xy
            case "xz":
                calc_flux = kn.calc_flux_xz
            case "yz":
                calc_flux = kn.calc_flux_yz
            case _:
                calc_flux = None

        history = wp.zeros(self._STEPS, dtype=wp.float32)
        wp.launch(
            calc_flux,
            dim=self._detectors[idx].shape_record,
            inputs=[self._detectors[idx].detector, history],
        )
        wp.synchronize()

        return history.numpy()

    @abstractmethod
    def __str__(self) -> str:
        pass
