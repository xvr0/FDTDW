from fdtdw import (
    ReferenceSimulation as Simulation,
    StandardMaterialModel,
    MaterialSimulation,
)
import numpy as np

NX, NY, NZ = 64, 64, 512
S = 0.56
sim = MaterialSimulation(
    Simulation(
        NX=NX,
        NY=NY,
        NZ=NZ,
        STEPS=1500,
        DEVICE="cuda:0",
        boundaries={
            "xmin": "PMC",
            "xmax": "PMC",
            "ymin": "PEC",
            "ymax": "PEC",
            "zmin": "PEC",
            "zmax": "PML",
        },
        PML_THICKNESS=10,
    ),
    StandardMaterialModel(dx=0.01, S=S),
)

NU, NV = NX - 2, NY - 2

DSX = 1
DSY = 1
DSZ = NZ // 2

width = 250.0
center = 600.0

OMEGA = 2.0 * np.pi / width
OMEGA_t = 2.0 * np.pi / width / S

evfunc = lambda t: (1.0 - 2.0 * (np.pi * ((t - center) / width)) ** 2) * np.exp(
    -((np.pi * ((t - center) / width)) ** 2)
)
eufunc = lambda n: np.zeros_like(n)
euprofile = np.ones((NU, NV), dtype=np.float32)
evprofile = np.ones((NU, NV), dtype=np.float32)
hufunc = lambda n: np.zeros_like(n)
huprofile = np.ones((NU, NV), dtype=np.float32)
hvfunc = lambda n: np.zeros_like(n)
hvprofile = np.ones((NU, NV), dtype=np.float32)

sim.init_source(
    eufunc=eufunc,
    euprofile=euprofile,
    evfunc=evfunc,
    evprofile=evprofile,
    hufunc=hufunc,
    huprofile=huprofile,
    hvfunc=hvfunc,
    hvprofile=hvprofile,
    OFFSETX=DSX,
    OFFSETY=DSY,
    OFFSETZ=DSZ,
    plane="xy",
)

NUD = NX
NVD = NY
DDX = 0
DDZ = 0
DDY = NY//2

sim.init_detector(
    shape=(NX, NZ), OFFSETX=DDX, OFFSETY=DDY, OFFSETZ=DDZ , plane="xz", full=True
)

sim.record_graphs()

sim.launch_forward()

sim.render_detector_video(field="Hu", filename="wave.mp4", limit=1, fps=240)
print(sim)
