import warp as wp
from .structs32 import EMState, Properties


@wp.kernel
def updateiso_h(state: EMState, props: Properties):
    i, j, k = wp.tid()
    nx = state.Ex.shape[0]
    ny = state.Ex.shape[1]
    nz = state.Ex.shape[2]

    if i >= nx - 1 or j >= ny - 1 or k >= nz - 1:
        return

    dEz_dy = state.Ez[i, j + 1, k] - state.Ez[i, j, k]
    dEy_dz = state.Ey[i, j, k + 1] - state.Ey[i, j, k]

    dEx_dz = state.Ex[i, j, k + 1] - state.Ex[i, j, k]
    dEz_dx = state.Ez[i + 1, j, k] - state.Ez[i, j, k]

    dEy_dx = state.Ey[i + 1, j, k] - state.Ey[i, j, k]
    dEx_dy = state.Ex[i, j + 1, k] - state.Ex[i, j, k]
    
    cha=props.CHA_X[i, j, k]
    chb=props.CHB_X[i, j, k]
    if props.is_PML[i, j, k]:

        psi_y = props.B_H_Y[j] * state.psi_hx_y[i, j, k] + props.C_H_Y[j] * dEz_dy
        psi_z = props.B_H_Z[k] * state.psi_hx_z[i, j, k] + props.C_H_Z[k] * dEy_dz
        state.psi_hx_y[i, j, k] = psi_y
        state.psi_hx_z[i, j, k] = psi_z

        state.Hx[i, j, k] = cha * state.Hx[i, j, k] - chb * ((dEz_dy * props.INV_K_Y[j] + psi_y) - (dEy_dz * props.INV_K_Z[k] + psi_z))

        psi_z = props.B_H_Z[k] * state.psi_hy_z[i, j, k] + props.C_H_Z[k] * dEx_dz
        psi_x = props.B_H_X[i] * state.psi_hy_x[i, j, k] + props.C_H_X[i] * dEz_dx
        state.psi_hy_z[i, j, k] = psi_z
        state.psi_hy_x[i, j, k] = psi_x

        state.Hy[i, j, k] = cha * state.Hy[i, j, k] - chb * ((dEx_dz * props.INV_K_Z[k] + psi_z) - (dEz_dx * props.INV_K_X[i] + psi_x))

        psi_x = props.B_H_X[i] * state.psi_hz_x[i, j, k] + props.C_H_X[i] * dEy_dx
        psi_y = props.B_H_Y[j] * state.psi_hz_y[i, j, k] + props.C_H_Y[j] * dEx_dy
        state.psi_hz_x[i, j, k] = psi_x
        state.psi_hz_y[i, j, k] = psi_y

        state.Hz[i, j, k] = cha * state.Hz[i, j, k] - chb * ((dEy_dx * props.INV_K_X[i] + psi_x) - (dEx_dy * props.INV_K_Y[j] + psi_y))

    else:

        state.Hx[i, j, k] = cha * state.Hx[i, j, k] - chb * (dEz_dy - dEy_dz)
        state.Hy[i, j, k] = cha * state.Hy[i, j, k] - chb * (dEx_dz - dEz_dx)
        state.Hz[i, j, k] = cha * state.Hz[i, j, k] - chb * (dEy_dx - dEx_dy)


@wp.kernel
def updateiso_e(state: EMState, props: Properties):
    i, j, k = wp.tid()
    nx = state.Ex.shape[0]
    ny = state.Ex.shape[1]
    nz = state.Ex.shape[2]

    if i == 0 or j == 0 or k == 0 or i >= nx or j >= ny or k >= nz:
        return

    dHz_dy = state.Hz[i, j, k] - state.Hz[i, j - 1, k]
    dHy_dz = state.Hy[i, j, k] - state.Hy[i, j, k - 1]

    dHx_dz = state.Hx[i, j, k] - state.Hx[i, j, k - 1]
    dHz_dx = state.Hz[i, j, k] - state.Hz[i - 1, j, k]

    dHy_dx = state.Hy[i, j, k] - state.Hy[i - 1, j, k]
    dHx_dy = state.Hx[i, j, k] - state.Hx[i, j - 1, k]

    cea=props.CEA_X[i, j, k]
    ceb=props.CEB_X[i, j, k]
    if props.is_PML[i, j, k]:

        psi_y = props.B_E_Y[j] * state.psi_ex_y[i, j, k] + props.C_E_Y[j] * dHz_dy
        psi_z = props.B_E_Z[k] * state.psi_ex_z[i, j, k] + props.C_E_Z[k] * dHy_dz
        state.psi_ex_y[i, j, k] = psi_y
        state.psi_ex_z[i, j, k] = psi_z

        state.Ex[i, j, k] = cea * state.Ex[i, j, k] + ceb * ((dHz_dy * props.INV_K_Y[j] + psi_y) - (dHy_dz * props.INV_K_Z[k] + psi_z))

        psi_z = props.B_E_Z[k] * state.psi_ey_z[i, j, k] + props.C_E_Z[k] * dHx_dz
        psi_x = props.B_E_X[i] * state.psi_ey_x[i, j, k] + props.C_E_X[i] * dHz_dx
        state.psi_ey_z[i, j, k] = psi_z
        state.psi_ey_x[i, j, k] = psi_x

        state.Ey[i, j, k] = cea * state.Ey[i, j, k] + ceb * ((dHx_dz * props.INV_K_Z[k] + psi_z) - (dHz_dx * props.INV_K_X[i] + psi_x))

        psi_x = props.B_E_X[i] * state.psi_ez_x[i, j, k] + props.C_E_X[i] * dHy_dx
        psi_y = props.B_E_Y[j] * state.psi_ez_y[i, j, k] + props.C_E_Y[j] * dHx_dy
        state.psi_ez_x[i, j, k] = psi_x
        state.psi_ez_y[i, j, k] = psi_y

        state.Ez[i, j, k] = cea * state.Ez[i, j, k] + ceb * ((dHy_dx * props.INV_K_X[i] + psi_x) - (dHx_dy * props.INV_K_Y[j] + psi_y))

    else:

        state.Ex[i, j, k] = cea * state.Ex[i, j, k] + ceb * (dHz_dy - dHy_dz)
        state.Ey[i, j, k] = cea * state.Ey[i, j, k] + ceb * (dHx_dz - dHz_dx)
        state.Ez[i, j, k] = cea * state.Ez[i, j, k] + ceb * (dHy_dx - dHx_dy)
