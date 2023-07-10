import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from numpy import cos, sin, tan, arctan, arccosh, pi, sqrt, exp, power, log, absolute, arctan2, arccos, sign
from scipy.integrate import solve_ivp
import math
import os
import time
import h5py
from scipy import integrate
from scipy.optimize import fsolve
from scipy.special import spherical_jn as Jl
from sympy.physics.wigner import gaunt, wigner_3j
from scipy.special import sph_harm
from scipy.special import lpmn as Plm
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


def Linear_Tunneling_Time(t_ion_initial, v_perp_initial, kz_initial):
    """
    Compute linear tunneling time.
    
    Arguments:
    t_ion_initial -- array of initial time values
    v_perp_initial -- array of initial perpendicular velocities
    kz_initial -- array of initial kz values
    
    Returns:
    t_ion_return -- filtered array of initial time values
    v_perp_return -- filtered array of initial perpendicular velocities
    kz_return -- filtered array of initial kz values
    tunneling_time_return -- computed tunneling times
    """

    Field_Norm = cos(omega * (t_ion_initial - tau/2))
    gamma_eff = omega * sqrt(2 * Ip + power(kz_initial, 2)) / \
        (Fo * Envelop_Fun(t_ion_initial))

    error_idx = ((Field_Norm * omega * v_perp_initial / (Fo * Envelop_Fun(t_ion_initial)))**2 +
                 (Field_Norm**4 - ell**2) * (1 + power(gamma_eff / Field_Norm, 2))) > 0

    t_ion_return = t_ion_initial[error_idx]
    v_perp_return = v_perp_initial[error_idx]
    kz_return = kz_initial[error_idx]
    Field_Norm = Field_Norm[error_idx]
    gamma_eff = gamma_eff[error_idx]

    tunneling_time_return = (Field_Norm**2) * sqrt((Field_Norm * omega * v_perp_return / (Fo * Envelop_Fun(t_ion_return)))**2 +
                                                   (Field_Norm**4) * (1 + power(gamma_eff / Field_Norm, 2)))
    tunneling_time_return *= 1 / (Field_Norm**4)

    greater_than_one_idx = tunneling_time_return >= 1.0
    tunneling_time_return = arccosh(
        tunneling_time_return[greater_than_one_idx]) / omega
    t_ion_return = t_ion_return[greater_than_one_idx]
    v_perp_return = v_perp_return[greater_than_one_idx]
    kz_return = kz_return[greater_than_one_idx]

    return t_ion_return, v_perp_return, kz_return, tunneling_time_return


def Circular_Tunneling_Time(t_ion_initial, v_perp_initial, kz_initial):
    """
    Compute circular tunneling time.
    
    Arguments:
    t_ion_initial -- array of initial time values
    v_perp_initial -- array of initial perpendicular velocities
    kz_initial -- array of initial kz values
    
    Returns:
    t_ion_return -- filtered array of initial time values
    v_perp_return -- filtered array of initial perpendicular velocities
    kz_return -- filtered array of initial kz values
    tunneling_time_return -- computed tunneling times
    """

    gamma_eff = omega * sqrt(2 * Ip + power(kz_initial, 2)) / \
        (Fo * Envelop_Fun(t_ion_initial))

    tunneling_time_return = 0.5 * (1 - omega * v_perp_initial / (ell * Fo * Envelop_Fun(t_ion_initial))) \
        + 0.5 * (1 + power(gamma_eff, 2)) / (1 - omega *
                                             v_perp_initial / (ell * Fo * Envelop_Fun(t_ion_initial)))

    greater_than_one_idx = tunneling_time_return >= 1.0
    tunneling_time_return = arccosh(
        tunneling_time_return[greater_than_one_idx]) / omega
    t_ion_return = t_ion_initial[greater_than_one_idx]
    v_perp_return = v_perp_initial[greater_than_one_idx]
    kz_return = kz_initial[greater_than_one_idx]

    return t_ion_return, v_perp_return, kz_return, tunneling_time_return


def Elliptical_Tunneling_Time(t_ion_initial, v_perp_initial, kz_initial):
    """
    Compute elliptical tunneling time.
    
    Arguments:
    t_ion_initial -- array of initial time values
    v_perp_initial -- array of initial perpendicular velocities
    kz_initial -- array of initial kz values
    
    Returns:
    t_ion_return -- filtered array of initial time values
    v_perp_return -- filtered array of initial perpendicular velocities
    kz_return -- filtered array of initial kz values
    tunneling_time_return -- computed tunneling times
    """

    Field_Norm = sqrt(power(cos(omega*(t_ion_initial - tau/2)), 2) +
                      ell*ell*power(sin(omega*(t_ion_initial - tau/2)), 2))
    gamma_eff = omega * sqrt(2 * Ip + power(kz_initial, 2)) / \
        (Fo * Envelop_Fun(t_ion_initial))

    A_idx = absolute(Field_Norm**2 - abs(ell)) < 1e-16

    tunneling_time_A = 0.5 * (1 - Field_Norm[A_idx] * omega * v_perp_initial[A_idx] / (ell * Fo * Envelop_Fun(t_ion_initial[A_idx]))) \
        + 0.5 * power(Field_Norm[A_idx] / ell, 2) * (1 + power(gamma_eff[A_idx] / Field_Norm[A_idx], 2)) / (
            1 - Field_Norm[A_idx] * omega * v_perp_initial[A_idx] / (ell * Fo * Envelop_Fun(t_ion_initial[A_idx])))

    greater_than_one_idx = tunneling_time_A >= 1.0
    tunneling_time_A = arccosh(tunneling_time_A[greater_than_one_idx]) / omega

    tunneling_time_return = tunneling_time_A
    t_ion_return = t_ion_initial[A_idx][greater_than_one_idx]
    v_perp_return = v_perp_initial[A_idx][greater_than_one_idx]
    kz_return = kz_initial[A_idx][greater_than_one_idx]

    B_idx = absolute(Field_Norm**2 - abs(ell)) > 1e-12
    B_idx_two = ((Field_Norm[B_idx] * omega * v_perp_initial[B_idx] / (Fo * Envelop_Fun(t_ion_initial[B_idx])) - ell)**2 +
                 (Field_Norm[B_idx]**4 - ell**2) * (1 + power(gamma_eff[B_idx] / Field_Norm[B_idx], 2))) > 0

    t_ion_initial = t_ion_initial[B_idx][B_idx_two]
    v_perp_initial = v_perp_initial[B_idx][B_idx_two]
    kz_initial = kz_initial[B_idx][B_idx_two]
    Field_Norm = Field_Norm[B_idx][B_idx_two]
    gamma_eff = gamma_eff[B_idx][B_idx_two]

    tunneling_time_B = ell * \
        (Field_Norm * omega * v_perp_initial /
         (Fo * Envelop_Fun(t_ion_initial)) - ell)
    tunneling_time_B += (Field_Norm**2) * sqrt((Field_Norm * omega * v_perp_initial / (Fo * Envelop_Fun(t_ion_initial)) - ell)**2 +
                                               (Field_Norm**4 - ell**2) * (1 + power(gamma_eff / Field_Norm, 2)))
    tunneling_time_B *= 1 / (Field_Norm**4 - ell**2)

    greater_than_one_idx = tunneling_time_B >= 1.0
    tunneling_time_B = arccosh(tunneling_time_B[greater_than_one_idx]) / omega

    tunneling_time_return = np.append(tunneling_time_return, tunneling_time_B)
    t_ion_return = np.append(t_ion_return, t_ion_initial[greater_than_one_idx])
    v_perp_return = np.append(
        v_perp_return, v_perp_initial[greater_than_one_idx])
    kz_return = np.append(kz_return, kz_initial[greater_than_one_idx])

    return t_ion_return, v_perp_return, kz_return, tunneling_time_return


def Cosh_Dependent_Terms(t_ion_array, v_perp, kz_array, tunneling_time):
    """
    Compute cosh-dependent terms.
    
    Arguments:
    t_ion_array -- array of initial time values
    v_perp -- initial perpendicular velocity
    kz_array -- array of kz values
    tunneling_time -- array of tunneling times
    
    Returns:
    t_ion_array -- filtered array of initial time values
    tunneling_time -- filtered array of tunneling times
    kx_array -- computed kx values
    ky_array -- computed ky values
    kz_array -- filtered array of kz values
    x_pos_array -- computed x positions
    y_pos_array -- computed y positions
    v_parall -- computed parallel velocities
    v_perp -- filtered array of initial perpendicular velocities
    """

    e_field_x = Fo * Envelop_Fun(t_ion_array) * \
        cos(omega * (t_ion_array - tau / 2))
    a_array = sqrt(power(cos(omega * (t_ion_array - tau / 2)), 2) +
                   ell * ell * power(sin(omega * (t_ion_array - tau / 2)), 2))

    v_parall = (1 - ell**2) * Fo * Envelop_Fun(t_ion_array) * sin(omega * (t_ion_array - tau / 2)) * cos(omega * (t_ion_array - tau / 2)) * \
               (np.cosh(omega * tunneling_time) - 1) / (a_array * omega)

    kx_array = v_parall / \
        sqrt(1 + power(ell * tan(omega * (t_ion_array - tau / 2)), 2))
    kx_array -= v_perp * ell * tan(omega * (t_ion_array - tau / 2)) / sqrt(
        1 + power(ell * tan(omega * (t_ion_array - tau / 2)), 2))
    kx_array *= np.sign(e_field_x + 1e-16)

    ky_array = v_parall * ell * tan(omega * (t_ion_array - tau / 2)) / sqrt(
        1 + power(ell * tan(omega * (t_ion_array - tau / 2)), 2))
    ky_array += v_perp / \
        sqrt(1 + power(ell * tan(omega * (t_ion_array - tau / 2)), 2))
    ky_array *= np.sign(e_field_x + 1e-16)

    px_array = kx_array + Fo * \
        Envelop_Fun(t_ion_array) * sin(omega * (t_ion_array - tau / 2)) / omega
    py_array = ky_array - ell * Fo * \
        Envelop_Fun(t_ion_array) * cos(omega * (t_ion_array - tau / 2)) / omega

    weight = -2 * ((px_array**2 + py_array**2 + kz_array**2) /
                   2 + Ip + Up) * tunneling_time
    weight += 2 * px_array * Fo * Envelop_Fun(t_ion_array) * sin(
        omega * (t_ion_array - tau / 2)) * np.sinh(omega * tunneling_time) / (omega**2)
    weight += -2 * py_array * ell * Fo * Envelop_Fun(t_ion_array) * cos(
        omega * (t_ion_array - tau / 2)) * np.sinh(omega * tunneling_time) / (omega**2)
    weight += (1 - (ell**2)) * power(Fo * Envelop_Fun(t_ion_array), 2) * cos(2 * omega *
                                                                             (t_ion_array - tau / 2)) * np.sinh(2 * omega * tunneling_time) / (4 * (omega**3))
    weight = exp(weight)

    weight_coefficent = (2 * lo + 1) / (4 * pi) * factorial(lo - abs(mo)) / \
        factorial(lo + abs(mo)) * (omega**2) / (4 * pi * pi)
    weight_coefficent *= absolute(Vcos_phi_array +
                                  1.0j * sign(mo) * Vsin_phi_array)**2

    if lo == 1 and mo == 0:
        weight_coefficent *= (kz_array**2) / (2 * Ip)

    if lo == 1 and (mo == 1 or mo == -1):
        weight_coefficent *= (1 + (kz_array**2) / (2 * Ip))

    weight *= weight_coefficent

    weight = sqrt(weight)
    weight = weight / weight.max()

    Y_val = np.random.uniform(0, 1, len(weight))
    green_points_idx = Y_val < weight

    t_ion_array = t_ion_array[green_points_idx]
    kx_array = kx_array[green_points_idx]
    ky_array = ky_array[green_points_idx]
    kz_array = kz_array[green_points_idx]

    tunneling_time = tunneling_time[green_points_idx]
    v_parall = v_parall[green_points_idx]
    v_perp = v_perp[green_points_idx]

    x_pos_array = Fo * Envelop_Fun(t_ion_array) * cos(omega * (
        t_ion_array - tau / 2)) * (1 - np.cosh(omega * tunneling_time)) / (omega**2)
    y_pos_array = ell * Fo * Envelop_Fun(t_ion_array) * sin(omega * (
        t_ion_array - tau / 2)) * (1 - np.cosh(omega * tunneling_time)) / (omega**2)

    return t_ion_array, tunneling_time, kx_array, ky_array, kz_array, x_pos_array, y_pos_array, v_parall, v_perp


def Depletion(t_ion_array, weight, kx_array, ky_array, kz_array, tunneling_time, v_parall, v_perp):
    weight_new = weight / np.sum(weight)
    ordering_idx = np.argsort(t_ion_array)

    weight_new_ordered = weight_new[ordering_idx]

    loss_array = []
    loss = 0

    weight_new_new = []

    for w in weight_new_ordered:
        loss += w
        loss_array.append(exp(-1 * loss))

        weight_new_new.append((1 - loss) * w)

    weight = weight[ordering_idx]
    weight *= np.array(loss_array)
    t_ion_array = t_ion_array[ordering_idx]
    kx_array = kx_array[ordering_idx]
    ky_array = ky_array[ordering_idx]
    kz_array = kz_array[ordering_idx]

    tunneling_time = tunneling_time[ordering_idx]
    v_parall = v_parall[ordering_idx]
    v_perp = v_perp[ordering_idx]

    return t_ion_array, weight, kx_array, ky_array, kz_array, tunneling_time, v_parall, v_perp, ordering_idx, np.array(weight_new_new)


def Test_Cosh_Term(t_ion_return, v_perp_return, kz_return, tunneling_time_return):
    e_field_x = Fo * Envelop_Fun(t_ion_return) * cos(omega * t_ion_return)
    a_array = sqrt(power(cos(omega * t_ion_return), 2) + ell *
                   ell * power(sin(omega * t_ion_return), 2))

    v_parall = (1 - ell**2) * Fo * Envelop_Fun(t_ion_return) * sin(omega * t_ion_return) * cos(omega * t_ion_return) * (
        np.cosh(omega * tunneling_time_return) - 1) / (a_array * omega)

    kx_array = v_parall / sqrt(1 + power(ell * tan(omega * t_ion_return), 2))
    kx_array -= v_perp_return * ell * \
        tan(omega * t_ion_return) / \
        sqrt(1 + power(ell * tan(omega * t_ion_return), 2))
    kx_array *= np.sign(e_field_x + 1e-16)

    ky_array = v_parall * ell * \
        tan(omega * t_ion_return) / \
        sqrt(1 + power(ell * tan(omega * t_ion_return), 2))
    ky_array += v_perp_return / \
        sqrt(1 + power(ell * tan(omega * t_ion_return), 2))
    ky_array *= np.sign(e_field_x + 1e-16)

    px_array = kx_array + Fo * \
        Envelop_Fun(t_ion_return) * sin(omega * t_ion_return) / omega
    py_array = ky_array - ell * Fo * \
        Envelop_Fun(t_ion_return) * cos(omega * t_ion_return) / omega

    gamma_eff = omega * sqrt(2 * Ip + power(v_perp_return, 2)) / np.absolute(
        Fo * Envelop_Fun(t_ion_return) * cos(omega * t_ion_return))

    vx = Fo * Envelop_Fun(t_ion_return) * sin(omega *
                                              t_ion_return) / omega * (sqrt(1 + gamma_eff ** 2) - 1)

    Px_array = kx_array + Fo * \
        Envelop_Fun(t_ion_return) * sin(omega * t_ion_return) / omega
    Py_array = ky_array - ell * Fo * \
        Envelop_Fun(t_ion_return) * cos(omega * t_ion_return) / omega
    Pz_array = kz_return

    Term_A = Px_array - Fo * Envelop_Fun(t_ion_return) * sin(omega * t_ion_return) * np.cosh(
        omega * tunneling_time_return) / omega - 1j * Fo * Envelop_Fun(t_ion_return) * cos(
        omega * t_ion_return) * np.sinh(omega * tunneling_time_return) / omega
    Term_B = Py_array + ell * Fo * Envelop_Fun(t_ion_return) * cos(omega * t_ion_return) * np.cosh(
        omega * tunneling_time_return) / omega - 1j * ell * Fo * Envelop_Fun(t_ion_return) * sin(
        omega * t_ion_return) * np.sinh(omega * tunneling_time_return) / omega

    check = Term_A ** 2 + Term_B ** 2 + Pz_array ** 2 + 2 * Ip

    plt.semilogy(np.absolute(check), '.')
    plt.savefig("SP.png")
    exit()


def Asymptotic_Momentum(final_position, final_momentum):
    """
    Calculate the asymptotic momentum given the final position and momentum.
    """
    r = np.linalg.norm(final_position)
    p = np.linalg.norm(final_momentum)

    if r == 0 or (p ** 2) / 2 - 1 * Z / r < 0:
        return [None, None, None]

    k = sqrt(p ** 2 - (2 * Z / r))

    L = np.cross(final_position, final_momentum)
    a_vec = np.cross(final_momentum, L) - (final_position / r)

    return_val = k * np.cross(L, a_vec) - a_vec
    return_val *= k / (1 + (k * np.linalg.norm(L)) ** 2)

    return return_val


def Phase(time_from_traj, position_x, position_y, position_z, velocity_x, velocity_y, velocity_z):
    """
    Calculate the phase using trajectory data.
    """
    momentum_array = velocity_x ** 2 + velocity_y ** 2 + velocity_z ** 2
    r_array = np.sqrt(position_x ** 2 + position_y ** 2 + position_z ** 2)

    return_value = -1 * (position_x[0] * velocity_x[0] + position_y[0]
                         * velocity_y[0] + position_z[0] * velocity_z[0])
    return_value += Ip * time_from_traj[0]
    return_value -= integrate.simps(momentum_array /
                                    2 - 2 * Z / r_array, time_from_traj)

    r_final = np.array([position_x[-1], position_y[-1], position_z[-1]])
    p_final = np.array([velocity_x[-1], velocity_y[-1], velocity_z[-1]])

    energy = pow(np.linalg.norm(p_final), 2) / 2 - Z / np.linalg.norm(r_final)
    L = np.linalg.norm(np.cross(r_final, p_final))

    b = 1 / (2 * energy)
    g = sqrt(1 + 2 * energy * L * L)

    return_value -= Z * \
        sqrt(b) * (log(g) + asinh(np.dot(r_final, p_final) / (g * sqrt(b))))
    return return_value


def Phase_QTMC(time_from_traj, position_x, position_y, position_z, velocity_x, velocity_y, velocity_z):
    """
    Calculate the phase using Quantum Trajectory Monte Carlo (QTMC) data.
    """
    momentum_array = velocity_x ** 2 + velocity_y ** 2 + velocity_z ** 2
    r_array = np.sqrt(position_x ** 2 + position_y ** 2 + position_z ** 2)

    return_value = -1 * (position_x[0] * velocity_x[0] + position_y[0]
                         * velocity_y[0] + position_z[0] * velocity_z[0])
    return_value += Ip * time_from_traj[0]
    return_value -= integrate.simps(momentum_array /
                                    2 - Z / r_array, time_from_traj)

    return return_value


def Derivative_RK45(to, phase_space):
    """
    Calculate the derivative of the phase space using the Runge-Kutta method (RK45).
    """
    e_field_z = 0
    e_field_x = Fo * (cos(omega * (to - tau / 2) + CEP) * sin(pi * to / tau) ** 2 +
                      sin(omega * (to - tau / 2) + CEP) * sin(2 * pi * to / tau) / (2 * cycle))
    e_field_y = ell * Fo * (sin(omega * (to - tau / 2) + CEP) * sin(pi * to / tau) ** 2 -
                            cos(omega * (to - tau / 2) + CEP) * sin(2 * pi * to / tau) / (2 * cycle))

    x, y, z = phase_space[0], phase_space[1], phase_space[2]
    r2 = x * x + y * y + z * z

    vx_derivative, vy_derivative, vz_derivative = -1 * e_field_x - x * Z / pow(r2 + 0.001, 3 / 2), \
        -1 * e_field_y - y * Z / pow(r2 + 0.001, 3 / 2), \
        -1 * e_field_z - z * Z / \
        pow(r2 + 0.001, 3 / 2)

    return [phase_space[3], phase_space[4], phase_space[5], vx_derivative, vy_derivative, vz_derivative]


def Sin_2(time_array):
    """
    Calculate the sine squared of an array.
    """
    return sin(time_array * pi / tau) ** 2


def Traj_Info_Of_Interest(position_x, position_y, position_z):
    """
    Calculate the trajectory information of interest given position data.
    """
    path = np.zeros(shape=(len(position_x), 3))
    path[:, 0], path[:, 1], path[:, 2] = position_x, position_y, position_z
    displacement = np.sqrt(np.sum(np.diff(path, axis=0) ** 2, axis=1))
    displacement = np.sum(displacement)

    return displacement


def Save_Phase_Space_Variables(time_of_ionization, time_of_tunneling, final_momentum_x, final_momentum_y,
                               final_momentum_z, phase_array, time_of_flight, displacement_array,
                               inital_momentum_x, inital_momentum_y, inital_momentum_z,
                               final_momentum_tau_x, final_momentum_tau_y, final_momentum_tau_z,
                               inital_position_x, inital_position_y, inital_position_z,
                               v_parall_save, v_perp_save):
    """
    Save various phase space variables to files.
    """
    file_name = file_name_location + "/Final_Mom_X_Axis_" + str(temp_rank)
    np.save(file_name, final_momentum_x)

    file_name = file_name_location + "/Final_Mom_Y_Axis_" + str(temp_rank)
    np.save(file_name, final_momentum_y)

    file_name = file_name_location + "/Final_Mom_Z_Axis_" + str(temp_rank)
    np.save(file_name, final_momentum_z)

    file_name = file_name_location + "/Phase_" + str(temp_rank)
    np.save(file_name, phase_array)

    file_name = file_name_location + "/Time_Flight_" + str(temp_rank)
    np.save(file_name, time_of_flight)

    file_name = file_name_location + "/Time_Ion_" + str(temp_rank)
    np.save(file_name, time_of_ionization)

    file_name = file_name_location + "/Time_Tunneling_" + str(temp_rank)
    np.save(file_name, time_of_tunneling)

    file_name = file_name_location + "/Displacment_" + str(temp_rank)
    np.save(file_name, displacement_array)

    file_name = file_name_location + "/Inital_Mom_X_Axis_" + str(temp_rank)
    np.save(file_name, inital_momentum_x)

    file_name = file_name_location + "/Inital_Mom_Y_Axis_" + str(temp_rank)
    np.save(file_name, inital_momentum_y)

    file_name = file_name_location + "/Inital_Mom_Z_Axis_" + str(temp_rank)
    np.save(file_name, inital_momentum_z)

    file_name = file_name_location + "/Final_Mom_Tau_X_Axis_" + str(temp_rank)
    np.save(file_name, final_momentum_tau_x)

    file_name = file_name_location + "/Final_Mom_Tau_Y_Axis_" + str(temp_rank)
    np.save(file_name, final_momentum_tau_y)

    file_name = file_name_location + "/Final_Mom_Tau_Z_Axis_" + str(temp_rank)
    np.save(file_name, final_momentum_tau_z)

    file_name = file_name_location + "/Inital_Pos_X_Axis_" + str(temp_rank)
    np.save(file_name, inital_position_x)

    file_name = file_name_location + "/Inital_Pos_Y_Axis_" + str(temp_rank)
    np.save(file_name, inital_position_y)

    file_name = file_name_location + "/Inital_Pos_Z_Axis_" + str(temp_rank)
    np.save(file_name, inital_position_z)

    file_name = file_name_location + "/V_Parall_" + str(temp_rank)
    np.save(file_name, v_parall_save)

    file_name = file_name_location + "/V_Perp_" + str(temp_rank)
    np.save(file_name, v_perp_save)


def Generate_Inital_Condition():
    """
    Generate initial conditions for the simulation.
    """
    t_ion_array_return, tunneling_time_array_return, kx_array_return, ky_array_return, kz_array_return, \
        x_pos_array_return, y_pos_array_return, v_parall_return, v_perp_return = \
        np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), \
        np.array([])

    for itr in range(200):
        if rank == 0:
            print("Current iteration:", itr)
        t_ion_inital = np.random.uniform(0, tau, total_traj)
        v_perp_inital = np.random.uniform(-4., 4., total_traj)
        kz_inital = np.random.uniform(-4., 4., total_traj)

        t_ion_return, v_perp_return, kz_return, tunneling_time_return = \
            Elliptical_Tunneling_Time(t_ion_inital, v_perp_inital, kz_inital)
        gc.collect()

        t_ion_array_temp, tunneling_time_temp, kx_array_temp, ky_array_temp, kz_array_temp, x_pos_array_temp, \
            y_pos_array_temp, v_parall_temp, v_perp_temp = Cosh_Dependent_Terms(t_ion_return, v_perp_return,
                                                                                kz_return, tunneling_time_return)
        gc.collect()

        t_ion_array_return = np.append(t_ion_array_return, t_ion_array_temp)
        tunneling_time_array_return = np.append(
            tunneling_time_array_return, tunneling_time_temp)
        kx_array_return = np.append(kx_array_return, kx_array_temp)
        ky_array_return = np.append(ky_array_return, ky_array_temp)
        kz_array_return = np.append(kz_array_return, kz_array_temp)

        x_pos_array_return = np.append(x_pos_array_return, x_pos_array_temp)
        y_pos_array_return = np.append(y_pos_array_return, y_pos_array_temp)

        v_parall_return = np.append(v_parall_return, v_parall_temp)
        v_perp_return = np.append(v_perp_return, v_perp_temp)

    return t_ion_array_return, tunneling_time_array_return, kx_array_return, ky_array_return, kz_array_return, \
        x_pos_array_return, y_pos_array_return, v_parall_return, v_perp_return


def CTMC():
    """
    Perform the CTMC simulation.
    """
    t_ion_array_return, tunneling_time_array_return, kx_array_return, ky_array_return, kz_array_return, \
        x_pos_array_return, y_pos_array_return, v_parall_return, v_perp_return = Generate_Inital_Condition()

    inital_position_x, inital_position_y, inital_position_z = [], [], []
    inital_momentum_x, inital_momentum_y, inital_momentum_z = [], [], []
    final_momentum_x, final_momentum_y, final_momentum_z = [], [], []
    final_momentum_tau_x, final_momentum_tau_y, final_momentum_tau_z = [], [], []
    v_parall_save, v_perp_save = [], []
    displacment_array, time_of_flight, phase_array = [], [], []
    time_of_ionization = []
    time_of_tunneling = []
    status_list = np.arange(0, len(t_ion_array_return),
                            int(len(t_ion_array_return) / 10))

    for i, t_ion in enumerate(t_ion_array_return):
        if i in status_list:
            print(temp_rank, i, len(t_ion_array_return))

        time_of_traj = np.arange(t_ion, tau + dt, dt)

        x_mom_inital = kx_array_return[i]
        y_mom_inital = ky_array_return[i]
        z_mom_inital = kz_array_return[i]

        x_pos_inital = x_pos_array_return[i]
        y_pos_inital = y_pos_array_return[i]
        z_pos_inital = 0

        r_pos_inital = sqrt(x_pos_inital ** 2 +
                            y_pos_inital ** 2 + z_pos_inital ** 2)

        if r_pos_inital <= 5:
            continue

        soln = solve_ivp(Derivative_RK45, (t_ion, tau + 2 * dt),
                         [x_pos_inital, y_pos_inital, z_pos_inital,
                             x_mom_inital, y_mom_inital, z_mom_inital],
                         t_eval=time_of_traj, rtol=1e-3, atol=1e-6)
        time_from_traj = soln.t
        position_x, position_y, position_z, velocity_x, velocity_y, velocity_z = soln.y[0], soln.y[1], soln.y[2], soln.y[3], \
            soln.y[4], soln.y[5]

        r_array = np.sqrt(position_x ** 2 + position_y ** 2 + position_z ** 2)

        if (r_array <= 5).any():
            continue

        r_end = [position_x[-1], position_y[-1], position_z[-1]]
        p_end = [velocity_x[-1], velocity_y[-1], velocity_z[-1]]

        asymptotic_momentum = Asymptotic_Momentum(r_end, p_end)
        fmx = asymptotic_momentum[0]
        fmy = asymptotic_momentum[1]
        fmz = asymptotic_momentum[2]

        if fmx is not None and fmy is not None and fmz is not None:
            final_momentum_x.append(fmx)
            final_momentum_y.append(fmy)
            final_momentum_z.append(fmz)
            phase_array.append(Phase(time_from_traj, position_x, position_y, position_z, velocity_x, velocity_y,
                                     velocity_z))

            final_momentum_tau_x.append(velocity_x[-1])
            final_momentum_tau_y.append(velocity_y[-1])
            final_momentum_tau_z.append(velocity_z[-1])

            inital_position_x.append(x_pos_inital)
            inital_position_y.append(y_pos_inital)
            inital_position_z.append(z_pos_inital)

            inital_momentum_x.append(x_mom_inital)
            inital_momentum_y.append(y_mom_inital)
            inital_momentum_z.append(z_mom_inital)

            v_parall_save.append(v_parall_return[i])
            v_perp_save.append(v_perp_return[i])

            time_of_flight.append(tau - t_ion)
            displacment_array.append(Traj_Info_Of_Interest(
                position_x, position_y, position_z))
            time_of_ionization.append(t_ion)
            time_of_tunneling.append(tunneling_time_array_return[i])

    Save_Phase_Space_Variables(time_of_ionization, time_of_tunneling, final_momentum_x, final_momentum_y, final_momentum_z, phase_array, time_of_flight,\
                            displacment_array, inital_momentum_x, inital_momentum_y, inital_momentum_z, final_momentum_tau_x,\
                            final_momentum_tau_y, final_momentum_tau_z, inital_position_x, inital_position_y, inital_position_z, v_parall_save, v_perp_save)        


def Initial_Momentum(t, v_perp_i):
    """
    Calculate the initial momentum.
    """
    gamma_new = omega * sqrt(2 * Ip + v_perp_i ** 2) / \
        abs(Fo * cos(omega * t) * (sin(t * pi / tau) ** 2))
    potential = -Fo * Envelop_Fun(t) * sin(omega * t) / omega

    return -potential * (sqrt(1 + gamma_new * gamma_new) - 1)


def Exit_Position(field_t):
    """
    Calculate the exit position.
    """
    field_t = field_t + 1e-10
    return_value = Ip + sqrt(Ip * Ip - (4 - sqrt(8 * Ip)) * field_t)
    return_value = return_value / (2 * field_t)
    return return_value


if __name__ == "__main__":

    temp_rank = rank 

    np.random.seed(temp_rank + 999)

    Ip = 0.834567
    Z = sqrt(2 * Ip)
    omega = 0.076
    cycle = 10
    intensity = 3e14
    ell = 0.7
    dt = 0.1
    total_traj = int(4.e7)
    Fo = pow(intensity / 3.51e16, 0.5) / sqrt(1 + ell * ell)
    pulse_period = 2 * np.pi / omega
    tau = 2 * np.pi * cycle / omega
    CEP = 0
    Up = (1 + (ell ** 2)) * Fo * Fo / (4 * (omega ** 2))
    Envelop_Fun = Sin_2

    file_name_location = "/mpdata/becker/yoge8051/Research/TDSE/Monte_Carlo/Data/Neon/Elliptical/600nm/Plus"

    CTMC()



