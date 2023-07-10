import numpy as np
import gc
from numpy import pi, sqrt

if __name__ == "__main__":
    # Initialize lists to store data
    major_axis, minor_axis, propagation_axis = [], [], []
    major_axis_tau, minor_axis_tau, propagation_axis_tau = [], [], []
    major_axis_initial, minor_axis_initial, propagation_axis_initial = [], [], []
    major_initial_position, minor_initial_position, propagation_initial_position = [], [], []
    phase_array, displacement, time_of_flight, time_of_ionization, time_of_tunneling = [], [], [], [], []
    v_parallel, v_perpendicular = [], []

    file_name_location = "/mpdata/becker/yoge8051/Research/TDSE/Monte_Carlo/Data/Neon/Elliptical/600nm/Plus"
    sorted_file_name_location = "/mpdata/becker/yoge8051/Research/TDSE/Monte_Carlo/Sorted_Data/Neon/Elliptical/600nm/Plus"

    for file_no in range(444):
        print("file", file_no)
        if file_no in range(282, 282 + 20) or file_no in range(324, 324 + 16 + 8):
            continue

        # Load phase data
        file_name = file_name_location + "/Phase_" + str(file_no) + ".npy"
        PH = np.load(file_name)
        phase_array += list(PH)

        # Load final momentum data
        file_name = file_name_location + \
            "/Final_Mom_X_Axis_" + str(file_no) + ".npy"
        MA_A = np.load(file_name)
        major_axis += list(MA_A)

        file_name = file_name_location + \
            "/Final_Mom_Y_Axis_" + str(file_no) + ".npy"
        MI_A = np.load(file_name)
        minor_axis += list(MI_A)

        file_name = file_name_location + \
            "/Final_Mom_Z_Axis_" + str(file_no) + ".npy"
        PR_A = np.load(file_name)
        propagation_axis += list(PR_A)

        # Load final momentum tau data
        file_name = file_name_location + \
            "/Final_Mom_Tau_X_Axis_" + str(file_no) + ".npy"
        MA_AT = np.load(file_name)
        major_axis_tau += list(MA_AT)

        file_name = file_name_location + \
            "/Final_Mom_Tau_Y_Axis_" + str(file_no) + ".npy"
        MI_AT = np.load(file_name)
        minor_axis_tau += list(MI_AT)

        file_name = file_name_location + \
            "/Final_Mom_Tau_Z_Axis_" + str(file_no) + ".npy"
        PR_AT = np.load(file_name)
        propagation_axis_tau += list(PR_AT)

        # Load remaining data
        file_name = file_name_location + \
            "/Displacement_" + str(file_no) + ".npy"
        DS = np.load(file_name)
        displacement += list(DS)

        file_name = file_name_location + "/Time_Ion_" + str(file_no) + ".npy"
        TI = np.load(file_name)
        time_of_ionization += list(TI)

        file_name = file_name_location + \
            "/Time_Tunneling_" + str(file_no) + ".npy"
        TT = np.load(file_name)
        time_of_tunneling += list(TT)

        file_name = file_name_location + \
            "/Time_Flight_" + str(file_no) + ".npy"
        TF = np.load(file_name)
        time_of_flight += list(TF)

        file_name = file_name_location + "/V_Parall_" + str(file_no) + ".npy"
        PAR = np.load(file_name)
        v_parallel += list(PAR)

        file_name = file_name_location + "/V_Perp_" + str(file_no) + ".npy"
        PER = np.load(file_name)
        v_perpendicular += list(PER)

        file_name = file_name_location + \
            "/Initial_Mom_X_Axis_" + str(file_no) + ".npy"
        IMA = np.load(file_name)
        major_axis_initial += list(IMA)

        file_name = file_name_location + \
            "/Initial_Mom_Y_Axis_" + str(file_no) + ".npy"
        MAI = np.load(file_name)
        minor_axis_initial += list(MAI)

        file_name = file_name_location + \
            "/Initial_Mom_Z_Axis_" + str(file_no) + ".npy"
        IPA = np.load(file_name)
        propagation_axis_initial += list(IPA)

        file_name = file_name_location + \
            "/Initial_Pos_X_Axis_" + str(file_no) + ".npy"
        MIP = np.load(file_name)
        major_initial_position += list(MIP)

        file_name = file_name_location + \
            "/Initial_Pos_Y_Axis_" + str(file_no) + ".npy"
        MIP = np.load(file_name)
        minor_initial_position += list(MIP)

        file_name = file_name_location + \
            "/Initial_Pos_Z_Axis_" + str(file_no) + ".npy"
        PIP = np.load(file_name)
        propagation_initial_position += list(PIP)

    # Convert lists to numpy arrays
    Final_K_X = np.array(major_axis)
    Final_K_Y = np.array(minor_axis)
    Final_K_Z = np.array(propagation_axis)
    phase_array = np.array(phase_array)
    T_Of_Flight = np.array(time_of_flight)
    T_Of_Ion = np.array(time_of_ionization)
    T_Of_Tun = np.array(time_of_tunneling)
    Disp = np.array(displacement)
    Final_K_X_Tau = np.array(major_axis_tau)
    Final_K_Y_Tau = np.array(minor_axis_tau)
    Final_K_Z_Tau = np.array(propagation_axis_tau)
    Int_K_Parl = np.array(v_parallel)
    Int_K_Perp = np.array(v_perpendicular)
    r_initial = np.zeros(shape=(len(major_initial_position), 3))
    r_initial[:, 0], r_initial[:, 1], r_initial[:,
                                                2] = major_initial_position, minor_initial_position, propagation_initial_position
    p_initial = np.zeros(shape=(len(major_axis_initial), 3))
    p_initial[:, 0], p_initial[:, 1], p_initial[:,
                                                2] = major_axis_initial, minor_axis_initial, propagation_axis_initial
    Angular_KZ = np.cross(r_initial, p_initial)
    Angular_KZ = np.array(Angular_KZ[:, 2])
    Int_K_X = np.array(major_axis_initial)
    Int_K_Y = np.array(minor_axis_initial)
    Int_K_Z = np.array(propagation_axis_initial)
    Int_Pos_X = np.array(major_initial_position)
    Int_Pos_Y = np.array(minor_initial_position)
    Int_Pos_Z = np.array(propagation_initial_position)

    # Clear unused lists from memory
    del major_axis, minor_axis, propagation_axis, major_axis_tau, minor_axis_tau, propagation_axis_tau
    del displacement, time_of_flight, time_of_ionization, time_of_tunneling, v_parallel, v_perpendicular
    del major_initial_position
