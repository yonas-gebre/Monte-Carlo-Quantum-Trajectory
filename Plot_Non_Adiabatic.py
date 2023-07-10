import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.signal import find_peaks
import os
import gc

# Set environment variable
os.environ.update(
    {"QT_QPA_PLATFORM_PLUGIN_PATH": "~/anaconda3/envs/research-headless/lib/python3.8/site-packages/PyQt5/Qt5/plugins/xcbglintegrations/libqxcb-glx-integration.so"})


def PAD(major_axis_passed, minor_axis_passed, propagation_axis_passed, phase_array_passed, plot_name):
    resolution = 0.01
    x_momentum = np.arange(-1.5, 1.5 + resolution, resolution)
    y_momentum = np.arange(-1.5, 1.5 + resolution, resolution)
    y_momentum = np.flip(y_momentum)
    z_momentum = np.arange(-1.5, 1.5 + resolution, resolution)

    pad_value = np.zeros((y_momentum.size, x_momentum.size))

    for i, px in enumerate(x_momentum):
        print(round(px, 3))
        # Select major axis values within the specified range
        first_idx = np.where(np.logical_and(
            major_axis_passed >= px - resolution/2, major_axis_passed <= px + (resolution/2)))[0]
        minor_axis_temp = minor_axis_passed[first_idx]
        propagation_axis_temp = propagation_axis_passed[first_idx]
        phase_temp = phase_array_passed[first_idx]

        for j, py in enumerate(y_momentum):
            # Select minor axis values within the specified range
            second_idx = np.where(np.logical_and(
                minor_axis_temp >= py - resolution/2, minor_axis_temp <= py + resolution/2))[0]
            propagation_axis_temp_two = propagation_axis_temp[second_idx]
            phase_temp_two = phase_temp[second_idx]

            for l, pz in enumerate(z_momentum):
                # Select propagation axis values within the specified range
                third_idx = np.where(np.logical_and(
                    propagation_axis_temp_two >= pz - resolution/2, propagation_axis_temp_two <= pz + resolution/2))[0]
                phase = phase_temp_two[third_idx]
                value = np.exp(-1.0j * phase)
                value = np.sum(value)

                pad_value[j, i] += np.absolute(value) ** 2

    pad_value = pad_value / pad_value.max() + 1e-12
    pos = plt.imshow(pad_value, cmap='jet',
                     extent=[-1.5, 1.5, -1.5, 1.5], norm=LogNorm(vmin=pow(10, -3), vmax=1))

    plt.xlabel(r'$P_{x}$')
    plt.ylabel(r'$P_{y}$')
    plt.tight_layout()
    plt.colorbar(pos)
    plt.grid(color="cyan")
    plt.savefig(plot_name)
    plt.clf()
    # plt.show()


def PAD_Angular_Two(Final_K, Final_Phi, Final_Thetha, phase_array):
    resolution = 0.05

    angle_small, angle_large = -270, 450
    phi_array = np.arange(angle_small * 2 * np.pi / 360,
                          angle_large * 2 * np.pi / 360, 0.05)

    energy_max = 20
    energy_array = np.arange(0, energy_max / 27.21, 0.005)
    phi_array = np.flip(phi_array)

    phi_array += 4 * np.pi
    theta_array = np.arange(np.pi / 2 - 0.15, np.pi / 2 + 0.15 + 0.005, 0.005)

    pad_value = np.zeros((phi_array.size, energy_array.size))
    pad_value_assymetry = np.zeros((phi_array.size, energy_array.size))

    for i, energy in enumerate(energy_array):
        print(round(energy, 3))
        k = np.sqrt(2 * energy)

        k_axis_idx = np.where(np.logical_and(
            Final_K >= k - resolution / 2, Final_K <= k + resolution / 2))[0]
        Final_Phi_temp = Final_Phi[k_axis_idx]
        Final_Thetha_temp = Final_Thetha[k_axis_idx]
        phase_array_temp = phase_array[k_axis_idx]

        for j, phi in enumerate(phi_array):
            phi_two = phi + np.pi
            phi = phi % (2 * np.pi)
            phi_two = phi_two % (2 * np.pi)

            phi_axis_idx = np.where(np.logical_and(
                Final_Phi_temp >= phi - resolution / 2, Final_Phi_temp <= phi + (resolution / 2)))[0]
            Final_Thetha_temp_two = Final_Thetha_temp[phi_axis_idx]
            phase_array_temp_two = phase_array_temp[phi_axis_idx]

            phi_axis_idx_B = np.where(np.logical_and(
                Final_Phi_temp >= phi_two - resolution / 2, Final_Phi_temp <= phi_two + (resolution / 2)))[0]
            Final_Thetha_temp_two_B = Final_Thetha_temp[phi_axis_idx_B]
            phase_array_temp_two_B = phase_array_temp[phi_axis_idx_B]

            pad_assymetry_value = 0.0

            for l, theta in enumerate(theta_array):
                theta_axis_idx = np.where(np.logical_and(
                    Final_Thetha_temp_two >= theta - resolution / 2, Final_Thetha_temp_two <= theta + resolution / 2))[0]
                phase = phase_array_temp_two[theta_axis_idx]

                value = np.exp(-1.0j * phase)
                value = np.sum(value)
                value = np.absolute(value) ** 2
                pad_value[j, i] += value

                theta_axis_idx = np.where(np.logical_and(
                    Final_Thetha_temp_two_B >= theta - resolution / 2, Final_Thetha_temp_two_B <= theta + resolution / 2))[0]
                phase = phase_array_temp_two_B[theta_axis_idx]

                value_two = np.exp(-1.0j * phase)
                value_two = np.sum(value_two)
                value_two = np.absolute(value_two) ** 2

                pad_assymetry_value += (value - value_two) / \
                    (value + value_two + 1e-16)

            pad_value_assymetry[j, i] += pad_assymetry_value / len(theta_array)

    pad_value = pad_value / pad_value.max() + 1e-16
    plt.rcParams.update({'font.size': 16})

    pos = plt.imshow(pad_value, cmap='bwr', extent=[
                     0, energy_max, angle_small, angle_large], aspect='auto', norm=LogNorm(vmin=pow(10, -3), vmax=1))
    plt.xlabel(r'$E_{kin}$')
    plt.ylabel(r'$\phi_{abs}(deg)$')
    plt.tight_layout()
    plt.colorbar(pos)
    plt.savefig("PAD400_Angular_Ell_Pulse_102.png")
    plt.clf()

    pos = plt.imshow(pad_value_assymetry, cmap='bwr', extent=[
                     0, energy_max, angle_small, angle_large], vmin=-1, vmax=1, aspect='auto')
    print(np.absolute(pad_value_assymetry).max())
    plt.xlabel(r'$E_{kin}$')
    plt.ylabel(r'$\phi_{abs}(deg)$')
    plt.tight_layout()
    plt.colorbar(pos)
    plt.savefig("PAD400_Assym_Ell_Pulse_102.png")
    plt.clf()


def PAD_Asym_At_90(Final_K, Final_Phi, Final_Thetha, phase_array):
    
    resolution = 0.01
    
    energy_max = 20
    energy_array = np.arange(0, energy_max/27.21, 0.005)
    pad_value = np.zeros(len(energy_array))
    theta_array = theta_array = np.arange(0, pi, 0.005)#np.arange(pi/2 - 0.15, pi/2 + 0.15 + 0.005, 0.005)

    for i, energy in enumerate(energy_array):
        print(round(energy,3))
        k = sqrt(2*energy)

        k_axis_idx = np.where(np.logical_and(Final_K >= k - resolution/2, Final_K <= k + resolution/2 ))[0]
        Final_Phi_temp = Final_Phi[k_axis_idx]
        Final_Thetha_temp = Final_Thetha[k_axis_idx]
        phase_array_temp = phase_array[k_axis_idx]

        phi = pi/2           
        phi_two = -pi/2

        phi_axis_idx = np.where(np.logical_and(Final_Phi_temp >= phi - resolution/2, Final_Phi_temp <= phi + (resolution/2)))[0]
        Final_Thetha_temp_two = Final_Thetha_temp[phi_axis_idx]
        phase_array_temp_two = phase_array_temp[phi_axis_idx]

        phi_axis_idx_B = np.where(np.logical_and(Final_Phi_temp >= phi_two - resolution/2, Final_Phi_temp <= phi_two + (resolution/2)))[0]
        Final_Thetha_temp_two_B = Final_Thetha_temp[phi_axis_idx_B]
        phase_array_temp_two_B = phase_array_temp[phi_axis_idx_B]

        pad_assymetry_value = 0.0

        for l, theta in enumerate(theta_array):  
        
            theta_axis_idx = np.where(np.logical_and(Final_Thetha_temp_two >= theta - resolution/2, Final_Thetha_temp_two <= theta + resolution/2 ))[0]
            phase = phase_array_temp_two[theta_axis_idx]

            value = np.exp(-1.0j*phase)
            value = np.sum(value)
            value = np.absolute(value)**2
            

            
            theta_axis_idx = np.where(np.logical_and(Final_Thetha_temp_two_B >= theta - resolution/2, Final_Thetha_temp_two_B <= theta + resolution/2 ))[0]
            phase = phase_array_temp_two_B[theta_axis_idx]

            value_two = np.exp(-1.0j*phase)
            value_two = np.sum(value_two)
            value_two = np.absolute(value_two)**2

            
            
            pad_assymetry_value += (value - value_two)/(value + value_two + 1e-16)
    
        pad_value[i] += pad_assymetry_value/len(theta_array)
        
    plt.rcParams.update({'font.size': 16})

    plt.plot(energy_array*27.21, pad_value)
    plt.xlabel(r'$E_{kin}$')
    plt.ylabel(r'Assymetry')
    plt.tight_layout()
    # plt.set_cmap('bwr') 
    plt.savefig("PAD400_NA_Assym_at_90_Ell2.png")

    plt.clf()

def PAD_Integrated(Final_K, Final_Phi, Final_Thetha, Phase):
    
    #0.005
        
    resolution = 0.005
    resolution_two = 0.005


    k_array = np.arange(0.0, 2., resolution)
    phi_array = np.arange(0, 2*pi, resolution_two)
    theta_array = np.arange(0 , pi, resolution_two)

    theta_array = np.array([pi/2])
    e_array = k_array*k_array/2
    pad_value = np.zeros(len(k_array))

    pad_value_2D = np.zeros((theta_array.size,phi_array.size))

    for i, k in enumerate(k_array):
        print(round(k, 5))

        k_axis_idx = np.where(np.logical_and(Final_K >= k - resolution/2, Final_K <= k + resolution/2 ))[0]
        
        Final_Phi_temp = Final_Phi[k_axis_idx]
        Final_Thetha_temp = Final_Thetha[k_axis_idx]
        Phase_temp = Phase[k_axis_idx]


        for j, phi in enumerate(phi_array):           
            
            phi_axis_idx = np.where(np.logical_and(Final_Phi_temp >= phi - resolution_two/2, Final_Phi_temp <= phi + (resolution_two/2)))[0]
            Final_Thetha_temp_two = Final_Thetha_temp[phi_axis_idx]
            Phase_temp_two = Phase_temp[phi_axis_idx]
            
            phase = Phase_temp_two
            value = np.exp(1.0j*phase)
            value = np.sum(value)
            value = np.absolute(value)**2
            pad_value[i] += value

    peaks, _ = find_peaks(pad_value, height = 0.2, threshold=0.2, distance=2)
    return k_array[peaks]

    E_int = 0.001
    k_int = sqrt(2*E_int)

    idx, elemnt = min(enumerate(k_array), key=lambda x: abs(x[1]-k_int))

    pad_value[idx:] = pad_value[idx:] / (k_array[idx:])**2

    pad_value = pad_value/pad_value.max()

    plt.plot(e_array, pad_value)
    plt.xlim(0, 1.5)
    # plt.ylim(1e-4, 1)
    plt.grid()
    # plt.savefig("Ell-71-500-Plus.png")

    # np.save("CTMC_Ell-71-500-E-Plus.npy", e_array)
    # np.save("CTMC_Ell-71-500-P-Plus.npy", pad_value)
    # exit()
    plt.clf()
    

def PAD_Energy_Angle(Final_K, Final_Phi, Final_Thetha, Phase):
    
    #0.005
        
    resolution = 0.005
    resolution_two = 0.005


    e_array = np.arange(0.0, 0.75, resolution)
    phi_array = np.arange(-pi/8, pi - pi/8, resolution_two)
    theta_array = np.arange(0 , pi, resolution_two)
    phi_array = np.flip(phi_array)

    pad_value_2D = np.zeros((phi_array.size,e_array.size))

    off_set_values = {}

    for i, e in enumerate(e_array):
        k = sqrt(2*e)
        print(round(k, 5))

        k_axis_idx = np.where(np.logical_and(Final_K >= k - resolution/2, Final_K <= k + resolution/2 ))[0]
        
        Final_Phi_temp = Final_Phi[k_axis_idx]
        Final_Thetha_temp = Final_Thetha[k_axis_idx]
        Phase_temp = Phase[k_axis_idx]

        phi_array_for_max = np.zeros(phi_array.size)

        for j, phi in enumerate(phi_array):           
            
            phi = phi + 2*pi
            phi = phi%(2*pi)

            phi_axis_idx = np.where(np.logical_and(Final_Phi_temp >= phi - resolution_two/2, Final_Phi_temp <= phi + (resolution_two/2)))[0]
            Final_Thetha_temp_two = Final_Thetha_temp[phi_axis_idx]
            Phase_temp_two = Phase_temp[phi_axis_idx]
            
            phase = Phase_temp_two
            value = np.exp(1.0j*phase)
            value = np.sum(value)
            value = np.absolute(value)**2

            phi_array_for_max[j] += value

            pad_value_2D[j, i] += value

        off_set_idx = np.argmax(phi_array_for_max/phi_array_for_max.max())

        off_set_values[i] = (e, phi_array[off_set_idx])

    pad_value_2D = pad_value_2D / pad_value_2D.max() + 1e-12
    pos = plt.imshow(pad_value_2D, cmap='jet', aspect='auto', extent=[e_array.min(), e_array.max(), phi_array.min(), phi_array.max()],norm=LogNorm(vmin=pow(10, -2), vmax=1))

    x_axis_data = np.array(list(off_set_values.values()))[:,0]
    y_axis_data = np.array(list(off_set_values.values()))[:,1]
    
    black_plot_idx = np.where(x_axis_data > 0.1)[0]


    plt.plot(np.array(list(off_set_values.values()))[:,0][black_plot_idx], np.array(list(off_set_values.values()))[:,1][black_plot_idx], color="black", linewidth=2)

    plt.xlabel(r'$E_{f}$')
    plt.ylabel(r'$\phi_{f}$')
    plt.tight_layout()
    plt.colorbar(pos)
    plt.grid(color = "cyan")
    plt.savefig("PAD-Ell-71-500-E-Vs-Phi-Plus2.png")
    # plt.show()
   
    plt.clf()

    # return k_array[peaks]

def PAD_K_VS_X(Final_K, Final_Phi, Final_Thetha, T_Of_Ion, phase_array):

    #0.005
        
    resolution = 0.025
    resolution_two = 0.01#resolution*2
    resolution_three = 1

    k_array = np.arange(0.0, 1.5, resolution)
    phi_array = np.arange(0, 2*pi, resolution_two)
    theta_array = np.arange(0 , pi, resolution_two)

    time_of_ion = np.arange(100, 450, resolution_three)

    e_array = k_array*k_array/2
    
    pad_value = np.zeros((time_of_ion.size,k_array.size))

    for l, t in enumerate(time_of_ion): 
        print(round(t, 5))
        
        t_axis_idx = np.where(np.logical_and(T_Of_Ion >= t - resolution_three/2, T_Of_Ion <= t + resolution_three/2 ))[0]

        Final_K_A = Final_K[t_axis_idx]
        Final_Phi_A = Final_Phi[t_axis_idx]
        Final_Thetha_A = Final_Thetha[t_axis_idx]
        phase_array_A = phase_array[t_axis_idx]
        
        for i, k in enumerate(k_array):
            

            k_axis_idx = np.where(np.logical_and(Final_K_A >= k - resolution/2, Final_K_A <= k + resolution/2 ))[0]
            
            Final_Phi_B = Final_Phi_A[k_axis_idx]
            Final_Thetha_B = Final_Thetha_A[k_axis_idx]
            phase_array_B = phase_array_A[k_axis_idx]

            for j, phi in enumerate(phi_array):           
                
                phi = phi%(2*pi)
                phi_axis_idx = np.where(np.logical_and(Final_Phi_B >= phi - resolution_two/2, Final_Phi_B <= phi + (resolution_two/2)))[0]
                
                Final_Thetha_C = Final_Thetha_B[phi_axis_idx]
                phase_array_C = phase_array_B[phi_axis_idx]
                

                for l, theta in enumerate(theta_array):  
                
                    theta_axis_idx = np.where(np.logical_and(Final_Thetha_C >= theta - resolution_two/2, Final_Thetha_C <= theta + resolution_two/2 ))[0]
                    phase = phase_array_C[theta_axis_idx]

                    value = np.exp(-1.0j*phase)
                    value = np.sum(value)
                    value = np.absolute(value)**2
                    
                    pad_value[l, i] += value

   
    
    pad_value = pad_value / pad_value.max() + 1e-12
    pos = plt.imshow(pad_value, cmap='jet', extent=[-1.5, 1.5, 100, 450],norm=LogNorm(vmin=pow(10, -2), vmax=1))

    plt.xlabel(r'$K$')
    plt.ylabel(r'$T_{ion}$')
    plt.tight_layout()
    plt.colorbar(pos)
    plt.grid(color = "cyan")
    plt.savefig("PES_K_T-ion.png")
    # plt.show()
   
    plt.clf()

    return 

def PAD_Integrated_New(Final_K, Final_Phi, Final_Thetha, phase_array):
    
        
    resolution = 0.005
    e_array = np.arange(0, 1.5, resolution)
    phi_array = np.arange(0, 2*pi, resolution*2)
    theta_array = np.arange(0 , pi, resolution*2)

    
    pad_value = np.zeros(len(e_array))


    for i, e in enumerate(e_array):
        print(round(e, 3))

        k = np.sqrt(2*e)
        k_axis_idx = np.where(np.logical_and(Final_K >= k - resolution/2, Final_K <= k + resolution/2 ))[0]
        
        Final_Phi_temp = Final_Phi[k_axis_idx]
        Final_Thetha_temp = Final_Thetha[k_axis_idx]
        phase_array_temp = phase_array[k_axis_idx]

        for j, phi in enumerate(phi_array):           
            
            phi = phi%(2*pi)
            phi_axis_idx = np.where(np.logical_and(Final_Phi_temp >= phi - resolution/2, Final_Phi_temp <= phi + (resolution/2)))[0]
            Final_Thetha_temp_two = Final_Thetha_temp[phi_axis_idx]
            phase_array_temp_two = phase_array_temp[phi_axis_idx]
            

            for l, theta in enumerate(theta_array):  
            
                theta_axis_idx = np.where(np.logical_and(Final_Thetha_temp_two >= theta - resolution/2, Final_Thetha_temp_two <= theta + resolution/2 ))[0]
                phase = phase_array_temp_two[theta_axis_idx]

                value = np.exp(-1.0j*phase)
                value = np.sum(value)
                value = np.absolute(value)**2
                pad_value[i] += value*np.sin(theta)


    E_int = 0.001
    k_int = sqrt(2*E_int)
    # idx, elemnt = min(enumerate(k_array), key=lambda x: abs(x[1]-k_int))
    # pad_value[idx:] = pad_value[idx:] / (k_array[idx:])**2

    pad_value /= pad_value.max()

    np.save("CTMC_Circular_500_E_Plus_Loss.npy", e_array)
    np.save("CTMC_Circular_500_P_Plus_Loss.npy", pad_value)

    exit()
    
    peaks, _ = find_peaks(pad_value, height = 1e-2, distance=2)
    peaks = np.delete(peaks, [0])
    plt.plot(k_array[peaks], pad_value[peaks], "x", label="peaks")

    plt.plot(k_array, pad_value)
    plt.savefig("PES.png")
    plt.clf()
    return k_array[peaks]

def ATI_Peak_Plots(Final_K, Final_Phi, Final_Thetha, phase_array, k_peaks, v_paral, v_perp, displacment_array, time_of_fligh, L):
    
    resolution = 0.01
    
    x_array = v_paral
    x_array_range = np.arange(-0.75, 0.75, resolution)
    # x_array_range = np.arange(150, 400, 1)
    
    plt.rcParams.update({'font.size': 16})
    for i in range(3):
        k_peak_idx = np.where(np.logical_and(Final_K >= k_peaks[i] - resolution/2, Final_K <= k_peaks[i]+(resolution/2)))[0]
        
        Final_Phi_temp = Final_Phi[k_peak_idx]
        Final_Thetha_temp = Final_Thetha[k_peak_idx]
        phase_array_temp = phase_array[k_peak_idx]
        x_array_temp = x_array[k_peak_idx]
                
        pad_value = Single_ATI_Peak_Plots(Final_Phi_temp, Final_Thetha_temp, phase_array_temp, x_array_temp, x_array_range, resolution)
        
        plt.plot(x_array_range, pad_value)#, label= "peak:" + str(i+1))
    
    plt.legend()
    
    # plt.title("Yield as a function of inital angular momentum")
    plt.xlabel(r'$P_{||}$')
    plt.ylabel(r'$Yield$')
    plt.tight_layout()
    plt.savefig("P_parallel3.png")
    plt.clf()
    
def Single_ATI_Peak_Plots(Final_Phi, Final_Thetha, phase_array, x_array, x_array_range, resolution):
    
    phi_array = np.arange(0, 2*pi, resolution*2)
    theta_array = np.arange(0 , pi, resolution*2)

    resolution_x = abs(x_array_range[1] - x_array_range[0])

    pad_value = np.zeros(len(x_array_range))
    
    for i, x in enumerate(x_array_range):
        print(round(x, 3))

        k_axis_idx = np.where(np.logical_and(x_array >= x - resolution_x/2, x_array <= x + resolution_x/2 ))[0]
        
        Final_Phi_temp = Final_Phi[k_axis_idx]
        Final_Thetha_temp = Final_Thetha[k_axis_idx]
        phase_array_temp = phase_array[k_axis_idx]

        for j, phi in enumerate(phi_array):           
            
            phi = phi%(2*pi)
            phi_axis_idx = np.where(np.logical_and(Final_Phi_temp >= phi - resolution/2, Final_Phi_temp <= phi + (resolution/2)))[0]
            Final_Thetha_temp_two = Final_Thetha_temp[phi_axis_idx]
            phase_array_temp_two = phase_array_temp[phi_axis_idx]
            

            for l, theta in enumerate(theta_array):  
            
                theta_axis_idx = np.where(np.logical_and(Final_Thetha_temp_two >= theta - resolution/2, Final_Thetha_temp_two <= theta + resolution/2 ))[0]
                phase = phase_array_temp_two[theta_axis_idx]

                value = np.exp(-1.0j*phase)
                value = np.sum(value)
                value = np.absolute(value)**2
                pad_value[i] += value

    pad_value /= pad_value.max()
    
    return pad_value
    
if __name__ == "__main__":
    
        
    file_name_location = "/mpdata/becker/yoge8051/Research/TDSE/Monte_Carlo/Data/Circular/New/500nm/Plus3"

    phase_array = []
    v_paral = []
    v_perp = []

    for file_no in range(200):
        print("file", file_no)

        file_name = file_name_location + "/Phase_" + str(file_no) + ".npy"
        phase_array.extend(np.load(file_name))

        file_name = file_name_location + \
            "/Final_Mom_X_Axis_" + str(file_no) + ".npy"
        v_paral.extend(np.load(file_name))

        file_name = file_name_location + \
            "/Final_Mom_Y_Axis_" + str(file_no) + ".npy"
        v_perp.extend(np.load(file_name))

    phase_array = np.array(phase_array)
    v_paral = np.array(v_paral)
    v_perp = np.array(v_perp)

    Final_K_X = np.array(v_paral)
    gc.collect()

    print("{:e}".format(len(Final_K_X)))

    Final_K_Y = np.array(v_perp)
    gc.collect()

    Final_K_Z = np.array(v_paral)
    gc.collect()

    Final_K = np.sqrt(Final_K_X ** 2 + Final_K_Y ** 2 + Final_K_Z ** 2)
    Final_K2 = np.sqrt(Final_K_X ** 2 + Final_K_Y ** 2)

    Final_Thetha = np.arctan2(Final_K2, Final_K_Z)
    Final_Phi = np.arctan2(Final_K_Y, Final_K_X)
    Final_Phi[Final_Phi < 0] += 2 * np.pi

    gc.collect()

    print("Finished")
    k_peaks = PAD_Integrated_New(Final_K, Final_Phi, Final_Thetha, phase_array)

    displacment_array, time_of_fligh, L = None, None, None

    ATI_Peak_Plots(
        Final_K, Final_Phi, Final_Thetha, phase_array, k_peaks, v_paral, v_perp, displacment_array, time_of_fligh, L
    )
