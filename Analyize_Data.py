import numpy as np
import sys
import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('TkAgg')
from numpy import pi, sqrt
from matplotlib.colors import LogNorm
from scipy.signal import find_peaks 
from scipy.stats import linregress

import os
import gc

# def Color_map():
#     cdict3 = {'red':  ((0.0, 0.0, 0.0),
#                    (0.25, 0.0, 0.0),
#                    (0.5, 0.8, 1.0),
#                    (0.75, 1.0, 1.0),
#                    (1.0, 0.4, 1.0)),
 
#          'green': ((0.0, 0.0, 0.0),
#                    (0.25, 0.0, 0.0),
#                    (0.5, 0.9, 0.9),
#                    (0.75, 0.0, 0.0),
#                    (1.0, 0.0, 0.0)),
 
#          'blue':  ((0.0, 0.0, 0.4),
#                    (0.25, 1.0, 1.0),
#                    (0.5, 1.0, 0.8),
#                    (0.75, 0.0, 0.0),
#                    (1.0, 0.0, 0.0))
#         }

#     plt.register_cmap(name ='BlueRed3',data = cdict3)

#     cdict4 = {**cdict3,
#             'alpha': ((0.0, 1.0, 1.0),
#                     #   (0.25, 1.0, 1.0),
#                         (0.5, 0.3, 0.3),
#                     #   (0.75, 1.0, 1.0),
#                         (1.0, 1.0, 1.0)),
#             }

   

#     plt.register_cmap(name ='BlueRedAlpha',
#                     data = cdict4)

def Read_Files(sorted_file_name_location):
    
    Int_K_X, Int_K_Y, Int_K_Z = None, None, None
    Final_K_X, Final_K_Y, Final_K_Z = None, None, None
    Final_K_X_Tau, Final_K_Y_Tau, Final_K_Z_Tau = None, None, None
    Int_K_Parl, T_Of_Ion, T_Of_Tun, Int_K, Angular_KZ, Int_K_Perp, Disp = None, None, None, None, None, None, None
    Final_K, Final_Phi, Final_Thetha = None, None, None
    # Int_K_X = np.load(sorted_file_name_location + "/Int_K_X.npy")
    # Int_K_Y = np.load(sorted_file_name_location + "/Int_K_Y.npy")
    # Int_K_Z = np.load(sorted_file_name_location + "/Int_K_Z.npy")

    # Final_K_X = np.load(sorted_file_name_location + "/Final_K_X.npy")
    # Final_K_Y = np.load(sorted_file_name_location + "/Final_K_Y.npy")
    # Final_K_Z = np.load(sorted_file_name_location + "/Final_K_Z.npy")

    # Final_K_X_Tau = np.load(sorted_file_name_location + "/Final_K_X_Tau.npy")
    # Final_K_Y_Tau = np.load(sorted_file_name_location + "/Final_K_Y_Tau.npy")
    # Final_K_Z_Tau = np.load(sorted_file_name_location + "/Final_K_Z_Tau.npy")


    Final_K =  np.load(sorted_file_name_location + "/Final_K.npy")
    Final_Phi =  np.load(sorted_file_name_location + "/Final_Phi.npy")   
    Final_Thetha =  np.load(sorted_file_name_location + "/Final_Thetha.npy")   
    Phase =  np.load(sorted_file_name_location + "/Phase.npy")   

    # Int_K_Parl =  np.load(sorted_file_name_location + "/Int_K_Parl.npy")


    # T_Of_Ion = np.load(sorted_file_name_location + "/T_Of_Ion.npy")   
    # T_Of_Tun = np.load(sorted_file_name_location + "/T_Of_Tun.npy")   

    # Int_K = np.load(sorted_file_name_location + "/Int_K.npy")   
    # Angular_KZ = np.load(sorted_file_name_location + "/Angular_KZ.npy")   
    # Int_K_Perp = np.load(sorted_file_name_location + "/Int_K_Perp.npy") 
    # Disp = np.load(sorted_file_name_location + "/Disp.npy")   


    return Final_K, Final_Phi, Final_Thetha, Phase, Int_K_X, Int_K_Y, Int_K_Z, Final_K_X, Final_K_Y, Final_K_Z, Final_K_X_Tau, Final_K_Y_Tau, Final_K_Z_Tau, Int_K_Parl, T_Of_Ion, T_Of_Tun, Int_K, Angular_KZ, Int_K_Perp, Disp

def PAD(Final_K_X, Final_K_Y, Final_K_Z, Phase, plot_name):
    
    resolution = 0.01
    x_momentum = np.arange(-1.5, 1.5 + resolution, resolution)
    y_momentum = np.arange(-1.5, 1.5 + resolution, resolution)
    # y_momentum = np.flip(y_momentum)
    
    z_momentum = np.arange(-1, 1 + resolution, resolution)
    # z_momentum = np.array([0])

    pad_value = np.zeros((y_momentum.size,x_momentum.size))
    
    for i, px in enumerate(x_momentum):
        
        print(round(px,3))
        first_idx = np.where(np.logical_and(Final_K_X >= px - resolution/2, Final_K_X<= px+(resolution/2)))[0]
        Final_K_Y_temp = Final_K_Y[first_idx]
        Final_K_Z_temp = Final_K_Z[first_idx]
        Phase_temp = Phase[first_idx]
        

        for j, py in enumerate(y_momentum):   
                
            second_idx = np.where(np.logical_and(Final_K_Y_temp >= py - resolution/2, Final_K_Y_temp <= py + resolution/2 ))[0]
            Final_K_Z_temp_two = Final_K_Z_temp[second_idx]
            Phase_temp_two = Phase_temp[second_idx]


            for l, pz in enumerate(z_momentum):
                k_val = np.sqrt(px**2 + py**2 +pz**2) + 0.01
                third_idx = np.where(np.logical_and(Final_K_Z_temp_two >= pz - resolution/2, Final_K_Z_temp_two <= pz + resolution/2 ))[0]    
                phase = Phase_temp_two[third_idx] 
               
                value = np.exp(-1.0j*phase)
                value = np.sum(value)
                
                # if k_val < 0.3:
                #     continue
                # pad_value[i, j] += (np.absolute(value)**2)
                 
                if k_val < 0.2:
                    pad_value[i, j] += (np.absolute(value)**2)
                
                else:
                    pad_value[i, j] += (np.absolute(value)**2)*(k_val**3)
            
    
    # pad_value = pad_value / pad_value.max() + 1e-12
    # outlier_idx = np.where(pad_value > 0.1)
    # pad_value[outlier_idx] = 0.1
    pad_value = pad_value / pad_value.max() + 1e-12
    
    np.save("400_Minus_Tau", pad_value)
    # plt.rcParams.update({'font.size': 16})

    pos = plt.imshow(pad_value, cmap='jet', extent=[-1.5, 1.5, -1.5, 1.5],norm=LogNorm(vmin=pow(10, -3), vmax=1))

    plt.xlabel(r'$P_{x}$', fontsize=20)
    plt.ylabel(r'$P_{y}$', fontsize=20)
    plt.tight_layout()
    plt.colorbar(pos)
    plt.grid(color = "cyan")
    plt.savefig(plot_name)
    # plt.show()
   
    plt.clf()

def PAD_Integrated(Final_K, Final_Phi, Final_Thetha, Phase, plot_name):
    
        
    resolution = 0.01

    resolution2 = 0.0025
    k_array = np.arange(resolution2, 2.0, resolution2)
    
    # e_array = np.arange(0, 1.5, resolution/2)
    
    e_array = k_array*k_array/2
    
    phi_array = np.arange(0, 2*pi, resolution)
    theta_array = np.arange(pi/4 , 3*pi/4, resolution)

    pad_value = np.zeros(len(e_array))


    for i, k in enumerate(k_array):
        
        print(round(k, 5))
        # k = sqrt(2*e)
        k_axis_idx = np.where(np.logical_and(Final_K >= k - resolution2/2, Final_K <= k + resolution2/2 ))[0]
        
        Final_Phi_temp = Final_Phi[k_axis_idx]
        Final_Thetha_temp = Final_Thetha[k_axis_idx]
        Phase_temp = Phase[k_axis_idx]

        for j, phi in enumerate(phi_array):           
            
            phi = phi%(2*pi)
            phi_axis_idx = np.where(np.logical_and(Final_Phi_temp >= phi - resolution/2, Final_Phi_temp <= phi + (resolution/2)))[0]
            Final_Thetha_temp_two = Final_Thetha_temp[phi_axis_idx]
            Phase_temp_two = Phase_temp[phi_axis_idx]
            

            for l, theta in enumerate(theta_array):  
            
                theta_axis_idx = np.where(np.logical_and(Final_Thetha_temp_two >= theta - resolution/2, Final_Thetha_temp_two <= theta + resolution/2 ))[0]
                phase = Phase_temp_two[theta_axis_idx]

                value = np.exp(-1.0j*phase)
                value = np.sum(value)
                value = np.absolute(value)**2
                pad_value[i] += value*np.sin(theta)

    pad_value /= pad_value.max()

    np.save("CTMC_E_600_Plus", e_array)
    np.save("CTMC_P_600_Plus", pad_value)
    
    
    plt.plot(e_array, pad_value)

    plt.xlabel(r'$E$')
    plt.ylabel(r'$Yield$')

    # plt.ylim(1e-2, 1)
    plt.xlim(0, 2)
    plt.tight_layout()
    plt.grid()
    # plt.show()
    plt.savefig(plot_name)
    plt.clf()

def ATI_Peak_Plots(Final_K, Final_Phi, Final_Thetha, Phase, k_peaks, Int_K_Parl, Int_K_Perp, Disp, Angular_KZ, plot_name):
    
    resolution = 0.01
    
    x_array = Int_K_Parl
    x_array_range = np.arange(-0.5, 0.5, resolution)
    # x_array_range = np.arange(150, 400, 1)
    
    x_data = []
    y_data = []

    plt.rcParams.update({'font.size': 16})
    for i in range(len(k_peaks)):
        k_peak_idx = np.where(np.logical_and(Final_K >= k_peaks[i] - resolution/2, Final_K <= k_peaks[i]+(resolution/2)))[0]
        
        Final_Phi_temp = Final_Phi[k_peak_idx]
        Final_Thetha_temp = Final_Thetha[k_peak_idx]
        Phase_temp = Phase[k_peak_idx]
        x_array_temp = x_array[k_peak_idx]
                
        pad_value = Single_ATI_Peak_Plots(Final_Phi_temp, Final_Thetha_temp, Phase_temp, x_array_temp, x_array_range, resolution)
        
        peak_idx = np.argmax(pad_value)

        x_data.append(k_peaks[i]*k_peaks[i]/2)
        y_data.append(x_array_range[peak_idx])

        # print(k_peaks[i]*k_peaks[i]/2, x_array_range[peak_idx])

        # exit()

        plt.plot(x_array_range, pad_value, label= "peak:" + str(i+1))
    
    np.save("PES_Parallel_Vs_Energy_peaks_Ell_0.71_X", x_data)
    np.save("PES_Parallel_Vs_Energy_peaks_Ell_0.71_Y", y_data)


    plt.legend()
    
    # plt.title("Yield as a function of inital angular momentum")
    plt.xlabel(r'$P_{||}$')
    plt.ylabel(r'$Yield$')
    plt.tight_layout()
    plt.savefig(plot_name)
    plt.clf()
    
def Single_ATI_Peak_Plots(Final_Phi, Final_Thetha, Phase, x_array, x_array_range, resolution):
    
    phi_array = np.arange(0, 2*pi, resolution*2)
    theta_array = np.arange(0 , pi, resolution*2)

    resolution_x = abs(x_array_range[1] - x_array_range[0])

    pad_value = np.zeros(len(x_array_range))
    
    for i, x in enumerate(x_array_range):
        print(round(x, 3))

        k_axis_idx = np.where(np.logical_and(x_array >= x - resolution_x/2, x_array <= x + resolution_x/2 ))[0]
        
        Final_Phi_temp = Final_Phi[k_axis_idx]
        Final_Thetha_temp = Final_Thetha[k_axis_idx]
        Phase_temp = Phase[k_axis_idx]

        for j, phi in enumerate(phi_array):           
            
            phi = phi%(2*pi)
            phi_axis_idx = np.where(np.logical_and(Final_Phi_temp >= phi - resolution/2, Final_Phi_temp <= phi + (resolution/2)))[0]
            Final_Thetha_temp_two = Final_Thetha_temp[phi_axis_idx]
            Phase_temp_two = Phase_temp[phi_axis_idx]
            

            for l, theta in enumerate(theta_array):  
            
                theta_axis_idx = np.where(np.logical_and(Final_Thetha_temp_two >= theta - resolution/2, Final_Thetha_temp_two <= theta + resolution/2 ))[0]
                phase = Phase_temp_two[theta_axis_idx]

                value = np.exp(-1.0j*phase)
                value = np.sum(value)
                value = np.absolute(value)**2
                pad_value[i] += value*np.sin(theta)

    pad_value /= pad_value.max()
    
    return pad_value

def PAD_Energy_Angle(Final_K, Final_Phi, Final_Thetha, Phase, plot_name):
    
    #0.005
        
    resolution = 0.005
    resolution_two = 0.005


    e_array = np.arange(resolution, 0.75, resolution)
    phi_array = np.arange(0 + pi/8, 2*pi + pi/8, resolution_two)
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
            
            for l, theta in enumerate(theta_array): 
                theta_axis_idx = np.where(np.logical_and(Final_Thetha_temp_two >= theta - resolution/2, Final_Thetha_temp_two <= theta + resolution/2 ))[0]
                phase = Phase_temp_two[theta_axis_idx]

                value = np.exp(1.0j*phase)
                value = np.sum(value)
                value = np.absolute(value)**2

                phi_array_for_max[j] += value*np.sin(theta)

                pad_value_2D[j, i] += value*np.sin(theta)

        off_set_idx = np.argmax(phi_array_for_max/phi_array_for_max.max())
        off_set_values[i] = (e, phi_array[off_set_idx])

    pad_value_2D = pad_value_2D / pad_value_2D.max() + 1e-12
    pos = plt.imshow(pad_value_2D, cmap='jet', aspect='auto', extent=[e_array.min(), e_array.max(), phi_array.min(), phi_array.max()],norm=LogNorm(vmin=pow(10, -2), vmax=1))

    np.save("CTMC_PAD_400_Minus_L", pad_value_2D)
    x_axis_data = np.array(list(off_set_values.values()))[:,0]
    y_axis_data = np.array(list(off_set_values.values()))[:,1]
    
    black_plot_idx = np.where(x_axis_data > 0.1)[0]

    plt.rcParams.update({'font.size': 16})
    # plt.plot(np.array(list(off_set_values.values()))[:,0][black_plot_idx], np.array(list(off_set_values.values()))[:,1][black_plot_idx], color="black", linewidth=2)
    # np.save("PES_Angle_Vs_Energy_peaks_Ell_0.71_X", np.array(list(off_set_values.values()))[:,0][black_plot_idx])
    # np.save("PES_Angle_Vs_Energy_peaks_Ell_0.71_Y", np.array(list(off_set_values.values()))[:,1][black_plot_idx])

    plt.xlabel(r'$E$')
    plt.ylabel(r'$\phi$')
    plt.tight_layout()
    plt.colorbar(pos)
    plt.grid(color = "cyan")
    plt.savefig(plot_name)
    plt.clf()

def Slope_Compare():

    E_Vs_A_X = np.load("PES_Angle_Vs_Energy_peaks_Ell_0.71_X.npy")
    E_Vs_A_Y = np.load("PES_Angle_Vs_Energy_peaks_Ell_0.71_Y.npy")

    E_Vs_P_X = np.load("PES_Parallel_Vs_Energy_peaks_Ell_0.71_X.npy")
    E_Vs_P_Y = np.load("PES_Parallel_Vs_Energy_peaks_Ell_0.71_Y.npy")

    fig, (ax1, ax2) = plt.subplots(1, 2)

    m1, b1 = linregress(E_Vs_A_X, E_Vs_A_Y/E_Vs_A_Y.max())[0], linregress(E_Vs_A_X, E_Vs_A_Y/E_Vs_A_Y.max())[1]
    m2, b2 = linregress(E_Vs_P_X, E_Vs_P_Y/E_Vs_P_Y.max())[0], linregress(E_Vs_P_X, E_Vs_P_Y/E_Vs_P_Y.max())[1]

    ax1.plot(E_Vs_A_X, E_Vs_A_Y/E_Vs_A_Y.max(), '.')
    ax2.plot(E_Vs_P_X, E_Vs_P_Y/E_Vs_P_Y.max(), '.')

    ax1.plot(E_Vs_A_X, E_Vs_A_X*m1 + b1)
    ax2.plot(E_Vs_P_X, E_Vs_P_X*m2 + b2)

    plt.xlim(0, 0.75)
    plt.show()

if __name__ == "__main__":

    sorted_file_name_location = "/mpdata/becker/yoge8051/Research/TDSE/Monte_Carlo/Sorted_Data/Neon/Elliptical/400nm/New/Minus"
    # Color_map()

    Final_K, Final_Phi, Final_Thetha, Phase, Int_K_X, Int_K_Y, Int_K_Z, Final_K_X, Final_K_Y, Final_K_Z, \
    Final_K_X_Tau, Final_K_Y_Tau, Final_K_Z_Tau, Int_K_Parl, T_Of_Ion, T_Of_Tun, Int_K, Angular_KZ, Int_K_Perp, Disp = \
    Read_Files(sorted_file_name_location)

    print("{:e}".format(len(Final_K)))  

    no_traj = 5.e8
    no_traj = int(no_traj)
    Final_K, Final_Phi, Final_Thetha, Phase = Final_K[:no_traj], Final_Phi[:no_traj], Final_Thetha[:no_traj], Phase[:no_traj]
    PAD_Energy_Angle(Final_K, Final_Phi, Final_Thetha, Phase, "PAD_400_Angle_Energy_Minus.png")
      
    # PAD(Final_K_X_Tau, Final_K_Y_Tau, Final_K_Z_Tau, Phase, "PAD_E_400_Minus.png")
    # PAD(Final_K_X, Final_K_Y, Final_K_Z, Phase, "PAD_E_600_Plus.png")
    # k_peaks = PAD_Integrated(Final_K, Final_Phi, Final_Thetha, Phase, "PES_E_600_Plus.png")
    # ATI_Peak_Plots(Final_K, Final_Phi, Final_Thetha, Phase, k_peaks, Int_K_Parl, Int_K_Perp, Disp, Angular_KZ, "Paralle_500_Ell_0.71.png")
  

    # Slope_Compare()