import numpy as np
import gc
from numpy import pi, sqrt

if __name__ == "__main__":

    if True:
    
        major_axis, minor_axis, propagation_axis = [], [], []   
        major_axis_tau, minor_axis_tau, propagation_axis_tau = [], [], [] 
        major_axis_inital,minor_axis_inital,propagation_axis_inital = [], [], []
        major_inital_position, minor_inital_position, propagation_inital_position = [], [], [] 
        phase_array, displacement, time_of_flight, time_of_ionization, time_of_tunneling  = [], [], [], [], []
        v_paral, v_perp = [], []


    file_name_location = "/mpdata/becker/yoge8051/Research/TDSE/Monte_Carlo/Data/Neon/Elliptical/600nm/Plus"

    sorted_file_name_location = "/mpdata/becker/yoge8051/Research/TDSE/Monte_Carlo/Sorted_Data/Neon/Elliptical/600nm/Plus"

    for file_no in range(444):

        print("file", file_no)
        if file_no in range(282, 282 + 20):
            continue
        
        if file_no in range(324, 324 + 16 + 8):
            continue
        
        # if file_no in [154, 155]:
        #     continue


        ###########################################################################################

        file_name = file_name_location + "/Phase_" + str(file_no)+ ".npy"
        PH = np.load(file_name)
        # continue
        phase_array += list(PH)

        file_name = file_name_location + "/Final_Mom_X_Axis_" + str(file_no)+ ".npy"
        MA_A = np.load(file_name)        
        major_axis += list(MA_A)
        
        file_name = file_name_location + "/Final_Mom_Y_Axis_" + str(file_no)+ ".npy"
        MI_A = np.load(file_name)
        minor_axis += list(MI_A)
        
        file_name = file_name_location + "/Final_Mom_Z_Axis_" + str(file_no)+ ".npy"
        PR_A = np.load(file_name)
        propagation_axis += list(PR_A)
        
        ###########################################################################################
        file_name = file_name_location + "/Final_Mom_Tau_X_Axis_" + str(file_no)+ ".npy"
        MA_AT = np.load(file_name)        
        major_axis_tau += list(MA_AT)
        
        file_name = file_name_location + "/Final_Mom_Tau_Y_Axis_" + str(file_no)+ ".npy"
        MI_AT = np.load(file_name)
        minor_axis_tau += list(MI_AT)
        
        file_name = file_name_location + "/Final_Mom_Tau_Z_Axis_" + str(file_no)+ ".npy"
        PR_AT = np.load(file_name)
        propagation_axis_tau += list(PR_AT)

        continue
    
    
        ########################################################################################### 
        file_name = file_name_location + "/Displacment_" + str(file_no)+ ".npy"
        DS = np.load(file_name)        
        displacement += list(DS)
        
        file_name = file_name_location + "/Time_Ion_" + str(file_no)+ ".npy"
        TI = np.load(file_name)        
        time_of_ionization += list(TI)

        file_name = file_name_location + "/Time_Tunneling_" + str(file_no)+ ".npy"
        TT = np.load(file_name)        
        time_of_tunneling += list(TT)

        file_name = file_name_location + "/Time_Flight_" + str(file_no)+ ".npy"
        TF = np.load(file_name)        
        time_of_flight += list(TF)

        file_name = file_name_location + "/V_Parall_" + str(file_no)+ ".npy"
        PAR = np.load(file_name)
        v_paral += list(PAR)
        
        file_name = file_name_location + "/V_Perp_" + str(file_no)+ ".npy"
        PER = np.load(file_name)
        v_perp += list(PER)
        
        ###########################################################################################
        file_name = file_name_location + "/Inital_Mom_X_Axis_" + str(file_no)+ ".npy"
        IMA = np.load(file_name)        
        major_axis_inital += list(IMA)
        
        file_name = file_name_location + "/Inital_Mom_Y_Axis_" + str(file_no)+ ".npy"
        MAI = np.load(file_name)
        minor_axis_inital += list(MAI)
        
        file_name = file_name_location + "/Inital_Mom_Z_Axis_" + str(file_no)+ ".npy"
        IPA = np.load(file_name)
        propagation_axis_inital += list(IPA)
        
        
        ###########################################################################################
        file_name = file_name_location + "/Inital_Pos_X_Axis_" + str(file_no)+ ".npy"
        MIP = np.load(file_name)        
        major_inital_position += list(MIP)
        
        file_name = file_name_location + "/Inital_Pos_Y_Axis_" + str(file_no)+ ".npy"
        MIP = np.load(file_name)
        minor_inital_position += list(MIP)
        
        file_name = file_name_location + "/Inital_Pos_Z_Axis_" + str(file_no)+ ".npy"
        PIP = np.load(file_name)
        propagation_inital_position += list(PIP)
        
        
      
        
     

    Final_K_X = np.array(major_axis)
    del(major_axis)
    gc.collect()

    print("{:e}".format(len(Final_K_X)))  

    Final_K_Y = np.array(minor_axis)
    del(minor_axis)
    gc.collect()

    Final_K_Z = np.array(propagation_axis)
    del(propagation_axis)
    gc.collect()

    phase_array = np.array(phase_array)
    gc.collect()

    T_Of_Flight = np.array(time_of_flight)
    del(time_of_flight)
    gc.collect()

    T_Of_Ion = np.array(time_of_ionization)
    del(time_of_ionization)
    gc.collect()

    
    T_Of_Tun = np.array(time_of_tunneling)
    del(time_of_tunneling)
    gc.collect()

    Disp = np.array(displacement)
    del(displacement)
    gc.collect()
    
    Final_K_X_Tau = np.array(major_axis_tau)
    del(major_axis_tau)
    gc.collect()
    
    Final_K_Y_Tau = np.array(minor_axis_tau)
    del(minor_axis_tau)
    gc.collect()

    Final_K_Z_Tau = np.array(propagation_axis_tau)
    del(propagation_axis_tau)
    gc.collect()

    Int_K_Parl = np.array(v_paral)
    del(v_paral)
    gc.collect()

    Int_K_Perp = np.array(v_perp)
    del(v_perp)
    gc.collect()
    
    

    r_inital = np.zeros(shape=(len(major_inital_position), 3))
    r_inital[:,0], r_inital[:,1], r_inital[:,2] = major_inital_position, minor_inital_position, propagation_inital_position
     
    p_inital = np.zeros(shape=(len(major_axis_inital), 3))
    p_inital[:,0], p_inital[:,1], p_inital[:,2] = major_axis_inital, minor_axis_inital, propagation_axis_inital
    
    Angular_KZ = np.cross(r_inital, p_inital)
    Angular_KZ= np.array(Angular_KZ[:, 2])

    Int_K_X = np.array(major_axis_inital)
    del(major_axis_inital)
    gc.collect()
    Int_K_Y = np.array(minor_axis_inital)
    del(minor_axis_inital)
    gc.collect()
    Int_K_Z = np.array(propagation_axis_inital)
    del(propagation_axis_inital)
    gc.collect()

    Int_Pos_X = np.array(major_inital_position)
    del(major_inital_position)
    gc.collect()
    Int_Pos_Y = np.array(minor_inital_position)
    del(minor_inital_position)
    gc.collect()
    Int_Pos_Z = np.array(propagation_inital_position)
    del(propagation_inital_position)
    gc.collect()

    
    file_name = sorted_file_name_location + "/Final_K_X" 
    np.save(file_name, Final_K_X)

    file_name = sorted_file_name_location + "/Final_K_Y" 
    np.save(file_name, Final_K_Y)

    file_name = sorted_file_name_location + "/Final_K_Z" 
    np.save(file_name, Final_K_Z)
    


    Final_K, Final_K2 = sqrt(Final_K_X**2 + Final_K_Y**2 + Final_K_Z**2), sqrt(Final_K_X**2 + Final_K_Y**2)

    Final_Thetha, Final_Phi = np.arctan2(Final_K2, Final_K_Z), np.arctan2(Final_K_Y, Final_K_X) 

    phi_negative_idx = np.where(Final_Phi < 0)[0]
    Final_Phi[phi_negative_idx] += 2*pi
    
    # del(Final_K_X, Final_K_Y, Final_K_Z, Final_K2)
    # gc.collect()
  
    file_name = sorted_file_name_location + "/Final_K" 
    np.save(file_name, Final_K)

    file_name = sorted_file_name_location + "/Final_Phi" 
    np.save(file_name, Final_Phi)

    file_name = sorted_file_name_location + "/Final_Thetha" 
    np.save(file_name, Final_Thetha)

    file_name = sorted_file_name_location + "/Phase" 
    np.save(file_name, phase_array)




    file_name = sorted_file_name_location + "/T_Of_Ion" 
    np.save(file_name, T_Of_Ion)

    file_name = sorted_file_name_location + "/Final_K_X_Tau" 
    np.save(file_name, Final_K_X_Tau)

    file_name = sorted_file_name_location + "/Final_K_Y_Tau" 
    np.save(file_name, Final_K_Y_Tau)

    file_name = sorted_file_name_location + "/Final_K_Z_Tau" 
    np.save(file_name, Final_K_Z_Tau)

    file_name = sorted_file_name_location + "/Int_K_X" 
    np.save(file_name, Int_K_X)

    file_name = sorted_file_name_location + "/Int_K_Y" 
    np.save(file_name, Int_K_Y)

    file_name = sorted_file_name_location + "/Int_K_Z" 
    np.save(file_name, Int_K_Z)

    file_name = sorted_file_name_location + "/Int_K_Parl" 
    np.save(file_name, Int_K_Parl)

    file_name = sorted_file_name_location + "/Int_K_Perp" 
    np.save(file_name, Int_K_Perp)

    file_name = sorted_file_name_location + "/Int_Pos_X" 
    np.save(file_name, Int_Pos_X)

    file_name = sorted_file_name_location + "/Int_Pos_Y" 
    np.save(file_name, Int_Pos_Y)

    file_name = sorted_file_name_location + "/Int_Pos_Z" 
    np.save(file_name, Int_Pos_Z)

    file_name = sorted_file_name_location + "/Disp" 
    np.save(file_name, Disp)

    file_name = sorted_file_name_location + "/Angular_KZ" 
    np.save(file_name, Angular_KZ)

    



