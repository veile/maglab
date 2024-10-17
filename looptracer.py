import os
import re
import glob

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

def reduce(f, R, P):
    '''
    This function takes in the raw values of frequency, amplitude and phase.
    All nan values are removed, and the mean values are taken for each frequency.
    
    Returns 3 arrays of averages values
    '''  
    # Have to use cartesian coordinates to take proper mean values - consider saving X/Y instead?
    # E.g. averaging -180° and 180°, should NOT result in 0. 
    Z = R*np.exp(1j*P)
    X = Z.real
    Y = Z.imag
        
    # Finding all the frequencies by looking at the difference.
    # First value will always be the the first frequency, so it is appended
    idx = np.append([0], np.where(np.diff(f) > 100)[0]+1)
    
    # Probably this can be done more efficiently but it does the trick
    f_sum = []
    X_sum = []
    Y_sum = []
    for i in range(len(idx)):
        if i == len(idx)-1:
            f_sum.append(np.mean(f[idx[i]:]))
            X_sum.append(np.mean(X[idx[i]:]))
            Y_sum.append(np.mean(Y[idx[i]:]))
            
        else:
            f_sum.append(np.mean(f[idx[i]:idx[i+1]]))
            X_sum.append(np.mean(X[idx[i]:idx[i+1]]))
            Y_sum.append(np.mean(Y[idx[i]:idx[i+1]]))
    
    f = np.array(f_sum)
    X = np.array(X_sum)
    Y = np.array(Y_sum)
    
    Z = X + 1j*Y 
    R = np.abs(Z)
    P = np.angle(Z)
    
    return f, R, P


def get_distortion_correction(f, I, V):
    f = np.abs(f)
    # The expected voltage is equal to frequency, current and some coil property constant.
    # The constant is not important as we calibrate the system later.
    Vexp = f*np.abs(I)
    # print(Vexp)
    # Magnitude transfer denotes the attenuation/rise of the true voltage due to the pick-up coils.
    # This constant needs to be multiplied onto the measured voltage
    mag_transfer = Vexp / np.abs(V)
    
    # Phase transfer denotes the phase shift imposed by the pick-up system
    # The phase shift needs to be added onto the measured phase.
    # A coil has the voltage V = L dI/dt, so the expected voltage is 90 degree shifted from the current
    Pexp = np.angle(I*np.exp(-1j*np.pi/2))

    # This is to correctly add angles
    # pi is added to get the phase in range of 0 - 2pi to use modulus and then subtract eh pi again.
    # phase_transfer = (Pexp - np.angle(V) +np.pi) % (2*np.pi) - np.pi
    phase_transfer = np.angle(np.exp(1j*Pexp)*np.exp(-1j*np.angle(V)))
    # phase_transfer = Pexp - np.angle(V)
    
    return interp1d(f, mag_transfer), interp1d(f, phase_transfer)

def get_coeff(df, mag_transfer = lambda f: 1, phase_transfer = lambda f: 0, phase_correction = 0):
    # Pickup signal
    f = df['Blank Pickup Frequency']
    R = df['Blank Pickup R']
    P = df['Blank Pickup P']
    
    fS = df['Sample Pickup Frequency']
    RS = df['Sample Pickup R']
    PS = df['Sample Pickup P']

    # Control signal
    f_C = df['Blank Control Frequency']
    R_C = df['Blank Control R']
    P_C = df['Blank Control P']

    fS_C = df['Sample Control Frequency']
    RS_C = df['Sample Control R']
    PS_C = df['Sample Control P']

    # Applied field is shifted to zero - shifting all signals with it
    # P  = np.angle(np.exp(1j*(P  - P_C + phase_transfer(f ))))
    # PS = np.angle(np.exp(1j*(PS - P_C + phase_transfer(fS))))

    P  = np.angle(np.exp(1j*(P  + phase_transfer(f ))))
    PS = np.angle(np.exp(1j*(PS + phase_transfer(fS))))

    
    n = fS/fS[0]
    
    # Applied field
    Hv = np.sqrt(2)*R_C/(2*np.pi*f_C)
    ϕv = np.angle(np.exp(1j*( PS_C+np.pi/2 )))
    # ϕv = np.pi/2

    # # Magnetic Moment
    phase_corrections = np.ones(f.size) * phase_correction

    V = (np.sqrt(2)*RS*mag_transfer(fS)*np.exp(1j*PS) -\
         np.sqrt(2)*R *mag_transfer(f )*np.exp(1j*P ))*np.exp(-1j*phase_corrections)

    M = np.abs(V)/(2*np.pi*fS)
    ϕ = np.angle(V*np.exp(1j*np.pi/2))

    return Hv, ϕv, M, ϕ

class LoopTracer():
    '''
    Attributes:
        path: filepath to folder with all experiment files
        weight: float value of sample weight in kg
        distortion: list of filepath to measured f, V, I values for a impedance that need to be specified in filename (HiZ or imp50)
        phase_correction: Dictionary with phase correction values for each capacitance
        name: string identifier saved in self.name. Defaults to folder name if not specified
    '''
    def __init__(self, path, distortion_path, phase_correction, name=None):
        self.path = path
        self.phase_correction = phase_correction

        self.distortion = self.create_distortion_dict(distortion_path)
        # self.distortion = {'imp50': {}, 'HiZ': {}}
        # self.distortion['HiZ']['Mag Transfer'], self.distortion['HiZ']['Phase Transfer'] = lambda f: 1, lambda f: 0
        
        self.foldername = path[path.rfind('/')+1:]
        
        if name:
            self.name = name
        else:
            self.name = self.foldername

        try:
            self.df = pd.read_pickle(path + '/' + self.foldername + '.pkl')
        except IOError:
            self.create_pickle()
            
    def create_distortion_dict(self, file_list):
        distortion = {'imp50': {}, 'HiZ': {}}
        
        for file in file_list:
            f, I, V = np.loadtxt(file, dtype=complex)
            if 'HiZ' in file:
                distortion['HiZ']['Mag Transfer'], distortion['HiZ']['Phase Transfer'] = get_distortion_correction(f, I, V)
            elif 'imp50' in file:
                distortion['imp50']['Mag Transfer'], distortion['imp50']['Phase Transfer'] = get_distortion_correction(f, I, V)
            else:
                raise Exception('No impedance specified in the filename!')

        return distortion
    
    def create_pickle(self):
        files = [s[:-12] for s in glob.glob(self.path+'/*_control.csv')]
        
        df_control = pd.concat((pd.read_csv(f+'_control.csv', delimiter=',', header=0, index_col=0) for f in files))
        # https://stackoverflow.com/questions/64767166/reducing-rows-in-pandas-dataframe-from-index
        df_control = (df_control.groupby((df_control.index == 0).cumsum()).agg(list)
              .map(lambda x: np.nan if np.isnan(np.array(x)).all()
                        else np.array(x)))

        df_pickup  = pd.concat((pd.read_csv(f+'_pickup.csv', delimiter=',', header=0, index_col=0) for f in files))
        # https://stackoverflow.com/questions/64767166/reducing-rows-in-pandas-dataframe-from-index
        df_pickup = (df_pickup.groupby((df_pickup.index == 0).cumsum()).agg(list)
              .map(lambda x: np.nan if np.isnan(np.array(x)).all()
                        else np.array(x)))
        
        df = pd.concat([df_control, df_pickup], axis=1)
        df = df.add_prefix('RAW-')
        
        df.index -= 1
        
        # Create columns for reduced data (average and no nan values)
        loop_df = df
        for i in range(0, 10, 3):
            raw_columns = df.columns[i:i+3]
            reduced_columns = [c[4:] for c in raw_columns]
            
            raw_df = df[raw_columns]
            reduced_df = pd.DataFrame(
                raw_df.apply(lambda row: reduce(row.iloc[0], row.iloc[1], row.iloc[2]), axis=1).to_list(),
                columns=reduced_columns)

            loop_df = pd.concat([reduced_df, loop_df], axis=1)
 
        df = loop_df


        # Creating columns denoting information in parameters
        df['File'] = files
        
        params = []
        for file in files:
            with open(file+'_parameters.txt', 'r') as f:
                lines = f.read().splitlines()
                params.append({line[2:line.find(':')]:line[line.find(':')+2:] for line in lines})
                
        df_params = pd.DataFrame(params).apply(pd.to_numeric, errors='ignore')
        
        df = pd.concat([df, df_params], axis=1)
        
        # Calculate the H and M values from df_row using get_HM
        def get_HM_wrapper(df_row):
            cap = df_row['capacitor']

            phase_correction = self.phase_correction[cap]*2*np.pi/360
            # phase_correction = np.mean(self.phase_correction[cap]['diff'], axis=0)

            if df_row['imp50']:
                mag_transfer = self.distortion['imp50']['Mag Transfer']
                phase_transfer =self.distortion['imp50']['Phase Transfer']
            else:
                mag_transfer = self.distortion['HiZ']['Mag Transfer']
                phase_transfer =self.distortion['HiZ']['Phase Transfer']
                
            Hv, ϕv, M, ϕ = get_coeff(df_row, mag_transfer, phase_transfer, phase_correction)
            
            return Hv, ϕv, M, ϕ

        df['H0*'], df['Hp'], df['Mn*'], df['Mpn'] = zip(*df.apply(get_HM_wrapper, axis=1))
        df['Mn*'] = df['Mn*']
        
        self.df = df
        
    def apply_calibration(self, cM, cH):
        self.df['H0'] = self.df['H0*']*cH
        self.df['Mn'] = self.df['Mn*']*cM

    def get_HM(self, i):
        H0 = self.df.loc[i]['H0']
        Hp = self.df.loc[i]['Hp']
        Mn = self.df.loc[i]['Mn']
        Mpn = self.df.loc[i]['Mpn']

        f = self.df.loc[i]['Blank Control Frequency']
        fS = self.df.loc[i]['Sample Pickup Frequency']
        
        # Constructing the time signals from Fourier series
        t = np.linspace(0, 1/f[0], 1000)      
        H = H0*np.sin(2*np.pi*f*t+Hp)
        M = (Mn*np.sin(np.outer(t, 2*np.pi*fS)+Mpn)).sum(axis=1)

        return H, M
    