import numpy as np
import pandas as pd
import glob

class GC():
    def __init__(self, folder, cal_file, cal_lines, composition):
        self.header = ['File', 'Path', 'Injected', 'Sample', 'Info', 'H2', 'N2', 'CO', 'FID methane', 'CH4', 'CO2']
        self.gasses = ['H2', 'N2', 'CO', 'CO2', 'CH4']
        
        # Get calibration constants
        cal_path = folder+'\\'+cal_file
        c = self.calibration(cal_path, cal_lines, composition)
        
        # Loading remaing files
        self.files = glob.glob(folder+"/*.xls")

        try:
            self.files.remove(folder+'\\'+cal_file)
        except FileNotFoundError:
            raise Exception("Calibration is not present in specified folder")

        gc = pd.concat((pd.read_excel(f, sheet_name='Area') for f in self.files))
        gc.columns = self.header
        
        gc = gc.sort_values(by='Injected')
        gc = gc.reset_index(drop=True)
        
        gc['Injected'] = pd.to_datetime(gc['Injected'])

        # Apply Calibration
        gc[self.gasses] = gc[self.gasses]*c
        
        # Calculates conversion ratio
        gc['Conversion'] = (gc.CO+gc.CO2)/(gc.CO+gc.CO2+gc.CH4)
        
        self.df = gc
    
    def calibration(self, cal_path, cal_lines, composition):
        """
        Calibrates the GC data according to known gas mixtures.
        The composition should be in a 2D-array of [N x [H2, N2, CO, CO2, CH4]] or 1D-array of [H2, N2, CO, CO2, CH4]
        """
        cal = pd.read_excel(cal_path,sheet_name='Area')
        cal = cal.iloc[cal_lines]

        cal.columns = self.header
                       
        cal['TotalArea'] = cal[self.gasses].sum(axis=1)
        
        N = np.size(cal_lines)
        composition = np.array(composition)
        if len(composition.shape) == 1:
            composition = np.tile(composition, (N, 1))
            
        
        cal = pd.concat([cal, pd.DataFrame(composition, columns=list(map(lambda x: x+' %', self.gasses)))], axis=1)
        
        
        def k(c, r, t):
            if c==0:
                return 0
            else:
                return r*t/c
        c = np.array([])
        for gas in self.gasses:
            c = np.append(c, np.mean( cal.apply(lambda x: k(x[gas], x[gas+' %'], x['TotalArea']), axis=1) ))
            
        
        return c
    
    def __repr__(self):
        msg = "The following files has been used for this GC dataset\n"
        return msg+"\n".join(self.files)