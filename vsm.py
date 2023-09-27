import pandas as pd
import numpy as np

import glob
import re

from scipy.interpolate import interp1d

def nearest_idx(arr, x):
    return np.abs(arr-x).argmin()

nearest=nearest_idx
# Class should be constructed differently with a load from file method and
# just a df+log init
class VSM():
    def __init__(self, folder, fmt='%Y-%m-%d %H:%M:%S', weight=None, name=None):
        self.fmt = fmt
        self.folder = folder
        self.weight = weight
        
        
        search = re.search(r'\/.*\/(.*) \(ProfileData\)', self.folder)
        self.filename = search.group(1)
        if not name:
            self.name = self.filename
        
        try:
            self.df = pd.read_pickle(self.folder+"/"+self.filename+".pkl")
        except IOError:
            self.make_pickle()
            
        # Adding name as a column (Implemented 27-09-2023)
        # Name is not saved in .PKL file
        self.df['Name'] = pd.Series([name]*len(self.df))
            
        self.retrieve_log()
    
    def make_pickle(self):
        files = glob.glob(self.folder+"/*.dat")
        
        df = pd.concat((pd.read_csv(f[:-4]+'.txt', delimiter='\s+',
                       header=8) for f in files))
        
        # https://stackoverflow.com/questions/64767166/reducing-rows-in-pandas-dataframe-from-index
        df = (df.groupby((df.index == 0).cumsum()).agg(list)
              .applymap(lambda x: np.nan if np.isnan(np.array(x)).all()
                        else np.array(x)))
        
        
        regex_oneliners = re.compile(r'^(\S+) (\S*)$', re.MULTILINE)
        time_start = []
        time_end = []
        props = []
        for file in files:
            # Retrieving Properties
            with open(file, 'r') as f:
                props.append( dict( re.findall(regex_oneliners, f.read()) ) )
                
            # Retrieving Times
            with open(file[:-4]+'.txt', 'r') as f:
                first_line = f.readline()
                times = re.split("\s{2,}", first_line)
                
                time_start.append(times[1])
                time_end.append(times[3])

        df['Started'] = time_start
        df['Completed'] = time_end
        df['Properties'] = props
        
        idx = np.array([re.findall('#(\d+)', f)[0] for f in files],
                       dtype=int)-1 #Set index 0 to 0 instead of 1
        df = df.set_index(idx)
        df.sort_index(inplace=True)
        
        # Expanding properties column and converts to numeric if possible
        df = pd.concat([df.drop('Properties', axis=1),
                        pd.json_normalize(df['Properties'])], axis=1)
        df = df.apply(pd.to_numeric, errors='ignore')
        
        
        # Convert time to datetime using self.fmt
        df['Started']= pd.to_datetime(df['Started'],
                              format=self.fmt)
        df['Completed']= pd.to_datetime(df['Completed'],
                                        format=self.fmt)

        # Adding average temperature information
        try:
            df['AvgTemperature'] = df['Temperature(C)'].apply(lambda x: np.around(np.mean(x), -1))
        except KeyError:
            df['AvgTemperature'] = np.nan

        df.to_pickle(self.folder+"/"+self.filename+".pkl")
        
        self.df = df
        
    def retrieve_log(self):
        filename = glob.glob(self.folder+'/*_Log.txt')[0]
        with open(filename, 'r') as f:
                content = f.readlines()
                
        profile_line = content[4]
        self.profile = re.search('\:\s+(.*)', profile_line).group(1)
        
        
        content = content[6:]
        
        experimentnames = np.array([], dtype=str)
        
        fileno = 1
        reduced = ""
        for i in range(1, len(content)):
            if content[i-1] == "\n":
                end = content[i].find(' began ')
                pre = str(fileno).zfill(3)
                fileno += 1
                
                
                
                if end == -1:
                    pre = '---'
                    fileno -= 1
                    end = content[i].find(' at ')
                    
                else:
                    experimentname = content[i][:end]
                    experimentname = experimentname.replace(' ', '')
                    experimentnames = np.append(experimentnames, experimentname.upper())
                
                reduced += pre + " " + content[i][:end] +"\n"
        self.log = reduced
        self.df['ExperimentName'] = experimentnames
        
    def print_log(self):
        try:
            print(self.log)
        except:
            self.retrieve_log()
            print(self.log)
    
    # Retrieving log after adding will reset log to self
    def __add__(self, other):
        start = int( re.findall(r'\d{3} ', self.log)[-1] )
        N = int( re.findall(r'\d{3} ', other.log)[-1] )
        
        add_log = re.sub(r'\d{3} ',
                         lambda x: str(int(x.group(0))+start).zfill(3)+" ",
                         other.log)
        
        self.log += add_log
        
        new_df = other.df.set_index(np.arange(start, start+N))
        
        self.df = pd.concat([self.df, new_df])
        
        return self
    
    def __repr__(self):
        return self.profile
        
        
    
    def calculate_area(self):
        self.df['Area'] = self.df.apply(
            lambda x: self.area(x['Field(Oe)'], x['Moment(emu)']), axis=1)
        
    def ty(self, key):
        def timespace(start, end, y):
            t = np.linspace(start.value, end.value, np.size(y))
            return pd.to_datetime(t)
        
        keys = self.df[self.df[key].notna()].apply(lambda x:timespace(
                                                  x['Started'],
                                                  x['Completed'],
                                                  x[key]), axis=1)
    
        t = np.concatenate(keys.values)
        y = np.concatenate(self.df[self.df[key].notna()][key].values)
        
        return t, y    
    
    # Function to return temperature and moment (degC and Am^2/kg)
    def curie(self, i=None):
        if not i:
            i = self.get_curie_index()
            
        T = self.df.iloc[i]['Temperature(C)']
        M = self.df.iloc[i]['Moment(emu)']/(self.weight*1e-3)

        return T, M
    
    def get_curie_index(self):
        # Experimental code to self-find the correct index
        exp = self.df.Experiment == 'TimeExp'
        field = self.df.MaxField == 100
        
        idx = np.logical_and(exp, field)
       
        # Checks if temperature has decreased atleast 100 C    
        curie = self.df['Temperature(C)'][idx].apply(lambda x: x[0]-x[-1]) > 100
        
        return curie.index[curie == True].tolist()[0]
     
    
    def hysteresis(self, field=None, temperature=None):
        pass
    
    
        
    def area(self, B, M):
        if np.size(B) <2:
            return np.nan
    
        idx = np.gradient(B)<0
        B1, M1 = np.flip(B[idx]), np.flip(M[idx])
        B2, M2 = B[np.logical_not(idx)], M[np.logical_not(idx)]
        
        
        inter = interp1d(B2, M2, fill_value='extrapolate')
        M2 = inter(B1)
        
        # Remove ends
        switch = (M1-M2) < 0
        if switch.sum () > 1:
            test = np.where(switch)[0] - int(switch.size/2)
            
            try:
                mid = int( np.argwhere(test >0)[0] ) 
                if   (test>0).sum() == test.size:
                    start = 0
                    end = test[0]+int(switch.size/2)
                else:
                    start = nearest(test[:mid], 0) + int(switch.size/2)
                    end = nearest(test[mid:], 0) + int(switch.size/2)
                
            except:
                start = test[-1]+int(switch.size/2)
                end=B1.size
                
        
        else:
            start = 0
            end = B1.size
        
        return np.trapz(M1[start:end], B1[start:end]) -\
               np.trapz(M2[start:end], B1[start:end])
