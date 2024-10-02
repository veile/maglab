import pandas as pd
import numpy as np
import re

class QMS():
    def __init__(self, filename, etype, fmt="%d/%m/%Y %I:%M:%S.%f %p", delay=None):
        self.delay = delay
        self.etype = etype.upper()
        self.fmt = fmt
        self.filename = filename
        
        try:
            self.df = pd.read_pickle(filename[:-4]+'.pkl')
        except FileNotFoundError:
            self.make_pickle()
            
    def make_pickle(self):
        # Multiple Ion Detection
        if self.etype == "MID":
            self.df = self.load_MID()
        
        # Analog Scan
        elif self.etype == "AS":
            if self.delay is not None:
                print("Analog Scan does not need delay")
            self.df = self.load_AS()
            
        else:
            raise NotImplementedError("Type of experiment not yet implemented")
            
        self.df.to_pickle(self.filename[:-4]+".pkl")
        
        
    def load_MID(self):
        # Reads line 6 for masses
        with open(self.filename, 'r') as file:
            i = 0 # Indexing rows
            for line in file:
                if i == 6:
                    mass = re.compile('\s+').split(line)
                    break
                i+=1
        
        mass = list(filter(None, mass))
        
        # Constructing column headers
        times = ['Time {}'.format(m) for m in mass]
        currents = ['Current {}'.format(m) for m in mass]
        rels = ['Rel. Time {}'.format(m) for m in mass]
        
        columns= np.array([column for sublist in zip(times, rels, currents)
                  for column in sublist])
        
        # Filters the data from doubles
        _, unique_idx = np.unique(columns, return_index=True)
        unique_idx = np.sort(unique_idx)
        
        # Reads data
        df = pd.read_csv(self.filename, delimiter='\t', skiprows=8,
                            usecols=unique_idx,
                            names=columns[unique_idx]).dropna()
        
        # Convert to datetime
        for t in times:
            df[t] = pd.to_datetime(df[t],
                                      format=self.fmt)
            
        if self.delay is not None:
            df[times] = df[times]+dt.timedelta(seconds=self.delay)
            
        return df
    
    def load_AS(self):
        data = pd.read_csv(self.filename, delimiter='\t', header=None)  

        data = data.groupby((data[0]=="Start Time").cumsum()).agg(list)
        df = data.applymap(lambda x: np.array(x[2:-3], dtype=float))
    
    
        df['Start Time'] = data.apply(lambda x: x[1][0], axis=1)
        df = df.rename(columns={0: 'Mass', 1: 'Current'})
        
        return df.iloc[2:]
    