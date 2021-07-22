# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 10:39:14 2021

@author: tveile
"""

import numpy as np
import pandas as pd
import re

class Magnetherm():
    def __init__(self, filename):
        with open(filename, 'r') as f:
            self.props = {}
            prop_names = ['frequency', 'set_current']
            
            s = f.readlines()
            gen = (line for line in s if line.startswith('#'))
            for i, line in enumerate(gen):
                val = re.search('\d+', line).group(0)
                self.props[prop_names[i]] = float(val)   

        # Getting the capacitance based on resonance frequency
        f2C = {'1': '200 nF', '2': '88 nF', '4': '26 nF',
               '5': '15 nF', '9': '6.2 nF'}
        self.props['capacitance'] = f2C[str(self.props['frequency'])[0]]


        self.df = pd.read_csv(filename, comment='#', delimiter='\t')
        
    # def split(self, idx, lims=[1, 0.5]):
    #     t, T = self.df.Time.iloc[idx], self.df['Temperature Sample'].iloc[idx]

    #     baseline = T[:5].mean()
    #     t, T = t-t[0], T-T[0]
    
    #     # from itertools import groupby
    #     # T = np.array([k for k,g in groupby(T) if k!=0])
    
    #     # Filter out bad data
    #     # selection = np.ones(T.size, dtype=booal)
    #     # selection[1:] = T[1:] != T[:-1]
    #     # selection &= T != 0
    
    #     # T = T[selection]
    #     # t = t[selection]
    
    #     # 09/01-2020 changed 0.5 to 1.0
    #     start = np.argmax(np.gradient(T, t)
    #                       > lims[0])
    #     end = np.argmax( (np.gradient(T[start:], t[start:])
    #                       < -lims[1]) )+start
    
    #     theat, Theat = t[start:end], T[start:end]
    #     tcool, Tcool = t[end:], T[end:]
        
    #     return t, T, theat, Theat, tcool, Tcool, start, end, baseline