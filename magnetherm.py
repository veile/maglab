# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 10:39:14 2021

@author: tveile
"""

import numpy as np
import pandas as pd
import re

from scipy.signal import savgol_filter

def nearest_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


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
        # Units of nF
        f2C = {'1': 200, '2': 88 , '4': 26,
               '5': 15, '9': 6.2}
        self.props['capacitance'] = f2C[str(self.props['frequency'])[0]]


        self.df = pd.read_csv(filename, comment='#', delimiter='\t')
        
   
    def split(self, tc='T0'):
        t, T = self.df['Time [s]'], self.df[tc+' [degC]']
        
        
        base_mask = self.df.State == 'BEFORE'
        if base_mask.sum() != 0:
            baseline = T[base_mask].mean()
        else:
            baseline = T[0]
            print('No baseline recorded! Baseline set to: %d' %baseline)
        
        T = T-baseline
        
        heat_mask = self.df.State == 'EXPOSING'
        theat, Theat = t[heat_mask], T[heat_mask]
        
        cool_mask = self.df.State == 'WAIT'
        tcool, Tcool = t[cool_mask], T[cool_mask]

        return t, T, theat, Theat, tcool, Tcool, baseline
    

class Power():
    def __init__(self, Magnetherm, C):
        self.m = Magnetherm
        self.C = C
        
    def loss(self):
        t, T, theat, Theat, tcool, Tcool, baseline = self.m.split()
        
        fp = savgol_filter(Tcool, 7, 2, deriv=1, delta=tcool.diff().mean())
        
        equiv = np.array([nearest_idx(Tcool, temperature)
                          for temperature in Theat])
        
        return fp[equiv]*self.C #Unit is K/s = W/kg
    
    def savgol(self):
        t, T, theat, Theat, tcool, Tcool, baseline = self.m.split()
        
        fp = savgol_filter(Theat, 11, 2, deriv=1, delta=theat.diff().mean())
        
        
        # import matplotlib.pyplot as plt
        # f = savgol_filter(Theat, 11, 2, deriv=0, delta=theat.diff().mean())
        # plt.figure()
        # plt.plot(theat, self.loss(), lw=3)
        # # plt.plot(theat, Theat)
        # plt.show()
        
        return theat, (fp*self.C-self.loss())
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        