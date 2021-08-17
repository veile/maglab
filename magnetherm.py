# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 10:39:14 2021

@author: tveile
"""

import numpy as np
import pandas as pd
import re

from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

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
        f2C = {'1': 200, '2': 88 , '3': 26, '4': 26,
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

        # Removing points in sharp decline
        tcool, Tcool = tcool[3:], Tcool[3:]

        return t.values, T.values, theat.values, Theat.values, tcool.values, Tcool.values, baseline
    

class Power():
    def __init__(self, Magnetherm, C):
        self.m = Magnetherm
        self.C = C
        
    def loss(self):
        t, T, theat, Theat, tcool, Tcool, baseline = self.m.split()
        
        fp = savgol_filter(Tcool, 9, 2, deriv=1, delta=np.diff(tcool).mean())

        equiv = np.array([nearest_idx(Tcool, temperature)
                          for temperature in Theat])
        
        return fp[equiv]*self.C
    
    def savgol(self, window=11, order=2, compensate=None, debug=False):
        t, T, theat, Theat, tcool, Tcool, baseline = self.m.split()
        
        fp = savgol_filter(Theat, window, order, deriv=1, delta=np.diff(theat).mean())
        P = (fp*self.C-self.loss())
        
        if compensate is not None:
            compensate = compensate.lower()
            start = np.where(t == theat[0])[0][0]
            end = np.where(t == theat[-1])[0][0]
            
            current = self.m.df['Current [A]'][start:end+1]
            set_current = self.m.props['set_current']
            
            if compensate=='linear':
                P = P*(set_current/current)
                
                
        
        if debug:
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9, 8))
            fheat = savgol_filter(Theat, window, order,
                                  delta=theat.diff().mean())
            fcool = savgol_filter(Tcool, 7, 2,
                                  delta=theat.diff().mean())
            
            ax1.plot(theat, Theat, 'k', alpha=0.5, lw=3)
            ax1.plot(theat, fheat, 'red')
            
            ax2.plot(tcool, Tcool, 'k', alpha=0.5, lw=3)
            ax2.plot(tcool, fcool, 'blue')
            
            ax3.plot(theat, fp*self.C, 'red')
            ax3b = ax3.twinx()
            ax3b.plot(theat, self.loss(), 'blue')

        
        return theat[window:], P[window:]
    
    # def savgol(self, window=11, order=2, debug=False):
    #     t, T, theat, Theat, tcool, Tcool, baseline = self.m.split()
        
    #     fp = savgol_filter(T, window, order, deriv=1, delta=np.diff(t).mean())
            
    #     start = np.where(t == theat[0])[0][0]
    #     end = np.where(t == theat[-1])[0][0]+1
        
    #     Ph = fp[start:end]
    #     Pc = fp[end:]
        
    #     equiv = np.array([nearest_idx(Tcool, temperature)
    #                       for temperature in Theat])

                
    #     if debug:
    #         fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9, 8))
            
    #         ax1.plot(t, T)
    #         ax2.plot(t, fp)
    #         ax3.plot(tcool, Tcool)
    #         ax3.plot(tcool[equiv], Tcool[equiv], 's', ms=1)
            



    #     return theat, self.C*(Ph-Pc[equiv])
    
        
        