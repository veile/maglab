# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 10:39:14 2021

@author: tveile
"""
import os

import numpy as np
import pandas as pd
import re

from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def nearest_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


class Magnetherm():
    def __init__(self, filename, name=None):
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
        
        
        if not name:
            self.name = os.path.basename(filename)
        else:
            self.name = name
   
    def split(self, tc='T0'):
        t, T = self.df['Time [s]'], self.df[tc+' [degC]']
        
        
        base_mask = self.df.State == 'BEFORE'
        if base_mask.sum() != 0:
            baseline = T[base_mask].mean()
        else:
            baseline = T[0]
            print('No baseline recorded! Baseline set to: %d' %baseline)
        
        #T = T-baseline
        
        heat_mask = self.df.State == 'EXPOSING'
        theat, Theat = t[heat_mask], T[heat_mask]
        
        cool_mask = self.df.State == 'WAIT'
        tcool, Tcool = t[cool_mask], T[cool_mask]

        # Removing points in sharp decline
        # tcool, Tcool = tcool[3:], Tcool[3:]

        return t.values, T.values, theat.values, Theat.values, tcool.values, Tcool.values, baseline
    

class Power():
    def __init__(self, Magnetherm, C, TC):
        self.m = Magnetherm
        self.C = C
        self.TC = TC
        
    def loss(self):
        t, T, theat, Theat, tcool, Tcool, baseline = self.m.split(self.TC)
        
        fp = savgol_filter(Tcool, 9, 2, deriv=1, delta=np.diff(tcool).mean())

        equiv = np.array([nearest_idx(Tcool, temperature)
                          for temperature in Theat])
        
        return fp[equiv]*self.C
    
    def power(self, method, compensate=None, **kwargs):
        method = method.lower()
        if method == 'savgol':
            tp, heat = self.savgol(**kwargs)
        
        elif method == 'box_lucas':
            tp, heat = self.box_lucas(**kwargs)
    
    
        loss = self.loss()
        P = heat - loss[loss.size-heat.size:]
    
    
        if compensate is not None:
            compensate = compensate.lower()
            start = np.where(t == theat[0])[0][0]
            end = np.where(t == theat[-1])[0][0]
            
            current = self.m.df['Current [A]'][start:end+1]
            set_current = self.m.props['set_current']
            
            if compensate=='linear':
                P = P*(set_current/current)
            
        return tp, P 
                
                
    def savgol(self, **kwargs):
        t, T, theat, Theat, tcool, Tcool, baseline = self.m.split(self.TC)
        
        fp = savgol_filter(Theat, 11, 2, deriv=1, delta=np.diff(theat).mean())
        heat = fp*self.C
        
        return theat[11:], heat[11:]
        
      
    def box_lucas(self, m=None, compensate=False, **kwargs):
        t, T, theat, Theat, tcool, Tcool, baseline = self.m.split(self.TC)
        
        if m:
            f = lambda t, L, P: P/L*(1-np.exp(-L/(m*self.C)*t))
            fp = lambda t, L, P: P/(m*self.C)*np.exp(-L/(m*self.C)*t)
        else:
            f = lambda t, L, P, m: P/L*(1-np.exp(-L/(m*self.C)*t))
            fp = lambda t, L, P, m: P/(m*self.C)*np.exp(-L/(m*self.C)*t)
        
        popt, pcov = curve_fit(f, theat, Theat, **kwargs)
        

        return theat, fp(theat, *popt)*self.C
        
        
    
        
        