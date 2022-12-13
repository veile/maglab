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

def get_slope(xpts, ypts):
    """Function that returns the slope of a fitted line through (xpts, ypts)."""
    popt, pcov = curve_fit(lambda x, a, b: a*x+b, xpts, ypts)
    P = popt[0]
    Perr = np.sqrt(np.diag(pcov))[0]
    return P, Perr

# fitted current and field expression
# B = k1*I
mu0 = 4 * np.pi * 1e-7
N = 18
L = 54e-3
R1 = 13e-3
R2 = 23e-3

k1 = 1/2 * mu0*N/(R2-R1)*np.log((np.sqrt(4*R2**2+L**2)+2*R2)/(np.sqrt(4*R1**2+L**2)+2*R1))*1e3

# Icoil = k2(f)*Iset
k2 = lambda f: -8.79797909e-07*f**2 - 8.52364188e-04*f + 4.99950719e+00
def coil_current(f, power_current):
    return k2(f)*power_current


def current_to_field(f, power_current, r=18e-3, L=53e-3, N=18):
    I = coil_current(f, power_current)
    return k1*I

def field_to_current(field, f):
    mu0 = 4*np.pi*1e-7
    I = field/k1

    return I/k2(f)



class Magnetherm():
    def __init__(self, filename, tc='T0', name=None):
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
            
        self.split(tc)
        
        
    def __repr__(self):
        return 'I=%.1f A, f=%.1f kHz' %(self.props['set_current'], self.props['frequency']*1e-3)  
    
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
        
        # Removing duplicate values from heating curve
        idx = Theat != np.roll(Theat, 1)
        theat, Theat = theat[idx], Theat[idx]        
        
        cool_mask = self.df.State == 'WAIT'
        tcool, Tcool = t[cool_mask], T[cool_mask]

        # Removing duplicates from cooling curve
        idx = Tcool != np.roll(Tcool, 1)
        tcool, Tcool = tcool[idx], Tcool[idx]
        
        
        split_df = pd.DataFrame({'theat [s]': theat, 'Theat [degC]': Theat,

                                 'tcool [s]': tcool, 'Tcool [degC]': Tcool} )
                                 
        self.df = pd.concat([self.df, split_df], axis=1)
        
    
    

class Power():
    def __init__(self, Magnetherm, C):
        self.m = Magnetherm
        self.C = C
    
    
    def loss(self, tcool, Tcool, T, N=5):
        i = nearest_idx(Tcool, T)
        
        start, end = i-int(np.floor(N/2)), i+int(np.ceil(N/2))
        if end > tcool.size:
            end = tcool.size
            start = end-N
        elif start < 0:
            start = 0
            end = start+N
        
        return get_slope(tcool[start:end], Tcool[start:end])
    
    def continuous_slope(self, N=5, return_seperate=False):
        # Extracting heat and cooling curves from Magnetherm
        theat, Theat = self.m.df['theat [s]'], self.m.df['Theat [degC]']
        theat, Theat = theat[~np.isnan(theat)], Theat[~np.isnan(Theat)]
        
        tcool, Tcool = self.m.df['tcool [s]'], self.m.df['Tcool [degC]']
        tcool, Tcool = tcool[~np.isnan(tcool)], Tcool[~np.isnan(Tcool)]
        
        Ptot, Ptot_err = zip(*[get_slope(theat[i:i+N], Theat[i:i+N]) for i in range(theat.size-N)])
        Ptot, Ptot_err = np.array(Ptot), np.array(Ptot_err)
        
        Tp = np.array([np.mean(Theat[i:i+N]) for i in range(theat.size-N)])
        
        Pout, Pout_err = zip(*[self.loss(tcool, Tcool, T, N) for T in Tp])
        
        Pout, Pout_err = np.array(Pout), np.array(Pout_err)
 
        if return_seperate:
            return Tp, Ptot*self.C, Pout*self.C
        
        return Tp, (Ptot-Pout)*self.C, np.sqrt(Ptot_err**2+Pout_err**2)*self.C
    
    # def loss(self):
        # t, T, theat, Theat, tcool, Tcool, baseline = self.m.split(self.TC)
        
        # fp = savgol_filter(Tcool, 9, 2, deriv=1, delta=np.diff(tcool).mean())

        # equiv = np.array([nearest_idx(Tcool, temperature)
                          # for temperature in Theat])
        
        # return fp[equiv]*self.C
    
    # def power(self, method, compensate=None, **kwargs):
        # method = method.lower()
        # if method == 'savgol':
            # tp, heat = self.savgol(**kwargs)
        
        # elif method == 'box_lucas':
            # tp, heat = self.box_lucas(**kwargs)
    
    
        # loss = self.loss()
        # P = heat - loss[loss.size-heat.size:]
    
    
        # if compensate is not None:
            # compensate = compensate.lower()
            # start = np.where(t == theat[0])[0][0]
            # end = np.where(t == theat[-1])[0][0]
            
            # current = self.m.df['Current [A]'][start:end+1]
            # set_current = self.m.props['set_current']
            
            # if compensate=='linear':
                # P = P*(set_current/current)
            
        # return tp, P 
                
                
    # def savgol(self, **kwargs):
        # t, T, theat, Theat, tcool, Tcool, baseline = self.m.split(self.TC)
        
        # fp = savgol_filter(Theat, 11, 2, deriv=1, delta=np.diff(theat).mean())
        # heat = fp*self.C
        
        # return theat[11:], heat[11:]
        
      
    # def box_lucas(self, m=None, compensate=False, **kwargs):
        # t, T, theat, Theat, tcool, Tcool, baseline = self.m.split(self.TC)
        
        # if m:
            # f = lambda t, L, P: P/L*(1-np.exp(-L/(m*self.C)*t))
            # fp = lambda t, L, P: P/(m*self.C)*np.exp(-L/(m*self.C)*t)
        # else:
            # f = lambda t, L, P, m: P/L*(1-np.exp(-L/(m*self.C)*t))
            # fp = lambda t, L, P, m: P/(m*self.C)*np.exp(-L/(m*self.C)*t)
        
        # popt, pcov = curve_fit(f, theat, Theat, **kwargs)
        

        # return theat, fp(theat, *popt)*self.C
        
        
    
        
        