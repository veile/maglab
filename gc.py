from __future__ import print_function, unicode_literals, division
from collections import defaultdict

import codecs
import re
import os
import time
import math
import struct
from struct import unpack
from xml.etree import ElementTree
import numpy as np
import pandas as pd
import glob

import matplotlib.pyplot as plt

# Constants used for binary file parsing
ENDIAN = '>'
STRING = ENDIAN + '{}s'
UINT8 = ENDIAN + 'B'
UINT16 = ENDIAN + 'H'
INT16 = ENDIAN + 'h'
INT32 = ENDIAN + 'i'

def find_nearest_idx(array, value):
    array = np.asarray(array)
    return (np.abs(array - value)).argmin()

def parse_utf16_string(file_, encoding='UTF16'):
    """Parse a pascal type UTF16 encoded string from a binary file object"""
    # First read the expected number of CHARACTERS
    string_length = unpack(UINT8, file_.read(1))[0]
    # Then read and decode
    parsed = unpack(STRING.format(2 * string_length),
                    file_.read(2 * string_length))
    return parsed[0].decode(encoding)


class CHFile(object):
    """Class that implementats the Agilent .ch file format version 179

    .. warning:: Not all aspects of the file header is understood, so there may and probably
       is information that is not parsed. See the method :meth:`._parse_header_status` for
       an overview of which parts of the header is understood.

    .. note:: Although the fundamental storage of the actual data has change, lots of
       inspiration for the parsing of the header has been drawn from the parser in the
       `ImportAgilent.m file <https://github.com/chemplexity/chromatography/blob/dev/
       Methods/Import/ImportAgilent.m>`_ in the `chemplexity/chromatography project
       <https://github.com/chemplexity/chromatography>`_ project. All credit for the parts
       of the header parsing that could be reused goes to the author of that project.

    Attributes:
        values (numpy.array): The internsity values (y-value) or the spectrum. The unit
            for the values is given in `metadata['units']`
        metadata (dict): The extracted metadata
        filepath (str): The filepath this object was loaded from

    """

    # Fields is a table of name, offset and type. Types 'x-time' and 'utf16' are specially
    # handled, the rest are format arguments for struct unpack
    fields = (
        ('sequence_line_or_injection', 252, UINT16),
        ('injection_or_sequence_line', 256, UINT16),
        ('start_time', 282, 'x-time'),
        ('end_time', 286, 'x-time'),
        ('version_string', 326, 'utf16'),
        ('description', 347, 'utf16'),
        ('sample', 858, 'utf16'),
        ('operator', 1880, 'utf16'),
        ('date', 2391, 'utf16'),
        ('inlet', 2492, 'utf16'),
        ('instrument', 2533, 'utf16'),
        ('method', 2574, 'utf16'),
        ('software version', 3601, 'utf16'),
        ('software name', 3089, 'utf16'),
        ('software revision', 3802, 'utf16'),
        ('units', 4172, 'utf16'),
        ('detector', 4213, 'utf16'),
        ('yscaling', 4732, ENDIAN + 'd')
    )
    # The start position of the data
    data_start = 6144
    # The versions of the file format supported by this implementation
    supported_versions = {179}

    def __init__(self, filepath):
        self.filepath = filepath
        self.metadata = {}
        with open(self.filepath, 'rb') as file_:
            self._parse_header(file_)
            self.values = self._parse_data(file_)

    def _parse_header(self, file_):
        """Parse the header"""
        # Parse and check version
        length = unpack(UINT8, file_.read(1))[0]
        parsed = unpack(STRING.format(length), file_.read(length))
        version = int(parsed[0])
        if not version in self.supported_versions:
            raise ValueError('Unsupported file version {}'.format(version))
        self.metadata['magic_number_version'] = version

        # Parse all metadata fields
        for name, offset, type_ in self.fields:
            file_.seek(offset)
            if type_ == 'utf16':
                self.metadata[name] = parse_utf16_string(file_)
            elif type_ == 'x-time':
                self.metadata[name] = unpack(ENDIAN + 'f', file_.read(4))[0] / 60000
            else:
                self.metadata[name] = unpack(type_, file_.read(struct.calcsize(type_)))[0]

        # Convert date
        self.metadata['datetime'] = time.strptime(self.metadata['date'], '%d-%b-%y, %H:%M:%S')

    def _parse_header_status(self):
        """Print known and unknown parts of the header"""
        file_ = open(self.filepath, 'rb')

        print('Header parsing status')
        # Map positions to fields for all the known fields
        knowns = {item[1]: item for item in self.fields}
        # A couple of places has a \x01 byte before a string, these we simply skip
        skips = {325, 3600}
        # Jump to after the magic number version
        file_.seek(4)

        # Initialize variables for unknown bytes
        unknown_start = None
        unknown_bytes = b''
        # While we have not yet reached the data
        while file_.tell() < self.data_start:
            current_position = file_.tell()
            # Just continue on skip bytes
            if current_position in skips:
                file_.read(1)
                continue

            # If we know about a data field that starts at this point
            if current_position in knowns:
                # If we have collected unknown bytes, print them out and reset
                if unknown_bytes != b'':
                    print('Unknown at', unknown_start, repr(unknown_bytes.rstrip(b'\x00')))
                    unknown_bytes = b''
                    unknown_start = None

                # Print out the position, type, name and value of the known value
                print('Known field at {: >4},'.format(current_position), end=' ')
                name, _, type_ = knowns[current_position]
                if type_ == 'x-time':
                    print('x-time, "{: <19}'.format(name + '"'),
                          unpack(ENDIAN + 'f', file_.read(4))[0] / 60000)
                elif type_ == 'utf16':
                    print(' utf16, "{: <19}'.format(name + '"'),
                          parse_utf16_string(file_))
                else:
                    size = struct.calcsize(type_)
                    print('{: >6}, "{: <19}'.format(type_, name + '"'),
                          unpack(type_, file_.read(size))[0])
            else:  # We do not know about a data field at this position If we have already
                # collected 4 zero bytes, assume that we are done with this unkonw field,
                # print and reset
                if unknown_bytes[-4:] == b'\x00\x00\x00\x00':
                    print('Unknown at', unknown_start, repr(unknown_bytes.rstrip(b'\x00')))
                    unknown_bytes = b''
                    unknown_start = None

                # Read one byte and save it
                one_byte = file_.read(1)
                if unknown_bytes == b'':
                    # Only start a new collection of unknown bytes, if this byte is not a
                    # zero byte
                    if one_byte != b'\x00':
                        unknown_bytes = one_byte
                        unknown_start = file_.tell() - 1
                else:
                    unknown_bytes += one_byte

        file_.close()

    def _parse_data(self, file_):
        """Parse the data"""
        # Go to the end of the file and calculate how many points 8 byte floats there are
        file_.seek(0, 2)
        n_points = (file_.tell() - self.data_start) // 8

        # Read the data into a numpy array
        file_.seek(self.data_start)
        return np.fromfile(file_, dtype='<d', count=n_points) * self.metadata['yscaling']

    
class GC():
    def __init__(self, folders):
        self.TCD = []
        self.FID = []

        # If folders is str input, convert to list
        if isinstance(folders, str):
            self.folders = [folders]
            
        else:
            self.folders = folders
        
        self.seq = [glob.glob(folder+'/*.D') for folder in self.folders]
        self.seq = [item for sublist in self.seq for item in sublist] # Flattens list
        
        for f in self.seq:
            ch_files = glob.glob(f+'/*.ch')
            
            for ch_file in ch_files:
                if 'FID' in ch_file:
                    self.FID.append(CHFile(ch_file))
                
                elif 'TCD' in ch_file:
                    self.TCD.append(CHFile(ch_file))
                    
                else:
                    raise Exception('No .ch file found!')
        
        if (len(self.TCD)+len(self.FID)) < 1:
            raise Exception('No data found in specified folders')
    
        injected = pd.to_datetime([data.metadata['date'] for data in self.TCD],
                                  format='%d-%b-%y, %H:%M:%S')
        self.injected = np.array(injected, dtype=np.datetime64)
    
    def __add__(self, other):
        new = GC(self.folder)
    
        new.seq += other.seq
        new.TCD += other.TCD
        new.FID += other.FID
        new.injected = np.append(new.injected, other.injected)
        
        return new
    
    def remove_entry(self, idx):
        self.seq.pop(idx)
        self.TCD.pop(idx)
        self.FID.pop(idx)
        self.injected = np.delete(self.injected, idx)
        
        
    def trapz(self, tleft, tright, det='TCD', idx=slice(None), plot=False):
    
        if det == 'TCD':
            signals = [data.values for data in self.TCD[idx]]
            times = [np.linspace(data.metadata['start_time'],
                                 data.metadata['end_time'],
                                 len(data.values)) for data in self.TCD[idx]]
        if det == 'FID':
            signals = [data.values for data in self.FID[idx]]
            times = [np.linspace(data.metadata['start_time'],
                                 data.metadata['end_time'],
                                 len(data.values)) for data in self.FID[idx]]
        
        if plot:
            fig, ax = plt.subplots(figsize=(9, 6))
            
            ax.set_xlabel('Time [min]')
            ax.set_ylabel('Voltage [µV]')
        
        areas = []
        for i in range(len(signals)):
            t = times[i]
            y = signals[i]
            
            # Finding idx for specified time values from user
            ileft = find_nearest_idx(t, tleft)
            iright = find_nearest_idx(t, tright)
            
            t, y = t[ileft:iright], y[ileft:iright]
            
            # # Finding baseline by fitting a linear fit between first
            # # and last point.
            # a = (y[-1]-y[0])/(t[-1]-t[0])
            # b = y[0]-a*t[0]
            
            # base = a*t+b
            base = np.linspace(y[0], y[-1], y.size)
            
            areas.append(np.trapz(y-base, t, dx=np.diff(t).mean())*60) # unit of µV*s
            
            if plot:
                ax.plot(t, y-base, label=f'Seq. {idx.start+i}')
        
        if plot:
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1.02))
            fig.tight_layout()
            plt.show()
       
        return np.array(areas)