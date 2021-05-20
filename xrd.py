import numpy as np
import xml.etree.ElementTree as ET

class XRD():
    def __init__(self, filepath):
        with open(filepath, 'r') as f:
            xml = ET.parse(f)
        root = xml.getroot()
        
        self.cts = np.array(root[2][5][1][4].text.split(' '), dtype=float)

        self.startAng = float(root[2][5][1][0][0].text)
        self.endAng = float(root[2][5][1][0][1].text)

        self.ang = np.linspace(self.startAng, self.endAng, self.cts.size)

        