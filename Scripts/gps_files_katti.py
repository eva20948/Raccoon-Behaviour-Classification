
"""
Filename: gps_files_katti.py
Author: Eva Reinhardt
Date: 2025-01-20
Version: 1.0
Description: This file extracts the gps columns from the general acc files of Katti's dataset.
"""

from raccoon_acc_setup import variables_simplefunctions as sim_func



files = sim_func.IMPORT_PARAMETERS['Katti']['filepath_acc']
for filepath in files:
    data = ['GPS,number of days,date,day_of_the_week,time,location-long,location-lat,height-above-ellipsoid,type of fix,status,flag of gpsclock,noiselevel,expended time to get fix,date of fix,weekday of fix,time of day of gps fix,battery voltage,temperature,ground-speed,heading,"speed inaccuracy estimate, m/s","horizontal inaccuracy estimate, m",pdop,satellite-count,"pressure, hPa",q0raw,q1raw,q2raw,q3raw,magnx,magny,magnz']
    output_path = filepath
    output_path = output_path.split('.')
    output_path[-2] = output_path[-2]+'_gps'
    output_path = '.'.join(output_path)
    if filepath[-4:] == '.txt':
        with open(filepath, 'r') as file:
            for line in file:
                if 'GPS' in line:
                    data.append(line)
        f = open(output_path, 'w')
        f.write('\n'.join(data))



