import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from copy import deepcopy
import json
from ast import literal_eval
from scipy.interpolate import splprep, splev

from mayavi import mla

import sys; sys.path.append('../../../')
from boatbuilding.panels import *

offsets = pd.read_excel('gc_draft_1_offsets.xlsx', sheet_name='points')

offsets['x'] = np.float_(offsets.x)
offsets['y'] = np.float_(offsets.y)
offsets['z'] = np.float_(offsets.z)

offsets['y'] *= 0.9
offsets['x'] *= 1.1

offsets['x'] -= offsets.x.min()

offsets['chine'].fillna('', inplace=True)

offsets['chine'] = offsets.apply(parse_chines, axis=1)

left_offsets = make_sym_offsets(offsets)

offsets = pd.concat((offsets, left_offsets), ignore_index=True)

for i, row in offsets.iterrows():
    if row.pt[-1] == 'c':
        if isinstance(row.chine, list):
            for c in row.chine:
                if c[-1] == 'r':
                    offsets.loc[i, 'chine'].append(r_to_l(c))


chine_list = []
for ch in offsets.chine:
    if isinstance(ch, str):
        if ch not in chine_list:
            chine_list.append(ch)
    elif isinstance(ch, list):
        for c in ch:
            if c not in chine_list:
                chine_list.append(c)

chine_dfs = {}
for ch in chine_list:
    chine_dfs[ch] = pd.DataFrame(columns=['x', 'y', 'z'],
                                 dtype=float)
    cd = chine_dfs[ch]
    for i, row in offsets.iterrows():
        if ch in row.chine:
            cd.loc[row.pt, :] = np.float_((row.x, row.y, row.z))
    cd.sort_values('x', inplace=True)


#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#
#    ax.plot(chine_dfs[ch].x, chine_dfs[ch].y, chine_dfs[ch].z)
#
#plt.axis('equal')
#
#plt.show()


with open('panel_pts.json') as fp:
    panels = json.load(fp)

for panel_key, panel_vals in panels.items():
    panels[panel_key] = {literal_eval(k):v for k, v in panel_vals.items()}

panels.update(make_sym_panels(panels))

chine_splines = {}

for ch, cdf in chine_dfs.items():
    try:
        tck, u = splprep([cdf.x, cdf.y, cdf.z])
        new_x, new_y, new_z = splev(np.linspace(0,1), tck)

        chine_splines[ch] = {'x': new_x,
                             'y': new_y,
                             'z': new_z}
    except:
        chine_splines[ch] = {'x': cdf.x.values,
                             'y': cdf.y.values,
                             'z': cdf.z.values}




for ch in chine_list:
    l = mlab.plot3d(chine_dfs[ch]['x'].values, 
                 chine_dfs[ch]['y'].values,
                 chine_dfs[ch]['z'].values,
                 color=(0.,0.,0.),
                 line_width=50.
                 )
    ll = mlab.plot3d(chine_splines[ch]['x'],
                     chine_splines[ch]['y'],
                     chine_splines[ch]['z'],
                     line_width=50.
                     )




mlab.show()