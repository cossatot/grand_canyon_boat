import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from copy import deepcopy
import json
from ast import literal_eval

from mayavi import mlab

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

#for i, row in offsets.iterrows():
#    if row.pt[-1] == 'c':
#        if isinstance(row.chine, list):
#            for c in row.chine:
#                if c[-1] == 'r':
#                    offsets.loc[i, 'chine'].append(r_to_l(c))


with open('panel_pts.json') as fp:
    panels = json.load(fp)

for panel_key, panel_vals in panels.items():
    panels[panel_key] = {literal_eval(k):v for k, v in panel_vals.items()}

panels.update(make_sym_panels(panels))

for p in panels.values():
    p_tri_coords, p_tri_list = make_panel_tris(p, offsets)

    p_tri_coords, p_tri_list = simplify_panel_tris(p_tri_coords, p_tri_list)

    mlab.triangular_mesh(p_tri_coords[:,0],
                         p_tri_coords[:,1],
                         p_tri_coords[:,2],
                         p_tri_list,
                         color=(0,1,1)
    )

#for ch in chine_list:
#   l = mlab.plot3d(chine_dfs[ch]['x'].values, 
#                 chine_dfs[ch]['y'].values,
#                 chine_dfs[ch]['z'].values,
#                 color=(0.,0.,0.),
#                 line_width=50.
#                 )
mlab.show()