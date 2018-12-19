import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

from mayavi import mlab

offsets = pd.read_excel('gc_draft_1_offsets.xlsx', sheet_name='points')

offsets['x'] = np.float_(offsets.x)
offsets['y'] = np.float_(offsets.y)
offsets['z'] = np.float_(offsets.z)

offsets['y'] *= 0.9
offsets['x'] *= 1.1


def parse_chines(row):
    if row.chine[0] == '(':
        chine_list = row.chine[1:-1].split(',')
    else:
        chine_list = row.chine

    return chine_list


def r_to_l(val):
    def replace_l(s):
        return s[:-1] + 'l'
    if isinstance(val, str):
        if val[-1] == 'r':
            return replace_l(val)
    elif isinstance(val, list):
        vv = deepcopy(val)
        for i, v in enumerate(vv):
            if v[-1] == 'r':
                vv[i] = replace_l(v)
        return vv


def make_sym_offsets(df):

    ldf = df[df.y != 0]

    ldf.y *= -1
    ldf['pt'] = [r_to_l(p) for p in ldf.pt]
    ldf['chine'] = [r_to_l(chine) for chine in ldf.chine]
    return ldf

#offsets['chine'] = offsets.apply(parse_chines, axis=1)

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

#chine_dfs = {}
#for ch in chine_list:
#    chine_dfs[ch] = pd.DataFrame(columns=['x', 'y', 'z'],
#                                 dtype=float)
#    cd = chine_dfs[ch]
#    for i, row in offsets.iterrows():
#        if ch in row.chine:
#            cd.loc[row.pt, :] = np.float_((row.x, row.y, row.z))
#    cd.sort_values('x', inplace=True)


#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#
#    ax.plot(chine_dfs[ch].x, chine_dfs[ch].y, chine_dfs[ch].z)
#
#plt.axis('equal')
#
#plt.show()

panels = {
    'p0': {
        1: {'a': '1l',
            'b': '1r'},
        2: {'a': '6l',
            'b': '6r'},
        3: {'a': '11l',
            'b': '11r'},
        4: {'a': '16l',
            'b': '16r'},
        5: {'c': '20c'},
        0: {'a': '28l',
            'b': '28r'},
        -1:{'a': '32l',
            'b': '32r'},
        -2:{'a': '36l',
            'b': '36r'},
        -5:{'a': '40l',
            'b': '40r'},
        -6:{'a': '45l',
            'b': '45r'},
       -6.5:{'a': '48l',
             'b': '48r'},
    },
    'p1r': {
        1: {'a': '1r',
            'b': '2r'},
        2: {'a': '6r',
            'b': '7r'},
        3: {'a': '12r',
            'b': '13r'},
        4: {'a': '17r',
            'b': '18r'},
        5: {'a': '21r',
            'b': '22r'},
        6: {'a': '24c',
            'b': '25r'},
        7: {'p': '27c'},
        0: {'a': '28r',
            'b': '29r'},
        -1:{'a': '32r',
            'b': '33r'},
        -2:{'a': '36r',
            'b': '37r'},
        -5:{'a': '41r',
            'b': '42r'},
        -6:{'a': '45r',
            'b': '46r'},
       -6.5:{'c': '49r'}
    },
    'p2r': {
        1: {'a': '2r',
            'b': '3r'},
        2: {'a': '7r',
            'b': '8r'},
        3: {'a': '13r',
            'b': '14r'},
        4: {'c': '18r'},
        0: {'a': '29r',
            'b': '30r'},
        -1:{'a': '33r',
            'b': '34r'},
        -2:{'a': '37r',
            'b': '38r'},
        -5:{'a': '42r',
            'b': '43r'},
        -6:{'c': '46r'}
    },
    'p3r': {
        0: {'c': '30r'},
        1: {'a': '3r',
            'b': '4r'},
        2: {'a': '8r',
            'b': '9r'},
        3: {'c': '14r'}
    },
    'p4r': {
        0: {'c': '30r'},
        1: {'a': '4r',
            'b': '5c'},
        2: {'a': '9r',
            'b': '10c'},
        3: {'a': '14r',
            'b': '15c'},
        4: {'a': '18r',
            'b': '19c'},
        5: {'a': '22r',
            'b': '23c'},
        6: {'a': '25r',
            'b': '26c'},
        7: {'c': '27c'}
    },
    'p5r': {
        2: {'c': '6r'},
        3: {'a': '11r',
            'b': '12r'},
        4: {'a': '16r',
            'b': '17r'},
        5: {'a': '20c',
            'b': '21r'},
        6: {'c': '24c'}
    },
    'p6r': {
        0: {'c': '30r'},
        -1:{'a': '34r',
            'b': '35c'},
        -2:{'a': '38r',
            'b': '39c'},
        -5:{'a': '43r',
            'b': '44c'},
        -6:{'a': '46r',
            'b': '47c'},
      -6.5:{'a': '49r',
            'b': '50c'},
        -7:{'a': '51r',
            'b': '52c'}
    },
    'p7r': {
        -2:{'c': '36r'},
        -5:{'a': '40r',
            'b': '41r'},
        -6:{'c': '45r'}
    }
}


def make_sym_panel(pan):
    pan_c = deepcopy(pan)
    for station, ends in pan_c.items():
        for end, point in ends.items():
            if point[-1] == 'r':
                pan_c[station][end] = r_to_l(point)
    return pan_c


def make_sym_panels(panels):
    new_panels = {}
    for k, v in panels.items():
        if k[-1] == 'r':
            new_panels[r_to_l(k)] = make_sym_panel(v)
    return new_panels

panels.update(make_sym_panels(panels))



def make_section_tris(pa, pb, point_df):
    point_list = list(pa.values()) + list(pb.values())

    coord_list = [point_df.loc[(point_df.pt == p), ['x', 'y', 'z']].values[0]
                  for p in point_list]

    coord_list = np.array(coord_list)

    if len(point_list) == 3:
        tri_list = [[0,1,2]]
    elif len(point_list) == 4:
        tri_list = [[0,2,1],
                    [1,3,2]]

    return coord_list, tri_list


def make_panel_tris(panel, point_df):
    ks = sorted(panel.keys())

    tri_tups = [make_section_tris(panel[k], panel[ks[i+1]], point_df)  
                for i, k in enumerate(ks[:-1])]
    
    tri_coords = np.vstack([t[0] for t in tri_tups])

    tri_list = []
    tri_line_counter = 0
    for t in tri_tups:
        tl = np.array(t[1])
        tl += tri_line_counter
        tri_list.append(tl)

        tri_line_counter += len(t[0])

    tri_list = np.vstack(tri_list)

    return tri_coords, tri_list


#for ch in chine_list:
    #l = mlab.plot3d(chine_dfs[ch]['x'].values, 
                  #chine_dfs[ch]['y'].values,
                  #chine_dfs[ch]['z'].values,
                  #color=(0.,0.,0.),
                  #line_width=50.
                  #)

def simplify_panel_tris(tri_coords, tri_list):
    new_tri_list = deepcopy(tri_list)
    unique_pts = []

    for i, row in enumerate(tri_coords):
        _row = tuple(row)
        if _row not in unique_pts:
            unique_pts.append(_row)
        else:
            prev_row = unique_pts.index(_row)
            new_tri_list[new_tri_list == i] = prev_row
            #new_tri_list[new_tri_list > i] -= 1

    return np.array(unique_pts), new_tri_list

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