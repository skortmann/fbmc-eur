"""
================
The Sankey class
================

Demonstrate the Sankey class by producing three basic diagrams.
"""

import matplotlib.pyplot as plt
from pandas import concat

from bidding_zones import BIDDING_ZONES_CWE
plt.style.use(['science'])
from matplotlib import cm, colors
# import matplotlib
# matplotlib.use("pgf")
# matplotlib.rcParams.update({
#     "pgf.texsystem": "pdflatex",
#     'font.family': 'serif',
#     'text.usetex': True,
#     'pgf.rcfonts': False,
# })

import color_blinded
plt.cm.register_cmap('sunset', color_blinded.tol_cmap('sunset'))

from matplotlib.sankey import Sankey


###############################################################################
# Example 1 -- Mostly defaults
#
# This demonstrates how to create a simple diagram by implicitly calling the
# Sankey.add() method and by appending finish() to the call to the class.

# Sankey(flows=[0.25, 0.15, 0.60, -0.20, -0.15, -0.05, -0.50, -0.10],
#        labels=['', '', '', 'First', 'Second', 'Third', 'Fourth', 'Fifth'],
#        orientations=[-1, 1, 0, 1, 1, 1, 0, -1]).finish()
# plt.title("The default settings produce a diagram like this.")

###############################################################################
# Notice:
#
# 1. Axes weren't provided when Sankey() was instantiated, so they were
#    created automatically.
# 2. The scale argument wasn't necessary since the data was already
#    normalized.
# 3. By default, the lengths of the paths are justified.


###############################################################################
# Example 2
#
# This demonstrates:
#
# 1. Setting one path longer than the others
# 2. Placing a label in the middle of the diagram
# 3. Using the scale argument to normalize the flows
# 4. Implicitly passing keyword arguments to PathPatch()
# 5. Changing the angle of the arrow heads
# 6. Changing the offset between the tips of the paths and their labels
# 7. Formatting the numbers in the path labels and the associated unit
# 8. Changing the appearance of the patch and the labels after the figure is
#    created

# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[],
#                      title="Flow Diagram of a Widget")
# sankey = Sankey(ax=ax, scale=0.01, offset=0.2, head_angle=180,
#                 format='%.0f', unit='%')
# sankey.add(flows=[25, 0, 60, -10, -20, -5, -15, -10, -40],
#            labels=['', '', '', 'First', 'Second', 'Third', 'Fourth',
#                    'Fifth', 'Hurray!'],
#            orientations=[-1, 1, 0, 1, 1, 1, -1, -1, 0],
#            pathlengths=[0.25, 0.25, 0.25, 0.25, 0.25, 0.6, 0.25, 0.25,
#                         0.25],
#            patchlabel="Widget\nA")  # Arguments to matplotlib.patches.PathPatch
# diagrams = sankey.finish()
# diagrams[0].texts[-1].set_color('r')
# diagrams[0].text.set_fontweight('bold')

###############################################################################
# Notice:
#
# 1. Since the sum of the flows is nonzero, the width of the trunk isn't
#    uniform.  The matplotlib logging system logs this at the DEBUG level.
# 2. The second flow doesn't appear because its value is zero.  Again, this is
#    logged at the DEBUG level.


###############################################################################
# Example 3
#
# This demonstrates:
#
# 1. Connecting two systems
# 2. Turning off the labels of the quantities
# 3. Adding a legend

# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[], title="Two Systems")
# flows = [0.25, 0.15, 0.60, -0.10, -0.05, -0.25, -0.15, -0.10, -0.35]
# sankey = Sankey(ax=ax, unit=None)
# sankey.add(flows=flows, label='one',
#            orientations=[-1, 1, 0, 1, 1, 1, -1, -1, 0])
# sankey.add(flows=[-0.25, 0.15, 0.1], label='two',
#            orientations=[-1, -1, -1], prior=0, connect=(0, 0))
# diagrams = sankey.finish()
# diagrams[-1].patch.set_hatch('/')
# plt.legend()

###############################################################################
# Notice that only one connection is specified, but the systems form a
# circuit since: (1) the lengths of the paths are justified and (2) the
# orientation and ordering of the flows is mirrored.

import create_dataframe
df_total = create_dataframe.create_dataframe_scheduled_exchanges(countries_scheduled_exchanges=list(BIDDING_ZONES_CWE.keys()))
df_total = df_total.groupby((df_total.index.to_period('Q'))).sum()
df_total.to_excel("./data/dataframes/scheduled_exchanges_group_by.xlsx")
import pandas as pd
pd.set_option("display.max_columns", 101)
print(df_total.tail(2))

## DE_LU
#DE_LU_AT       7076019.0
# DE_LU_BE       3652804.0
# DE_LU_FR       5208275.0
# DE_LU_NL       1442580.0

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[], title="Cross border trading in CWE region for 2021Q4")
sankey = Sankey(ax=ax, scale=0.02, offset=0.4,
                format='%.0f', unit='%')
sankey.add(flows=[(17309915/17309915)*100, (-7051596/17309915)*100, (-3642772/17309915)*100, (-5181693/17309915)*100, (-1433854/17309915)*100], 
        labels=['DE_LU', "AT", "BE", "FR", "NL"],
        orientations=[0, -1, 1, 0, 1],
        trunklength=3)
# sankey.add(flows=[0.25, -0.25], label='France',
#            orientations=[0, 0], prior=0, connect=(1, 0))
# sankey.add(flows=[0.25, -0.25], label='two',
#            orientations=[0, 0], prior=2, connect=(1, 0))
diagrams = sankey.finish()
# diagrams[-1].patch.set_hatch('/')
plt.legend()

# ## NL

# fig = plt.figure(figsize=(10,10))
# ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[], title="Cross border trading in CWE region for 2021Q4")
# sankey = Sankey(ax=ax, scale=0.01, offset=0.2,
#                 format='%.0f', unit='%')
# sankey.add(flows=[(1311211/1311211)*100, (-1173408/1311211)*100, (-137803/1311211)*100], 
#         labels=['NL', "BE", "DE_LU"],
#         orientations=[0, -1, 0])
# # sankey.add(flows=[0.25, -0.25], label='France',
# #            orientations=[0, 0], prior=0, connect=(1, 0))
# # sankey.add(flows=[0.25, -0.25], label='two',
# #            orientations=[0, 0], prior=2, connect=(1, 0))
# diagrams = sankey.finish()
# # diagrams[-1].patch.set_hatch('/')
# plt.legend()

# ## BE

# fig = plt.figure(figsize=(10,10))
# ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[], title="Cross border trading in CWE region for 2021Q4")
# sankey = Sankey(ax=ax, scale=0.01, offset=0.2,
#                 format='%.0f', unit='%')
# sankey.add(flows=[(1117073/1117073)*100, (-495737/1117073)*100, (-295944/1117073)*100, (-325392/1117073)*100], 
#         labels=['BE', "FR", "DE_LU", "NL"],
#         orientations=[0, -1, 0, 1])
# # sankey.add(flows=[0.25, -0.25], label='France',
# #            orientations=[0, 0], prior=0, connect=(1, 0))
# # sankey.add(flows=[0.25, -0.25], label='two',
# #            orientations=[0, 0], prior=2, connect=(1, 0))
# diagrams = sankey.finish()
# # diagrams[-1].patch.set_hatch('/')
# plt.legend()

# ## FR

# fig = plt.figure(figsize=(10,10))
# ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[], title="Cross border trading in CWE region for 2021Q4")
# sankey = Sankey(ax=ax, scale=0.01, offset=0.2,
#                 format='%.0f', unit='%')
# sankey.add(flows=[(100060/100060)*100, (-17683/100060)*100, (-82377/100060)*100], 
#         labels=['FR', "BE", "DE_LU"],
#         orientations=[0, 1, 0])
# # sankey.add(flows=[0.25, -0.25], label='France',
# #            orientations=[0, 0], prior=0, connect=(1, 0))
# # sankey.add(flows=[0.25, -0.25], label='two',
# #            orientations=[0, 0], prior=2, connect=(1, 0))
# diagrams = sankey.finish()
# # diagrams[-1].patch.set_hatch('/')
# plt.legend()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.sankey import Sankey

# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[],
#                      title="Vereinfachtes Kraftwerksmodell")
# sankey = Sankey(ax=ax, unit=None)
# sankey.add(flows=[1.0, -0.3, -0.1, -0.1, -0.5],
#            labels=['P$el$', 'Q$ab,vd$', 'P$vl,vd$', 'P$vl,mot$', ''],
#            label='Laden',
#            orientations=[0, -1, 1, 1, 0])
# sankey.add(flows=[0.5, 0.1, 0.1, -0.1, -0.1, -0.1, -0.1, -0.3], fc='#37c959',
#            label='Entladen',
#            labels=['P$mech$', 'Q$zu,ex$', 'Q$zu,rekup$', 'P$vl,tb$', 'P$vl,gen$',         'Q$ab,tb$', 'Q$ab,rekup$', 'P$nutz$'],
#            orientations=[0, -1, -1, 1, 1, -1, -1, 0], prior=0, connect=(4, 0))
# sankey.add(flows=[-0.1, 0.1],
#            label='Rekuperator',
#            #labels=['bla'],
#            orientations=[-1,-1], prior=1, connect=(2, 0))
# diagrams = sankey.finish()
# diagrams[-1].patch.set_hatch('/')
# plt.legend(loc='lower right')

plt.show()


#############################################################################
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.sankey`
#    - `matplotlib.sankey.Sankey`
#    - `matplotlib.sankey.Sankey.add`
#    - `matplotlib.sankey.Sankey.finish`
