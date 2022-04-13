import pandas as pd
import numpy as np

from entsoe import EntsoePandasClient

from datetime import date, datetime

import warnings
warnings.filterwarnings("ignore")

import bidding_zones

client = EntsoePandasClient(api_key="e3547493-771f-47ae-9df6-d61cee2a5ffa")

start = pd.Timestamp('20140101', tz='Europe/Brussels')
end = pd.Timestamp(datetime.strftime(datetime.today(), "%Y%m%d"), tz='Europe/Brussels')

# for country in list(bidding_zones.BIDDING_ZONES_CWE.keys()):
#     globals()[f"df_{country}"] = client.query_installed_generation_capacity(country_code=country, start=start,end=end, psr_type=None)
#     globals()[f"df_{country}"].index = globals()[f"df_{country}"].index.year
#     globals()[f"df_{country}"].index = pd.MultiIndex.from_product([globals()[f"df_{country}"].index, [f'{country}']])

# df = pd.concat([globals()[f"df_{country}"] for country in list(bidding_zones.BIDDING_ZONES_CWE.keys())], axis=0)
# df = df.groupby(level=[0,1]).sum()
# print(df)
# print(df.columns)

# column_names = ['Wind Offshore', 'Wind Onshore', 'Solar', 'Other renewable', 'Geothermal', 'Biomass', 
# 'Hydro Pumped Storage', 'Hydro Run-of-river and poundage','Hydro Water Reservoir', 'Waste',
# 'Nuclear', 'Fossil Brown coal/Lignite',  'Fossil Hard coal', 'Fossil Coal-derived gas',
# 'Fossil Gas', 'Fossil Oil', 'Other']

# df = df.reindex(columns=column_names)

# df.to_excel('data/dataframes/generation_capacity.xlsx')

df = pd.read_excel('data/dataframes/generation_capacity.xlsx', index_col=[0,1], header=0)
df = df.div(df.sum(axis=1), axis=0).swaplevel(0, 1).multiply(100)
df = df.sort_index(axis=0, level=0)
print(df)

# Q1 = df.quantile(0.05)
# Q3 = df.quantile(0.95)
# IQR = Q3 - Q1
# df = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]

import matplotlib.pyplot as plt
plt.style.use(['science'])
from matplotlib import cm, colors
import matplotlib
# matplotlib.use("pgf")
# matplotlib.rcParams.update({
#     "pgf.texsystem": "pdflatex",
#     'font.family': 'serif',
#     'text.usetex': True,
#     'pgf.rcfonts': False,
#     'font.size' : 20,
# })

import color_blinded
plt.cm.register_cmap('rainbow_discrete', color_blinded.tol_cmap('rainbow_discrete'))

fig, ax = plt.subplots(figsize=(21,8))
df.plot.bar(stacked=True, ax=ax, cmap="rainbow_discrete")
# plt.suptitle('Generation capacity in CWE region', fontsize=30)
ax.set_xlabel('(bidding zone, year)', fontsize=24)
ax.set_ylim(0,100)
ax.set_ylabel('share of generation capacity $[\%]$', fontsize=24)
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1,
                box.width * 0.9 , box.height * 0.9])

# Put a legend below current axis
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35),
        fancybox=True, shadow=True, ncol=9, fontsize=12)
# ax.grid(True)
fig.subplots_adjust(bottom=0.4)
plt.tight_layout()
plt.savefig("./plots/price_spread/generation_capacity.pdf", dpi=1200)
plt.show()
# plt.close('all')