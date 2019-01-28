#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 15:21:47 2019

@author: mpleinti
"""

fig, axs = plt.subplots(3, 7, figsize=(10,4.5))

row = np.arange(3)
col = np.arange(7)
idx = 0

for r in row:
    for c in col:
        ax = axs[r][c]
        try:
            ax.imshow(rsp_e7767[idx], alpha=1)
            ax.set_title('det {}'.format(idx))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.scatter([47], [47], color='white', s=50)
            ax.scatter([47], [47], color='#ef8836', edgecolor='black', s=20)
            idx += 1
        except IndexError:
            ax.set_visible(False)

fig.tight_layout()

fig.savefig('/Users/mpleinti/Python/Plots/Images/pyspi/spi_irf_grp_0024_e7767_source.pdf')