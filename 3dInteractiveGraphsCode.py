# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 21:58:32 2026

@author: nicho
"""

import numpy as np
import plotly.graph_objects as go
import pandas as pd
import plotly.io as pio

'''Generate interactive 3D surfaces for either the symbolic regression codes
or Neural Network outputs'''

#set output for plotly to open in browser
pio.renderers.default = "browser"

#select beam energy
BeamE = '800'
#BeamE = 'xytrain90'

#Set input (string1 and 2) and output variables (string3)
string1 = 'X'
string2 = 'K'
string3 = 'F1'

#select trymunber and grid divisions
trynumber = 4
N = 100

#Set column names
columns = ['xsection','incompressability','f1n','v2n','f1','v2']
#where the simulation data file is
datasetDirectory = fr"C:\Users\nicho\Downloads\Research\Dr. Li\DNN+Bayes\Traing data\E{BeamE}train120.dat"
#import data and set up names for variables for later use
dataset = pd.read_csv(datasetDirectory, names = columns)
XValues = dataset[['xsection']].to_numpy()
KValues = dataset[['incompressability']].to_numpy()
f1Values = dataset[['f1']].to_numpy()  
v2Values = dataset[['v2']].to_numpy()

#find and import DNN grid data 
gridColumns = ['xsection', 'incompressability', 'f1_pred', 'v2_pred']
dnnGridDataDir = fr"C:\Users\nicho\Downloads\Research\Dr. Li\Symbolic Regression\DNNTimeTest1million\E{BeamE}\try{trynumber}\gridPrediction{BeamE}try{trynumber}.dat"
dnnGrid = pd.read_csv(dnnGridDataDir, names = gridColumns, sep = r'\s+', skiprows = 1)
XGridValues = dnnGrid[['xsection']].to_numpy()
KGridValues = dnnGrid[['incompressability']].to_numpy()
f1GridValues = dnnGrid[['f1_pred']].to_numpy()  
v2GridValues = dnnGrid[['v2_pred']].to_numpy()

#Reshape the DNN grid values so it works
XGrid = XGridValues.reshape(N, N)
KGrid = KGridValues.reshape(N, N)
F1Grid = f1GridValues.reshape(N, N)
V2Grid = v2GridValues.reshape(N, N)

#Import DNN predictions for scatter plot
dnnPredColumns = ['xsection', 'incompressability', 'f1_pred', 'v2_pred', 'f1_true', 'v2_true']
dnnPredsDir = fr"C:\Users\nicho\Downloads\Research\Dr. Li\Symbolic Regression\DNNTimeTest1million\E{BeamE}\try{trynumber}\modelPrediction{BeamE}try{trynumber}.dat"
dnnPred = pd.read_csv(dnnPredsDir, names = dnnPredColumns, sep = r'\s+', skiprows = 1)
Zpred = dnnPred[['f1_pred']].to_numpy()


#This is the code for generating the symbolic regression 3D plots
'''
#set up grid
x = np.linspace(0,1,100)
y = np.linspace(-0.25,0.05,100)
X,Y = np.meshgrid(x,y)
'''
'''
#Use equation to generate the symbolic regression surface, as well as predict points
Zequ = -7464.2363*Y - 1077.7682*(-X+Y)**2 - X/0.29975963 + 285.1789326
Zpred = -7464.2363*v2Values - 1077.7682*(-f1Values+v2Values)**2 - f1Values/0.29975963 + 285.1789326
'''

#Calculate error (divided by standard deviation)
Zerr = (Zpred-f1Values)/0.019
Zerr = Zerr.ravel()

#Set Z axis values
ZValues = f1Values

#make the semitransparent surface
trace_surface = go.Surface(
    z = F1Grid,
    x = XGrid,
    y = KGrid,
    colorscale=[[0, 'gray'], [1, 'gray']],
    showscale=False,
    opacity = 0.5)

#Set scatterplot values
scatter_x = XValues
scatter_y = KValues
scatter_z = f1Values

#Generate scatterplot on the same surface
trace_scatter = go.Scatter3d(
    x = scatter_x.ravel(),
    y = scatter_y.ravel(),
    z = scatter_z.ravel(),
    mode = 'markers',
    marker = dict(size = 4, 
                  color = Zerr, 
                  symbol = 'circle', 
                  colorscale='bluered',
                  cmin = -3,
                  cmax = 3,
                  colorbar=dict(title=f'{string3} Error in Experimental Standard Deviations'),
                  showscale=True)
    )

#Combine plot elements
fig = go.Figure(data = [trace_surface, trace_scatter])

#Set up axes, and title
fig.update_layout(
    title = f'{string3} DNN for {BeamE} MeV try {trynumber}',
    scene = dict(
        xaxis_title = string1,
        yaxis_title = string2,
        zaxis_title = string3,
        xaxis=dict(range=[0,2], dtick = 0.5),
        yaxis=dict(range=[150,450], dtick = 50),
        zaxis=dict(range=[0, 1], dtick=0.2)
        )
    )

#Add R2 score annotation
fig.add_annotation(
    text="R² = 0.9931",
    x=0.02,
    y=0.02,
    xref="paper",
    yref="paper",
    showarrow=False
)

#Write to file
fig.write_html(f'{string3}{BeamE}MeV.html')