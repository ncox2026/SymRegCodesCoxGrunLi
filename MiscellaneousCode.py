# -*- coding: utf-8 -*-
"""
Created on Wed May 20 13:23:53 2026

@author: nicho
"""

import pandas as pd
import numpy as np
from sympy import symbols,lambdify, sympify
import scatter3d as s3d
import os
import matplotlib.pyplot as plt

'''
The first code takes the list of equations made by the symbolic regression and
converts it into a latex table
'''

#Set up energy list and test size list and column names
energyList = [150,250,400,600,800,1000,1200]
testsizeList = [0.5,0.25,0.15,0.1,0.05]
#R2File = open("r2file.log","w")
columnNames = ["Type", "Beam Energy", "TestSizeLabel", "TestSize", "Trylabel", "SympyModel", "R2Label", "R2"]

#Import datasets
r"""
columnNames = ["Type", "Beam Energy", "TestSizeLabel", "TestSize", "Trylabel", "SympyModel", "R2Label", "R2"]
dataaccuracy = pd.read_csv(fr"C:\Users\nicho\Downloads\Research\Dr. Li\Symbolic Regression\n400size20v2\accuracy.log", delimiter = "|", names = columnNames)
dataanorm = pd.read_csv(fr"C:\Users\nicho\Downloads\Research\Dr. Li\Symbolic Regression\n400size20v2\accuracy_norm.log", delimiter = "|", names = columnNames)
databest = pd.read_csv(fr"C:\Users\nicho\Downloads\Research\Dr. Li\Symbolic Regression\n400size20v2\best.log", delimiter = "|", names = columnNames)
databnorm = pd.read_csv(fr"C:\Users\nicho\Downloads\Research\Dr. Li\Symbolic Regression\n400size20v2\best_norm.log", delimiter = "|", names = columnNames)

#Import r2 scores for each type of symbolic regression
r2arraya = dataaccuracy["R2"].to_numpy()
r2arrayan = dataanorm["R2"].to_numpy()
r2arrayb = databest["R2"].to_numpy()
r2arraybn = databnorm["R2"].to_numpy()

m = 0

#Average the 5 r2 scores of each type and print it into a file
for i in energyList:
    print(i)
    for j in testsizeList:
        r2a = 0
        r2an = 0
        r2b = 0
        r2bn = 0
        
        for k in range(5):
            r2a += r2arraya[m]
            r2an += r2arrayan[m]
            r2b += r2arrayb[m]
            r2bn += r2arraybn[m]
            m += 1
            
        r2a = r2a/5
        r2an = r2an/5
        r2b = r2b/5
        r2bn = r2bn/5
        
        print(f"Beam Energy {i} | TestSize {j} | Avg Accuracy R2:",r2a, "| Avg Normalized Accuracy R2:",r2an, "| Average 'Best' R2:", r2b,"| Average Normalized 'Best' R2:", r2bn , file = R2File)
        R2File.flush()

R2File.close()
"""

#convert the file created above into a latex table
r"""
columnR2Names = ["BeamEnergyLabel", "Beam Energy", "TestSizeLabel", "TestSize", "AccLabel", "AccR2", "AccNLabel", "AccNR2", "BestLabel","BestR2", "BestNLabel", "BestNR2"]
R2Big = pd.read_csv(fr'C:\Users\nicho\Downloads\Research\Dr. Li\Symbolic Regression\n400size20v2\r2file.log', delimiter = "|", names = columnR2Names)

R2table = R2Big[["Beam Energy", "TestSize", "AccR2", "AccNR2", "BestR2", "BestNR2"]].copy()
R2tableLatex = R2table.to_latex('R2table.tex', index=False)
"""

###################################################################################

'''
This code generates surfaces and error plots of the symbolic regression generated
equations
'''
#import data
dataaccuracy = pd.read_csv(fr"C:\Users\nicho\Downloads\Research\Dr. Li\Symbolic Regression\n400size20K\accuracyHADES.log", delimiter = "|", names = columnNames)
sympymodels = dataaccuracy["SympyModel"]

#Import the equation and find all of its stats
for i in range(len(sympymodels)):
    test = sympymodels[i]
    x0,x1 = symbols("x0 x1")
    func = sympify(test, evaluate=False)
    func_lam = lambdify((x0,x1),func)
    BeamE = dataaccuracy["Beam Energy"][i]
    TestS = dataaccuracy["TestSize"][i]
    TryL = dataaccuracy["Trylabel"][i]
    R22 = dataaccuracy["R2"][i]
    
    #Determine which experimental data is applicable
    """    
    if BeamE <= 200:
        #150MeV FOPI data
        exp_data = (0.279, -0.005)
        exp_sigma = (0.012, 0.005)
    elif ((BeamE > 200) and (BeamE <= 325)):
        #250MeV FOPI data
        exp_data = (0.373,-0.055)
        exp_sigma = (0.015,0.004)
    elif ((BeamE > 325) and (BeamE <= 500)):
        #400MeV FOPI data
        exp_data = (0.453, -.071)
        exp_sigma = (0.016, 0.005)
    elif ((BeamE > 500) and (BeamE <= 700)):
        #600MeV FOPI data
        exp_data = (0.498, -0.074)
        exp_sigma = (0.016, 0.004)
    elif ((BeamE > 700) and (BeamE <= 900)):
        #800MeV FOPI data
        exp_data= (0.516, -0.072)
        exp_sigma = (0.019, 0.005)
    elif ((BeamE > 900) and (BeamE <= 1100)):
        #1000MeV FOPI data
        exp_data = (0.518, -0.066)
        exp_sigma = (0.017, 0.005)
    elif ((BeamE > 1100) and (BeamE <= 1215)):
        #1200MeV FOPI data
        exp_data = (0.54, -0.058)
        exp_sigma = (0.019, 0.005)
        """
    exp_data = (0.458, -0.06)
    exp_sigma = (0.03, 0.01)
    
    #import the data for the corresponding beam energy
    columns = ['xsection','incompressability','f1n','v2n','f1','v2']
    datasetDirectory = fr"C:\Users\nicho\Downloads\Research\Dr. Li\DNN+Bayes\Traing data\{BeamE}train90.dat"
    dataset = pd.read_csv(datasetDirectory, names = columns)
    dataset = dataset.to_numpy()
    
    #set input and output values
    Xdat = dataset[:,4:]
    Ydat = dataset[:,1]
    
    #Set up strings for later use and graphing
    string1, string2, string3 = s3d.stringgenerator(Xdat[0, 0], Xdat[0, 1], Ydat[0],
                                                          dataset[0, 0], dataset[0, 1], dataset[0, 4], dataset[0, 5])
    if string3 == 'F_1':
        sigma = exp_sigma[0]
    elif string3 == 'v_2':
        sigma = exp_sigma[1]
    else: 
        sigma = 0
    
    #Make grid and compute grid values with the given model
    X,Y = s3d.makeGrid(Xdat[:,0],Xdat[:,1])
    Z = func_lam(X,Y)
    
    #predict points from the input data
    zModel = func_lam(Xdat[:,0],Xdat[:,1])
    
    #Make graph titles
    title = fr"{string3}BeamEnergy{BeamE}TestSize{TestS}{TryL}3D"
    title2 = fr"{string3}BeamEnergy{BeamE}TestSize{TestS}{TryL}Error"
    
    #Make filenames for graphs
    filename = fr"{string3}BeamEnergy{BeamE}TestSize{TestS}{TryL}3D.pdf"
    filename2 = fr"{string3}BeamEnergy{BeamE}TestSize{TestS}{TryL}Error.pdf"
    
    #Ensure no bad characters are in the filename
    for bad in r'\/:*?"<>| ':
        filename = filename.replace(bad, "")
        filename2 = filename2.replace(bad, "")
    
    #Set up figure and axes parameters for surface, save figure
    fig,ax = s3d.surf3d(Xdat[:,0],Xdat[:,1],Ydat,zModel,
                              Z,string1,string2,string3,proj=False,title = title, textstr = test, r2 = R22, sigma = sigma, guy = string3)
    fig.savefig(filename, format="pdf")
    plt.close(fig)
    
    #Set up figure and axes parameters for error plot, save figure
    fig,ax = s3d.plotErr(Ydat,zModel,xlab=string3,title = title2,sigma = sigma)
    fig.savefig(filename2, format="pdf")
    plt.close(fig)

####################################################################################

'''
This code plots the 3D surfaces from our linear regression from our previous paper
for comparison with surfaces generated by symbolic regression
'''
r"""  
#Define symbols and functions 
x0,x1 = symbols("x0 x1")   
functionF1 = -0.0251 + 0.206*x0 + 0.000663*x1
functionv2 = 0.0519 - 0.0400*x0 - 0.000158*x1
functionX = -0.130 + 2.92*x0 - 1.34*x1
functionK = 403 - 881*x0 - 6400*x1

#Import data
datasetDirectory = fr"C:\Users\nicho\Downloads\Research\Dr. Li\DNN+Bayes\Traing data\E1200train120.dat"
columns = ['xsection','incompressability','f1n','v2n','f1','v2']
dataset = pd.read_csv(datasetDirectory, names = columns)
dataset = dataset.to_numpy()

#Select input and output data
Xdat = dataset[:,4:]
Ydat = dataset[:,1]

#1200MeV FOPI data
exp_data = (0.54, -0.058)
exp_sigma = (0.019, 0.005)

#Generate appropriate strings for later use
string1, string2, string3 = s3d.stringgenerator(Xdat[0, 0], Xdat[0, 1], Ydat[0],     
                                                     dataset[0, 0], dataset[0, 1], dataset[0, 4], dataset[0, 5])

#Choose appropriate function for the data
if string3 == 'F_1':
    function = functionF1
    sigma = exp_sigma[0]
elif string3 == 'v_2':
    function = functionv2
    sigma = exp_sigma[1]
elif string3 == 'X': 
    function = functionX
    sigma = 0
else:
    function = functionK
    sigma = 0

#Make function callable
func = sympify(function, evaluate=False)
func_lam = lambdify((x0,x1),func)

#Create grid and calculate grid values
X,Y = s3d.makeGrid(Xdat[:,0],Xdat[:,1])
Z = func_lam(X,Y)
 
#Calculate model values of input data   
zModel = func_lam(Xdat[:,0],Xdat[:,1])

#Create titles and filenames
filename = "functionK3D.pdf"
filename2 = "functionKError.pdf"
title = "PRC K Equation for HADES data 3D"
title2 = "PRC K Equation for HADES data Error"

#Calculate r2 scores
npredictsse = np.sum( np.square( zModel - Ydat))
npredicttss = np.sum( np.square( Ydat - Ydat.mean()))
npredictr2_score = 1-npredictsse/npredicttss
print("SymReg _R2:", npredictr2_score)
R22 = npredictr2_score

#Set figure and axes parameters for surface plot and save figure
fig,ax = s3d.surf3d(Xdat[:,0],Xdat[:,1],Ydat,zModel,
                          Z,string1,string2,string3,proj=False,title = title, textstr = function, r2 = R22, sigma = sigma)
fig.savefig(filename, format="pdf")
plt.close(fig)

Set figure and axes parameters for error plot and save figure
fig,ax = s3d.plotErr(Ydat,zModel,xlab=string3,title = title2,sigma = sigma)
fig.savefig(filename2, format="pdf")
plt.close(fig)
"""







