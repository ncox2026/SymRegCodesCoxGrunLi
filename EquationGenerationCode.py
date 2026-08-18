"""
Author: Xavier Grundler and Nicholas Cox
Title: run.py
Description: Run symbolic regression and make plots.
"""

import numpy as np
import symReg as sr
import scatter3d
from sympy import symbols,lambdify
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

'''
This code generates equations using normalized and unnormalized data with
different training sizes in order to see if normalization or testing data
size plays a role in accuracy of the equations.
'''

#Set Maximum Complexity and number of iterations for the model to converge
maxcomplexity = 20
nofiterations = 400

#list all energies and test sizes
#energyList = [150,250,400,600,800,1000,1200]
energyList = ['xy']
testsizeList = [0.5,0.25,0.15,0.1,0.05]

#open files for recording data
AccuracyFile = open("accuracyHADES.log","w")
AccuracyFile_Normalized = open("accuracy_normHADES.log","w")
BestFile = open("bestHADES.log","w")
BestFile_Normalized = open("best_normHADES.log","w")
            
#start the first loop that cycles through every beam energy
for i in energyList:
    #load IBUU simulation data
    #select input and output data
    data = np.loadtxt(fr"C:\Users\nicho\Downloads\Research\Dr. Li\Symbolic Regression\Traing Data\{i}train90.dat", delimiter=",")
    Xdat = data[:,4:]
    Ydat = data[:,1]
    
    #Generate string names
    string1, string2, string3 = scatter3d.stringgenerator(Xdat[0, 0], Xdat[0, 1], Ydat[0],
                                                   data[0, 0], data[0, 1], data[0, 4], data[0, 5])
    
    #Start the second loop that cycles through test size splits 
    for j in testsizeList:
        #Start third loop that cycles each beam energy and size 5 times
        for k in range(5):
            #print combination for tracking progress
            print(i,j,k)
            
            #Setup training and testing data
            xtrain, xtest, ytrain, ytest = train_test_split(Xdat, Ydat, test_size = j)
            
            #Set up one variable of input data
            X0dat = xtrain[:,0]
            X0mean = np.mean(X0dat)
            X0std = np.std(X0dat)
            normalizeX0dat = (X0dat - X0mean)/X0std
            normalizeX0test = (xtest[:,0] - X0mean)/X0std
            
            #Set up another variable of input data 
            X1dat = xtrain[:,1]
            X1mean = np.mean(X1dat)
            X1std = np.std(X1dat)
            normalizeX1dat = (X1dat - X1mean)/X1std
            normalizeX1test = (xtest[:,1] - X1mean)/X1std
            
            #Set up normalized input data
            normalizeXdattrain = np.stack((normalizeX0dat, normalizeX1dat), axis = 1)
            normalizeXdattest = np.stack((normalizeX0test, normalizeX1test), axis = 1)
            
            #Set up output data
            Ymean = np.mean(ytrain)
            Ystd = np.std(ytrain)
            normalizeYdat = (ytrain - Ymean) / Ystd
            normalizeYtest = (ytest - Ymean) / Ystd            
            
            #create the models, forced to be rational function here
            nmodel = sr.rational(size=maxcomplexity, niter=nofiterations, choice='best')
            model = sr.rational(size=maxcomplexity, niter=nofiterations, choice='best')
            
            #fit one model to the normalized data
            nmodel.fit(normalizeXdattrain,normalizeYdat)
            nmodelr2 = nmodel.score(normalizeXdattest, normalizeYtest)
            
            #fit the other model to the unnormalized data
            model.fit(xtrain, ytrain)
            modelr2 = model.score(xtest,ytest)
            
            #print models in a latex table, turned off to save time
            """
            with open(f"modelEq{i}{j}{k}.tex","w") as eq:
                print(model.latex_table(), file=eq)
                
            with open("nmodelEq{i}{j}{k}.tex", "w") as neq:
                print(nmodel.latex_table(), file=neq)
            """
            
            #get the prediction for all data points
            #zModel = model.predict(Xdat)
            
            #create a callable function for each model
            func = model.sympy()
            nfunc = nmodel.sympy()
            
            #Find the most accurate trained model with unnormalized data and save it
            acc_idx = model.equations_["loss"].idxmin()
            funca = model.sympy(index=acc_idx)
            
            #Find the most accurate trained model with normalized data and save it
            nacc_idx = nmodel.equations_["loss"].idxmin()
            nfunca = nmodel.sympy(index=nacc_idx)
            
            #make collable functions
            #x0,x1 = symbols("x0 x1")
            #func_lam = lambdify((x0,x1),func)
 
            #print r2 scores for comparison
            """
            print("Model without normalization:", func)
            print("Model without normalization R2:", modelr2)
            
            print("Model with normalization:", nfunc)
            print("Model with normalization R2 (normalized space):", nmodelr2)
            """
           
            #Get predictions from the 'best' model with normalized data and score it
            nZpred_norm = nmodel.predict(normalizeXdattest)
            nZpred = nZpred_norm * Ystd + Ymean
            sse = np.sum((nZpred - ytest)**2)
            tss = np.sum((ytest - np.mean(ytest))**2)
            nmodelr2_true = 1 - sse/tss
            
            #Get predictions from the most accurate model with normalized data and score it
            nZpred_norma = nmodel.predict(normalizeXdattest, index=nacc_idx)
            nZpreda = nZpred_norma * Ystd + Ymean
            sse_best = np.sum((nZpreda - ytest)**2)
            tss_best = np.sum((ytest - np.mean(ytest))**2)
            nmodelr2_trueb = 1 - sse_best/tss_best
            
            #Get predictions from the most accurate model with unnormalized data and score it
            zacc = model.predict(xtest, index=acc_idx)
            sse_acc = np.sum((zacc - ytest)**2)
            tss_acc = np.sum((ytest - np.mean(ytest))**2)
            modelr2_acc = 1 - sse_acc/tss_acc
            
            #Write to files based on model type
            print(f"{string3} Beam Energy|{i}|TestSize|{j}|try {k} Model: ",func , "R2:", modelr2, file = BestFile, sep = "|")
            print(f"{string3} Beam Energy|{i}|TestSize|{j}|try {k} Model: ",nfunc , "R2:", nmodelr2_true, file = BestFile_Normalized, sep = "|")
            print(f"{string3} Beam Energy|{i}|TestSize|{j}|try {k} Model: ",funca , "R2:", modelr2_acc, file = AccuracyFile, sep = "|")
            print(f"{string3} Beam Energy|{i}|TestSize|{j}|try {k} Model: ",nfunca , "R2:", nmodelr2_trueb, file = AccuracyFile_Normalized, sep = "|")
            
            #Flush files to avoid slowdown
            BestFile.flush()
            BestFile_Normalized.flush()
            AccuracyFile.flush()
            AccuracyFile_Normalized.flush()
            
#close files
BestFile.close()
BestFile_Normalized.close()
AccuracyFile.close()
AccuracyFile_Normalized.close()

   