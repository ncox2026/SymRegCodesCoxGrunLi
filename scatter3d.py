"""
Author: Xavier Grundler
Title: scatter3d.py
Description: Provide function to create 3d scatter plot.
"""

import matplotlib.pyplot as plt
import numpy as np

#calculate DNN error
def calcError(target,prediction,sigma = 0):
    if sigma == 0:
        return np.abs(target - prediction)
    else:
        return np.abs(target - prediction) / sigma

#create 3d scatter plot
def scatter3d(x,y,z,zM,xlab="x",ylab="y",zlab="z",
              proj=False, title = "", textstr = "", r2 = "", sigma = 0, guy = ''):

    #calculate difference between simulation and function
    c = calcError(z,zM, sigma) #used for color
    
    if sigma != 0:
        vmax = 3
    elif sigma == 0:
        if guy == "X":
            vmax = 0.1
        else:
            vmax = 50
    
    
    #create figure
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    #plot the points
    scatter = ax.scatter(x,y,z, c=c,vmin=0,
    vmax= vmax, cmap="jet", depthshade=False)

    #add colorbar to plot the error
    fig.colorbar(scatter, shrink=0.5, pad=0.15)

    #set the axis labels
    ax.set_xlabel(fr"{xlab}")
    ax.set_ylabel(fr"{ylab}")
    ax.set_zlabel(fr"{zlab}")

    #set the limits to the max and minimum values
    ax.set_xlim(np.min(x),np.max(x))
    ax.set_ylim(np.min(y),np.max(y))
    ax.set_zlim(np.min(z),np.max(z))

    #if we want to project the points onto each plane
    if proj:
        #get use the grid limits for projections
        xlim = np.ones_like(x) * ax.get_xlim3d()[0]
        ylim = np.ones_like(y) * ax.get_ylim3d()[1]
        zlim = np.ones_like(z) * ax.get_zlim3d()[0]

        #plot the projections of the data on each plane
        ax.scatter(xlim,y,z,c=c,cmap="gray",depthshade=False)
        ax.scatter(x,ylim,z,c=c,cmap="gray",depthshade=False)
        ax.scatter(x,y,zlim,c=c,cmap="gray",depthshade=False)
        
    ax.set_title(title)
    r2 = str(r2)
    r2 = fr"R2: {r2}"
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text2D(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props, wrap = True)
    ax.text2D(0.95, 0.05, r2, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props, wrap = True)


    return fig,ax

def makeGrid(x,y):
    #get the limits of the x and y values
    xb = np.min(x)
    xu = np.max(x)
    yb = np.min(y)
    yu = np.max(y)

    #set up the grid
    X = np.linspace(xb,xu,100)
    Y = np.linspace(yb,yu,100)
    X,Y = np.meshgrid(X,Y)

    return X,Y

#create 3d scatter plot with surface
def surf3d(x,y,z,zM,Z,xlab="x",ylab="y",zlab="z",proj=False, color = "gray", alpha = 0.4, title = "", textstr = "", r2 = "", sigma = 0, guy = ''):
    #make a grid
    X,Y = makeGrid(x,y)

    #get a scatter plot first
    fig,ax = scatter3d(x,y,z,zM,xlab,ylab,zlab,proj,title, textstr, r2 = r2, sigma = sigma, guy = guy)

    #plot surface, slightly transparent
    ax.plot_surface(X,Y,Z,color=color,linewidth=0,
                    alpha=alpha)

    return fig,ax

def plotErr(z,zM,xlab="Count",ylab="Error",title="", sigma = 0):
    #calculate error
    err = calcError(z,zM, sigma)

    #create a count for the x-axis
    #count = np.linspace(1,len(z),len(z))

    fig = plt.figure()
    ax = fig.add_subplot()

    ax.scatter(z,err,c="r",marker="o")

    ax.set_yscale("linear")

    ax.set_xlabel(fr"{xlab}")
    ax.set_ylabel(fr"{ylab}")

    ax.set_title(fr"{title}")

    return fig,ax

def stringgenerator(x0,x1,y,datx,datk,datF1,datv2):
    testlist = [datx,datk,datF1,datv2]
    stringlist = ["X","K","F_1","v_2"]
    
    string1,string2,string3 = "","",""
    
    for num,var in enumerate(testlist):
        if x0 == var:
            string1 = stringlist[num]
        elif x1 == var:
            string2 = stringlist[num]
        elif y == var:
            string3 = stringlist[num]
            
    return string1,string2,string3
            
