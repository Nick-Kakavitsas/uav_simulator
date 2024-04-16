import numpy as np
import math
import matplotlib.pyplot as plt
import pandas
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.collections import PatchCollection

def plotCoords(coords, plotFlag):
    if plotFlag == True:
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter3D(coords[:, 0], coords[:, 1], coords[:, 2])
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_zlabel('z (m)')
        ax.set_title('Desired xyz position setpoints')
        plt.show()

def plot1x3new(allData, ylabs, figName, yLims='skip'):
    '''
    Input a tuple of data ('allData') contianing the following:
        * data set 1
            - time vector corresponding to data
            - x data 1
            - y data 1
            - z data 1
            - legend entry
        * data set 2
            - time vector corresponding to data
            - x data 2
            - y data 2
            - z data 2
            - legend entry
        * etc

    ylabs = tuple with ylabel1, ylabel2, ylabel3

    yLims = tuple with low and high value

    Can also input additional data and labels in t2/3, x2/3, y2/3, z2/3
    '''

    # Setup the figure with 3 subplots
    fig, axes = plt.subplots(3, 1)

    # Loop through data and plot each set on each axis
    for i, dataSet in enumerate(allData):
        if not dataSet == 'skip':
            # Unpack data for this set
            t, x, y, z, legend = dataSet

            # Set the x limits based on the first set of data
            if i == 0:
                # Set xlims
                x0 = min(t) - max(t) / 100
                xf = max(t)

            # Plot the data
            if legend == 'skip':
                axes[0].plot(t, x)
                axes[1].plot(t, y)
                axes[2].plot(t, z)
            else:
                axes[0].plot(t, x)
                axes[1].plot(t, y)
                axes[2].plot(t, z, label=legend)

    # Set plot characteristics
    for i, yLabel in enumerate(ylabs):
        axes[i].set_ylabel(yLabel)
        axes[i].set_xlabel('t (s)')
        axes[i].set_xlim(x0, xf)
        if not yLims == 'skip':
            y0, yf = yLims
            axes[i].set_ylim(y0, yf)
        axes[i].grid()
    # Show the legend on the third plot
    axes[2].legend()

    # Set tight layout
    fig.tight_layout()

    # Save plot
    fig.savefig(figName)

def rotMat(ang, radius, alt):
    R = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]])
    xoriginal = radius
    yoriginal = 0
    newCoords = R@np.array([[xoriginal],[yoriginal]])
    newCoordsXYZ = np.vstack([newCoords, alt])
    return newCoordsXYZ.flatten()

def circleCoords(n, desAlt, desRadius, plotFlag):
    wp_ini = np.array([0, 0, 0])
    wp_up = np.array([0, 0, desAlt])
    angles = np.array(np.linspace(0,2*math.pi,n))
    coords = wp_up.copy()
    for i in range(len(angles)):
        newCoords = rotMat(angles[i],desRadius, desAlt)
        coords = np.vstack([coords, newCoords])
    coords = np.vstack([coords, wp_up, wp_ini])

    # 3d plot
    plotCoords(coords, plotFlag)

    # Return the desired coordinates
    return(coords)

def rrtCoords(desAlt, plotFlag, IC):
    # Get the RRT data
    csvFile = np.array(pandas.read_csv("RRTData/rrtTrack.csv",header=None))
    newRow = np.array([desAlt] * np.max(csvFile.shape)).reshape(1,-1)
    newVec = np.vstack([csvFile,newRow]).T

    # Set the initial and final waypoints
    x0, y0, z0 = newVec[0,:]
    xf, yf, zf = newVec[-1,:]
    wp0 = np.array([x0, y0, 0])
    wpUp0 = np.array([x0, y0, desAlt])
    wpF = np.array([xf, yf, 0])
    wpUpF = np.array([xf, yf, desAlt])
    ICUp = np.array([IC[0], IC[1], desAlt])

    # Add on initial point and final point
    coords = np.vstack([IC, ICUp, wpUp0, newVec, wpUpF, ICUp, IC])
    # coords = np.vstack([IC, wp0, wpUp0, newVec, wpUpF, wpF, IC])

    # 3d plot
    plotCoords(coords, plotFlag)

    # Return the desired coordinates
    return (coords)

def lawnmowerCoords(desAlt):
    csvFile = np.array(pandas.read_csv("RRTData/lawnmowerTrack.csv", header=None))
    newRow = np.array([desAlt] * np.max(csvFile.shape)).reshape(1, -1)
    lawnmowerData = np.vstack([csvFile, newRow]).T
    return lawnmowerData

def plot_polygon(ax, poly, **kwargs):
    # From https://stackoverflow.com/questions/55522395/how-do-i-plot-shapely-polygons-and-objects-using-matplotlib#:~:text=24-,Geometries,-can%20be%20Point
    path = Path.make_compound_path(
        Path(np.asarray(poly.exterior.coords)[:, :2]),
        *[Path(np.asarray(ring.coords)[:, :2]) for ring in poly.interiors])

    patch = PathPatch(path, **kwargs)
    collection = PatchCollection([patch], **kwargs)

    ax.add_collection(collection, autolim=True)
    ax.autoscale_view()
    return collection

def b1Calc(R,e1):
    # Based on https://github.com/fdcl-gwu/uav_simulator/blob/b91cd04fe41759f6e1cd782d049c04b8df14c797/scripts/trajectory.py#L166
    b1 = R.dot(e1)
    theta = np.arctan2(b1[1], b1[0])
    return np.array([np.cos(theta), np.sin(theta), 0.0])