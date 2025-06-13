import numpy as np
from scipy.interpolate import interpn

def transferFunction(x,rhos):
    """
    Transfer function for volume rendering.

    **Args:**

    - `x`: numpy array of values to be mapped to colors.

    - `rhos`: list of density values where the colors change.

    **Returns:**

    - `r`, `g`, `b`, `a`: numpy arrays of red, green, blue, and alpha values.
    """

    sigmarho = np.array([0.1,0.1,0.15])*rhos[-1]

    offwhite = np.array([247,247,247])/255
    red = np.array([191,33,47])/255 
    yellow = np.array([249,167,62])/255
    blue = np.array([38,75,150])/255 

    color = [blue,yellow,red]

    r = offwhite[0]
    g = offwhite[1]
    b = offwhite[2]
    a = 0.0

    for i in range(len(rhos)):
        r += -(offwhite[0]-color[i][0])*np.exp( -0.5*(x-rhos[i])**2/sigmarho[i]**2 )
        g += -(offwhite[1]-color[i][1])*np.exp( -0.5*(x-rhos[i])**2/sigmarho[i]**2 )
        b += -(offwhite[2]-color[i][2])*np.exp( -0.5*(x-rhos[i])**2/sigmarho[i]**2 )
        a += 0.4*np.exp( -0.5*(x-rhos[i])**2/sigmarho[i]**2 )
	
    return r,g,b,a

def volumerender(L, rho, phi = -30.0, theta = 0.0, psi = -15.0, Ngrid = 256, min_value = 0.0, max_value = 1.0):
    """
    Volume rendering of a 3D data cube.
    **Args:**

    - `L`: tuple of length 3, the dimensions of the data cube (Lx, Ly, Lz).

    - `rho`: numpy array of shape (Nx, Ny, Nz), the data cube to be rendered.

    - `phi`: float, rotation around x-axis in degrees (default: -30.0).

    - `theta`: float, rotation around y-axis in degrees (default: 0.0).

    - `psi`: float, rotation around z-axis in degrees (default: -15.0).

    - `Ngrid`: int, number of grid points for rendering (default: 256).

    - `min_value`: float, minimum value for transfer function (default: 0.0).

    - `max_value`: float, maximum value for transfer function (default: 1.0).

    **Returns:**

    - `image`: numpy array of shape (Ngrid, Ngrid, 3), the rendered image.
    """

    # Dimetric view: phi=45,theta=0,psi=-15.5
    # Trimetric view: phi=62.5,theta=0,psi=-15.5
    # Isometric view: phi=45,theta=0,psi=-35.264

    Ngridarray = rho.shape

    # Datacube Grid
    x = np.linspace(-0.5*L[0], 0.5*L[0], Ngridarray[0])
    y = np.linspace(-0.5*L[1], 0.5*L[1], Ngridarray[1])
    z = np.linspace(-0.5*L[2], 0.5*L[2], Ngridarray[2])
    points = (x, y, z)

    # Camera Grid / Query Points -- rotate camera view
    phi = phi*np.pi/180 # rotation around x
    theta = theta*np.pi/180 # rotation around y
    psi = psi*np.pi/180 # rotation around z
    # rotation matrix
    Rx = np.array([[1,0,0],[0.0,np.cos(phi),np.sin(phi)],[0.0,-np.sin(phi),np.cos(phi)]])
    Ry = np.array([[np.cos(theta),0.0,-np.sin(theta)],[0,1,0],[np.sin(theta),0,np.cos(theta)]])
    Rz = np.array([[np.cos(psi),np.sin(psi),0],[-np.sin(psi),np.cos(psi),0],[0,0,1]])
    Rot = np.matmul(Ry,np.matmul(Rx,Rz))

    cx = np.linspace(-0.8*L[0], 0.8*L[0], Ngrid)
    cy = np.linspace(-0.8*L[1], 0.8*L[1], Ngrid)
    cz = np.linspace(-0.8*L[2], 0.8*L[2], Ngrid)
    qx, qy, qz = np.meshgrid(cx,cy,cz)
    qxR = Rot[0,0]*qx + Rot[0,1]*qy + Rot[0,2]*qz
    qyR = Rot[1,0]*qx + Rot[1,1]*qy + Rot[1,2]*qz
    qzR = Rot[2,0]*qx + Rot[2,1]*qy + Rot[2,2]*qz
    qi = np.array([qxR.ravel(), qyR.ravel(), qzR.ravel()]).T

    # Interpolate onto Camera Grid
    camera_grid = interpn(points, rho, qi, method='linear', bounds_error=False, fill_value=0.0).reshape((Ngrid,Ngrid,Ngrid))

    # Do Volume Rendering
    image = np.ones((camera_grid.shape[1],camera_grid.shape[2],3))

    for dataslice in camera_grid:
        r,g,b,a = transferFunction(dataslice,[min_value,0.5*(min_value+max_value),max_value])
        image[:,:,0] = a*r + (1-a)*image[:,:,0]
        image[:,:,1] = a*g + (1-a)*image[:,:,1]
        image[:,:,2] = a*b + (1-a)*image[:,:,2]

    image[:] = np.clip(image,0.0,1.0)

    return image