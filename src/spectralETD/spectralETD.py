import numpy as np
import torch

class spectralETD():
    """
    A class for spectral Exponential Time Differencing (ETD) methods for Phase-Field Model Equation in three dimensions in a GPU-accelerated framework.
    """
    def __init__(self,ndim=2,method = 'ETD',device='cuda'):    
        """
        Initialize the spectralETD class.
        
        **Parameters:**

        - `ndim` (int): Number of dimensions (1, 2, or 3).

        - `method` (str): Time integration method ('IMEX', 'IF', or 'ETD').

        - `device` (str): Device to run the computations on ('cpu' or 'cuda').
        """
        # Set the number of dimensions
        if ndim not in [1, 2, 3]:
            raise ValueError("ndim must be 1, 2, or 3.")
        self.ndim = ndim
        print('Using dimensions:', ndim)
        # Set the method for time integration
        if method not in ['IMEX', 'IF', 'ETD']:
            raise ValueError("method must be 'IMEX', 'IF', or 'ETD'.")
        self.method = method
        print('Using method:', method)
        # Set the device for computations
        if device not in ['cpu', 'cuda']:
            raise ValueError("Device must be 'cpu' or 'cuda'.")
        # Use "cuda" if you have GPU from Nvidia        
        elif device == 'cuda' and not torch.cuda.is_available():
            print('CUDA is not available, switching to CPU.')
            device = 'cpu'
        elif device == 'cpu' and torch.cuda.is_available():
            print('CUDA is available, but using CPU as requested.')
        
        else:
            self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print('Using device:', device)


    # Method to Set the Box of Geometry by user
    def Set_Geometry(self,lengths,gridsize=None):
        """
        Set the geometry of the simulation box.

        **Parameters:**

        - `lengths` (tuple): Lengths of the box in each dimension.
        - `gridsize` (tuple): Number of grid points in each dimension. If None, defaults to 128.
        """
        # Test if lengths is a tuple or list
        if not isinstance(lengths, (tuple, list)):
            # use the single float as the length of the box
            lengths = (lengths,) * self.ndim
        # Test if gridsize is a tuple or list
        if not isinstance(gridsize, (tuple, list)):
            # use the single float as the gridsize of the box
            gridsize = (gridsize,) * self.ndim if gridsize is not None else (128,) * self.ndim
        
        self.L = tuple(lengths)
        self.Ngrid = tuple(gridsize)
        self.dx = tuple([self.L[i]/self.Ngrid[i] for i in range(self.ndim)]) # grid spacing

        # Define the Geometry 
        if self.ndim == 1:
            self.x = np.arange(0.0,self.L[0],self.dx[0]) 
            self.X = self.x.copy()
            self.dV = self.dx[0]
            self.Vol = self.L[0]

            kx = torch.fft.rfftfreq(self.Ngrid[0], d=self.dx[0])*2*np.pi
            self.Kx = kx.clone().to(self.device)
            self.K2 = self.Kx**2

            # Dealising matrix
            self.dealias = (torch.abs(self.Kx) < kx.max()*2/3 ).to(self.device)
            del kx

            self.fft = lambda x: torch.fft.rfft(x)
            self.ifft = lambda x: torch.fft.irfft(x)

            # The auxiliary variable to FFT
            self.n_k = torch.empty(self.Ngrid[0]//2+1, dtype=torch.complex64,device=self.device)
            self.n = torch.empty(self.Ngrid[0], dtype=torch.float32,device=self.device)

        elif self.ndim == 2:
            self.x = np.arange(0.0,self.L[0],self.dx[0])
            self.y = np.arange(0.0,self.L[1],self.dx[1])
            self.X,self.Y = np.meshgrid(self.x,self.y,indexing ='ij')
            self.dV = self.dx[0]*self.dx[1]
            self.Vol = self.L[0]*self.L[1]

            kx = torch.fft.fftfreq(self.Ngrid[0], d=self.dx[0])*2*np.pi
            ky = torch.fft.rfftfreq(self.Ngrid[1], d=self.dx[1])*2*np.pi
            self.Kx,self.Ky = torch.meshgrid(kx,ky,indexing ='ij')
            self.Kx = self.Kx.to(self.device)
            self.Ky = self.Ky.to(self.device)
            self.K = torch.stack((self.Kx,self.Ky)).to(self.device)
            self.K2 = torch.sum(self.K*self.K, dim=0)
            
            # Dealising matrix
            self.dealias = ((torch.abs(self.Kx) < kx.max()*2/3)*(torch.abs(self.Ky) < ky.max()*2/3)).to(self.device)
            del kx, ky

            self.fft = lambda x: torch.fft.rfft2(x)
            self.ifft = lambda x: torch.fft.irfft2(x)

            # preallocate complex spectra for rFFT output shape
            out_shape = (self.Ngrid[0], self.Ngrid[1]//2 + 1)

            # The auxiliary variable to FFT
            self.n_k = torch.empty(out_shape, dtype=torch.complex64,device=self.device)
            self.n = torch.empty(self.Ngrid[0],self.Ngrid[1], dtype=torch.float32,device=self.device)

        else:
            self.x = np.arange(0.0,self.L[0],self.dx[0])
            self.y = np.arange(0.0,self.L[1],self.dx[1])
            self.z = np.arange(0.0,self.L[2],self.dx[2])
            self.X,self.Y,self.Z = np.meshgrid(self.x,self.y,self.z,indexing ='ij')
            self.dV = self.dx[0]*self.dx[1]*self.dx[2]
            self.Vol = self.L[0]*self.L[1]*self.L[2]

            kx = torch.fft.fftfreq(self.Ngrid[0], d=self.dx[0])*2*np.pi
            ky = torch.fft.fftfreq(self.Ngrid[1], d=self.dx[1])*2*np.pi
            kz = torch.fft.rfftfreq(self.Ngrid[2], d=self.dx[2])*2*np.pi
            self.Kx,self.Ky,self.Kz = torch.meshgrid(kx,ky,kz,indexing ='ij')
            self.Kx = self.Kx.to(self.device)
            self.Ky = self.Ky.to(self.device)
            self.Kz = self.Kz.to(self.device)
            self.K = torch.stack((self.Kx,self.Ky,self.Kz)).to(self.device)
            self.K2 = torch.sum(self.K*self.K, dim=0)

            # Dealising matrix
            self.dealias = ((torch.abs(self.Kx) < kx.max()*2/3)*(torch.abs(self.Ky) < ky.max()*2/3)*(torch.abs(self.Kz) < kz.max()*2/3)).to(self.device)
            del kx, ky, kz

            self.fft = lambda x: torch.fft.rfftn(x)
            self.ifft = lambda x: torch.fft.irfftn(x)

            # preallocate complex spectra for rFFT output shape
            out_shape = (self.Ngrid[0], self.Ngrid[1], self.Ngrid[2]//2 + 1)

            # The auxiliary variable to FFT
            self.n_k = torch.empty(out_shape, dtype=torch.complex64,device=self.device)
            self.n = torch.empty(self.Ngrid[0],self.Ngrid[1],self.Ngrid[2], dtype=torch.float32,device=self.device)         

    def Set_Field(self,n):
        """
        Set the initial field configuration.

        **Parameters:**

        - `n` (numpy.ndarray): Initial field configuration, must match the grid size.
        """
        self.n[:] = torch.from_numpy(n) 
        self.n_k[:] =  self.fft(self.n)

    def Set_LinearOperator(self,Loperator_k,params):
        """
        Set the linear operator in Fourier space.

        **Parameters:**

        - `Loperator_k` (callable): Function that computes the linear operator in Fourier space.

        - `params` (dict): Parameters required by the linear operator function.
        """
        if Loperator_k == None:
            self.Loperator_k = lambda n: torch.zeros_like(self.n_k)
        else:
            self.Loperator_k = Loperator_k(params)

    def Set_NonLinearOperator(self,Noperator_func,params):
        """
        Set the nonlinear operator in Fourier space.

        **Parameters:** 

        - `Noperator_func` (callable): Function that computes the nonlinear operator in Fourier space.

        - `params` (dict): Parameters required by the nonlinear operator function.
        """
        # Check if Noperator_func is None
        if Noperator_func == None:
            self.Noperator_func = lambda n: torch.zeros_like(self.n_k)
        elif not callable(Noperator_func):
            raise ValueError("Noperator_func must be a callable function.")
        else:
            self.Noperator_func = lambda n: Noperator_func(n,params)

    def Set_TimeStep(self,h=0.01):
        """
        Set the time step for the simulation.

        **Parameters:**

        `h` (float): Time step size.
        """
        if not isinstance(h, (int, float)):
            raise ValueError("h must be a numeric value.")
        if h <= 0:
            raise ValueError("h must be a positive value.")
        # Set the time step
        self.h = float(h)
        print('Using time step:', self.h)
        # Defining the time marching operators arrays
        if self.method == 'IMEX':
            self.Tlinear_k = 1.0/(1.0-self.h*self.Loperator_k) 
            self.Tnon_k = self.h*self.dealias/(1.0-self.h*self.Loperator_k) 
        elif self.method == 'IF':
            self.Tlinear_k = torch.exp(self.h*self.Loperator_k) 
            self.Tnon_k = self.h*self.dealias*self.Tlinear_k
        elif self.method == 'ETD':
            self.Tlinear_k = torch.exp(self.h*self.Loperator_k) 
            self.Tnon_k = self.h*self.dealias*torch.where(torch.abs(self.Tlinear_k)==1.0,1.0,(self.Tlinear_k-1.0)/torch.log(self.Tlinear_k))
        else: print('ERROR: Undefined Integrator')

    # time step function
    def Calculate_TimeStep(self):
        """
        Perform a single time step of the simulation.

        This function updates the field configuration using the specified time integration method.
        """
        # updating in time
        self.n_k[:] = self.n_k*self.Tlinear_k + self.Noperator_func(self.n)*self.Tnon_k 
        # IFT to next step
        self.n[:] = self.ifft(self.n_k).real 

    # Dynamics integration
    # Calculate dynamics from ti to tf and make output at each 10 time steps

    def Calculate_Dynamics(self,ti,tf, print_every = 10):
        """
        Integrate the dynamics from initial time ti to final time tf.

        **Parameters:**

        - `ti` (float): Initial time.

        - `tf` (float): Final time.

        - `print_every` (int): Print progress every 'print_every' steps.

        **Returns:**

        numpy.ndarray: The field configuration at the final time.
        """
        if not isinstance(ti, (int, float)):
            raise ValueError("ti must be a numeric value.")
        if not isinstance(tf, (int, float)):
            raise ValueError("tf must be a numeric value.")
        if ti >= tf:
            raise ValueError("tf must be greater than ti.")
        if not hasattr(self, 'h'):
            raise ValueError("Time step h must be set before calling Calculate_Dynamics.")
        
        Nsteps = int((tf-ti)/self.h)

        t = [ti]
        output = [self.n.cpu().numpy().copy()] # list to store output at each print_every step
        for i in range(1,Nsteps+1):
            self.Calculate_TimeStep()
            if i % print_every == 0:
                print(f"Step {i}, Time: {ti + i*self.h:.2f}")
                output.append(self.n.cpu().numpy().copy()) 
                t.append(ti + i*self.h)
        print(f"Final Time: {tf:.2f}")

        return np.array(t), np.array(output)
