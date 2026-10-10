# Code title: MatLEA as a Linear Elastic Analysis of pavement structure.
# Further description: This Code will solve the Burmister's equations to analyse the stress and strains in the pavement
#           structure. Full description will be added later <<<<<<<<<<<<<<
#
#
#
# Firstly developed by Dr. M. Emin Kutay (kutay@msu.edu) as a Matlab code.
# Converted to Python by Farhad Abdollahi (abdolla4@msu.edu) under supervision of Dr. Kutay.
#       Details of any imported functions are provided within the function definition.
#
#
# Date: 06/11/2020


# Importing the required function and packages.
from numpy import cumsum, append, insert, insert, pi, inf, sqrt, sort, zeros, ones, diff, delete, array, arange, linspace, \
    exp, repeat, squeeze, gradient, meshgrid
from numpy import sum as arrsum
from scipy.special import jv
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from time import time
# from MatLEA_Sub02_Plot_Contour import Plot_Contour

start_time = time()                                     # temporary parameter to evaluate the run time.
# from numba import jit, njit

# MAIN FUNCTION => MatLEA
# @njit
def MatLEA(ZCoord, RCoord, CP, CR, E, PR, h_layer, plot_type=''):
    """
    This function is developed by Dr. M. Emin Kutay (kutay@msu.edu) in Matlab environment and then converted to Python
        by Farhad Abdollahi (abdolla4@msu.edu). This function is the MAIN FUNCTION of the MatLEA.

    Code description: This code calculates the pavement responses under a circular loading at the coordination points
        defined by the user. The logic of this code is relied on the Burmister's Layered Elastic Theory (LET). Finally,
        this code will return all pavement responses at the pre-defined points in a dictionary.

    Input and output parameters of this function are as follows:
    :param ZCoord: A list of Z-coordination for calculation of pavement responses in which:
        ZCoord[0]: A matrix of different z-coordination relative to the pavement surface.
        ZCoord[1]: A matrix of different z-coordination relative to the top of containing layer.
        ZCoord[2]: A matrix of the layer's number containing each point. The toppest layer number is '1', increasing as
            going downward.
    :param RCoord: A matrix of different radial coordination to calculate the pavement responses.
    :param CP: The contact pressure of the applied circular load (mostly known as 'q' param.).
    :param CR: The contact radius of the applied circular load (mostly known as 'a' param.).
    :param E: A matrix of different layer's elastic moduli.
    :param PR: A matrix of different layer's Poisson ratios.
    :param h_layer: A matrix of different layer's height.
    :param PLTSTR:
    :return: the 'out' param. which is a dictionary of pavement analysis responses (stresses, strains, and displacements)
        at the (r,z) coordinations defined by user, pavement structure properties, and the program's runtime.
    """

    len_ZC = len(ZCoord[0])     # Number of Z-coordination.
    len_RC = len(RCoord)        # Number of R-coordination.
    n_layer = len(E)            # Number of layers. Number of unknown parameters are 4 * n_layer.

    k = besselzero(0, 50,1)
    ms_zeros = zeros(len(k)+1)
    ms_zeros[1:] = k / (CR * 3)
    # ms_zeros2 = insert(k / (CR * 3), 0, 0)
    len_ms_zeros = len(ms_zeros)

    n1 = array([20,15,10,10])   # An array of number of elements to create the integral parameter 'm'.
    nc = cumsum(n1)             # Cumulative summation of array 'n1'.

    len_ms = (arrsum(n1[0:3]) - 2) + (len_ms_zeros - 4) * (n1[-1] - 1)
    ms = zeros(len_ms)

    ms[0:nc[0]] =     linspace(ms_zeros[0], ms_zeros[1], num=n1[0])
    ms[nc[0]-1:nc[1]-1] = linspace(ms_zeros[1], ms_zeros[2], num=n1[1])
    ms[nc[1]-1:nc[2]-1] = linspace(ms_zeros[2], ms_zeros[3], num=n1[2])

    for i in range(4, len_ms_zeros):
        istart = nc[2] - 3 + (i - 4) * (n1[3] - 1)
        iend =   nc[2] - 2 + (i - 3) * (n1[3] - 1)
        ms[istart:iend] = linspace(ms_zeros[i-1], ms_zeros[i], num=n1[3])

    ms = delete(ms, 0)
    len_ms -= 1

    # start_time = time()
    # Creating the 3-dimensional M matrix.
    M = mat_calculator(ms, E, PR, h_layer)
    # print(f'mat_calculator tel - {time()-start_time}')

    # Creating the 2-dimensional Y matrix (Loading matrix).
    Y = zeros([4 * n_layer - 2, len_ms])
    Y[0] = CP * CR * jv(1, ms * CR) / (ms ** 2)

    # start_time = time()
    # Solve the linear system of equations, where [M][X] = [Y].
    X = solvelinsys3(M, Y)
    # print(f'solvelinsys3 tel - {time()-start_time}')

    # Reshape the X matrix to create the ABCD one.
    ABCD = X[:,0:-2,:].reshape([4, n_layer - 1, len_ms], order='F')
    temporary = zeros([4, n_layer, len_ms])                             # A temporary variable for adding a new column.
    temporary[:,:-1,:] = ABCD; ABCD = temporary.copy()                  # The new column is added to the ABCD matrix.
    # ABCD[0,n_layer - 1,:] = zeros([1, len_ms])                        # This row already is equal to zero.
    ABCD[1,n_layer - 1,:] = X[0,-2,:]
    # ABCD[2,n_layer - 1,:] = zeros([1, len_ms])                        # This row already is equal to zero.
    ABCD[3,n_layer - 1,:] = X[0,-1,:]

    A = squeeze(ABCD[0,:,:])
    B = squeeze(ABCD[1,:,:])
    C = squeeze(ABCD[2,:,:])
    D = squeeze(ABCD[3,:,:])

    Ai = array([A[i-1,:] for i in ZCoord[2]])
    Bi = array([B[i-1,:] for i in ZCoord[2]])
    Ci = array([C[i-1,:] for i in ZCoord[2]])
    Di = array([D[i-1,:] for i in ZCoord[2]])


    # Calculate the Henkel transforms of the stresses and strains.
    nui = array([ones([len_ms]) * PR[i-1] for i in ZCoord[2]])                  # The poisson ratio of each layer containing ZCoord.
    mxzi = (ms.reshape(-1,1) @ array(ZCoord[1]).reshape(1,-1)).transpose()      # The product of 'ms' and 'ZCoord[1]', relative Z-coordination.
    mxZ =  (ms.reshape(-1,1) @ ones([len_ZC]).reshape(1,-1)).transpose()        # The product of 'ms' and ones. Representing the subgrade.
    emxzi = exp(mxzi)

    # Henkel transform of vertical stress (Sigma_z)
    fi_sz = (Ai * mxZ * emxzi + Bi * mxZ / emxzi - Ci * (1 - 2 * nui - mxzi) * emxzi + Di * (1 - 2 * nui + mxzi) / emxzi)

    # Henkel transform of radial stress (Sigma_r)
    fi_sr1 = (Ai * mxZ * emxzi + Bi * mxZ / emxzi + Ci * (1 + 2 * nui + mxzi) * emxzi - Di * (1 + 2 * nui - mxzi) / emxzi)
    fi_sr2 = (Ai * mxZ * emxzi + Bi * mxZ / emxzi + Ci * (1 + mxzi) * emxzi - Di * (1 - mxzi) / emxzi)

    # Henkel transform of tangent stress (Sigma_t)
    fi_st1 = (2 * Ci * nui * emxzi - 2 * Di * nui / emxzi)
    fi_st2 = (Ai * mxZ * emxzi + Bi * mxZ / emxzi + Ci * (1 + mxzi) * emxzi - Di * (1 - mxzi) / emxzi)

    # Henkel transform of shear stress of tao_rz
    fi_srz = (Ai * mxZ * emxzi - Bi * mxZ / emxzi + Ci * (2 * nui + mxzi) * emxzi + Di * (2 * nui - mxzi) / emxzi)

    # Henkel transform of vertical displacement (w)
    fi_w = (-Ai * mxZ * emxzi + Bi * mxZ / emxzi + Ci * (2 - 4 * nui - mxzi) * emxzi + Di * (2 - 4 * nui + mxzi) / emxzi)

    # Henkel transform of horizontal displacement (u)
    fi_u = (Ai * mxZ * emxzi + Bi * mxZ / emxzi + Ci * (1 + mxzi) * emxzi - Di * (1 - mxzi) / emxzi)


    # Creating the response matrices.
    Sig_z = zeros([len_ZC, len_RC])                 # A response matrix of vertical stress (Sigma_z).
    Sig_r = Sig_z.copy()                            # A response matrix of radial stress (Sigma_r).
    Sig_t = Sig_z.copy()                            # A response matrix of tangential stress (Sigma_t).
    Tau_rz = Sig_z.copy()                           # A response matrix of shear stress in rz direction.

    Disp_z = Sig_z.copy()                           # A response matrix of vertical displacement (w).
    Disp_r = Sig_z.copy()                           # A response matrix of horizontal displacement (u).

    Eps_z = Sig_z.copy()                            # A response matrix of vertical strain (eps_z).
    Eps_r = Sig_z.copy()                            # A response matrix of radial strain (eps_r).
    Eps_t = Sig_z.copy()                            # A response matrix of tangential strain (eps_t).
    Eps_rz = Sig_z.copy()                           # A response matrix of shear strain in rz direction.


    # Calculation of the inverse Henkel transform.
    dmxZ = gradient(mxZ)[1]                                     # The gradient in respect to the columns.
    Ei = array([E[i-1] for i in ZCoord[2]])                     # The matrix of elastic moduli.
    nui = array([PR[i-1] for i in ZCoord[2]])                   # The matrix of Poisson ratios.
    Gi = Ei / (2 * (1 + nui))                                   # The matrix of Shear modulus.

    for i in range(len_RC):                                     # This loop will calculate all responses.
        RCoordi = RCoord[i]
        if RCoordi == 0:                                        # This is to eliminate singularity at r=0
            RCoordi = 1e-6

        J0 = jv(0, mxZ * RCoordi)                             # Calculate the first order bessel value of 'mxZ * R[j]'.
        J1 = jv(1, mxZ * RCoordi)                             # Calculate the second order bessel value of 'mxZ * R[j]'.

        Sig_z[:,i]  = -arrsum(J0 * fi_sz * dmxZ * (mxZ ** 2), axis=1)
        Sig_r[:,i]  =  arrsum(J0 * fi_sr1 * dmxZ * (mxZ ** 2), axis=1) - \
                         (1 / RCoordi) * arrsum(J1 * fi_sr2 * dmxZ * mxZ, axis=1)
        Sig_t[:,i]  =  arrsum(J0 * fi_st1 * dmxZ * (mxZ ** 2), axis=1) + \
                         (1 / RCoordi) * arrsum(J1 * fi_st2 * dmxZ * mxZ, axis=1)
        Tau_rz[:,i] =  arrsum(J1 * fi_srz * dmxZ * (mxZ ** 2), axis=1)

        Disp_z[:,i] = (1 + nui) / Ei * arrsum(J0 * fi_w * dmxZ * mxZ, axis=1)
        Disp_r[:,i] = (1 + nui) / Ei * arrsum(J1 * fi_u * dmxZ * mxZ, axis=1)

        Eps_z[:,i]  = (1 / Ei) * (Sig_z[:,i] - nui * (Sig_r[:,i] + Sig_t[:,i]))
        Eps_r[:,i]  = (1 / Ei) * (Sig_r[:,i] - nui * (Sig_z[:,i] + Sig_t[:,i]))
        Eps_t[:,i]  = (1 / Ei) * (Sig_t[:,i] - nui * (Sig_z[:,i] + Sig_r[:,i]))
        Eps_rz[:,i] = (1 / Gi) * Tau_rz[:,i]


    # Preparing data for being stored in the output.
    reshapemat = [1, len_ZC * len_RC]
    RC_vec, ZC_vec = meshgrid(RCoord, ZCoord[0])
    RC_vec = squeeze(RC_vec.reshape(reshapemat, order='F'))
    ZC_vec = squeeze(ZC_vec.reshape(reshapemat, order='F'))
    Sig_z  = squeeze(Sig_z.reshape(reshapemat, order='F'))
    Sig_r  = squeeze(Sig_r.reshape(reshapemat, order='F'))
    Sig_t  = squeeze(Sig_t.reshape(reshapemat, order='F'))
    Tau_rz = squeeze(Tau_rz.reshape(reshapemat, order='F'))
    Disp_z = squeeze(Disp_z.reshape(reshapemat, order='F'))
    Disp_r = squeeze(Disp_r.reshape(reshapemat, order='F'))
    Eps_z  = squeeze(Eps_z.reshape(reshapemat, order='F'))
    Eps_r  = squeeze(Eps_r.reshape(reshapemat, order='F'))
    Eps_t  = squeeze(Eps_t.reshape(reshapemat, order='F'))
    Eps_rz = squeeze(Eps_rz.reshape(reshapemat, order='F'))

    # Save data in an output dictionary.
    out = {'stress_z' :  Sig_z,                 # Saving the analysis results (pavement responses).
           'stress_r' :  Sig_r,
           'stress_t' :  Sig_t,
           'stress_rz':  Tau_rz,
           'strain_z' :  Eps_z,
           'strain_r' :  Eps_r,
           'strain_t' :  Eps_t,
           'strain_rz':  Eps_rz,
           'disp_z'   :  Disp_z,
           'disp_r'   :  Disp_r,

           'r'        :  RC_vec,                # Saving the 'r' and 'z' coordination matrices as a map for pavement responses.
           'z'        :  ZC_vec,
                                                # Saving the pavement structure properties.
           'thick'    :  h_layer,               # Thicknesses of different layers.
           'PR'       :  PR,                    # Poisson ratio of different layers.
           'E'        :  E,                     # Elastic moduli of different layers.
           'CP'       :  CP,                    # Loading contact pressure.
           'CR'       :  CR,                    # Loading contact radius.
           'RCoord'   :  RCoord,                # The required radial points for analysis.
           'ZCoord'   :  ZCoord,                # The required Z-coordination points for analysis (3 layer matrix).

           'runtime'  :  time() - start_time}   # Runtine of the program in second.


    # # Check for the plotting the results, and plot them if required.
    # if plot_type in ['STRAINS', 'STRESSES', 'DISPLACEMENTS']:
    #     Plot_Contour(out, plot_type)            # Call the 'Plot_Contour' function to plot the use specified response contour.

    # Return the result.
    return out








########################################## Defining Required Functions #################################################
# @njit
def besselzero(n, k, kind=1):
    """
    This function is initially developed by "Greg von Winckel" (gregvw@chtm.unm.edu)- 01/25/05. This code is then
        converted to Python by "Farhad Abdollahi" (abdolla4@msu.edu).

    Code description: This code will find the first 'k' positive zeros of the 'n'th order bessel function. As only the
        first kind of the bessel function is used in MatLEA, the first kind bessel part of the code is included here.
        This code uses the Halley's method for its calculation. This method is defined in another function just after
        this function.

    The input and output parameters of this function are as follows:
    :param n: The order of the bessel function, which is shown by 'v' in the notation of Jv(z).
    :param k: number of first positive zeros of bessel function to be found.
    :param kind: The kind of the bessel function (This parameter is suppose to be 1, because we use only the first kind
            of the bessel function in this method).
    :return: x param. which is a list of the first k positive zeros of the bessel function.
    """

    # Creating the 'x' param.
    k3 = 3 * k
    x = zeros(k3)

    for i in range(k3):

        # Initial guess of zeros.
        x0 = 1 + sqrt(2) + i * pi + n + n ** 0.4

        # Implementation of Halley's method.
        x[i] = halley(n, x0, kind)

        if x[i] == inf:
            print('Bad guess!')

    x = sort(x)
    dx = insert(diff(x), 0, 1)
    counter = 0
    while (counter < k):
        if dx[counter] < 1e-8:
            delete(dx,counter)
            delete(x,counter)
        else:
            counter += 1
    x = x[:k]
    # Returning the result.
    return x

# ___________________________________________________
# @njit
def halley(n, x0, kind=1):
    """
    This code is the follow-up function used for the previous function, named 'besselzero'. Please refer to that one for
        more details.

    The input and output parameters of this function are as follows:
    :param n: The order of the bessel function for evaluation.
    :param x0: The initial guess by which the Halley's method implemented.
    :param kind: This parameter specify the kind of the bessel function which is first kind in our application.
    :return: The new assigned 'x0' param.
    """

    # Specify the input parameters for Halley's method.
    tol = 1e-12         # Specify the Tolerance.
    maxiter = 100       # Specify the maximum number of iterations.
    err = 1             # The initial error used in the following loop.
    counter = 0         # The iteration counter used in the following loop.

    # The main code of the Halley's method.
    while (abs(err) > tol) and (counter < maxiter):
        a = jv(n,     x0)
        b = jv(n + 1, x0)

        err = 2 * a * x0 * (n * a - b * x0) / \
              (2 * (b ** 2) * (x0 ** 2) - a * b * x0 * (4 * n + 1) + (n * (n + 1) + (x0 ** 2)) * (a ** 2))
        x0 = x0 - err
        counter += 1

    if counter > maxiter - 1:
        return inf

    return x0
# ======================================================================================================================
# @njit
def mat_calculator(m, E, PR, h_layer):
    """
    This function is firstly developed by "Dr. M. Emin Kutay" (kutay@msu.edu) within a Matlab code of MatLEA. This code
        is then converted to the python code by "Farhad Abdollahi" (abdolla4@msu.edu).

    Code description: This code creates the 3-dimensional M matrix which is used in solving the Burmister's equation.
        For more details, please refer to the TRR paper entitled "MnLayer: High-performance layered elastic analysis
        program" by 'Lev Khazanovich' and 'Qiang (Chuck) Wang' which was published at 2007.

    The input and output parameters of this function are as follows:
    :param m:
    :param E:
    :param PR:
    :param h_layer:
    :return:
    """

    # Create the shape of M matrix.
    num_m = len(m)                      # Specify the number of m, as integral parameter.
    n_layer = len(PR)                   # Specify the number of layers.
    # M = zeros([4 * n_layer - 2, 4 * n_layer - 2, num_m])
    M = zeros((4 * n_layer - 2, 4 * n_layer - 2, num_m))

    # Implementing the coefficients of the first layer.
    nu = PR[0] * ones(num_m)
    M[0][0] = m;  M[0][1] =  m;  M[0][2] = -1 + 2 * nu;  M[0][3] = 1 - 2 * nu
    M[1][0] = m;  M[1][1] = -m;  M[1][2] =      2 * nu;  M[1][3] =     2 * nu

    # Implementing the coefficients of the intermediate layers.
    for i in range(1, n_layer-1):
        hi = h_layer[i-1]               # The height of the 'i-1'th layer.
        expmhi = exp(hi * m)

        nuit = ones(num_m) * PR[i-1]    # The Poisson ratio matrix of the top layer in 'i'th interface.
        nuib = ones(num_m) * PR[i]      # The Poisson ratio matrix of the bottom layer in the 'i'th interface.
        Eit = E[i-1]                    # The elastic modulus of the top layer in the 'i'th interface.
        Eib = E[i]                      # The elastic modulus of the bottom layer in the 'i'th interface.
        alphai = ((1 + nuib) * Eit) / \
                 ((1 + nuit) * Eib)     # The alpha parameter for the 'i'th interface.

        M[4*(i-1)+2][4*(i-1)+0] = m * expmhi
        M[4*(i-1)+2][4*(i-1)+1] = m / expmhi
        M[4*(i-1)+2][4*(i-1)+2] = (-1 + hi * m + 2 * nuit) * expmhi
        M[4*(i-1)+2][4*(i-1)+3] = ( 1 + hi * m - 2 * nuit) / expmhi
        M[4*(i-1)+2][4*(i-1)+4] = -m
        M[4*(i-1)+2][4*(i-1)+5] = -m
        M[4*(i-1)+2][4*(i-1)+6] =  1 - 2 * nuib
        M[4*(i-1)+2][4*(i-1)+7] = -1 + 2 * nuib

        M[4*(i-1)+3][4*(i-1)+0] =  m * expmhi
        M[4*(i-1)+3][4*(i-1)+1] = -m / expmhi
        M[4*(i-1)+3][4*(i-1)+2] = (-2 + hi * m + 4 * nuit) * expmhi
        M[4*(i-1)+3][4*(i-1)+3] = (-2 - hi * m + 4 * nuit) / expmhi
        M[4*(i-1)+3][4*(i-1)+4] = -alphai * m
        M[4*(i-1)+3][4*(i-1)+5] =  alphai * m
        M[4*(i-1)+3][4*(i-1)+6] = (2 - 4 * nuib) * alphai
        M[4*(i-1)+3][4*(i-1)+7] = (2 - 4 * nuib) * alphai

        M[4*(i-1)+4][4*(i-1)+0] =  m * expmhi
        M[4*(i-1)+4][4*(i-1)+1] = -m / expmhi
        M[4*(i-1)+4][4*(i-1)+2] = ( hi * m + 2 * nuit) * expmhi
        M[4*(i-1)+4][4*(i-1)+3] = (-hi * m + 2 * nuit) / expmhi
        M[4*(i-1)+4][4*(i-1)+4] = -m
        M[4*(i-1)+4][4*(i-1)+5] =  m
        M[4*(i-1)+4][4*(i-1)+6] = -2 * nuib
        M[4*(i-1)+4][4*(i-1)+7] = -2 * nuib

        M[4*(i-1)+5][4*(i-1)+0] = m * expmhi
        M[4*(i-1)+5][4*(i-1)+1] = m / expmhi
        M[4*(i-1)+5][4*(i-1)+2] = ( 1 + hi * m) * expmhi
        M[4*(i-1)+5][4*(i-1)+3] = (-1 + hi * m) / expmhi
        M[4*(i-1)+5][4*(i-1)+4] = -alphai * m
        M[4*(i-1)+5][4*(i-1)+5] = -alphai * m
        M[4*(i-1)+5][4*(i-1)+6] = -alphai
        M[4*(i-1)+5][4*(i-1)+7] =  alphai

    # Implementing the coefficients of the subgrade layer.
    i = n_layer - 1                         # To address the last element of the M matrix.
    hi = h_layer[i -1]                      # The height of the one layer above the subgrade.
    expmhi = exp(hi * m)

    nuit = ones(num_m) * PR[i - 1]          # The Poisson ratio matrix of the layer above the subgrade.
    nuib = ones(num_m) * PR[i]              # The Poisson ratio matrix of the subgrade.
    Eit = E[i - 1]                          # The elastic modulus of the layer above the subgrade.
    Eib = E[i]                              # The elastic modulus of the subgrade.
    alphai = ((1 + nuib) * Eit) / \
             ((1 + nuit) * Eib)             # The alpha parameter for the last interface.

    M[4*(i-1)+2][4*(i-1)+0] = m * expmhi
    M[4*(i-1)+2][4*(i-1)+1] = m / expmhi
    M[4*(i-1)+2][4*(i-1)+2] = (-1 + hi * m + 2 * nuit) * expmhi
    M[4*(i-1)+2][4*(i-1)+3] = ( 1 + hi * m - 2 * nuit) / expmhi
    M[4*(i-1)+2][4*(i-1)+4] = -m
    M[4*(i-1)+2][4*(i-1)+5] = -1 + 2 * nuib

    M[4*(i-1)+3][4*(i-1)+0] =  m * expmhi
    M[4*(i-1)+3][4*(i-1)+1] = -m / expmhi
    M[4*(i-1)+3][4*(i-1)+2] = (-2 + hi * m + 4 * nuit) * expmhi
    M[4*(i-1)+3][4*(i-1)+3] = (-2 - hi * m + 4 * nuit) / expmhi
    M[4*(i-1)+3][4*(i-1)+4] = alphai * m
    M[4*(i-1)+3][4*(i-1)+5] = alphai * (2 - 4 * nuib)

    M[4*(i-1)+4][4*(i-1)+0] =  m * expmhi
    M[4*(i-1)+4][4*(i-1)+1] = -m / expmhi
    M[4*(i-1)+4][4*(i-1)+2] = ( hi * m + 2 * nuit) * expmhi
    M[4*(i-1)+4][4*(i-1)+3] = (-hi * m + 2 * nuit) / expmhi
    M[4*(i-1)+4][4*(i-1)+4] =  m
    M[4*(i-1)+4][4*(i-1)+5] = -2 * nuib

    M[4*(i-1)+5][4*(i-1)+0] = m * expmhi
    M[4*(i-1)+5][4*(i-1)+1] = m / expmhi
    M[4*(i-1)+5][4*(i-1)+2] = ( 1 + hi * m) * expmhi
    M[4*(i-1)+5][4*(i-1)+3] = (-1 + hi * m) / expmhi
    M[4*(i-1)+5][4*(i-1)+4] = -alphai * m
    M[4*(i-1)+5][4*(i-1)+5] =  alphai

    return M                                # Return the result
# ======================================================================================================================
# @njit
def solvelinsys3(M, Y):
    """
    This function is firstly developed by Bruno Luong (brunoluong@yahoo.com) as a Matlab code accessible in the Matlab
        file exchange. This function is used in the MatLEA by 'Dr. M. Emin Kutay' (kutay@msu.edu) and is then converted
        to Python by 'Farhad Abdollahi' (abdolla4@msu.edu).

    Code description: The main purpose of this function is to solve the multiple linear equation systems using the least
        square method. The equation systems should have the same size.

    The input and output parameters of this function are as follows:
    :param M: is the grand M matrix, a 3D one, used for solution of A,B,C, and D in Burmister theory.
    :param Y: is the loading matrix.
    :return: The X matrix which contains the combination of the Ai, Bi, Ci, and Di.

    Comments from the Author (In Matlab):
        % function X = MultiSolver(M, RHS)
        %
        % PURPOSE: Multiple linear (least-square) solver (usually small) for
        %          systems having the same size
        %
        % INPUT
        %   M  : 3D array (m x n x p)
        %   RHS: 3D array (m x p x q)
        % OUTPUT
        %   X  : 3D array (n x p x q) if p>1,
        %      : 2D array (n x q) if p==1 (squeezed, and equal to standard
        %        backslash M\RHS)
        %
        % Solve p systems of linear equations:
        %   M(:,:,k) * squeeze(X(:,k,:)) = squeeze(RHS(:,k,:)) for all k=1,2,...,p
        %
        % NOTE 1 -- This function is likely used with q == 1: single RHS for each
        %           system). In this case
        %       M  : 3D array (m x n x p)
        %       RHS: 2D array (m x p)
        %       X  : 2D array (n x p)
        %       M(:,:,k) * X(:,k) = RHS(:,k) for all k=1,2,...,p
        %
        % NOTE 2 -- Special call with squeezed RHS: common RHS for all systems.
        %   Input argument RHS can be squeezed as (m x q), or leave as 3D with
        %   singleton in the second dimension: (m x 1 x q).
        %       M   is 3D array (m x n x p)
        %       RHS is 2D array (m x q) or (m x 1 x q)
        %       X   is 2D array (n x p x q)
        %       M(:,:,k) * squeeze(X(:,k,:)) = squeeze(RHS) for all k=1,2,...,p
        %
        % NOTE  3 -- Underdetermined system (more unknowns than equation)
        %   The solution is basic solution obtained with sparse mldivide
        %   which is not the same as basic solution when calling for full matrix.
        %
        % See also: SliceMultiSolver, MultiProd
        %
        % Author: Bruno Luong <brunoluong@yahoo.com>
        % History: original 26-May-2009
        %          11-Auguts-2009, add help
        % Acknowlegement: Tim Davis for the tip of using sparse to speedup
        %                 from FOR loop method
    """
    m, n, p = M.shape                   # The size of the M matrix.

    # Check for the compatibility of the size of Matrices.
    if Y.shape[0] != m:
        print('M and Y matrices are not compatible in their first dimension.')
        return
    if Y.shape[1] != p:
        if Y.ndim == 2:
            Y = Y.reshape((m, 1, -1))
        elif Y.shape[1] != 1:
            print('M and Y matrices are not compatible in their second dimension.')
            return
        Y = repeat(Y, p, axis=1)                # Y is now a [m x p x q] matrix.

    Y = Y.transpose().reshape((m * p, -1))      # Y is now a [m*p x q] matrix.
    q = Y.shape[1]                              # Finding the 'q' parameter.


    # Building the sparse matrix to increase the computation speed.
    row = repeat(arange(m*p).reshape(-1,m) , n, axis=0).reshape(-1,1)       # row is a row coordination matrix.
    col = repeat(arange(n*p).reshape(-1,1), m, axis=1).reshape(-1,1)        # col is a column coordination matrix.
    M = M.transpose().reshape(m*n*p, -1)                                    # This variable is the 'M(:)' variable in Matlab.
    M = coo_matrix((M[:,0], (row[:,0], col[:,0])))
    X = spsolve(M, Y).reshape([p,n,q]).transpose()

    # Squeeze for single system.
    if p == 1:              # This code maybe need revision as: use ln 381: X = spsolve(M, Y).reshape([n,q]).transpose()
        X = X.reshape([n, q])

    return X
# ======================================================================================================================