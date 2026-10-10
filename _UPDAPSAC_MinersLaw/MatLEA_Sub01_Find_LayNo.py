# Code title: Find the layer number and relative coordinates in each number.
# Further description: This code get the Z-coordination and then simply calculate the relative Z-coordination form the
#        the top of each layer. The results of this function is a three-dimensional list in which the first element is a
#        list of Z-coordination, second element is a list of Z-coordination relative to the top of each layer, and the
#        third element is a list of layer number attribute to each Z-coordination.
#
#
# Firstly developed by 'Dr. M. Emin Kutay' (kutay@msu.edu), in Matlab.
# Converted to Python by 'Farhad Abdollahi' (abdolla4@msu.edu)
#
# Date: 06/11/2020

from numpy import cumsum, append, insert

def Modify_coordinates(ZCoord, Thick):
    """
    The input and output parameters of this function are as follows:
    :param ZCoord: a list of the Z-coordinations.
    :param Thick: a list of the layer thicknesses in the same dimension as ZCoord param. The layer thicknesses are
                sorted from the top layer to the bottom one.
    :return: the revised version of the ZCoord param.
    """

    # Modify Thick param. as an array with the first element is '0' and the cumulative height of the layers and the
    #       last element is 'inf'.
    Thick = insert(append(cumsum(Thick), float('inf')), 0, 0)
    data_no = len(ZCoord)
    ZCoord = [ZCoord.copy(), ZCoord.copy(), [0 for i in range(data_no)]]
    for i in range(data_no):
        for j in range(1,len(Thick)):
            test = ZCoord[0]
            test2 = ZCoord[0][1]
            if ZCoord[0][i] < Thick[j]:
                ZCoord[1][i] -= Thick[j-1]
                ZCoord[2][i] = j
                break


    return ZCoord
