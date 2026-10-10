import numpy as np
from updapsutil import strcmp

def ff_sublayer_MEPDGmethod(TH_in, layerType):

    # Edum = dummy modulus variable. Not actually used in the layered elastic analysis
    # Edum is needed in loading frequency calculation from speed
    Edum = np.zeros(len(layerType))
    Edum[strcmp(layerType, 'AC')] = 511000  # AC layer
    Edum[strcmp(layerType, 'EAC')] = 711000  # Existing AC layer
    Edum[strcmp(layerType, 'CSM')] = 100000  # Chemically stabilized layer
    Edum[strcmp(layerType, 'BASE')] = 49888  # Base or Subbase
    Edum[strcmp(layerType, 'SUB')] = 10000  # Subgrade

    if TH_in[0]>=1.5:
        if np.floor(TH_in[0]) == TH_in[0]:
            TH_insub1 = np.append([0.5, 0.5], np.ones(int(TH_in[0]-1)))
            Edum_sub1 = np.append([Edum[0], Edum[0]], Edum[0] * np.ones(int(TH_in[0]-1)))
            LayNo_sub = np.ones(len(Edum_sub1))
        else:
            THint = int(np.floor(TH_in[0]))
            TH_insub1 =np.append([0.5, 0.5, (TH_in[0] - THint)],  np.ones(THint-1))
            Edum_sub1 = np.append([Edum[0], Edum[0], Edum[0]], Edum[0] * np.ones(THint-1))
            LayNo_sub = np.ones(len(Edum_sub1))

        TH_insub = TH_insub1.copy()
        Edum_sub = Edum_sub1.copy()
    else:
        TH_insub = TH_in[0]
        Edum_sub = Edum[0]
        if isinstance(Edum_sub, (list, tuple, np.ndarray)):
            LayNo_sub = np.ones(len(Edum_sub))
        else:
            LayNo_sub = 1

    # % Rest of the layers are divided into 2" intervals

    if len(TH_in)>=2:
        for j in range(1, len(TH_in)):
            # % thick but not a CSM Layer
            if TH_in[j]>2 and Edum[j] != 100000:
                if np.floor(TH_in[j]/2) == TH_in[j]/2:
                    TH_insub2 = 2*np.ones(int(TH_in[j]/2))
                    Edum_sub2 = Edum[j] * np.ones(int(TH_in[j]/2))
                    LayNo_sub2 = np.zeros(len(Edum_sub2)) + j + 1
                else:
                    THint = np.floor(TH_in[j]/2)
                    TH_insub2 = np.append([(TH_in[j] - THint*2)],  2*np.ones(int(THint)))
                    Edum_sub2 = np.append([Edum[j]],  Edum[j] * np.ones(int(THint)))
                    LayNo_sub2 = np.zeros(len(Edum_sub2)) + j + 1

                    TH_insub3 = TH_insub2
                    TH_insub3[0] = sum(TH_insub3[0:2])
                    TH_insub3 = np.delete(TH_insub3, 1)

                    Edum_sub2 = np.delete(Edum_sub2, 1)
                    LayNo_sub2 = np.delete(LayNo_sub2, 1)
                    TH_insub2 = TH_insub3.copy()

            # % thin but not a CSM Layer
            elif TH_in[j]<=2 and Edum[j] !=100000:
                TH_insub2   = TH_in[j]
                Edum_sub2   = Edum[j]
                if isinstance(Edum_sub2, (list, tuple, np.ndarray)):
                    LayNo_sub2 = np.zeros(len(Edum_sub2)) + j + 1
                else:
                    LayNo_sub2 = j + 1
                # %error(['Layer - ' num2str(j) ' thickness is less than 2"'])

            # % CSM layer - no sublayering
            elif Edum[j] == 100000:
                TH_insub2   = TH_in[j]
                Edum_sub2   = Edum[j]
                LayNo_sub2 = j + 1

            TH_insub  = np.append(TH_insub, TH_insub2)
            Edum_sub  = np.append(Edum_sub, Edum_sub2)
            LayNo_sub = np.append(LayNo_sub, LayNo_sub2)
    # % lastly add the subgrade modulus.
    Edum_sub = np.append(Edum_sub, [Edum[-1]])
    LayNo_sub = np.append(LayNo_sub, [LayNo_sub[-1]+1])

    LayNo_sub = LayNo_sub.astype(int)-1
    return TH_insub, Edum_sub, LayNo_sub

# TH_in = [4,6.5,4,12,12,12]
# Edum  = [511000,511000,511000,49888,49888,49888,10000]
# [TH_insub, Edum_sub, LayNo_sub] = ff_sublayer_MEPDGmethod(TH_in, Edum)
# print(TH_insub)
# print(TH_insub.shape)
# print(Edum_sub)
# print(Edum_sub.shape)
# print(LayNo_sub)
# print(LayNo_sub.shape)
