# Code title: Plot the pavement response contours as user specified.
# Further description: This code get the Z-coordination and then simply calculate the relative Z-coordination form the
#        the top of each layer. The results of this function is a three-dimensional list in which the first element is a
#        list of Z-coordination, second element is a list of Z-coordination relative to the top of each layer, and the
#        third element is a list of layer number attribute to each Z-coordination.
#
#
# Firstly developed by 'Dr. M. Emin Kutay' (kutay@msu.edu), in Matlab. However, the code is completely revised to be
#   converted to Python by 'Farhad Abdollahi' (abdolla4@msu.edu).
#
# Date: 06/14/2020

# Importing the required libraries and packages.
import matplotlib.pyplot as plt
from matplotlib import ticker
from scipy.interpolate import griddata
from numpy import mgrid, amax, array, cumsum


def Plot_Contour(data, type):
    """
    The input and output parameters of this function are as follows:
    :param data: The results of running the MatLEA program are stored in the 'data' dictionary. These results are used
        for plotting the results.
    :param type: The type of the contours which could be 'STRAINS', 'STRESSES', 'DISPLACEMENTS'.
    :return: This function will return nothing except plotting the required contours.
    """

    # Reading the coordination points in which pavement responses are calculated.
    Coord = array([data['r'], data['z']]).transpose()                   # A coordination matrix in which the first column
                                                                        #    1st column: R-coordination.
                                                                        #    2nd column: Z-coordination.
    RCmax = amax(data['RCoord'])                                        # The maximum R-coordination entered by user.
    ZCmax = cumsum(data['thick'])[-1]                                   # The maximum Z-coordination is the height of last layer.
    # Creating the R and Z-coordinate grid data points for using in linear interpolation (150x150 grid).
    RC_grid, ZC_grid = mgrid[0:RCmax:150j, 0:ZCmax:150j]


#   # The coordination points in which pavement responses are calculated.
#   reshapemat = [len(data['ZCoord'][0]), len(data['RCoord'])]          # The shape of the reshaped matrices of RC and ZC.
#   RC_vec = data['r'].reshape(reshapemat, order='F')                   # The reshaped R-coordination matrix.
#   ZC_vec = data['z'].reshape(reshapemat, order='F')                   # The reshaped Z-coordination matrix.
#   plt.pcolor(RC_vec, ZC_vec, Sigz)
#   plt.imshow(Sigz, origin='lower', interpolation='bilinear')
#   plt.show()

    # Defining the plot setting.
    Colormap = plt.get_cmap('plasma')                   # Setting the color map. More details at:
                                                        #   https://matplotlib.org/3.2.1/tutorials/colors/colormaps.html
                                                        #   Some of the most frequently used color maps:
                                                        #       'viridis', 'plasma', 'inferno', 'magma', 'cividis'
                                                        #       'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
                                                        #       'hot', 'cool', 'summer', 'brg'



    # Check for the type of the response.
    if type == 'STRAINS':
        # Read the strain data from the input dictionary and linearly interpolate the response data points to the
        #   well-organized meshgrid (Preparing for plot).
        Eps_z  = griddata(Coord, data['strain_z'],  (RC_grid, ZC_grid), method='linear')
        Eps_r  = griddata(Coord, data['strain_r'],  (RC_grid, ZC_grid), method='linear')
        Eps_t  = griddata(Coord, data['strain_t'],  (RC_grid, ZC_grid), method='linear')
        Eps_rz = griddata(Coord, data['strain_rz'], (RC_grid, ZC_grid), method='linear')
        Responses = [[Eps_z, Eps_r], [Eps_t, Eps_rz]]               # A variable for storing responses as a matrix.
        Responses_titles = [['MatLEA Vertical strain, (-):Compression',  'MatLEA Radial strain, (-):Compression'],
                            ['MatLEA Tangential strain, (-):Compression', 'MatLEA Shear strain (rz), (-):Compression']]

        # Creating a figure in which a 2x2 subplot are prepared for displaying the strains.
        fig, axes = plt.subplots(2,2)

        # Plotting each strain response in the following loop (details are provided in comments).
        for i, j in [[indx1, indx2] for indx1 in range(2) for indx2 in range(2)]:
            # Create the response contours.
            draw = axes[i,j].imshow(Responses[i][j].transpose(),  extent=(0,RCmax,ZCmax,0), cmap=Colormap, aspect='auto')

            # Draw the loading and write its properties on the axes.
            axes[i,j].plot([0, data['CR']], [-1, -1], color = 'r', lw = 5)
            axes[i,j].text(0.05 * RCmax + data['CR'], -2.5, "CP = %.2f psi & CR = %.2f in" % (data['CP'], data['CR']), va = 'center')

            # Draw the layer interfaces and write the layer properties.
            temporary_height = 0                        # A temporary variable for storing the height of previous layer.
            lay_heights = data['thick']                 # The list of layer heights.
            cum_lay_height = cumsum(lay_heights)        # The cumulative list of layer heights.
            lay_E = data['E']                           # The list of layer's elastic moduli.
            lay_PR = data['PR']                         # The list of layer's Poisson ratios.
            for k in range(len(lay_heights)):           # Drawing the layers interface.
                axes[i,j].plot([0, RCmax], [cum_lay_height[k], cum_lay_height[k]], color = 'k')
                information = "h = %d in, E = %d psi, PR = %.2f" % (lay_heights[k], lay_E[k], lay_PR[k])
                axes[i,j].text(0.1 * RCmax, 0.5 * (cum_lay_height[k] + temporary_height), information, va = 'center')
                temporary_height = cum_lay_height[k]    # Updating the temporary variable.
            # For the last layer as a Rock bed.
            information = "h = inf, E = %d psi, PR = %.2f" % (lay_E[-1], lay_PR[-1])
            axes[i,j].text(0.1 * RCmax, 0.5 * (ZCmax + 11 + temporary_height), information, va = 'center')

            # Set the Axis ticks, labels and color bar.
            axes[i,j].set_yticks([m for m in range(-5, int(ZCmax+11), 5)])      # Set the number of 'Y' ticks.
            axes[i,j].set_ylabel('Z (in)'); axes[i,j].set_xlabel('R (in)')      # Set the 'X' and 'Y' labels.
            axes[i,j].set_title(Responses_titles[i][j])                         # Set the title of the axes.
            colorbar = fig.colorbar(draw, ax=axes[i,j], format = '%1.1e')       # Create a color bar legend.
            colorbar.locator = ticker.MaxNLocator(nbins=8)                      # Set the number of labels on the color bar legend.
            colorbar.update_ticks()                                             # Update the y-ticks.


        # Set the figure size and adjust the subplot locations in the figure.
        fig.set_figheight(6.5)
        fig.set_figwidth(11)
        fig.canvas.set_window_title('MatLEA: Strain Responses')
        plt.subplots_adjust(0.05, 0.1, 0.95, 0.95, 0.15, 0.32)


    elif type == 'STRESSES':
        # Read the stress data from the input dictionary and linearly interpolate the response data points to the
        #   well-organized meshgrid (Preparing for plot).
        Sig_z  = griddata(Coord, data['stress_z'],  (RC_grid, ZC_grid), method='linear')
        Sig_r  = griddata(Coord, data['stress_r'],  (RC_grid, ZC_grid), method='linear')
        Sig_t  = griddata(Coord, data['stress_t'],  (RC_grid, ZC_grid), method='linear')
        Tau_rz = griddata(Coord, data['stress_rz'], (RC_grid, ZC_grid), method='linear')
        Responses = [[Sig_z, Sig_r], [Sig_t, Tau_rz]]               # A variable for storing responses as a matrix.
        Responses_titles = [['MatLEA Vertical stress, (-):Compression',  'MatLEA Radial stress, (-):Compression'],
                            ['MatLEA Tangential stress, (-):Compression', 'MatLEA Shear stress (rz), (-):Compression']]

        # Creating a figure in which a 2x2 subplot are prepared for displaying the stresses.
        fig, axes = plt.subplots(2,2)

        # Plotting each stress response in the following loop (details are provided in comments).
        for i, j in [[indx1, indx2] for indx1 in range(2) for indx2 in range(2)]:
            # Create the response contours.
            draw = axes[i,j].imshow(Responses[i][j].transpose(),  extent=(0,RCmax,ZCmax,0), cmap=Colormap, aspect='auto')

            # Draw the loading and write its properties on the axes.
            axes[i,j].plot([0, data['CR']], [-1, -1], color = 'r', lw = 5)
            axes[i,j].text(0.05 * RCmax + data['CR'], -2.5, "CP = %.2f psi & CR = %.2f in" % (data['CP'], data['CR']), va = 'center')

            # Draw the layer interfaces and write the layer properties.
            temporary_height = 0                        # A temporary variable for storing the height of previous layer.
            lay_heights = data['thick']                 # The list of layer heights.
            cum_lay_height = cumsum(lay_heights)        # The cumulative list of layer heights.
            lay_E = data['E']                           # The list of layer's elastic moduli.
            lay_PR = data['PR']                         # The list of layer's Poisson ratios.
            for k in range(len(lay_heights)):           # Drawing the layers interface.
                axes[i,j].plot([0, RCmax], [cum_lay_height[k], cum_lay_height[k]], color = 'k')
                information = "h = %d in, E = %d psi, PR = %.2f" % (lay_heights[k], lay_E[k], lay_PR[k])
                axes[i,j].text(0.1 * RCmax, 0.5 * (cum_lay_height[k] + temporary_height), information, va = 'center')
                temporary_height = cum_lay_height[k]    # Updating the temporary variable.
            # For the last layer as a Rock bed.
            information = "h = inf, E = %d psi, PR = %.2f" % (lay_E[-1], lay_PR[-1])
            axes[i,j].text(0.1 * RCmax, 0.5 * (ZCmax + 11 + temporary_height), information, va = 'center')

            # Set the Axis ticks, labels and color bar.
            axes[i,j].set_yticks([m for m in range(-5, int(ZCmax+11), 5)])      # Set the number of 'Y' ticks.
            axes[i,j].set_ylabel('Z (in)'); axes[i,j].set_xlabel('R (in)')      # Set the 'X' and 'Y' labels.
            axes[i,j].set_title(Responses_titles[i][j])                         # Set the title of the axes.
            colorbar = fig.colorbar(draw, ax=axes[i,j], format = '%1.1e')       # Create a color bar legend.
            colorbar.locator = ticker.MaxNLocator(nbins=8)                      # Set the number of labels on the color bar legend.
            colorbar.update_ticks()                                             # Update the y-ticks.


        # Set the figure size and adjust the subplot locations in the figure.
        fig.set_figheight(6.5)
        fig.set_figwidth(11)
        fig.canvas.set_window_title('MatLEA: Stress Responses')
        plt.subplots_adjust(0.05, 0.1, 0.95, 0.95, 0.15, 0.32)


    elif type == 'DISPLACEMENTS':
        # Read the displacement data from the input dictionary and linearly interpolate the response data points to the
        #   well-organized meshgrid (Preparing for plot).
        Disp_z  = griddata(Coord, data['disp_z'],  (RC_grid, ZC_grid), method='linear')
        Disp_r  = griddata(Coord, data['disp_r'],  (RC_grid, ZC_grid), method='linear')
        Responses = [Disp_z, Disp_r]                          # A variable for storing responses as a matrix.
        Responses_titles = ['MatLEA Vertical Displacement (in)',  'MatLEA Horizontal Displacement (in)']

        # Creating a figure in which a 1x2 subplot are prepared for displaying the displacements.
        fig, axes = plt.subplots(1,2)

        # Plotting each displacement response in the following loop (details are provided in comments).
        for i in range(2):
            # Create the response contours.
            draw = axes[i].imshow(Responses[i].transpose(),  extent=(0,RCmax,ZCmax,0), cmap=Colormap, aspect='auto')

            # Draw the loading and write its properties on the axes.
            axes[i].plot([0, data['CR']], [-1, -1], color = 'r', lw = 5)
            axes[i].text(0.05 * RCmax + data['CR'], -2.5, "CP = %.2f psi & CR = %.2f in" % (data['CP'], data['CR']), va = 'center')

            # Draw the layer interfaces and write the layer properties.
            temporary_height = 0                        # A temporary variable for storing the height of previous layer.
            lay_heights = data['thick']                 # The list of layer heights.
            cum_lay_height = cumsum(lay_heights)        # The cumulative list of layer heights.
            lay_E = data['E']                           # The list of layer's elastic moduli.
            lay_PR = data['PR']                         # The list of layer's Poisson ratios.
            for k in range(len(lay_heights)):           # Drawing the layers interface.
                axes[i].plot([0, RCmax], [cum_lay_height[k], cum_lay_height[k]], color = 'k')
                information = "h = %d in, E = %d psi, PR = %.2f" % (lay_heights[k], lay_E[k], lay_PR[k])
                axes[i].text(0.1 * RCmax, 0.5 * (cum_lay_height[k] + temporary_height), information, va = 'center')
                temporary_height = cum_lay_height[k]    # Updating the temporary variable.
            # For the last layer as a Rock bed.
            information = "h = inf, E = %d psi, PR = %.2f" % (lay_E[-1], lay_PR[-1])
            axes[i].text(0.1 * RCmax, 0.5 * (ZCmax + 11 + temporary_height), information, va = 'center')

            # Set the Axis ticks, labels and color bar.
            axes[i].set_yticks([m for m in range(-5, int(ZCmax+11), 5)])        # Set the number of 'Y' ticks.
            axes[i].set_ylabel('Z (in)'); axes[i].set_xlabel('R (in)')          # Set the 'X' and 'Y' labels.
            axes[i].set_title(Responses_titles[i])                              # Set the title of the axes.
            colorbar = fig.colorbar(draw, ax=axes[i], format = '%1.1e')         # Create a color bar legend.
            colorbar.locator = ticker.MaxNLocator(nbins=8)                      # Set the number of labels on the color bar legend.
            colorbar.update_ticks()                                             # Update the y-ticks.


        # Set the figure size and adjust the subplot locations in the figure.
        fig.set_figheight(5)
        fig.set_figwidth(11)
        fig.canvas.set_window_title('MatLEA: Stress Responses')
        plt.subplots_adjust(0.05, 0.1, 0.95, 0.95, 0.15, 0.32)


    # Show the figure to the user.
    plt.show()
# ======================================================================================================================