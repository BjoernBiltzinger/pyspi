import numpy as np
try:
    from numba import njit
    has_numba = True
except:
    has_numba = False


if has_numba:
    # If numba is available use it here. Makes this about 10 times faster!
    @njit
    def polar2cart(ra,dec):

        x = np.cos(np.deg2rad(ra)) * np.cos(np.deg2rad(dec))
        y = np.sin(np.deg2rad(ra)) * np.cos(np.deg2rad(dec))
        z = np.sin(np.deg2rad(dec))

        return np.array([x,y,z])


    @njit
    def cart2polar(vector):

        ra = np.arctan2(vector[1],vector[0]) 
        dec = np.arcsin(vector[2])

        return np.rad2deg(ra), np.rad2deg(dec)

else:
    
    def polar2cart(ra,dec):

        x = np.cos(np.deg2rad(ra)) * np.cos(np.deg2rad(dec))
        y = np.sin(np.deg2rad(ra)) * np.cos(np.deg2rad(dec))
        z = np.sin(np.deg2rad(dec))

        return np.array([x,y,z])



    def cart2polar(vector):

        ra = np.arctan2(vector[1],vector[0]) 
        dec = np.arcsin(vector[2])

        return np.rad2deg(ra), np.rad2deg(dec)
    
    
