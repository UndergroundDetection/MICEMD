import numpy as np
import MicEMD.fdem as fdem
from MicEMD.handler import FDEMHandler

# Create and Initial the target, detector, collection class
target = fdem.Target(conductivity=5.71e7, permeability=1.26e-6, radius=0.2, pitch=0,
                     roll=0, length=1, position_x=0, position_y=0, position_z=-5)
detector = fdem.Detector(radius=0.4, current=20, frequency=1000, pitch=0, roll=0)
collection = fdem.Collection(spacing=0.5, height=0.1, SNR=30, x_min=-2, x_max=2,
                             y_min=-2, y_max=2, collection_direction='z-axis')
# call the interface of the fdem forward modeling
fwd_res = fdem.simulate(target, detector, collection, 'simpeg')


# set the inputs and parameters of the inversion
inv_inputs = (fwd_res, target, detector)
x0 = np.array([0.0, 0.0, -2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
inv_para = {'x0': x0, 'iterations': 10, 'tol': 1e-9}

# call the interface of the inversion
inv_res = fdem.inverse(inv_inputs, 'BFGS', inv_para)
# print(inv_res['error'], inv_res['pred'], inv_res['true'])


# create the FDEMHandler and call the methods to show and save the results
# set the FDEMHandler without parameters to save the results
# the file path of the results is generated by your settings

handler = FDEMHandler()
handler.save_fwd_data(fwd_res, 'magdata.csv')
# if you want get other axial mag_map, you can creat another collection with other
# collection_direction parameter, and input to the method show_mag_map
handler.show_mag_map(fwd_res, collection, show=True, save=True, file_name='mag_map.png')
handler.show_detection_scenario(target, collection, show=True, save=True)
handler.save_inv_res(inv_res, 'BFGS.csv')
handler.show_inv_res(inv_res, show=True, save=True, file_name='inv_res.png')

# set the FDEMHandler with parameters to save the results by default
# the file path of the results is generated according to the parameters of FDEMHandler
handler = FDEMHandler(target=target, collection=collection)
handler.save_fwd_data_default(fwd_res)
handler.show_detection_scenario_default(show=True)
handler.show_mag_map_default(fwd_res, show=True)
handler.save_inv_res_default(inv_res, 'BFGS')


