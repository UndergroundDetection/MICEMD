"""
The thread class, To prevent the interface from dying, we use multithreading
to call various time-consuming interface

Class:
- ThreadCalFdem: the thread to call the fdem simulation interface
- ThreadInvFdem: the thread to call the fdem inversion interface
- ThreadCalTdem: the thread to call the tdem simulation interface
- ThreadClsTdem: the thread to call the tdem classification interface
"""
from PyQt5.QtCore import QTranslator, QThread, pyqtSignal
import MicEMD.fdem as f
import MicEMD.tdem as t
import numpy as np

from MicEMD.handler import TDEMHandler
from MicEMD.preprocessor import data_prepare

import warnings

warnings.filterwarnings("ignore")  # 忽略版本问题


class ThreadCalFdem(QThread):
    """Fdem forward simulation thread.
    Use the mutilThread to avoid that the mainWindow play died because of
    the long time of complicated calculation.The simulation generates the
    forward data by the input condition of the simulation.

    Parameters
    ----------
    detector: class
        The detector class consists of the information about the detector
    target: class
        The target class consists of the information about the detector
    collection: class
        The collection class consists of the information about collecting
        the response of the target
    save: bool
        Whether to save the data of the forward simulation

    """

    # use the signal to get the result of the simulation thread
    trigger = pyqtSignal(tuple)

    def __init__(self):
        super(ThreadCalFdem, self).__init__()
        self.detector = None
        self.target = None
        self.collection = None
        self.save = None

    def run(self):
        """Use the information about the initial parameters to simulate"""

        forward_result = (f.simulate(self.target, self.detector, self.collection, 'simpeg'), )
        self.trigger.emit(forward_result)


class ThreadInvFdem(QThread):
    """Fdem inversion thread.
    Use the mutilThread to avoid that the mainWindow play died because of
    the long time of complicated calculation of the inversion.The inversion
    inverses by the input condition.

    Parameters
    ----------
    method: str
        The method of inversion
    iterations: int
        The maximum number of numeric optimization method
    tol: int
        The minimum cost of numeric optimization method
    x0: ndarray
        The initial parameters of inversion
    forward_result: tuple
        The result of simulation,consists of the information of the simulation
    save: bool
        Whether to save the result of inversion

    """
    # use the signal to get the result of the inversion thread
    trigger = pyqtSignal(dict)

    def __init__(self):
        super(ThreadInvFdem, self).__init__()
        self.method = None
        self.iterations = None
        self.tol = None
        self.x0 = np.array([0.0, 0.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.forward_result = None
        self.target = None
        self.detector = None
        self.save = False

    def run(self):
        """
        Use the information about the condition of inversion to inverse,
        and utilize the signal to transfer the inv_result.

        """

        data = (self.forward_result[0], self.target, self.detector)
        inv_para = {'x0': self.x0, 'iterations': self.iterations, 'tol': self.tol}

        inv_result = f.inverse(data, self.method, inv_para)

        self.trigger.emit(inv_result)


class ThreadCalTdem(QThread):
    """tdem forward simulation thread.
    Use the mutilThread to avoid that the mainWindow play died because of
    the long time of complicated calculation.The simulation generates the
    forward data by the input condition of the simulation.

    Parameters
    ----------
    detector: class
        The detector class consists of the information about the detector
    target: class
        The target class consists of the information about the detector
    collection: class
        The collection class consists of the information about collecting
        the response of the target
    save: bool
        Whether to save the data of the forward simulation

    """

    # use the signal to get the result of the simulation thread
    trigger = pyqtSignal(tuple)

    def __init__(self):
        super(ThreadCalTdem, self).__init__()
        self.detector = None
        self.target = None
        self.collection = None
        self.save = None

    def run(self):
        """Use the information about the initial parameters to simulate"""

        forward_result = t.simulate(self.target, self.detector, self.collection)
        self.trigger.emit(forward_result)


class ThreadClsTdem(QThread):
    """Tdem inversion thread.
    Use the mutilThread to avoid that the mainWindow play died because of
    the long time of complicated calculation of the inversion.The inversion
    inverses by the input condition.

    Parameters
    ----------
    method: str
        The method of inversion
    iterations: int
        The maximum number of numeric optimization method
    tol: int
        The minimum cost of numeric optimization method
    x0: ndarray
        The initial parameters of inversion
    forward_result: tuple
        The result of simulation,consists of the information of the simulation
    save: bool
        Whether to save the result of inversion

    """
    # use the signal to get the result of the inversion thread
    trigger = pyqtSignal(dict)

    def __init__(self):
        super(ThreadClsTdem, self).__init__()
        self.dir_method = None
        self.task = None
        self.cls_method = None
        self.forward_result = None
        self.target = None
        self.collection = None
        self.save = False

    def run(self):
        """
        Use the information about the condition of inversion to inverse,
        and utilize the signal to transfer the inv_result.

        """

        response = data_prepare(self.forward_result[0], self.task)

        res = t.preprocess(response, self.dir_method)
        handler = TDEMHandler(target=self.target, collection=self.collection)
        handler.save_preparation_default(response[0], response[1], self.task)
        handler.save_dim_reduction_default(res[0], res[1], self.task, self.dir_method)

        para = {'solver': 'lbfgs', 'hidden_layer_sizes': (50,), 'activation': 'tanh'}

        cls_res = t.classify(res, self.cls_method, para)

        self.trigger.emit(cls_res)
