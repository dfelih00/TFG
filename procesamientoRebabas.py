from matplotlib import pyplot as plt
from imagen import Imagen
import cv2
import numpy as np
import os
import csv
import datetime
from csvFile import CsvFile

def rindex(mylist, myvalue):
    return len(mylist) - mylist[::-1].index(myvalue) - 1

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except (ValueError, msg):
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')

def procesarImagen(imagen, lista):
    print("Procesando imagen "+imagen.ruta)
    prueba = imagen.ecualizar()
    otsu1 = prueba.segmentacionOtsu()
    #secciones = otsu1.secciones(ancho=2)

    imgErosion = otsu1.erosion(iteraciones=10)
    #seccionesO = imgErosion.secciones(ancho=2)
    imgErosion2 = imgErosion.erosion(iteraciones=10)
    #secciones1 = imgErosion2.secciones(ancho=2)
    secciones = imgErosion2.secciones(ancho=100)
    data = []
    for seccion in secciones:
        data.append(seccion.porcentage_pixeles_blancos())
    suavizada = savitzky_golay(np.asarray(data),5,1)
    #Descriptores
    info_partial = [imagen.ruta, imagen.pieza, imagen.tipo]
    lista.append(info_partial+suavizada.tolist())

#Variables
info_ok = []

#Leer imagenes
ruta = '../fotosPiezas';
for pieza in os.listdir(ruta):
    rutaPieza = ruta+'/'+pieza
    if os.path.isdir(rutaPieza):
        for tipo in os.listdir(rutaPieza):
            if tipo=="Rebabas":
                rutaTipo = rutaPieza +'/'+tipo
                if os.path.isdir(rutaTipo):
                    for imagen in os.listdir(rutaTipo):
                        rutaImagen = rutaTipo + '/'+imagen
                        if imagen.endswith('.tif'):
                            img = Imagen(rutaImagen, blancoNegro=True, pieza=pieza, tipo = tipo)
                            procesarImagen(img, info_ok)

print("Se han encontrado "+str(len(info_ok))+" imágenes del tipo Rebabas")

#Guardar los datos
print("Guardando datos Rebabas")
okFile = CsvFile(info_ok,"Rebabas")
print("Los datos Rebabas han sido guardados")
