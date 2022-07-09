import cv2 as cv
import numpy as np
import imutils

def get_coordinates(val, filtro):
    threshold = val
    filter = filtro
    lista = []

    res = cv.cvtColor(filter, cv.COLOR_HSV2BGR)
    src_gray = cv.cvtColor(res, cv.COLOR_BGR2GRAY)
    src_gray = cv.blur(src_gray, (3, 3))

    canny_output = cv.Canny(src_gray, threshold, threshold * 2)
    contours, _ = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    contours_poly = [None] * len(contours)
    centers = [None] * len(contours)
    radius = [None] * len(contours)

    for i, c in enumerate(contours):
        contours_poly[i] = cv.approxPolyDP(c, 3, True)
        centers[i], radius[i] = cv.minEnclosingCircle(contours_poly[i])

    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)

    for i in range(len(contours)):
        cx = int(centers[i][0])
        cy = int(centers[i][1])
        if int(radius[i]) < 10 and cx != 0 and cy != 0:  # ESTE < 600 SIRVE PARA ELIMINAR PUNTOS QUE ESTÉN FUERA DEL CAMPO Y SEAN INUTILES
            lista.append(str(cx) + ',' + str(cy))

    # ELIMINAMOS PUNTOS DOBLES PARA DEJAR LA LISTA LIMPIA
    lista = set(lista)
    lista = list(lista)

    return lista;

src = cv.imread('./images/offside_completo.png')
src = imutils.resize(src, height=640)
cv.imshow('src', src)
src = cv.cvtColor(src, cv.COLOR_BGR2HSV) #BGR TO RGB

drawing = np.zeros((src.shape[0], src.shape[1], 3), dtype=np.uint8)

max_thresh = 255
thresh = 40  # initial threshold
cv.createTrackbar('Canny thresh:', 'src', thresh, max_thresh, get_coordinates)

# Elegimos el umbral de ROJO en HSV
mask1 = cv.inRange(src, (170,100,100), (179,255,255)) # METEMOS DOS FILTROS ROJOS PA MEJORAR LA CALIDAD DE SUSTRACCIÓN
mask2 = cv.inRange(src, (0,100,100), (10,255,255))
mask = mask1 + mask2
res = cv.bitwise_and(src, src, mask=mask)
lista = get_coordinates(thresh, res)
for c in lista:
    lista_aux = c.split(sep=',')
    cx = int(lista_aux[0])
    cy = int(lista_aux[1])
    cv.circle(drawing, (cx, cy), 5, (0,0,255), 2)

# Elegimos el umbral de AZUL en HSV
mask1 = cv.inRange(src, (101,50,38), (110,255,255)) # METEMOS DOS FILTROS AZULES PA MEJORAR LA CALIDAD DE SUSTRACCIÓN
mask2 = cv.inRange(src, (101,50,38), (110,255,255))
mask = mask1 + mask2
res = cv.bitwise_and(src, src, mask=mask)
lista = get_coordinates(thresh, res)
for c in lista:
    lista_aux = c.split(sep=',')
    cx = int(lista_aux[0])
    cy = int(lista_aux[1])
    cv.circle(drawing, (cx, cy), 5, (255,0,0), 2)

cv.imshow('Contours', drawing)
cv.waitKey()