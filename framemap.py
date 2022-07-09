import cv2 as cv
import numpy as np
import imutils

drawing = False  # true if mouse is pressed
src_x, src_y = -1,-1
dst_x, dst_y = -1,-1

src_list = [[326, 68], [13, 791], [1366, 69], [1681, 791], [846, 69], [847, 792], [848, 443], [848, 260], [849, 347], [1227, 168], [1350, 578], [1588, 579], [1409, 170], [1448, 262], [1386, 261], [1458, 444], [1529, 443], [1348, 346]];
dst_list = [[54, 28], [56, 786], [1224, 29], [1225, 788], [640, 28], [641, 786], [641, 510], [640, 305], [639, 405], [1041, 185], [1043, 633], [1222, 632], [1223, 183], [1223, 305], [1162, 307], [1162, 508], [1224, 511], [1101, 409]];

# mouse callback function
def select_points_src(event,x,y,flags,param):
    global src_x, src_y, drawing
    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        src_x, src_y = x,y
        cv.circle(src_copy,(x,y),5,(0,0,255),-1)
    elif event == cv.EVENT_LBUTTONUP:
        drawing = False

# mouse callback function
def select_points_dst(event,x,y,flags,param):
    global dst_x, dst_y, drawing
    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        dst_x, dst_y = x,y
        cv.circle(dst_copy,(x,y),5,(0,0,255),-1)
    elif event == cv.EVENT_LBUTTONUP:
        drawing = False


def get_coordinates(filtro):
    filter = filtro
    threshold = 40
    lista = []

    res = cv.cvtColor(filter, cv.COLOR_HSV2BGR)
    src_gray = cv.cvtColor(res, cv.COLOR_BGR2GRAY)
    src_gray = cv.blur(src_gray, (3, 3))

    canny_output = cv.Canny(src_gray, threshold, threshold * 2)
    cv.imshow('aa', canny_output)
    contours, _ = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    contours_poly = [None] * len(contours)
    centers = [None] * len(contours)
    radius = [None] * len(contours)

    for i, c in enumerate(contours):
        contours_poly[i] = cv.approxPolyDP(c, 3, True)
        centers[i], radius[i] = cv.minEnclosingCircle(contours_poly[i])

    for i in range(len(contours)):
        cx = int(centers[i][0])
        cy = int(centers[i][1])
        if int(radius[i]) < 10 and cx > 100 and cy > 60 and cx < 1100:  # ESTE < 1100 SIRVE PARA ELIMINAR PUNTOS QUE ESTÉN FUERA DEL CAMPO Y SEAN INUTILES
            lista.append(str(cx) + ',' + str(cy))

    # ELIMINAMOS PUNTOS DOBLES PARA DEJAR LA LISTA LIMPIA
    lista = set(lista)
    lista = list(lista)
    print(lista)

    return lista;


def get_plan_view(src, dst):
    src_pts = np.array(src_list).reshape(-1, 1, 2)
    dst_pts = np.array(dst_list).reshape(-1, 1, 2)
    H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    plan_view = cv.warpPerspective(src, H, (dst.shape[1], dst.shape[0]))
    cv.imwrite('./images/homografia.png', plan_view)
    plan_view = cv.imread('./images/homografia.png')

    # Elegimos el umbral de ROJO en HSV
    plan_view_aux = plan_view
    plan_view_aux = cv.cvtColor(plan_view_aux, cv.COLOR_BGR2HSV)  # BGR TO RGB
    mask1 = cv.inRange(plan_view_aux, (170, 100, 100), (179, 255, 255))  # METEMOS DOS FILTROS ROJOS PA MEJORAR LA CALIDAD DE SUSTRACCIÓN
    mask2 = cv.inRange(plan_view_aux, (0, 100, 100), (10, 255, 255))
    mask = mask1 + mask2
    res = cv.bitwise_and(plan_view_aux, plan_view_aux, mask=mask)
    cv.imwrite('./images/filtroRED.png', res)
    lista_teamA = get_coordinates(res)

    # Elegimos el umbral de AZUL en HSV
    mask1 = cv.inRange(plan_view_aux, (101, 50, 38), (110, 255, 255))  # METEMOS DOS FILTROS AZUL PA MEJORAR LA CALIDAD DE SUSTRACCIÓN
    mask2 = cv.inRange(plan_view_aux, (101, 50, 38), (110, 255, 255))
    mask = mask1 + mask2
    final = cv.bitwise_and(plan_view_aux, plan_view_aux, mask=mask)
    cv.imwrite('./images/filtroBLUE.png', final)
    lista_teamB = get_coordinates(final)

    return plan_view, lista_teamA, lista_teamB;


def merge_views(src, dst):
    plan_view, _, _ = get_plan_view(src, dst)
    for i in range(0, dst.shape[0]):
        for j in range(0, dst.shape[1]):
            if (plan_view.item(i, j, 0) == 0 and \
                    plan_view.item(i, j, 1) == 0 and \
                    plan_view.item(i, j, 2) == 0):
                plan_view.itemset((i, j, 0), dst.item(i, j, 0))
                plan_view.itemset((i, j, 1), dst.item(i, j, 1))
                plan_view.itemset((i, j, 2), dst.item(i, j, 2))
    return plan_view;

def paint_points(dst):
    _, lista_teamA, lista_teamB = get_plan_view(src, dst)

    cx_offside1 = 0
    cx_offside2 = 0

    for c in lista_teamA:
        lista_aux = c.split(sep=',')
        cx = int(lista_aux[0])
        cy = int(lista_aux[1])
        cv.circle(dst, (cx, cy), 5, (0, 0, 255), 2)
        if cx > cx_offside1:
            if cx_offside1 < 1784:
                cx_offside1 = cx

    for c in lista_teamB:
        lista_aux = c.split(sep=',')
        cx = int(lista_aux[0])
        cy = int(lista_aux[1])
        cv.circle(dst, (cx, cy), 5, (255, 0, 0), 2)
        if cx > cx_offside2:
            if cx_offside2 < 1784:
                cx_offside2 = cx

    cv.line(dst, (cx_offside1 + 6, 0), (cx_offside1 + 6, 808), (0, 255, 255), 2)
    cv.line(dst, (cx_offside2 + 6, 0), (cx_offside2 + 6, 808), (255, 255, 0), 2)
    if cx_offside2 > cx_offside1:
        print("")
        print("")
        print("")
        print("FUERA DE JUEGO")
        print("")
        print("")
        print("")

    return dst;

src = cv.imread('./images/offside_completo.png', -1)
src_copy = src.copy()
cv.namedWindow('src')
cv.moveWindow("src", 80,80);
cv.setMouseCallback('src', select_points_src)

dst = cv.imread('./images/campo.png', -1)
dst_copy = dst.copy()
cv.namedWindow('dst')
cv.moveWindow("dst", 780,80);
cv.setMouseCallback('dst', select_points_dst)

while(1):
    cv.imshow('src',src_copy)
    cv.imshow('dst',dst_copy)
    k = cv.waitKey(1) & 0xFF
    if k == ord('s'):
        print('save points')
        cv.circle(src_copy,(src_x,src_y),5,(0,255,0),-1)
        cv.circle(dst_copy,(dst_x,dst_y),5,(0,255,0),-1)
        src_list.append([src_x,src_y])
        dst_list.append([dst_x,dst_y])
        print("src points:")
        print(src_list);
        print("dst points:")
        print(dst_list);
    elif k == ord('h'):
        print('create plan view')
        plan_view, _, _ = get_plan_view(src, dst)
        cv.imshow("plan view", plan_view)
    elif k == ord('m'):
        print('merge views')
        merge = merge_views(src, dst)
        cv.imshow("merge", merge)
    elif k == ord('p'):
        print('mapping')
        paint = paint_points(dst)
        cv.imshow('map', paint)
    elif k == 27:
        break
cv.destroyAllWindows()
