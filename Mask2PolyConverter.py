import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import shapely as sh
from collections import defaultdict

"""Conversion d'un masque binaire en multypolygon
    mask: le mask à convertir
    min_area: le nombre de pixels minimums pour qu'une forme soit retenue
    displayIntermediateSteps: Afficher les contours obtenus, utile pour débugger

    NOTE:
    - Fonctionne sur différentes formes, y compris les formes avec des "trous" dedans
    - Fonctionne sur les formes qui touchent le bord de l'image MAIS supprime 1 pixel d'épaisseur sur le bord de l'image
"""
def mask_to_polygons(mask, min_area=10., displayIntermediateSteps = False):

    mask = cv.cvtColor(mask, cv.COLOR_RGB2GRAY)

    border = 1 #Taille de la bordure de l'image
    emptyImg = np.ones((mask.shape[0]-border*2,mask.shape[1]-border*2,1), np.uint8) # Cadre vide
    emptyBorderedImg = cv.copyMakeBorder(emptyImg,border,border,border,border,cv.BORDER_CONSTANT,value=255) #cadre vide avec une border de pixels de valeur 255
    
    temp = cv.bitwise_and(mask,emptyBorderedImg) # Boolean And entre notre image à etudier et l'image vide avec bordure
    temp = cv.bitwise_not(temp) # Inversion

    mask = cv.min(temp,mask) #Minimum entre temp et mask

    if(displayIntermediateSteps):
        print("[Bordered image]")
        plt.imshow(mask)
    
    kernel = np.ones((3,3),np.uint8)
    erodedMask = cv.erode(mask,kernel)
    mask = cv.bitwise_xor(mask,erodedMask)
    

    contours, hierarchy = cv.findContours(mask,
                                  cv.RETR_CCOMP,
                                  cv.CHAIN_APPROX_NONE)
    if not contours:
        return sh.MultiPolygon()
    
    # now messy stuff to associate parent and child contours
    cnt_children = defaultdict(list)
    child_contours = set()
    assert hierarchy.shape[0] == 1
  
    for idx, (_, _, _, parent_idx) in enumerate(hierarchy[0]):
        if parent_idx != -1:
            child_contours.add(idx)
            cnt_children[parent_idx].append(contours[idx])
    # create actual polygons filtering by area (removes artifacts)
    all_polygons = []
    for idx, cnt in enumerate(contours):
        if idx not in child_contours and cv.contourArea(cnt) >= min_area:
            assert cnt.shape[1] == 1
            poly = sh.Polygon(
                shell=cnt[:, 0, :],
                holes=[c[:, 0, :] for c in cnt_children.get(idx, [])
                       if cv.contourArea(c) >= min_area])
            all_polygons.append(poly)
    all_polygons = sh.MultiPolygon(all_polygons)
    return all_polygons



###############################################################################################################
'''
Transforme un array Multipolygon en mask binaire
    polygons: l'array Multypolygon à exploiter
    im_size:taille de l'image à créer

    NOTE:
    - Fonctionne sur différentes formes, y compris les formes avec des "trous" dedans
    - Fonctionne sur les formes qui touchent le bord de l'image MAIS supprime 1 pixel d'épaisseur sur le bord de l'image
'''
def Polygons_To_Mask(polygons, im_size):
    
    img_mask = np.zeros(im_size, np.uint8)
    if not polygons:
        return img_mask
    int_coords = lambda x: np.array(x).round().astype(np.int32)
    exteriors = [int_coords(poly.exterior.coords) for poly in polygons]
    interiors = [int_coords(pi.coords) for poly in polygons
                 for pi in poly.interiors]
    cv.fillPoly(img_mask, exteriors, 255)
    #cv.fillPoly(img_mask, interiors, 150)
    return img_mask








