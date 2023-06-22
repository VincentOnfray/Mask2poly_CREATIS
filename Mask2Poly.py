import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import shapely as sh
from collections import defaultdict
import imageio.v3 as imio

def mask_to_polygons(mask, first_threshold, second_threshold, min_area=10.):
    border = 1
    """Convert a mask ndarray (binarized image) to Multipolygons"""
    # first, find contours with cv2: it's much faster than shapely
    #mask = cv.copyMakeBorder(mask,border,border,border,border,cv.BORDER_CONSTANT,value=255)
    
    emptyImg = np.ones((mask.shape[0]-border*2,mask.shape[1]-border*2,1), np.uint8)
    emptyBorderedImg = cv.copyMakeBorder(emptyImg,border,border,border,border,cv.BORDER_CONSTANT,value=255)
    cv.imshow("emptyBordered",emptyBorderedImg)
    print("shape of emptyBordered:" +str(emptyBorderedImg.shape[0])+" "+str(emptyBorderedImg.shape[1]))

    temp = cv.bitwise_and(mask,emptyBorderedImg)
    temp = cv.bitwise_not(temp)

    mask = cv.min(temp,mask)


    cv.waitKey(0)
    cv.imshow("Bordered",mask)
    #print("Bordered: "+mask.shape.x)
    #mask = cv.Canny(mask, first_threshold, second_threshold)
    kernel = np.ones((3,3),np.uint8)
    erodedMask = cv.erode(mask,kernel)
    mask = cv.bitwise_xor(mask,erodedMask)
    cv.imshow("edged",mask)
    #cv.imwrite("edged.png",mask)
    #print("edged: "+mask.shape.x)
    #cv.waitKey(0)


    
    cv.waitKey(0)
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

def Polygons_To_Mask(polygons, im_size):
    """Convert a polygon or multipolygon list back to
       an image mask ndarray"""
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

#LOAD original
im = cv.imread('Sample_3.png')
cv.imshow("Original",im)
assert im is not None, "file could not be read, check with os.path.exists()"

#PREPARE image to find contours
imgray = cv.cvtColor(im, cv.COLOR_RGB2GRAY)
#cv.imshow("gray",imgray)


# FIND edges


#CONVERT edges to polygons
polys = mask_to_polygons(imgray,1,250)
print("Nb de polygones "+str(len(polys.geoms)))

#CONVERT the polygons back to a mask image to validate that all went well
mask = Polygons_To_Mask(polys.geoms, imgray.shape[:2])

#cv.waitKey(0)
cv.imshow("final mask",mask)
cv.imwrite("Result.png",mask)
cv.waitKey(0)



