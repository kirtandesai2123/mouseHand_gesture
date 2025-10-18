import numpy as np

def get_angle(a,b,c):
    radians = np.arctan2(c[1]-b[1],c[0]-b[0])-np.arctan2(a[1]-b[1],a[0]-b[0])
    angle = np.abs(np.degrees(radians))
    return angle

def get_distance(landMarks_list):
    if len(landMarks_list) < 2:
        return
    (x1, y1), (x2, y2) = landMarks_list[0], landMarks_list[1]
    L = np.hypot(x2 - x1, y2-y1)#euclidian distance
    return np.interp(L,[0,1],[0,1000])#interpolation into 1 to 1000 (multiplying)
