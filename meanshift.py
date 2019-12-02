# pure numpy implementation of MeanShift algorithm
import numpy as np
import math

DBL_EPSILON = 2 * 10**15

def rect_intersect(rect1, rect2):
    return (max(rect1['x'], rect2['x']), max(rect1['y'], rect2['y']),
            min(rect1['x']+rect1['width'], rect2['x']+rect2['width']),
            min(rect1['y']+rect1['height'], rect2['y']+rect2['height']))


def moments2e(image):
    """
    This function calculates the raw, centered and normalized moments
    for any image passed as a numpy array.
    Further reading:
    https://en.wikipedia.org/wiki/Image_moment
    https://en.wikipedia.org/wiki/Central_moment
    https://en.wikipedia.org/wiki/Moment_(mathematics)
    https://en.wikipedia.org/wiki/Standardized_moment
    http://opencv.willowgarage.com/documentation/cpp/structural_analysis_and_shape_descriptors.html#cv-moments

    compare with:
    import cv2
    cv2.moments(image)
    """
    assert len(image.shape) == 2  # only for grayscale images
    x, y = np.mgrid[:image.shape[0], :image.shape[1]]
    moments = {}
    moments['mean_x'] = np.sum(x * image) / np.sum(image)
    moments['mean_y'] = np.sum(y * image) / np.sum(image)

    # raw or spatial moments
    moments['m00'] = np.sum(image)
    moments['m01'] = np.sum(x * image)
    moments['m10'] = np.sum(y * image)
    moments['m11'] = np.sum(y * x * image)
    moments['m02'] = np.sum(x ** 2 * image)
    moments['m20'] = np.sum(y ** 2 * image)
    moments['m12'] = np.sum(x * y ** 2 * image)
    moments['m21'] = np.sum(x ** 2 * y * image)
    moments['m03'] = np.sum(x ** 3 * image)
    moments['m30'] = np.sum(y ** 3 * image)

    # central moments
    # moments['mu01']= np.sum((y-moments['mean_y'])*image) # should be 0
    # moments['mu10']= np.sum((x-moments['mean_x'])*image) # should be 0
    moments['mu11'] = np.sum((x - moments['mean_x']) * (y - moments['mean_y']) * image)
    moments['mu02'] = np.sum((y - moments['mean_y']) ** 2 * image)  # variance
    moments['mu20'] = np.sum((x - moments['mean_x']) ** 2 * image)  # variance
    moments['mu12'] = np.sum((x - moments['mean_x']) * (y - moments['mean_y']) ** 2 * image)
    moments['mu21'] = np.sum((x - moments['mean_x']) ** 2 * (y - moments['mean_y']) * image)
    moments['mu03'] = np.sum((y - moments['mean_y']) ** 3 * image)
    moments['mu30'] = np.sum((x - moments['mean_x']) ** 3 * image)

    # opencv versions
    # moments['mu02'] = sum(image*(x-m01/m00)**2)
    # moments['mu02'] = sum(image*(x-y)**2)

    # wiki variations
    # moments['mu02'] = m20 - mean_y*m10
    # moments['mu20'] = m02 - mean_x*m01

    # central standardized or normalized or scale invariant moments
    moments['nu11'] = moments['mu11'] / np.sum(image) ** (2 / 2 + 1)
    moments['nu12'] = moments['mu12'] / np.sum(image) ** (3 / 2 + 1)
    moments['nu21'] = moments['mu21'] / np.sum(image) ** (3 / 2 + 1)
    moments['nu20'] = moments['mu20'] / np.sum(image) ** (2 / 2 + 1)
    moments['nu03'] = moments['mu03'] / np.sum(image) ** (3 / 2 + 1)  # skewness
    moments['nu30'] = moments['mu30'] / np.sum(image) ** (3 / 2 + 1)  # skewness
    return moments


def meanshift(input, window, termination_criteria):
    eps = round(termination_criteria['eps'] ** 2)
    new_window = window.copy()
    for _ in range(termination_criteria['max_iter']):
        new_window = rect_intersect(new_window, {'x':0, 'y':0, 'width':input.shape[0], 'height':input.shape[1]})
        m = moments2e(input[new_window['x']:new_window['x']+new_window['width'],
                      new_window['y']:new_window['y']+new_window['height']])

        if m['m00'] < DBL_EPSILON:
            break

        dx = round(m['m10']/m['m00']-window['width']*0.5)
        dy = round(m['m01']/m['m00']-window['height']*0.5)

        nx = min(max(new_window['x']+dx, 0), input.shape[0]-new_window['width'])
        ny = min(max(new_window['y']+dy, 0), input.shape[1]-new_window['height'])

        dx = nx - new_window['x']
        dy = ny - new_window['y']
        new_window['x'] = nx
        new_window['y'] = ny

        if dx**2+dy**2 < eps:
            break

    return new_window


def conv2d(a, f):
    s = f.shape + tuple(np.subtract(a.shape, f.shape) + 1)
    strd = np.lib.stride_tricks.as_strided
    subM = strd(a, shape=s, strides=a.strides * 2)
    return np.einsum('ij,ijkl->kl', f, subM)


def backproject(image, hist_template):
    image = convert2hst(image)
    I, _, _ = np.histogram2d(image[:, :, 0].ravel(), image[:, :, 1].ravel(),
                                       bins=(range(0, 180), range(0, 256)), density=True)
    R = hist_template / I
    B = R[image[:, :, 0].ravel(), image[:, :, 1].ravel()]
    B = np.minimum(B, 1)
    B = B.reshape(image.shape[:2])

    disc = np.array([[0, 0, 1, 0, 0],
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                    [0, 0, 1, 0, 0]], dtype=np.uint8)
    B = conv2d(B, disc)
    B = np.uint8(B)
    np.interp(B, (B.min(), B.max()), (0, 255))
    B[B < 50] = 0
    return B


def camshift(input, window, termination_criteria):
    TOLERANCE = 10
    window = meanshift(input, window, termination_criteria)
    window['x'] = max(window['x']-TOLERANCE, 0)
    window['y'] = max(window['y'-TOLERANCE], 0)

    window['width'] += 2 * TOLERANCE
    if window['x']+window['width'] > input.shape[0]: window['width'] = input.shape[0] - window['x']
    window['height'] += 2 * TOLERANCE
    if window['y'] + window['height'] > input.shape[1]: window['height'] = input.shape[0] - window['y']

    m = moments2e(input[window['x']:window['x'] + window['width'],
                  window['y']:window['y'] + window['height']])

    xc = round(m['m10']/m['m00']+window['x'])
    yc = round(m['01']/m['m00']+window['y'])

    square = (4*(m['mu11']/m['m00'])**2+((m['mu20']-m['mu02'])/m['m00'])**2)**0.5
    theta = math.atan2(2*m['mu11']/m['m00'], (m['mu20']-m['mu02'])/m['m00']+square)

    cs = math.cos(theta)
    sn = math.sin(theta)

    rotate_a = max(0, cs*cs*m['mu20']+2*cs*sn*m['mu11']+sn*sn*m['mu02'])
    rotate_c = max(0, sn*sn*m['mu20']+2*cs*sn*m['mu11']+cs*cs*m['mu02'])

    length = (rotate_a/m['m00'])**0.5*4
    width = (rotate_c/m['m00'])**0.5*4

    if length < width:
        length, width = width, length
        cs, sn = sn, cs
        theta = math.pi/2 - theta

    t0 = round(length * cs)
    t1 = round((width * sn))
    t0 = max(t0, t1)+2
    window['width'] = min(t0, (input.shape[0]-round(xc))*2)
    window['height'] = min(t0, (input.shape[1]-round(yc))*2)
    window['x'] = max(0, round(xc)-window['width']/2)
    window['y'] = max(0, round(yc)-window['height']/2)
    window['width'] = max(input.shape[0]-window['x'], window['width'])
    window['height'] = max(input.shape[1]-window['y'], window['height'])

    box = {}
    box['height'] = length
    box['width'] = width
    box['angle'] = (math.pi/2+theta)*180/math.pi
    while box['angle'] < 0: box['angle'] += 360
    while box['angle'] > 360: box['angle'] -= 360
    if box['angle'] > 180: box['angle'] -= 180
    box['center'] = (window['x']+window['width']/2, window['y']+window['height']/2)
    return box