import cv2
import numpy as np

def image_stats(image):
    # split the image into its L*a*b* channels
    (l, a, b) = cv2.split(image)

    # compute the mean and standard deviation of each channel
    (lMean, lStd) = (l.mean(), l.std())
    (aMean, aStd) = (a.mean(), a.std())
    (bMean, bStd) = (b.mean(), b.std())

    # return the color statistics
    return (lMean, lStd, aMean, aStd, bMean, bStd)


def color_transfer_skin_tone(source, target):
    # convert the images from the RGB to L*ab* color space
    source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")
    target = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32")
    
    # compute color statistics for the source and target images
    (lMeanSrc, lStdSrc, aMeanSrc, aStdSrc, bMeanSrc, bStdSrc) = image_stats(source)
    (lMeanTar, lStdTar, aMeanTar, aStdTar, bMeanTar, bStdTar) = image_stats(target)


    # define the skin tone color range in the source image
    lower_skin = np.array([0, 100, 0])
    upper_skin = np.array([255, 140, 255])

    # extract the skin tone pixels from the source image
    skin_mask = cv2.inRange(source, lower_skin, upper_skin)
    skin = cv2.bitwise_and(source, source, mask=skin_mask)

    # compute color statistics for the skin pixels in the source image
    (lMeanSrc, lStdSrc, aMeanSrc, aStdSrc, bMeanSrc, bStdSrc) = image_stats(skin)

    # subtract the means from the target image
    (l, a, b) = cv2.split(target)
    d = target.shape[0]*target.shape[1]
    l -= lMeanTar
    a -= aMeanTar
    b -= bMeanTar

    # scale by the standard deviations
    l = (lStdTar / lStdSrc)*l
    a = (aStdTar / aStdSrc)*a
    b = (bStdTar / bStdSrc)*b

    # add themeans back to the target image
    l += lMeanSrc
    a += aMeanSrc
    b += bMeanSrc

    # clip any values below 0 or above 255
    l = np.clip(l, 0, 255)
    a = np.clip(a, 0, 255)
    b = np.clip(b, 0, 255)

    # merge the channels back together
    transfer = cv2.merge([l, a, b])

    # convert the image back to the RGB color space
    transfer = cv2.cvtColor(transfer.astype("uint8"), cv2.COLOR_LAB2BGR)

    # return the color transferred image
    return transfer

# load the source and target images
source = cv2.imread("1.jpg")
target = cv2.imread("2.jpg")

# transfer the skin tone from the source image to the target image
transfer = color_transfer_skin_tone(source, target)

# show the original and color transferred images
#cv2.imshow("Original", np.hstack([source, target]))
cv2.imshow("Transfer", transfer)
cv2.waitKey(0)

cv2.destroyAllWindows()