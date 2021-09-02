import cv2
import numpy as np
import utils
def strokeEdges(src,dst,blurKsize=7,edgeKsize=5):
    if blurKsize>=3:
        blurredSrc=cv2.medianBlur(src,blurKsize)
        graySrc=cv2.cvtColor(blurredSrc,cv2.COLOR_BGRA2GRAY)
    else:
        graySrc=cv2.cvtColor(src,cv2.COLOR_BGRA2GRAY)
    cv2.Laplacian(graySrc,cv2.CV_8U,graySrc,ksize=edgeKsize)
    normalizedInverseAlpha=(1.0/255)*(255-graySrc)
    channels=cv2.split(src)
    for channel in channels:
        channel[:]=channel*normalizedInverseAlpha
    cv2.merge(channels,dst)

class VConvolutionFilter(object):
    '''A filter that applies a convolution to all BGR'''
    def __init__(self,kernel):
        self._kernel=kernel
    def apply(self,src,dst):
        '''apply the filter with a BGR or gray source'''
        cv2.filter2D(src,-1,self._kernel,dst)

class SharpenFilter(VConvolutionFilter):
    def __init__(self):
        kernel=np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
        VConvolutionFilter.__init__(self,kernel)

class FindEdgesFilter(VConvolutionFilter):
    '''an edge-finding filter with a 2-pixel radius'''
    def __init__(self, kernel):
        kernel=np.full((5,5),0.04)
        super().__init__(self,kernel)

class EmbossFilter(VConvolutionFilter):
    '''an emboss filter with a 1 pixel radius'''
    def __init__(self, kernel):
        kernel=np.array([[-2,-1,0],[-1,1,1],[0,1,2]])
        super().__init__(self,kernel)
        

class BGRFuncFilter(object):

    def __init__(self, vFunc = None, bFunc = None, gFunc = None, rFunc = None,dtype = np.uint8) :

        length = np.iinfo(dtype).max + 1
        self._bLookupArray = utils.createLookupArray(utils.createCompositeFunc(bFunc, vFunc), length)
        self._gLookupArray = utils.createLookupArray(utils.createCompositeFunc(gFunc, vFunc), length)
        self._rLookupArray = utils.createLookupArray(utils.createCompositeFunc(rFunc, vFunc), length)

    def apply(self, src, dst) :

        """Apply the filter with a BGR source/destination."""
        b, g, r = cv2.split(src)
        utils.applyLookupArray(self._bLookupArray, b, b)
        utils.applyLookupArray(self._gLookupArray, g, g)
        utils.applyLookupArray(self._rLookupArray, r, r)
        cv2.merge([ b, g, r ], dst)



class BGRCurveFilter(BGRFuncFilter):

    def __init__(self, vPoints = None, bPoints = None,gPoints = None, rPoints = None, dtype = np.uint8):
        BGRFuncFilter.__init__(self, utils.createCurveFunc(vPoints), utils.createCurveFunc(bPoints), utils.createCurveFunc(gPoints), utils.createCurveFunc(rPoints), dtype)



class BGRPortraCurveFilter(BGRCurveFilter):
    def __init__(self, dtype = np.uint8):
        BGRCurveFilter.__init__(
            self,
            vPoints = [ (0, 0), (23, 20), (157, 173), (255, 255) ],
            bPoints = [ (0, 0), (41, 46), (231, 228), (255, 255) ],
            gPoints = [ (0, 0), (52, 47), (189, 196), (255, 255) ],
            rPoints = [ (0, 0), (69, 69), (213, 218), (255, 255) ],
            dtype = dtype)






