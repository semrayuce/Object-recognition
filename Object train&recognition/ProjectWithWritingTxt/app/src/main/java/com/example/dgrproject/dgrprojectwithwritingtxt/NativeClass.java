package com.example.dgrproject.dgrprojectwithwritingtxt;


public class NativeClass {
    public native static void findContourAndCropping(long addrSrc, long addrDst);
    public native static void scalingWithAspectRatio(long matAddrRgb, int width, int height);
    public native static void lbpWithHistogram(long addrSrc, long addrDst);

}
