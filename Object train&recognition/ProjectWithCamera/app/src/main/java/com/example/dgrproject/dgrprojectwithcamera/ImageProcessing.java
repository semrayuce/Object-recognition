package com.example.dgrproject.dgrprojectwithcamera;

import android.graphics.Bitmap;
import android.os.AsyncTask;

import org.opencv.android.Utils;
import org.opencv.core.Mat;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;


public class ImageProcessing extends AsyncTask {
    private Mat croppingImage, sobelImage, lbpImage, testImage, predictMat, trainImages, finalImageMat;
    private int width = 300, height = 400;
    private MainActivity mainActivity;
    private Bitmap finalImage;

    public ImageProcessing(Mat testImage, Mat trainImages, MainActivity mainActivity) {

        this.testImage = testImage;
        this.trainImages = trainImages;
        this.mainActivity = mainActivity;
    }

    @Override
    protected Object doInBackground(Object[] objects) {

        try {
            int match = nativeFunction();
            updateUI(match, finalImage);
        } catch (IOException e) {
            e.printStackTrace();
        }

        return null;
    }


    private int nativeFunction() throws IOException {

        croppingImage = new Mat();
        sobelImage = new Mat();
        lbpImage = new Mat();
        predictMat = new Mat();
        for (int i = 0; i < 1; i++) {
            NativeClass.findContourAndCropping(testImage.getNativeObjAddr(), croppingImage.getNativeObjAddr());
            NativeClass.scalingWithAspectRatio(croppingImage.getNativeObjAddr(), width, height);
            NativeClass.lbpWithHistogram(croppingImage.getNativeObjAddr(), lbpImage.getNativeObjAddr());
        }


        NativeClass.svm(predictMat.getNativeObjAddr(), trainImages.getNativeObjAddr(), lbpImage.getNativeObjAddr());

        System.out.println(predictMat.dump());
        int matchedId = (int) predictMat.get(0, 0)[0];
        finalImage = Bitmap.createBitmap(croppingImage.cols(), croppingImage.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(croppingImage, finalImage);
        return matchedId;
    }

    private void updateUI(final int matchedId, final Bitmap finalImage) {
        mainActivity.runOnUiThread(new Runnable() {
            @Override
            public void run() {
                mainActivity.displayOnScreen(matchedId, finalImage);

            }
        });
    }

}


