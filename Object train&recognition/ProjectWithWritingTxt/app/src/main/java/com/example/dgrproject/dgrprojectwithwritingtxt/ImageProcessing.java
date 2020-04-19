package com.example.dgrproject.dgrprojectwithwritingtxt;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.AsyncTask;
import android.os.Environment;
import android.util.Log;
import android.widget.TextView;
import android.widget.Toast;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.io.BufferedWriter;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Vector;


public class ImageProcessing extends AsyncTask {
    private Mat croppingImage, lbpImage;
    private List<Mat> rawImageSets;
    private int width = 300, height = 400; // resimler yan olduğu için width 'i height'ten büyük olduğu için
    private MainActivity mainActivity;
    List<String> trainArray=new ArrayList<>();

    public ImageProcessing(List<Mat> rawTrainingSets, MainActivity mainActivity) {

        this.rawImageSets = rawTrainingSets;
        this.mainActivity = mainActivity;
    }

    @Override
    protected Object doInBackground(Object[] objects) {

        try {
            updateUI(nativeFunction());
        } catch (IOException e) {
            e.printStackTrace();
        }

        return null;
    }


    private boolean nativeFunction() throws IOException {
        List<Mat> processedImgSets = new ArrayList<>();

        for (int i = 0; i < rawImageSets.size(); i++) {
            croppingImage = new Mat();
            lbpImage = new Mat();
            NativeClass.findContourAndCropping(rawImageSets.get(i).getNativeObjAddr(), croppingImage.getNativeObjAddr());
            NativeClass.scalingWithAspectRatio(croppingImage.getNativeObjAddr(), width, height);
            NativeClass.lbpWithHistogram(croppingImage.getNativeObjAddr(), lbpImage.getNativeObjAddr());

            processedImgSets.add(i,lbpImage );
            rawImageSets.get(i).release();
        }

        for (int i = 0; i < processedImgSets.size(); i++) {
           trainArray.add(i,processedImgSets.get(i).dump());
        }


        FileOutputStream fos = null;

        try {
            final File dir = new File(Environment.getExternalStorageDirectory().getAbsolutePath() + "/DGR/" );

            if (!dir.exists())
            {
                if(!dir.mkdirs()){
                    Log.e("ALERT","could not create the directories");
                }
            }

            final File myFile = new File(dir, "trainimages" + ".txt");

            if (!myFile.exists())
            {
                myFile.createNewFile();
            }

            fos = new FileOutputStream(myFile);

            PrintWriter printWriter = new PrintWriter(fos);
            for(String str: trainArray) {

                printWriter.println(str);
            }
            printWriter.close();
            fos.close();
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
            return  false;
        }


        return  true;
    }

    private void updateUI(final boolean isWrite) {
        mainActivity.runOnUiThread(new Runnable() {
            @Override
            public void run() {
                mainActivity.displayOnScreen(isWrite);

            }
        });
    }
}


