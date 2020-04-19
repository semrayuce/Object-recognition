package com.example.dgrproject.dgrprojectwithwritingtxt;

import android.app.Activity;
import android.content.Context;
import android.os.Bundle;
import android.widget.TextView;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;


public class MainActivity extends Activity {

    static {
        System.loadLibrary("native-lib");
    }

    TextView textview1;
    Context context = this;
    List<Mat> rawImageSets = new ArrayList<>();
    int resourceId;
    Mat mat;
    String[] images = {
            "ananas1", "armut1", "ceviz1", "dbiber1", "domates1", "elma1", "kayisi1", "kbahar1", "lahana1",
            "muz1", "nar1", "onion1", "patlican1", "portakal1", "uzum1",

            "ananas2", "armut2", "ceviz2", "dbiber2", "domates2", "elma2", "kayisi2", "kbahar2", "lahana2",
            "muz2", "nar2", "onion2", "patlican2", "portakal2", "uzum2",

            "ananas3", "armut3", "ceviz3", "dbiber3", "domates3", "elma3", "kayisi3", "kbahar3", "lahana3",
            "muz3", "nar3", "onion3", "patlican3", "portakal3", "uzum3",

            "ananas4", "armut4", "ceviz4", "dbiber4", "domates4", "elma4", "kayisi4", "kbahar4", "lahana4",
            "muz4", "nar4", "onion4", "patlican4", "portakal4", "uzum4",

            "ananas5", "armut5", "ceviz5", "dbiber5", "domates5", "elma5", "kayisi5", "kbahar5", "lahana5",
            "muz5", "nar5", "onion5", "patlican5", "portakal5", "uzum5"
    };

    private BaseLoaderCallback mOpenCVCallBack = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                    break;
                default:
                    super.onManagerConnected(status);
                    break;
            }
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        textview1 = (TextView) findViewById(R.id.textview1);

        //resimleri drawable klasöründen alıyoruz
        for (int j = 0; j < images.length; j++) {
            resourceId = getResources().getIdentifier(images[j], "drawable", context.getPackageName());
            mat = new Mat();
            try {
                mat = Utils.loadResource(this, resourceId, CvType.CV_8UC4);
            } catch (IOException e) {
                e.printStackTrace();
            }
            rawImageSets.add(mat);
        }
        processImage(rawImageSets);
    }

    @Override
    protected void onResume() {
        super.onResume();
        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_2_0, this, mOpenCVCallBack);
    }

    public void onDestroy() {
        super.onDestroy();
        if (rawImageSets != null)
            rawImageSets.clear();
    }

    private void processImage(List<Mat> rawTrainingSets) {
        new ImageProcessing(rawTrainingSets, this).execute();
    }

    public void displayOnScreen(final boolean isWrite) {
        if (isWrite == true)
            textview1.setText(".txt başarıyla yazıldı");
        else
            textview1.setText(".txt başarısız");

    }
}
