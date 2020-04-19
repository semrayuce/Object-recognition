package com.example.dgrproject.dgrprojectwithcamera;

import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.content.res.Configuration;
import android.graphics.Bitmap;
import android.hardware.Camera;
import android.os.Build;
import android.os.Bundle;
import android.util.Log;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.RelativeLayout;
import android.widget.TextView;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Properties;

public class MainActivity extends Activity {

    public static final String LOG_TAG = "NiceCameraExample";

    /**
     * 'camera' is the object that references the hardware device
     * installed on your Android phone.
     */
    private Camera camera;

    /**
     * Phone can have multiple cameras, so 'cameraID' is a
     * useful variable to store which one of the camera is active.
     * It starts with value -1
     */
    private int cameraID;

    /**
     * 'camPreview' is the object that prints the data
     * coming from the active camera on the GUI, that is... frames!
     * It's an instance of the 'CameraPreview' class, more information
     * in {@link CameraPreview}
     */
    private CameraPreview camPreview;

    static {
        System.loadLibrary("native-lib");
    }

    Context context = this;
    private static Mat testImage;
    TextView textview1;
    private static List<Mat> rawTrainingSets = new ArrayList<>();
    String matchedName;
    String[] imagesName = {"Ananas", "Armut", "Ceviz", "Dolmalık Biber", "Domates", "Elma", "Kayısı", "Karnıbahar", "Lahana", "Muz", "Nar", "Soğan", "Patlıcan", "Portakal", "Üzüm"};
    List<String> dataOfTxt;
    String[] numberStrs;
    int[][] matrixOfTxt;
    int[] arrayOfMatrix;
    private static ImageView image;
    static RelativeLayout preview;


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

    /**
     * In the onCreate() we could initialize the camera preview, by calling
     * {@link #setCameraInstance()}. That's not strongly necessary to call it right here,
     * the camera preview may start in a secondary moment.
     */
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        textview1 = (TextView) findViewById(R.id.matchedName);
        testImage = new Mat();

        dataOfTxt = readTextFileAsList(R.raw.trainimages);

        String columnSize[] = dataOfTxt.get(0).split(", ");

        matrixOfTxt = new int[dataOfTxt.size()][columnSize.length];

        for (int i = 0; i < dataOfTxt.size(); i++) {
            numberStrs = dataOfTxt.get(i).split(", ");
            for (int j = 0; j < numberStrs.length; j++) {
                matrixOfTxt[i][j] = Integer.parseInt(numberStrs[j]);
            }
        }
        int idx = 0;
        arrayOfMatrix = new int[matrixOfTxt.length * columnSize.length];
        for (int row = 0; row < matrixOfTxt.length; row++) {
            for (int column = 0; column < columnSize.length; column++) {

                arrayOfMatrix[idx] = matrixOfTxt[row][column];
                // increment index
                idx++;
            }
        }

        final Mat trainImages = new Mat(dataOfTxt.size(), columnSize.length, CvType.CV_32S);
        int row = 0, col = 0;
        trainImages.put(row, col, arrayOfMatrix);

        System.gc();

        // first, we check is it's possible to get an "instance" of the hardware camera
        if (setCameraInstance() == true) {
            // everything's OK, we can go further and create the preview object
            this.camPreview = new CameraPreview(this, this.camera, this.cameraID);
        } else {
            // error here! we can print something or just cry.
            this.finish();
        }

        // if the preview is set, we add it to the contents of our activity.
        preview = (RelativeLayout) findViewById(R.id.preview_layout);
        preview.addView(this.camPreview);

        image = (ImageView) findViewById(R.id.imageView1);
        image.setVisibility(View.INVISIBLE);
        // also we set some layout properties
        RelativeLayout.LayoutParams previewLayout = (RelativeLayout.LayoutParams) camPreview.getLayoutParams();
        previewLayout.width = FrameLayout.LayoutParams.MATCH_PARENT;
        previewLayout.height = FrameLayout.LayoutParams.MATCH_PARENT;
        this.camPreview.setLayoutParams(previewLayout);

        // on the main activity there's also a "capture" button, we set its behavior
        // when it gets clicked here
        Button captureButton = (Button) findViewById(R.id.button_capture);
        captureButton.setOnClickListener(
                new View.OnClickListener() {
                    @Override
                    public void onClick(View v) {
                        camera.takePicture(null, null, camPreview); // request a picture
                    }
                }
        );
        // at last, a call to set the right layout of the elements (like the button)
        // depending on the screen orientation (if it's changeable).
        Button matchButton = (Button) findViewById(R.id.button_match);
        matchButton.setOnClickListener(
                new View.OnClickListener() {
                    @Override
                    public void onClick(View v) {
                        //  if (testImage!=null) {
                        processImage(testImage, trainImages);
                        //  }
                    }
                }
        );

    }

    public List<String> readTextFileAsList(int resourceId) {
        InputStream inputStream = context.getResources().openRawResource(resourceId);
        InputStreamReader inputreader = new InputStreamReader(inputStream);
        BufferedReader bufferedreader = new BufferedReader(inputreader);
        String line;
        List<String> list = new ArrayList<String>();
        int j = 0;
        try {
            while ((line = bufferedreader.readLine()) != null) {
                list.add(line);
            }
        } catch (IOException e) {
            return null;
        }
        return list;
    }

    public static void addTestImage(Mat testImageMat, Bitmap bitmap) {
        testImage = testImageMat;

    }

    @Override
    protected void onResume() {
        super.onResume();
        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_2_0, this,
                mOpenCVCallBack);
        if (setCameraInstance() == true) {
            // TODO: camPreview.refresh...
        } else {
            Log.e(MainActivity.LOG_TAG, "onResume(): can't reconnect the camera");
            this.finish();
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        releaseCameraInstance();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (rawTrainingSets != null)
            rawTrainingSets.clear();
        releaseCameraInstance();
    }


    /**
     * This method is added in order to detect changes in orientation.
     * If we want we can react on this and change the position of
     * some GUI elements (see {fixElementsPosition(int)}
     * method).
     */
    public void onConfigurationChanged(Configuration newConfig) {
        super.onConfigurationChanged(newConfig);

    }

    /**
     * [IMPORTANT!] The most important method of this Activity: it asks for an instance
     * of the hardware camera(s) and save it to the private field {@link #camera}.
     *
     * @return TRUE if camera is set, FALSE if something bad happens
     */
    private boolean setCameraInstance() {
        if (this.camera != null) {
            // do the job only if the camera is not already set
            Log.i(MainActivity.LOG_TAG, "setCameraInstance(): camera is already set, nothing to do");
            return true;
        }

        // warning here! starting from API 9, we can retrieve one from the multiple
        // hardware cameras (ex. front/back)
        if (Build.VERSION.SDK_INT >= 9) {

            if (this.cameraID < 0) {
                // at this point, it's the first time we request for a camera
                Camera.CameraInfo camInfo = new Camera.CameraInfo();
                for (int i = 0; i < Camera.getNumberOfCameras(); i++) {
                    Camera.getCameraInfo(i, camInfo);

                    if (camInfo.facing == Camera.CameraInfo.CAMERA_FACING_BACK) {
                        // in this example we'll request specifically the back camera
                        try {
                            this.camera = Camera.open(i);
                            this.cameraID = i; // assign to cameraID this camera's ID (O RLY?)
                            return true;
                        } catch (RuntimeException e) {
                            // something bad happened! this camera could be locked by other apps
                            Log.e(MainActivity.LOG_TAG, "setCameraInstance(): trying to open camera #" + i + " but it's locked", e);
                        }
                    }
                }
            } else {
                // at this point, a previous camera was set, we try to re-instantiate it
                try {
                    this.camera = Camera.open(this.cameraID);
                } catch (RuntimeException e) {
                    Log.e(MainActivity.LOG_TAG, "setCameraInstance(): trying to re-open camera #" + this.cameraID + " but it's locked", e);
                }
            }
        }

        // we could reach this point in two cases:
        // - the API is lower than 9
        // - previous code block failed
        // hence, we try the classic method, that doesn't ask for a particular camera
        if (this.camera == null) {
            try {
                this.camera = Camera.open();
                this.cameraID = 0;
            } catch (RuntimeException e) {
                // this is REALLY bad, the camera is definitely locked by the system.
                Log.e(MainActivity.LOG_TAG,
                        "setCameraInstance(): trying to open default camera but it's locked. "
                                + "The camera is not available for this app at the moment.", e
                );
                return false;
            }
        }

        // here, the open() went good and the camera is available
        Log.i(MainActivity.LOG_TAG, "setCameraInstance(): successfully set camera #" + this.cameraID);
        return true;
    }

    /**
     * [IMPORTANT!] Another very important method: it releases all the resources and the locks
     * we created while using the camera. It MUST be called everytime the app exits, crashes,
     * is paused or whatever. The order of the called methods are the following: <br />
     * <p>
     * 1) stop any preview coming to the GUI, if running <br />
     * 2) call {@link Camera#release()} <br />
     * 3) set our camera object to null and invalidate its ID
     */
    private void releaseCameraInstance() {
        if (this.camera != null) {
            try {
                this.camera.stopPreview();
            } catch (Exception e) {
                Log.i(MainActivity.LOG_TAG, "releaseCameraInstance(): tried to stop a non-existent preview, this is not an error");
            }

            this.camera.setPreviewCallback(null);
            this.camera.release();
            this.camera = null;
            this.cameraID = -1;
            Log.i(MainActivity.LOG_TAG, "releaseCameraInstance(): camera has been released.");
        }
    }

    public Camera getCamera() {
        return this.camera;
    }

    public int getCameraID() {
        return this.cameraID;
    }


    private void processImage(Mat testImage, Mat trainImages) {
        new ImageProcessing(testImage, trainImages, this).execute();
    }

    public void displayOnScreen(final int matchedId, final Bitmap imageFinal) {
        preview.setVisibility(View.INVISIBLE);
        image.setVisibility(View.VISIBLE);
        image.setImageBitmap(imageFinal);

        switch (matchedId) {
            case 0:
                matchedName = imagesName[0];
                break;
            case 1:
                matchedName = imagesName[1];
                break;
            case 2:
                matchedName = imagesName[2];
                break;
            case 3:
                matchedName = imagesName[3];
                break;
            case 4:
                matchedName = imagesName[4];
                break;
            case 5:
                matchedName = imagesName[5];
                break;
            case 6:
                matchedName = imagesName[6];
                break;
            case 7:
                matchedName = imagesName[7];
                break;

            default:
                System.out.println("Eşleşme Bulunamadı");
                break;
        }
        textview1.setText(matchedName);

    }
}
