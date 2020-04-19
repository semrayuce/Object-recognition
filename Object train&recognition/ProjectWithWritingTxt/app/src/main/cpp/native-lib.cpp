#include <jni.h>
#include <string>
#include <opencv2/opencv.hpp>
#include "opencv2/objdetect.hpp"
#include <stdlib.h>
#include <stdio.h>

using namespace cv;
using namespace std;
using namespace cv::ml;

extern "C"
{


//----------------------------------------FIND-CROP-CONTOUR---------------------------------------------

//findContourAndCropping fonksiyonunuzda resmimize aşağıdaki işlemleri uyguluyoruz ve Elde ettiğimiz yeni resmi addrDst adresine yazıyoruz.
// addrDst adresine yazdığımız için ImageProcessing'de bu adrese karşılık gelen değişkene yeni resmimizi atmış oluyoruz
//Convert Gray , Add GaussianBlur, Sharpen Image , Find Contours, Find Largest Area of Contours , Draw Rectangle around largest area and Crop Largest Area


void JNICALL
Java_com_example_dgrproject_dgrprojectwithwritingtxt_NativeClass_findContourAndCropping(JNIEnv *env,
                                                                                        jobject instance,
                                                                                        jlong addrSrc,
                                                                                        jlong addrDst) {
    Mat threshold_output;
    Mat &src = *(Mat *) addrSrc;
    Mat &dst = *(Mat *) addrDst;
    int thresh = 100;
    vector <vector<Point>> contours;
    vector <Vec4i> hierarchy;
    int largest_area = 0;
    int largest_contour_index = 0;
    Rect bounding_rect;
    Mat gray;

    cv::rotate(src, src, ROTATE_90_CLOCKWISE);
    cvtColor(src, gray, CV_BGR2GRAY);

    // Sharp işlemi için gri resme filtre uyguluyoruz, böylece Largest area bulma işleminde daha iyi sonuç aldık
    float kdata[] = {1, 4, 6, -1, 3, 5, -1, -2, 2};
    Mat kernel(3, 3, CV_32F, kdata), filter;
    filter2D(gray, filter, dst.depth(), kernel, Point(-1, -1), 0, BORDER_DEFAULT);

    GaussianBlur(gray, gray, Size(3, 3), 0, 0, BORDER_DEFAULT);

    addWeighted(gray, 1.5, filter, -0.5, 0,
                dst); //Blurlanmış ve Filtrelenmiş resmimizin ağırlıkları toplamını dst'ye atıyoruz

    gray.release(); // İşimiz biten değişkenleri bellekten siliyoruz
    filter.release();

    /// Detect edges using Threshold
    threshold(dst, threshold_output, thresh, 255, THRESH_BINARY_INV + THRESH_OTSU); //
    /// Find contours
    findContours(threshold_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE,
                 Point(0, 0));
    threshold_output.release();
    /// Approximate contours to polygons + get bounding rects
    vector <vector<Point>> contours_poly(contours.size());
    vector <Rect> boundRect(contours.size());
    vector <Point2f> center(contours.size());
    vector<float> radius(contours.size());

    for (int i = 0; i < contours.size(); i++) {
        approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
        boundRect[i] = boundingRect(Mat(contours_poly[i]));
        minEnclosingCircle((Mat) contours_poly[i], center[i], radius[i]);
    }


    for (int i = 0; i < contours.size(); i++) // iterate through each contour.
    {
        double a = contourArea(contours[i], false);  //  Find the area of contour
        if (a > largest_area) {
            largest_area = a;
            largest_contour_index = i;     //Store the index of largest contour
            bounding_rect = boundingRect(
                    contours[i]); // Find the bounding rectangle for biggest contour
            //bounding_rect içerisinde sol üst köşenin x - y noktalarını ve sağ alt köşenin x-y noktalarını tutmaktadır.
        }

    }

    /// Draw polygonal contour + bonding rects
    drawContours(dst, contours_poly, largest_contour_index, Scalar(0, 0, 0), 2, 8, vector<Vec4i>(),
                 0, Point());
    //rectangle(dst, bounding_rect, Scalar(255, 255, 255), 0, 0, 0); //dst üzerinde bounding_rect'in tuttuğu kordinatlara uygun bir dikdörtgen çiziliyor
    dst = dst(
            bounding_rect);  // dst içerisinden bounding_rect alanını kırpıyoruz ve yine dst 'ye atıyoruz
}




//----------------------------------------SCALE---------------------------------------------



void JNICALL
Java_com_example_dgrproject_dgrprojectwithwritingtxt_NativeClass_scalingWithAspectRatio(JNIEnv *env,
                                                                                        jobject instance,
                                                                                        jlong image,
                                                                                        jint targetWidth,
                                                                                        jint targetHeight) {

    Mat &scaleImage = *(Mat *) image;

    int width = scaleImage.cols;
    int height = scaleImage.rows;

    cv::Mat resize = cv::Mat(targetHeight, targetWidth, scaleImage.type(), Scalar(255, 255, 255));


    // int max_dim = (width >= height) ? width : height;
    float scale = ((float) targetWidth) / width;
    float scale1 = ((float) targetHeight) / height;
    cv::Rect roi;

    if (width < 300 && height < 400) {
        roi.width = width;
        roi.height = height;
        roi.y = (targetHeight - height) / 2;
        roi.x = (targetWidth - width) / 2;
        cv::resize(scaleImage, resize(roi), roi.size());
    } else if (width >= height) {
        roi.width = targetWidth;
        roi.height = height * scale;
        roi.y = (targetHeight - roi.height) / 2;
        roi.x = 0;
        if (roi.height > targetHeight) {
            float scale2 = ((float) targetHeight) / roi.height;
            roi.width = roi.width * scale2;
            roi.height = targetHeight;
            roi.x = (targetWidth - roi.width) / 2;
            roi.y = 0;
        }
        cv::resize(scaleImage, resize(roi), roi.size());
    } else {
        roi.height = targetHeight;
        roi.width = width * scale1;
        roi.x = (targetWidth - roi.width) / 2;
        roi.y = 0;
        if (roi.width > targetWidth) {
            float scale3 = ((float) targetWidth) / roi.width;
            roi.width = targetWidth;
            roi.height = roi.height * scale3;
            roi.x = 0;
            roi.y = (targetHeight - roi.height) / 2;
        }
        cv::resize(scaleImage, resize(roi), roi.size());
    }

    scaleImage = resize;

}




//----------------------------------------SOBEL---------------------------------------------


// Kenarların belirginleşmesi için sobel alroritması uyguluyoruz.

void JNICALL
Java_com_example_dgrproject_dgrprojectwithwritingtxt_NativeClass_sobelFilter(JNIEnv *env,
                                                                             jobject instance,
                                                                             jlong addrSrc,
                                                                             jlong addrDst) {

    Mat &src = *(Mat *) addrSrc;
    Mat &dst = *(Mat *) addrDst;
    int scale = 1;
    int delta = 0;
    int ddepth = CV_16S;

    /// Generate grad_x and grad_y
    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;

    /// Gradient X
    //Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
    Sobel(src, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
    convertScaleAbs(grad_x, abs_grad_x);

    /// Gradient Y
    //Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
    Sobel(src, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
    convertScaleAbs(grad_y, abs_grad_y);

    /// Total Gradient (approximate)
    addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, dst);

}

//----------------------------------------LBPWithHistogram---------------------------------------------




void LBP(const Mat &src, Mat &dst);
void
uniformPatternSpatialHistogram(const Mat &src, Mat &hist, int numPatterns, int gridX, int gridY,
                               int overlap);
Mat getFeatureVectorMat(Mat &spatial_hist);

void JNICALL
Java_com_example_dgrproject_dgrprojectwithwritingtxt_NativeClass_lbpWithHistogram(JNIEnv *env,
                                                                                  jobject instance,
                                                                                  jlong addrSrc,
                                                                                  jlong addrDst) {
    Mat &src = *(Mat *) addrSrc;
    Mat &dst = *(Mat *) addrDst;
    Mat lbp_image, spatial_histogram, feature_vector;

    LBP(src, lbp_image);//Klasik LBP
    // LBP yöntemi ile resim üzerinde daha belirgin pixelli resim elde ediyoruz

    uniformPatternSpatialHistogram(lbp_image, spatial_histogram, 256, 10, 10, 0);
    //LBP resmi üzerinden histogram çıkarıyoruz

    feature_vector = getFeatureVectorMat(spatial_histogram);
    // LBP resminden çıkan histogram içerisindeki featureları buluyoruz ve Mat objesi olarak alıp dst'ye atıyoruz

    dst = feature_vector;
}


// LBP yöntemi ile resim üzerinde daha belirgin pixelli resim elde ediyoruz
void LBP(const Mat &src, Mat &dst) {

    dst = Mat::zeros(src.rows - 2, src.cols - 2, CV_8UC1);
    for (int i = 1; i < (src.rows - 1); i++) {
        for (int j = 1; j < (src.cols - 1); j++) {
            uchar center = src.at<uchar>(i, j);
            unsigned char code = 0;
            code |= (src.at<uchar>(i - 1, j - 1) > center) << 7;
            code |= (src.at<uchar>(i - 1, j) > center) << 6;
            code |= (src.at<uchar>(i - 1, j + 1) > center) << 5;
            code |= (src.at<uchar>(i, j + 1) > center) << 4;
            code |= (src.at<uchar>(i + 1, j + 1) > center) << 3;
            code |= (src.at<uchar>(i + 1, j) > center) << 2;
            code |= (src.at<uchar>(i + 1, j - 1) > center) << 1;
            code |= (src.at<uchar>(i, j - 1) > center) << 0;
            dst.at<uchar>(i - 1, j - 1) = code;
        }
    }
}




//-------LBP-Histogramını çıkarmak için işlemler

vector<int> convertToBinary(int x) {
    vector<int> result(8, 0);

    int idx = 0;
    while (x != 0) {
        result[idx] = x % 2;
        ++idx;
        x /= 2;
    }

    reverse(result.begin(), result.end());
    return result;
}

int countTransitions(vector<int> x) {
    int result = 0;
    for (int i = 0; i < 8; ++i)
        result += (x[i] != x[(i + 1) % 8]);
    return result;
}

Mat uniformPatternHistogram(const Mat &src, int numPatterns) {
    Mat hist;
    hist = Mat::zeros(1, (numPatterns + 1), CV_32SC1);

    for (int i = 0; i < numPatterns; ++i) {
        if (countTransitions(convertToBinary(i)) > 2)
            hist.at<int>(0, i) = -1;
    }

    for (int i = 0; i < src.rows; ++i) {
        for (int j = 0; j < src.cols; ++j) {
            int bin = src.at<uchar>(i, j);
            if (hist.at<int>(0, bin) == -1)
                hist.at<int>(0, numPatterns) += 1;
            else
                hist.at<int>(0, bin) += 1;
        }
    }
    return hist;
}

void uniformPatternSpatialHistogram(const Mat &src, Mat &hist, int numPatterns,
                                    int gridX, int gridY, int overlap) {

    int width = src.cols;
    int height = src.rows;
    vector <Mat> histograms;

    Size window = Size(static_cast<int>(floor(src.cols / gridX)),
                       static_cast<int>(floor(src.rows / gridY)));

    for (int x = 0; x <= (width - window.width); x += (window.width - overlap)) {
        for (int y = 0; y <= (height - window.height); y += (window.height - overlap)) {
            Mat cell = Mat(src, Rect(x, y, window.width, window.height));
            histograms.push_back(uniformPatternHistogram(cell, numPatterns));
        }
    }

    hist.create(1, histograms.size() * (numPatterns + 1), CV_32SC1);
    for (int histIdx = 0; histIdx < histograms.size(); ++histIdx) {
        for (int valIdx = 0; valIdx < (numPatterns + 1); ++valIdx) {
            int y = (histIdx * (numPatterns + 1)) + valIdx;
            hist.at<int>(0, y) = histograms[histIdx].at<int>(valIdx);
        }
    }
}

// LBP resminden çıkan histogram içerisindeki featureları buluyoruz ve Mat objesi olarak return ediyoruz
Mat getFeatureVectorMat(Mat &spatial_hist) {
    Mat feature_vector;
    for (int j = 0; j < spatial_hist.cols; ++j) {
        if (spatial_hist.at<int>(0, j) != -1) {
            feature_vector.push_back(spatial_hist.at<int>(0, j));
        }

    }
    cv::Mat feature_vector_tp = cv::Mat(feature_vector.cols, feature_vector.rows, CV_32F);
    cv::transpose(feature_vector, feature_vector_tp);
    return feature_vector_tp;
}


//----------------------------------------SVM---------------------------------------------

Mat asRowMatrix(const vector <Mat> &src, int rtype);
void SVMevaluate(Mat &testResponse, float &count, float &accuracy, vector<int> &testLabels);

float JNICALL
Java_com_example_dgrproject_dgrprojectwithwritingtxt_NativeClass_svm(JNIEnv *env, jobject instance,
                                                                     jlongArray tempobjadrOfTrain,
                                                                     jlongArray tempobjadrOfTest) {

    vector <Mat> trainimgs;
    vector <Mat> testimgs;
    vector<int> trainLabels, testLabels;
    //ImageProcessing sınıfından trainsetlerin adreslerini gönderdik



    jsize a_len = env->GetArrayLength(tempobjadrOfTrain);
    jlong *traindata = env->GetLongArrayElements(tempobjadrOfTrain, 0);

    //Bu adreslerde bulunan objeleri alıyoruz ve buradaki vector<Mat>  türündeki trainimgs listemize ekliyoruz
    for (int k = 0; k < a_len; k++) {
        Mat &newimage = *(Mat *) traindata[k];
        trainimgs.push_back(newimage);
    }


    jsize a_lenTest = env->GetArrayLength(tempobjadrOfTest);
    jlong *testdata = env->GetLongArrayElements(tempobjadrOfTest, 0);

    //Bu adreslerde bulunan objeleri alıyoruz ve buradaki vector<Mat>  türündeki trainimgs listemize ekliyoruz
    for (int k = 0; k < a_lenTest; k++) {
        Mat &newimage2 = *(Mat *) testdata[k];
        testimgs.push_back(newimage2);
    }

    Mat trainimgsMat = asRowMatrix(trainimgs,
                                   CV_32FC1);// vector<Mat>'ı Mat türünde bir objeye çevitiyoruz
    Mat testimgsMat = asRowMatrix(testimgs,
                                  CV_32FC1); // test resmimiz Mat türünde fakat trainimgsMat ile aynı type 'ta olması gerektiği için bu fonksiyona yolluyoruz
    //

    //train Label'larını oluşturuyoruz
    float digitClassNumber = 0;
    for (int z = 0; z < trainimgs.size(); z++) {

        if (digitClassNumber == 8) { // her örnekten 4 resmimiz olduğu için
            digitClassNumber = 0;
        }
        trainLabels.push_back(digitClassNumber);
        digitClassNumber = digitClassNumber + 1;
    }

    digitClassNumber = 0;
    for (int z = 0; z < testimgs.size(); z++) {
        testLabels.push_back(digitClassNumber);
        digitClassNumber = digitClassNumber + 1;
    }


    Mat testResponse;

    // svm parametrelerini oluşturuyoruz
    Ptr <SVM> svm = SVM::create();
    svm->setType(SVM::C_SVC);
    svm->setKernel(SVM::INTER);
    svm->setC(1);
    svm->setGamma(0.5);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 240, 1e-6));

    // train işlemini gerçekleştiriyoruz
    svm->train(trainimgsMat, ROW_SAMPLE, trainLabels);

    //predict ile test resmimiz ile eşleşen train kümesindeki resmin label değerini elde ediyoruz
    svm->predict(testimgsMat, testResponse);


    float count = 0;
    float accuracy = 0;
    SVMevaluate(testResponse, count, accuracy, testLabels);

    return accuracy;

}


//vector<Mat> objesini Mat objesine çeviriyoruz

Mat asRowMatrix(const vector <Mat> &src, int rtype) {
    // Number of samples:
    double alpha = 1;
    double beta = 0;
    size_t n = src.size();
    // Return empty matrix if no matrices given:
    if (n == 0)
        return Mat();
    // dimensionality of (reshaped) samples
    size_t d = src[0].total();
    // Create resulting data matrix:
    Mat data(n, d, rtype);
    // Now copy data:
    for (int i = 0; i < n; i++) {
        //
        if (src[i].empty()) {
            string error_message = format(
                    "Image number %d was empty, please check your input data.", i);
            CV_Error(CV_StsBadArg, error_message);
        }
        // Make sure data can be reshaped, throw a meaningful exception if not!
        if (src[i].total() != d) {
            string error_message = format(
                    "Wrong number of elements in matrix #%d! Expected %d was %d.", i, d,
                    src[i].total());
            CV_Error(CV_StsBadArg, error_message);
        }
        // Get a hold of the current row:
        Mat xi = data.row(i);
        // Make reshape happy by cloning for non-continuous matrices:
        if (src[i].isContinuous()) {
            src[i].reshape(1, 1).convertTo(xi, rtype, alpha, beta);
        } else {
            src[i].clone().reshape(1, 1).convertTo(xi, rtype, alpha, beta);
        }
    }
    return data;
}


void SVMevaluate(Mat &testResponse, float &count, float &accuracy, vector<int> &testLabels) {

    for (int i = 0; i < testResponse.rows; i++) {
        //cout << testResponse.at<float>(i,0) << " " << testLabels[i] << endl;
        if (testResponse.at<float>(i, 0) == testLabels[i]) {
            count = count + 1;
        }
    }
    accuracy = (count / testResponse.rows) * 100;
}

void JNICALL
Java_com_example_dgrproject_dgrprojectwithwritingtxt_NativeClass_asRow(JNIEnv *env,
                                                                       jobject instance,
                                                                       jlong addrDst,
                                                                       jlongArray traindataaddr) {

    Mat &dst = *(Mat *) addrDst;
    vector <Mat> trainimgs;
    //ImageProcessing sınıfından trainsetlerin adreslerini gönderdik
    jsize a_len = env->GetArrayLength(traindataaddr);
    jlong *traindata = env->GetLongArrayElements(traindataaddr, 0);

    //Bu adreslerde bulunan objeleri alıyoruz ve buradaki vector<Mat>  türündeki trainimgs listemize ekliyoruz
    for (int k = 0; k < a_len; k++) {
        Mat &newimage = *(Mat *) traindata[k];
        trainimgs.push_back(newimage);
    }

    dst = asRowMatrix(trainimgs, CV_32FC1);// vector<Mat>'ı Mat türünde bir objeye çevitiyoruz

}

}


