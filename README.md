# Object-recognition
Object recognition projects: Creating train set and recognizing the object.

Used technologies: Android, OpenCv 3.2.0
There are 2 projects here. In the ProjectWithTxtFile, used data set as Mat to print .txt file. 
In the ProjectWithCamera all image processing steps are implemented and the train file that is written in mat form is being read. The images that are read through camera are processed and compared with train file to be written according to the object name that matches train string.

NOTES:
1)When creating a new Android project, you must click the Include C++ support part to create the CMake file on the initial creation screen.
2)After the project is created or when the path of the existing project changes, we need to change the path that we give to include_directories() in the Cmake file. Ex:  include_directories(C:/YOUR_PATH/OpenCV-android-sdk/sdk/native/jni/include)
