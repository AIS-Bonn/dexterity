#include "cvrviewer.h"
#include <unistd.h>


// Default constructor (needed for stack-allocation in Cython)
CVRViewer::CVRViewer() {}

// Actual constructor that receives commandline args
CVRViewer::CVRViewer(int argc, char **argv, uint32_t renderSize[2]) {
    InitOpenVR();
    InitDeviceIdx();
    if (deviceIdx.find("tracker_0") != deviceIdx.end()) {
        tracker = OVRDevice(VRSystem, deviceIdx["tracker_0"], false);
        trackerFound = true;
    }
    if (deviceIdx.find("headset") != deviceIdx.end()) {
        headset = OVRHeadset(VRSystem, deviceIdx["headset"], false, renderSize, argc, argv);
        headsetFound = true;
    }
    gloves = CSGTracker();

}

void CVRViewer::InitOpenVR() {
    VRSystem = vr::VR_Init(&InitError, vr::VRApplication_Scene);
    if (InitError != vr::VRInitError_None){
        VRSystem = NULL;
        std::cout << "Unable to initialise VR runtime: " << vr::VR_GetVRInitErrorAsEnglishDescription(InitError) << "\n";
        exit(EXIT_FAILURE);
    }
}

void CVRViewer::InitDeviceIdx() {

    std::map<std::string, int> numDevices;

    int maxAttempts = 1000;
    int connectionAttempts = 0;
    bool devicesFound = false;
    do {
        numDevices["headset"] = 0;
        numDevices["tracker"] = 0;
        numDevices["controller"] = 0;
        // Iterate through device indices and count what devices we find
        for (uint32_t idx=0; idx <  vr::k_unMaxTrackedDeviceCount; idx++) {
            if (!VRSystem->IsTrackedDeviceConnected(idx))
                continue;
            vr::ETrackedDeviceClass trackedDeviceClass = VRSystem->GetTrackedDeviceClass(idx);
            if (trackedDeviceClass == vr::ETrackedDeviceClass::TrackedDeviceClass_HMD) {
                deviceIdx["headset"] = idx;
                numDevices["headset"]++;
                //std::cout << "headset found";
            }
            else if (trackedDeviceClass == vr::ETrackedDeviceClass::TrackedDeviceClass_GenericTracker) {
                deviceIdx["tracker_" + std::to_string(numDevices["tracker"])] = idx;
                numDevices["tracker"]++;
                //std::cout << "tracker found";
            }
            else if (trackedDeviceClass == vr::ETrackedDeviceClass::TrackedDeviceClass_Controller) {
                deviceIdx["controller_" + std::to_string(numDevices["controller"])] = idx;
                numDevices["controller"]++;
            }
        }

        devicesFound = numDevices["headset"] == 1 && numDevices["tracker"] >= 1;
        if (!devicesFound) {
            std::cout << "Waiting for Tracker to be connected (Connection attempts: " << connectionAttempts << ")" << "\r" << std::flush;
            usleep(500000);
            connectionAttempts++;
        }
    } while (!devicesFound && connectionAttempts < maxAttempts);

    if (!devicesFound) {
        std::cout << "Could not connect to VR Headset and Tracker";
        exit(EXIT_FAILURE);
    }
}

float* CVRViewer::GetHeadsetPose() {
    if (headsetFound) {
        return headset.GetDevicePose();
    }
    else {
        std::cout << "No headset was found. Cannot get headset pose.";
        return defaultPose;
    }

}

float* CVRViewer::GetTrackerPose() {
    if (trackerFound) {
        return tracker.GetDevicePose();
    }
    else {
        std::cout << "No tracker was found. Cannot get tracker pose.";
        return defaultPose;
    }
}

float* CVRViewer::GetLeftEyeTransform() {
    return headset.GetLeftEyeTransform();
}

float* CVRViewer::GetRightEyeTransform() {
    return headset.GetRightEyeTransform();
}

float CVRViewer::GetLeftEyeFov() {
    return headset.GetLeftEyeFov();
}

float CVRViewer::GetRightEyeFov() {
    return headset.GetRightEyeFov();
}

void CVRViewer::SubmitVRCameraImages(uint8_t leftEyeImage[], uint8_t rightEyeImage[]) {
    headset.SubmitImages(leftEyeImage, rightEyeImage);
}

std::vector<std::vector<float>> CVRViewer::GetGloveSensorAngles(bool isRight) {
    return gloves.GetSensorAngles(isRight);
}

std::vector<float> CVRViewer::GetGloveFlexions(bool isRight) {
    return gloves.GetFlexions(isRight);
}

void CVRViewer::SendHapticFeedback(bool isRight, std::vector<int> Buzz, std::vector<int> ForceFeedback) {
    return gloves.SendHapticFeedback(isRight, Buzz, ForceFeedback);
}


// Destructor
CVRViewer::~CVRViewer() {
    //if (VRSystem != NULL){
    //    vr::VR_Shutdown();
    //    VRSystem = NULL;
    //}
}
