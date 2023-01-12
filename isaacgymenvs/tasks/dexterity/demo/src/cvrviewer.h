#ifndef CVRVIEWER_H
#define CVRVIEWER_H

#include <cmath>
#include <iostream>
#include <string>
#include <thread>
#include <chrono>
#include <map>
#include <GL/glew.h>
#include <GL/glut.h>
#include "../contrib/openvr.h"
#include "../contrib/SenseGlove-API/Core/SGConnect/incl/SGConnect.h"
#include "../contrib/SenseGlove-API/Core/SGCoreCpp/incl/SenseGlove.h"

#include "ovr_headset.cpp"
#include "csg_tracker.cpp"


class CVRViewer {
    private:
        // VR System
        vr::IVRSystem* VRSystem = NULL;
        vr::EVRInitError InitError;
        vr::EVRCompositorError CompositorError;

        // Define OpenVR and SenseGlove devices
        OVRDevice tracker;
        OVRHeadset headset;
        CSGTracker gloves;

        bool trackerFound = false;
        bool headsetFound = false;

        float defaultPose[7] = {0., 0., 0., 1., 0., 0., 0.};

        // 4 is arbitrary number of max tracked devices
        std::map<std::string, uint32_t> deviceIdx;
        void InitOpenVR();
        void InitDeviceIdx();

    public:
        CVRViewer();
        CVRViewer(int argc, char **argv, uint32_t renderSize[2]);
        ~CVRViewer();
        // OpenVR
        float* GetHeadsetPose();
        float* GetTrackerPose();
        float* GetLeftEyeTransform();
        float* GetRightEyeTransform();
        float GetLeftEyeFov();
        float GetRightEyeFov();
        void SubmitVRCameraImages(uint8_t leftEyeImage[], uint8_t rightEyeImage[]);
        // SenseGlove
        std::vector<std::vector<float>> GetGloveSensorAngles(bool isRight);
        std::vector<float> GetGloveFlexions(bool isRight);
};

#endif
