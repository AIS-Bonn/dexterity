#include "../contrib/openvr.h"
#include <cmath>
#include <iostream>

/* Wrapper class that returns pose information of tracked devices */
class OVRDevice {
    private:
        vr::TrackedDeviceIndex_t deviceIndex;
        vr::ETrackedDeviceClass deviceClass;
        vr::TrackedDevicePose_t devicePose;
        vr::VRControllerState_t deviceState;
        float devicePoseArr[7] = {0., 0., 0., 1., 0., 0., 0.};

        /* Retrieve current pose from VRSystem */
        void GetVRDevicePose() {
            if (deviceClass == vr::ETrackedDeviceClass::TrackedDeviceClass_HMD) {
                VRSystem->GetDeviceToAbsoluteTrackingPose(vr::TrackingUniverseStanding, 0, &devicePose, 1);
            }
            else if (deviceClass == vr::ETrackedDeviceClass::TrackedDeviceClass_GenericTracker) {
                VRSystem->GetControllerStateWithPose(vr::TrackingUniverseStanding, deviceIndex, &deviceState, sizeof(deviceState), &devicePose);
            }
            else {
                std::cout << "Unknown deviceClass" << deviceClass;
            }
        }

        /* Update array storing the current device pose */
        void UpdateDevicePoseArr() {
            vr::HmdMatrix34_t & m34 = devicePose.mDeviceToAbsoluteTracking;
            float x = m34.m[0][3];
            float y = m34.m[1][3];
            float z = m34.m[2][3];
            float q_w = sqrt(fmax(0, 1 + m34.m[0][0] + m34.m[1][1] + m34.m[2][2])) / 2;
            float q_x = sqrt(fmax(0, 1 + m34.m[0][0] - m34.m[1][1] - m34.m[2][2])) / 2;
            float q_y = sqrt(fmax(0, 1 - m34.m[0][0] + m34.m[1][1] - m34.m[2][2])) / 2;
            float q_z = sqrt(fmax(0, 1 - m34.m[0][0] - m34.m[1][1] + m34.m[2][2])) / 2;
            q_x = copysign(q_x, m34.m[2][1] - m34.m[1][2]);
            q_y = copysign(q_y, m34.m[0][2] - m34.m[2][0]);
            q_z = copysign(q_z, m34.m[1][0] - m34.m[0][1]);
            devicePoseArr[0] = x;
            devicePoseArr[1] = y;
            devicePoseArr[2] = z;
            devicePoseArr[3] = q_w;
            devicePoseArr[4] = q_x;
            devicePoseArr[5] = q_y;
            devicePoseArr[6] = q_z;
        }

    protected:
        vr::IVRSystem* VRSystem;
        bool verboseInfo;

    public:
        /* Default constructor */
        OVRDevice() {}

        /* Actual constructor */
        OVRDevice(vr::IVRSystem* VRSys, vr::TrackedDeviceIndex_t deviceIdx, bool verbose) {
            VRSystem = VRSys;
            deviceIndex = deviceIdx;
            deviceClass = VRSystem->GetTrackedDeviceClass(deviceIndex);
            verboseInfo = verbose;
        }

        /* Destructor */
        ~OVRDevice() {}

        /* Return pose of the device being tracked */
        float* GetDevicePose() {
            if (VRSystem->IsTrackedDeviceConnected(deviceIndex)) {
                GetVRDevicePose();
                // Update the devicePoseArr if the received pose is valid
                if (devicePose.bPoseIsValid) {
                    UpdateDevicePoseArr();
                }
            }
            // Warn if device is not connected
            else {
                std::cout << "deviceIndex " << deviceIndex << " is not connected." << std::endl;
            }
            return devicePoseArr;
        }
};