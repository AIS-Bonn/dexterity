from libc.stdint cimport uint8_t
from libc.stdint cimport uint32_t
from libcpp cimport bool, string
from libcpp.vector cimport vector

cdef extern from "cvrviewer.cpp":
    pass

# Declare the class with cdef
cdef extern from "cvrviewer.h":
    cdef cppclass CVRViewer:
        CVRViewer() except +
        CVRViewer(int argc, char **argv, uint32_t renderSize[2]) except +
        float* GetHeadsetPose() except +
        float* GetTrackerPose() except +
        float* GetLeftEyeTransform() except +
        float* GetRightEyeTransform() except +
        float GetLeftEyeFov() except +
        float GetRightEyeFov() except +
        void SubmitVRCameraImages(uint8_t leftEyeImage[], uint8_t rightEyeImage[]) except +
        vector[vector[float]] GetGloveSensorAngles(bool isRight) except +
        vector[float] GetGloveFlexions(bool isRight) except +
