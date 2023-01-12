# distutils: language = c++


from cvrviewer cimport CVRViewer
import numpy as np
from libc.stdlib cimport malloc, free
from libc.stdint cimport uint8_t
from libc.stdint cimport uint32_t



cdef class VRViewer:
    cdef CVRViewer cvrviewer
    cdef uint32_t recommended_width
    cdef uint32_t recommended_height
    cdef uint8_t[:, :, :] left_eye_image_memview
    cdef uint8_t[:, :, :] right_eye_image_memview
    def __init__(self, render_size, left_eye_image, right_eye_image):
        # Define command line args needed for graphics libraries
        args = []
        cdef char ** c_argv
        args = [b'calling_from_cython'] + [bytes(x) for x in args]
        c_argv = < char ** > malloc(sizeof(char *) * len(args))
        for idx, s in enumerate(args):
            c_argv[idx] = s

        cdef uint32_t c_render_size[2]
        c_render_size = render_size
        self.cvrviewer = CVRViewer(len(args), c_argv, c_render_size)
        self._init_eye_image_memview(left_eye_image, right_eye_image)

    def _init_eye_image_memview(self, left_eye_image, right_eye_image):
        self.left_eye_image_memview = left_eye_image
        self.right_eye_image_memview = right_eye_image

    def get_headset_pose(self):
        cdef float c_pose_array[7]
        c_pose_array = self.cvrviewer.GetHeadsetPose()
        cdef float [:] c_pose_array_memview = c_pose_array
        return np.array(c_pose_array_memview)

    def get_tracker_pose(self):
        cdef float c_pose_array[7]
        c_pose_array = self.cvrviewer.GetTrackerPose()
        cdef float [:] c_pose_array_memview = c_pose_array
        return np.array(c_pose_array_memview)

    def get_left_eye_transform(self):
        cdef float c_left_eye_transform[3]
        c_left_eye_transform = self.cvrviewer.GetLeftEyeTransform()
        cdef float [:] c_left_eye_transform_view = c_left_eye_transform
        return np.array(c_left_eye_transform_view)

    def get_right_eye_transform(self):
        cdef float c_right_eye_transform[3]
        c_right_eye_transform = self.cvrviewer.GetRightEyeTransform()
        cdef float [:] c_right_eye_transform_view = c_right_eye_transform
        return np.array(c_right_eye_transform_view)

    def get_left_eye_fov(self):
        return self.cvrviewer.GetLeftEyeFov()

    def get_right_eye_fov(self):
        return self.cvrviewer.GetRightEyeFov()

    def submit_vr_camera_images(self):
        self.cvrviewer.SubmitVRCameraImages(
            &self.left_eye_image_memview[0, 0, 0],
            &self.right_eye_image_memview[0, 0, 0]
        )

    def get_glove_sensor_angles(self, is_right):
        sensor_angles = self.cvrviewer.GetGloveSensorAngles(is_right)
        return sensor_angles

    def get_glove_flexions(self, is_right):
        flexions = self.cvrviewer.GetGloveFlexions(is_right)
        return flexions