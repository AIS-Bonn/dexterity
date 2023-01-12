#include "ovr_device.cpp"

/* Extends OVRDevice by functionality to submit images be displayed on the Headset */
class OVRHeadset: public OVRDevice {
    private:
        void InitEyeProperties() {
            InitEyeTransforms();
            InitEyeFovs();
        }

        void InitEyeTransforms() {
            // Get Eye Transforms from VRSystem
            vr::HmdMatrix34_t leftEyeTf= VRSystem->GetEyeToHeadTransform(vr::Eye_Left);
            vr::HmdMatrix34_t rightEyeTf= VRSystem->GetEyeToHeadTransform(vr::Eye_Right);
            // Save transforms to float arrays
            leftEyeTransform[0] = leftEyeTf.m[0][3];
            leftEyeTransform[1] = leftEyeTf.m[1][3];
            leftEyeTransform[2] = leftEyeTf.m[2][3];
            rightEyeTransform[0] = rightEyeTf.m[0][3];
            rightEyeTransform[1] = rightEyeTf.m[1][3];
            rightEyeTransform[2] = rightEyeTf.m[2][3];
        }

        void InitEyeFovs() {
            // Get Eye Projections from VRSystem
            vr::HmdMatrix44_t leftEyeProjection = VRSystem->GetProjectionMatrix(vr::Eye_Left, 0.1f, 30.0f);
            vr::HmdMatrix44_t rightEyeProjection = VRSystem->GetProjectionMatrix(vr::Eye_Right, 0.1f, 30.0f);
            // Calculate FoV and save to attribute
            leftEyeFov = fabs(atan(leftEyeProjection.m[2][2] / rightEyeProjection.m[1][1]))*2.0;
            rightEyeFov = fabs(atan(rightEyeProjection.m[2][2] / rightEyeProjection.m[1][1]))*2.0;
        }

        void InitGraphicsLibraries(int argc, char **argv) {
            InitGlut(argc, argv);
            InitGLEW();
            InitGlutTextures();
        }

        void InitGlut(int argc, char **argv) {
            glutInit(&argc, argv);
            glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
            glutInitWindowSize(renderSize[0], renderSize[1]);
            glutInitWindowPosition(0, 0);
            glutCreateWindow("Sim_App");
            glutHideWindow();
        }

        void InitGLEW() {
            GLenum res = glewInit();
            if (res != GLEW_OK){
                std::cout << "GLEW initialisation error: " << glewGetErrorString(res) << "\n";
                exit(EXIT_FAILURE);
            }
        }

        void InitGlutTextures() {
            glClearColor(0, 0, 0, 1);
            glClear(GL_COLOR_BUFFER_BIT);
            glGenTextures(1, &leftEyeTexture);
            glGenTextures(1, &rightEyeTexture);
            glFinish();
        }

        void RenderTextures() {
            // Initial Compositor check
            vr::TrackedDevicePose_t pose[vr::k_unMaxTrackedDeviceCount];
            CompositorError = vr::VRCompositor()->WaitGetPoses(pose, vr::k_unMaxTrackedDeviceCount, NULL, 0);
            CheckCompositorError();

            //Create texture from buffers
            vr::Texture_t leftEyeTex;
            leftEyeTex.handle = (void*)(uintptr_t) leftEyeTexture;
            leftEyeTex.eColorSpace = vr::ColorSpace_Linear;
            leftEyeTex.eType = vr::TextureType_OpenGL;

            vr::Texture_t rightEyeTex;
            rightEyeTex.handle = (void*)(uintptr_t) rightEyeTexture;
            rightEyeTex.eColorSpace = vr::ColorSpace_Linear;
            rightEyeTex.eType = vr::TextureType_OpenGL;

            // Submit textures
            CompositorError = vr::VRCompositor()->Submit(vr::Eye_Left, &leftEyeTex);
            CheckCompositorError();
            CompositorError = vr::VRCompositor()->Submit(vr::Eye_Right, &rightEyeTex);
            CheckCompositorError();
        }

        void CheckCompositorError() {
            if (CompositorError != vr::EVRCompositorError::VRCompositorError_None){
                VRSystem = NULL;
                std::cout << "Unable to manipulate compositor: " << static_cast<int>(CompositorError) << "\n";
                //exit(EXIT_FAILURE);
            }
        }

        vr::EVRCompositorError CompositorError;
        uint32_t renderSize[2];
        float leftEyeTransform[3];
        float rightEyeTransform[3];
        float leftEyeFov;
        float rightEyeFov;
        bool verbose;
        GLuint leftEyeTexture;
        GLuint rightEyeTexture;
        uint8_t* leftEyeBuffer = NULL;
        uint8_t* rightEyeBuffer = NULL;

    public:
        /* Default constructor */
        OVRHeadset() {}

        /* Call constructor of superclass and initialize rendering */
        OVRHeadset(vr::IVRSystem* VRSys, vr::TrackedDeviceIndex_t deviceIdx, bool verbose, uint32_t* rendSize, int argc, char **argv) : OVRDevice(VRSys, deviceIdx, verbose) {
            renderSize[0] = *rendSize;
            renderSize[1] = *(rendSize + 1);
            InitEyeProperties();
            InitGraphicsLibraries(argc, argv);
        }

        ~OVRHeadset() {}

        void SubmitImages(uint8_t leftEyeImage[], uint8_t rightEyeImage[]) {
            glBindTexture(GL_TEXTURE_2D, leftEyeTexture);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, renderSize[0], renderSize[1], 0, GL_RGBA, GL_UNSIGNED_BYTE, leftEyeImage);
            if (verboseInfo) {
                std::cout << "Manipulating OpenGL texture for left eye: " << glGetError() << "\n";
            }

            glBindTexture(GL_TEXTURE_2D, rightEyeTexture);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, renderSize[0], renderSize[1], 0, GL_RGBA, GL_UNSIGNED_BYTE, rightEyeImage);
            if (verboseInfo) {
                std::cout << "Manipulating OpenGL texture for right eye: " << glGetError() << "\n";
            }

            RenderTextures();
        }

        float* GetLeftEyeTransform() {
            return leftEyeTransform;
        }

        float* GetRightEyeTransform() {
            return rightEyeTransform;
        }

        float GetLeftEyeFov() {
            return leftEyeFov;
        }

        float GetRightEyeFov() {
            return rightEyeFov;
        }
};
