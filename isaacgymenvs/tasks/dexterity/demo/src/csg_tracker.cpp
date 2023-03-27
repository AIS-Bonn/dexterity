class CSGTracker {
    private:
        // SenseGlove objects
        SGCore::SG::SenseGlove rightSenseGlove;
        SGCore::SG::SenseGlove leftSenseGlove;

        // SG structs storing sensor data
        SGCore::SG::SG_SensorData rightSG_SensorData;
        SGCore::SG::SG_SensorData leftSG_SensorData;
        SGCore::HandPose rightHandPose;
        SGCore::HandPose leftHandPose;

        // Models required for normalized flexion
        SGCore::SG::SG_HandProfile rightHandProfile = SGCore::SG::SG_HandProfile::Default(true);
        SGCore::SG::SG_HandProfile leftHandProfile = SGCore::SG::SG_HandProfile::Default(false);
        SGCore::Kinematics::BasicHandModel rightHandModel = SGCore::Kinematics::BasicHandModel::Default(true);
        SGCore::Kinematics::BasicHandModel leftHandModel = SGCore::Kinematics::BasicHandModel::Default(false);

        std::vector<float> rightFlexions;
        std::vector<float> leftFlexions;
        std::vector<std::vector<float>> rightSensorAngles;
        std::vector<std::vector<float>> leftSensorAngles;

        /* Connect to all available SenseGloves and retrieve the corresponding
         * SGCore::SG::SenseGlove objects to interact with them. */
        void InitSenseGloves() {
            // Start scanning if scan is not already running
            bool alreadyRunning = SGConnect::ScanningActive();
            if (!alreadyRunning) {
                SGConnect::Init();
            }

            int maxAttempts = 20;
            int connectionAttempts = 0;
            std::vector<SGCore::SG::SenseGlove> senseGloves;
            bool senseGlovesConnected = false;
            do {
                senseGloves = SGCore::SG::SenseGlove::GetSenseGloves();
                senseGlovesConnected = !senseGloves.empty();
                connectionAttempts++;
                if (!senseGlovesConnected) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(500));
                }
            }
            // Wait for SenseGloves to be found
            while (!senseGlovesConnected && connectionAttempts < maxAttempts);
            if (senseGlovesConnected) {
                // Fill class attributes with found SenseGloves
                for (int gloveIdx = 0; gloveIdx < senseGloves.size(); gloveIdx++) {
                    if (senseGloves[gloveIdx].IsRight()) {
                        rightSenseGlove = senseGloves[gloveIdx];
                    }
                    else {
                        leftSenseGlove = senseGloves[gloveIdx];
                    }
                }
            }
            else {
                std::cout << "Could not connect to SenseGlove.";
            }
        }

    public:
        /* Connect to available SenseGloves */
        CSGTracker() {
            InitSenseGloves();
        }

        ~CSGTracker() {}

        std::vector<std::vector<float>> GetSensorAngles(bool isRight) {
            if (isRight) {
                if (rightSenseGlove.GetSensorData(rightSG_SensorData)) {
                    rightSensorAngles = rightSG_SensorData.sensorAngles;
                }
                else {
                    std::cout << "Unable to GetSensorData for right SenseGlove.";
                }
                return rightSensorAngles;
            }
            else {
                if (leftSenseGlove.GetSensorData(leftSG_SensorData)) {
                    leftSensorAngles = leftSG_SensorData.sensorAngles;
                }
                else {
                    std::cout << "Unable to GetSensorData for left SenseGlove.";
                }
                return leftSensorAngles;
            }
        }

        std::vector<float> GetFlexions(bool isRight) {
            if (isRight) {
                if (rightSenseGlove.GetHandPose(rightHandModel, rightHandProfile, rightHandPose)) {
                    rightFlexions = rightHandPose.GetNormalizedFlexion();
                }
                else {
                    std::cout << "Unable to GetHandPose for right SenseGlove.";
                }
                return rightFlexions;
            }
            else {
                if (leftSenseGlove.GetHandPose(leftHandModel, leftHandProfile, leftHandPose)) {
                    leftFlexions = leftHandPose.GetNormalizedFlexion();
                }
                else {
                    std::cout << "Unable to GetHandPose for left SenseGlove.";
                }
                return leftFlexions;
            }
        }

        void SendHapticFeedback(bool isRight, std::vector<int> Buzz, std::vector<int> ForceFeedback) {
            if (isRight) {
                rightSenseGlove.SendHaptics(SGCore::Haptics::SG_BuzzCmd(Buzz));
                rightSenseGlove.SendHaptics(SGCore::Haptics::SG_FFBCmd(ForceFeedback));
            }
            else {
                rightSenseGlove.SendHaptics(SGCore::Haptics::SG_BuzzCmd(Buzz));
                rightSenseGlove.SendHaptics(SGCore::Haptics::SG_FFBCmd(ForceFeedback));
            }
        }

};
