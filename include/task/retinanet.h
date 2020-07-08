#ifndef CHEETAH_INFER_TASK_RETINANET_H
#define CHEETAH_INFER_TASK_RETINANET_H

#include <vector>

#include <opencv2/opencv.hpp>
#include <NvOnnxParser.h>

#include "trtcommon/buffers.h"
#include "base_task.h"

namespace cheetahinfer{

    class RetinaNet: public BaseTask
    {
        template <typename T>
            using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

        public:
        RetinaNet(const samplesCommon::OnnxSampleParams& params);
        bool infer(const std::vector<std::string>& fps) override;
        bool verifyOutput() override;

        private:
        // variables
        int top_n_;
        int detections_per_im_;
        float score_thresh_;
        float nms_thresh_;
        std::vector<std::vector<float>> anchors_;

        // functions
        void setHp();
        void addPlugin(SampleUniquePtr<nvinfer1::INetworkDefinition> &network) override;
    };

} // namespace cheetahinfer

#endif
