#ifndef CHEETAH_INFER_TASK_CLASSIFIER_H
#define CHEETAH_INFER_TASK_CLASSIFIER_H

#include <NvOnnxParser.h>

#include "trtcommon/buffers.h"
#include "base_task.h"

namespace cheetahinfer{

class Classifier: public BaseTask
{
    template <typename T>
    using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

public:
    Classifier(const samplesCommon::OnnxSampleParams& params);
    bool infer(const std::vector<std::string>& fps);
    bool verifyOutput();
};

} // namespace cheetahinfer

#endif
