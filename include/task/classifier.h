#ifndef CHEETAH_INFER_CLASSIFIER_H
#define CHEETAH_INFER_CLASSIFIER_H

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
    //    : mParams(params)
    //    , mEngine(nullptr)
    //{
    //	_prepare();
    //}
    bool infer(std::string fp);
    bool verifyOutput(const samplesCommon::BufferManager& buffers);

};

}
#endif
