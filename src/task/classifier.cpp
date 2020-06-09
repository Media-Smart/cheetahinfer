//#include <cstdlib>
//#include <fstream>
#include <iostream>
//#include <sstream>

#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <NvOnnxConfig.h>
#include <NvOnnxParser.h>

#include "trtcommon/argsParser.h"
#include "trtcommon/buffers.h"
#include "trtcommon/logger.h"

#include "utils/misc.h"
#include "task/classifier.h"
#include "task/base_task.h"

namespace cheetahinfer{


Classifier::Classifier(const samplesCommon::OnnxSampleParams& params):
    BaseTask(params)
{
}

bool Classifier::infer(std::string img_fp)
{
    return inferCommon(img_fp, buffers_);
}

bool Classifier::verifyOutput()
{
    const int output_size = output_dims_vec_[0].d[0];
    float* output = static_cast<float*>(buffers_.getHostBufferByIndex(1));

    // Calculate Softmax
    float sum{0.0f};
    for (int i = 0; i < output_size; i++)
    {
        output[i] = exp(output[i]);
        sum += output[i];
    }

    gLogInfo << "Output:" << std::endl;
    for (int i = 0; i < output_size; i++)
    {
        output[i] /= sum;

        gLogInfo << " Prob " << i << "  " << std::fixed << std::setw(5) << std::setprecision(4) << output[i] << " "
                 << "Class " << i << ": " << std::string(int(std::floor(output[i] * 10 + 0.5f)), '*') << std::endl;
    }
    gLogInfo << std::endl;

    return true;
}

} // namespace cheetahinfer
