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

#include "timer.h"
#include "task/classifier.h"
#include "task/base_task.h"

namespace cheetahinfer{


Classifier::Classifier(const samplesCommon::OnnxSampleParams& params):
    BaseTask(params)
{
}

bool Classifier::infer(std::string img_fp)
{
    // Create RAII buffer manager object
	//std::cout<<"xxxxxxx buffer def"<<std::endl;
    samplesCommon::BufferManager buffers(engine_, params_.batch_size);

    inferCommon(img_fp, buffers);
    // Verify results
    if (!verifyOutput(buffers))
    {
        return false;
    }

    return true;
}

bool Classifier::verifyOutput(const samplesCommon::BufferManager& buffers)
{
    const int outputSize = output_dims_vec_[0].d[0];
    float* output = static_cast<float*>(buffers.getHostBufferByIndex(1));

    // Calculate Softmax
    float sum{0.0f};
    for (int i = 0; i < outputSize; i++)
    {
        output[i] = exp(output[i]);
        sum += output[i];
    }

    std::cout << "Output:" << std::endl;
    for (int i = 0; i < outputSize; i++)
    {
        output[i] /= sum;

        std::cout << " Prob " << i << "  " << std::fixed << std::setw(5) << std::setprecision(4) << output[i] << " "
                 << "Class " << i << ": " << std::string(int(std::floor(output[i] * 10 + 0.5f)), '*') << std::endl;
                 //<< "Class " << i << ": " << std::endl;
    }
    //gLogInfo << std::endl;

    return true;
}
}
