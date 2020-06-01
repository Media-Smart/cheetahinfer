#ifndef CHEETAH_INFER_CLASSIFIER_H
#define CHEETAH_INFER_CLASSIFIER_H

#include <NvOnnxParser.h>

#include "trtcommon/buffers.h"

namespace cheetahinfer{

class Classifier
{
    template <typename T>
    using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

public:
    Classifier(const samplesCommon::OnnxSampleParams& params)
        : mParams(params)
        , mEngine(nullptr)
    {
		_prepare();
    }

    bool build();

    bool infer();

private:
    samplesCommon::OnnxSampleParams mParams; //!< The parameters for the sample.

    nvinfer1::Dims mInputDims;  //!< The dimensions of the input to the network.
    nvinfer1::Dims mOutputDims; //!< The dimensions of the output to the network.
    int mNumber{0};             //!< The number to classify
    cudaStream_t _stream = nullptr;
	void _prepare();

    std::shared_ptr<nvinfer1::ICudaEngine> mEngine; //!< The TensorRT engine used to run the network

    bool constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
        SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
        SampleUniquePtr<nvonnxparser::IParser>& parser);

	bool readImage(const std::string fp, float* host_data);

    bool processInput(const samplesCommon::BufferManager& buffers);

    bool verifyOutput(const samplesCommon::BufferManager& buffers);
};

}
#endif
