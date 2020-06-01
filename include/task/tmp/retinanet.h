#ifndef CHEETAH_INFER_CLASSIFIER_H
#define CHEETAH_INFER_CLASSIFIER_H

#include <vector>

#include <opencv2/opencv.hpp>
#include <NvOnnxParser.h>

#include "trtcommon/buffers.h"


namespace cheetahinfer{

class RetinaNet
{
    template <typename T>
    using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

public:
    RetinaNet(const samplesCommon::OnnxSampleParams& params)
        : mParams(params)
        , mEngine(nullptr)
    {
        _set_hp();
		_prepare();
    }

    bool build(bool isFromOnnx);

    bool infer();

private:
    samplesCommon::OnnxSampleParams mParams; //!< The parameters for the sample.

    nvinfer1::Dims mInputDims;  //!< The dimensions of the input to the network.
    std::vector<nvinfer1::Dims> mOutputDimsVec; //!< The dimensions of the output to the network.
    int mNumber{0};             //!< The number to classify
    cudaStream_t _stream = nullptr;
	void _prepare();
	void _set_hp();

    std::shared_ptr<nvinfer1::ICudaEngine> mEngine; //!< The TensorRT engine used to run the network

    int top_n_;
	int detections_per_im_;
    float score_thresh_;
    float nms_thresh_;
    std::vector<std::vector<float>> anchors_;

    cv::Mat orig_image_;

    bool constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
        SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
        SampleUniquePtr<nvonnxparser::IParser>& parser);

	bool readImage(const std::string fp, float* host_data);

    bool processInput(const samplesCommon::BufferManager& buffers);

    bool verifyOutput(const samplesCommon::BufferManager& buffers);

    bool buildFromOnnx();

    bool buildFromPlan();
};

}
#endif
