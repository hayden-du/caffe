#ifdef USE_CUDNN
#include <algorithm>
#include <cfloat>
#include <vector>
#include <cudnn.h>
#include <cudnnSharedTest.h>

#include "thrust/device_vector.h"

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype, typename Mtype>
void CuDNNBatchNormLayer<Dtype,Mtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const void* scale_data = this->blobs_[0]->gpu_data();
  const void* bias_data = this->blobs_[1]->gpu_data();

  cudnnTensorDescriptor_t sbdesc = scale_bias_mean_var_desc_;
  Blob<float> f_scale_data, f_bias_data;
  cudnnTensorStruct fdesc;
  if (sizeof(Dtype) < 4) {
    CUDNN_CHECK(cudnnSetTensorNdDescriptor(&fdesc,
        CUDNN_DATA_FLOAT, sbdesc->nbDims, sbdesc->dimA, sbdesc->strideA));
    sbdesc = &fdesc;
    f_scale_data.ReshapeLike(*this->blobs_[0]);
    f_bias_data.ReshapeLike(*this->blobs_[1]);
    caffe_gpu_convert(f_scale_data.count(), this->blobs_[0]->gpu_data(),
        f_scale_data.mutable_gpu_data());
    caffe_gpu_convert(f_bias_data.count(), this->blobs_[1]->gpu_data(),
        f_bias_data.mutable_gpu_data());
    scale_data = f_scale_data.gpu_data();
    bias_data = f_bias_data.gpu_data();
  }

  Dtype* top_data = top[0]->mutable_gpu_data();
  Dtype* save_mean = save_mean_.mutable_gpu_data();
  Dtype* save_inv_var = save_inv_var_.mutable_gpu_data();

  if (this->phase_ == TRAIN) {
    // Call Batch normalization forward
    CUDNN_CHECK(cudnnBatchNormalizationForwardTraining(
      Caffe::cudnn_handle(),
      mode_,
      cudnn::dataType<Dtype>::one,
      cudnn::dataType<Dtype>::zero,
      bottom_desc_,
      bottom_data,
      bottom_desc_,
      top_data,
      sbdesc, //scale_bias_mean_var_desc_,
      scale_data,
      bias_data,
      1-this->moving_average_fraction_,
      this->blobs_[3]->mutable_gpu_data(),  // mean
      this->blobs_[4]->mutable_gpu_data(),  // variance
      epsilon_,
      save_mean,
      save_inv_var));
  } else if (this->phase_ == TEST) {
    CUDNN_CHECK(cudnnBatchNormalizationForwardInference(
      Caffe::cudnn_handle(),
      mode_,
      cudnn::dataType<Dtype>::one,
      cudnn::dataType<Dtype>::zero,
      bottom_desc_,
      bottom_data,
      bottom_desc_,
      top_data,
      sbdesc, //scale_bias_mean_var_desc_,
      scale_data,
      bias_data,
      this->blobs_[3]->gpu_data(),  // mean
      this->blobs_[4]->gpu_data(),  // variance
      epsilon_));
  } else {
    LOG(FATAL) << "Unknown phase";
  }
}

template <typename Dtype, typename Mtype>
void CuDNNBatchNormLayer<Dtype,Mtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_data = top[0]->gpu_data();
  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* save_mean = save_mean_.gpu_data();
  const Dtype* save_inv_var = save_inv_var_.gpu_data();

  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const void* scale_data = this->blobs_[0]->gpu_data();
  void* scale_diff = this->blobs_[0]->mutable_gpu_diff();
  void* bias_diff = this->blobs_[1]->mutable_gpu_diff();

  cudnnTensorDescriptor_t sbdesc = scale_bias_mean_var_desc_;
  Blob<float> f_scale_data, f_scale_diff, f_bias_diff;
  cudnnTensorStruct fdesc;
  if (sizeof(Dtype) < 4) {
    CUDNN_CHECK(cudnnSetTensorNdDescriptor(&fdesc,
        CUDNN_DATA_FLOAT, sbdesc->nbDims, sbdesc->dimA, sbdesc->strideA));
    sbdesc = &fdesc;
    f_scale_data.ReshapeLike(*this->blobs_[0]);
    f_scale_diff.ReshapeLike(*this->blobs_[0]);
    f_bias_diff.ReshapeLike(*this->blobs_[1]);
    caffe_gpu_convert(f_scale_data.count(), this->blobs_[0]->gpu_data(),
        f_scale_data.mutable_gpu_data());
    caffe_gpu_convert(f_scale_diff.count(), this->blobs_[0]->gpu_diff(),
        f_scale_diff.mutable_gpu_diff());
    caffe_gpu_convert(f_bias_diff.count(), this->blobs_[1]->gpu_diff(),
        f_bias_diff.mutable_gpu_diff());
    scale_data = f_scale_data.gpu_data();
    scale_diff = f_scale_diff.mutable_gpu_diff();
    bias_diff = f_bias_diff.mutable_gpu_diff();
  }

  // call Batch Normalization Backward
  CUDNN_CHECK(cudnnBatchNormalizationBackward(
      Caffe::cudnn_handle(),
      mode_,
      cudnn::dataType<Dtype>::one,
      cudnn::dataType<Dtype>::zero,
      bottom_desc_,
      bottom_data,
      bottom_desc_,
      top_diff,
      bottom_desc_,
      bottom_diff,
      sbdesc, //scale_bias_mean_var_desc_,
      scale_data,
      scale_diff,
      bias_diff,
      this->epsilon_,
      save_mean,
      save_inv_var));

    if (sizeof(Dtype) < 4) {
      caffe_gpu_convert(f_scale_diff.count(), f_scale_diff.gpu_diff(),
          this->blobs_[0]->mutable_gpu_diff());
      caffe_gpu_convert(f_bias_diff.count(), f_bias_diff.gpu_diff(),
          this->blobs_[1]->mutable_gpu_diff());
    }
}

INSTANTIATE_LAYER_GPU_FUNCS(CuDNNBatchNormLayer);

}  // namespace caffe
#endif
