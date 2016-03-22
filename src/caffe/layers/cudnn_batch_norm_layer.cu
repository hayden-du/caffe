#ifdef USE_CUDNN
#include <algorithm>
#include <cfloat>
#include <vector>

#include "thrust/device_vector.h"

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype, typename Mtype>
void CuDNNBatchNormLayer<Dtype,Mtype>::Forward_gpu(
    const vector<BlobBase*>& bottom,
    const vector<BlobBase*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data_base<Dtype>();
  const void* scale_data = this->blobs_[0]->template gpu_data_base<Dtype>();
  const void* bias_data = this->blobs_[1]->template gpu_data_base<Dtype>();

  Blob<float> f_scale_data, f_bias_data;
  if (sizeof(Dtype) < 4) {
    f_scale_data.ReshapeLike(*this->blobs_[0]);
    f_bias_data.ReshapeLike(*this->blobs_[1]);
    caffe_gpu_convert(f_scale_data.count(),
        this->blobs_[0]->template gpu_data_base<Dtype>(),
        f_scale_data.mutable_gpu_data());
    caffe_gpu_convert(f_bias_data.count(),
        this->blobs_[1]->template gpu_data_base<Dtype>(),
        f_bias_data.mutable_gpu_data());
    scale_data = f_scale_data.gpu_data();
    bias_data = f_bias_data.gpu_data();
  }

  Dtype* top_data = top[0]->mutable_gpu_data_base<Dtype>();
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
      scale_bias_mean_var_desc_,
      scale_data,
      bias_data,
      1-this->moving_average_fraction_,
      this->blobs_[3]->template mutable_gpu_data_base<Dtype>(),  // mean
      this->blobs_[4]->template mutable_gpu_data_base<Dtype>(),  // variance
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
      scale_bias_mean_var_desc_,
      scale_data,
      bias_data,
      this->blobs_[3]->template gpu_data_base<Dtype>(),  // mean
      this->blobs_[4]->template gpu_data_base<Dtype>(),  // variance
      epsilon_));
  } else {
    LOG(FATAL) << "Unknown phase";
  }
}

template <typename Dtype, typename Mtype>
void CuDNNBatchNormLayer<Dtype,Mtype>::Backward_gpu(
    const vector<BlobBase*>& top,
    const vector<bool>& propagate_down, const vector<BlobBase*>& bottom) {
  const Dtype* top_data = top[0]->gpu_data_base<Dtype>();
  const Dtype* top_diff = top[0]->gpu_diff_base<Dtype>();
  const Dtype* bottom_data = bottom[0]->gpu_data_base<Dtype>();
  const Dtype* save_mean = save_mean_.gpu_data();
  const Dtype* save_inv_var = save_inv_var_.gpu_data();

  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff_base<Dtype>();
  const void* scale_data = this->blobs_[0]->template gpu_data_base<Dtype>();
  void* scale_diff = this->blobs_[0]->template mutable_gpu_diff_base<Dtype>();
  void* bias_diff = this->blobs_[1]->template mutable_gpu_diff_base<Dtype>();

  Blob<float> f_scale_data, f_scale_diff, f_bias_diff;
  if (sizeof(Dtype) < 4) {
    f_scale_data.ReshapeLike(*this->blobs_[0]);
    f_scale_diff.ReshapeLike(*this->blobs_[0]);
    f_bias_diff.ReshapeLike(*this->blobs_[1]);
    caffe_gpu_convert(f_scale_data.count(),
        this->blobs_[0]->template gpu_data_base<Dtype>(),
        f_scale_data.mutable_gpu_data());
    caffe_gpu_convert(f_scale_diff.count(),
        this->blobs_[0]->template gpu_diff_base<Dtype>(),
        f_scale_diff.mutable_gpu_diff());
    caffe_gpu_convert(f_bias_diff.count(),
        this->blobs_[1]->template gpu_diff_base<Dtype>(),
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
#if CUDNN_VERSION >= 4005
      cudnn::dataType<Dtype>::one,
      cudnn::dataType<Dtype>::one,
#endif
      bottom_desc_,
      bottom_data,
      bottom_desc_,
      top_diff,
      bottom_desc_,
      bottom_diff,
      scale_bias_mean_var_desc_,
      scale_data,
      scale_diff,
      bias_diff,
      this->epsilon_,
      save_mean,
      save_inv_var));

    if (sizeof(Dtype) < 4) {
      caffe_gpu_convert(f_scale_diff.count(), f_scale_diff.gpu_diff(),
          this->blobs_[0]->template mutable_gpu_diff_base<Dtype>());
      caffe_gpu_convert(f_bias_diff.count(), f_bias_diff.gpu_diff(),
          this->blobs_[1]->template mutable_gpu_diff_base<Dtype>());
    }
}

INSTANTIATE_LAYER_GPU_FUNCS(CuDNNBatchNormLayer);

}  // namespace caffe
#endif
