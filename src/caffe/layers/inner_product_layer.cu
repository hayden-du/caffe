#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype, typename Mtype>
void InnerProductLayer<Dtype,Mtype>::Forward_gpu(const vector<BlobBase*>& bottom,
    const vector<BlobBase*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data_base<Dtype>();
  Dtype* top_data = top[0]->mutable_gpu_data_base<Dtype>();
  const Dtype* weight = this->blobs_[0]->template gpu_data_base<Dtype>();
  if (M_ == 1) {
    caffe_gpu_gemv<Dtype,Mtype>(CblasNoTrans, N_, K_, (Mtype)1.,
                         weight, bottom_data, (Mtype)0., top_data);
    if (bias_term_)
      caffe_gpu_axpy<Dtype,Mtype>(N_, bias_multiplier_.cpu_data()[0],
                            this->blobs_[1]->template gpu_data_base<Dtype>(), top_data);
  } else {
    caffe_gpu_gemm<Dtype,Mtype>(CblasNoTrans, CblasTrans, M_, N_, K_, (Mtype)1.,
                          bottom_data, weight, (Mtype)0., top_data);
    if (bias_term_)
      caffe_gpu_gemm<Dtype,Mtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Mtype)1.,
                            bias_multiplier_.gpu_data(),
                            this->blobs_[1]->template gpu_data_base<Dtype>(), (Mtype)1., top_data);
  }
}

template <typename Dtype, typename Mtype>
void InnerProductLayer<Dtype,Mtype>::Backward_gpu(const vector<BlobBase*>& top,
    const vector<bool>& propagate_down,
    const vector<BlobBase*>& bottom) {
  if (this->param_propagate_down_[0]) {
    const Dtype* top_diff = top[0]->gpu_diff_base<Dtype>();
    const Dtype* bottom_data = bottom[0]->gpu_data_base<Dtype>();
    // Gradient with respect to weight
    caffe_gpu_gemm<Dtype,Mtype>(CblasTrans, CblasNoTrans, N_, K_, M_, (Mtype)1.,
        top_diff, bottom_data, (Mtype)1., this->blobs_[0]->template mutable_gpu_diff_base<Dtype>());
  }
  if (bias_term_ && this->param_propagate_down_[1]) {
    const Dtype* top_diff = top[0]->gpu_diff_base<Dtype>();
    // Gradient with respect to bias
    caffe_gpu_gemv<Dtype,Mtype>(CblasTrans, M_, N_, (Mtype)1., top_diff,
        bias_multiplier_.gpu_data(), (Mtype)1.,
        this->blobs_[1]->template mutable_gpu_diff_base<Dtype>());
  }
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff_base<Dtype>();
    // Gradient with respect to bottom data
    caffe_gpu_gemm<Dtype,Mtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, (Mtype)1.,
        top_diff, this->blobs_[0]->template gpu_data_base<Dtype>(), (Mtype)0.,
        bottom[0]->mutable_gpu_diff_base<Dtype>());
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(InnerProductLayer);

}  // namespace caffe
