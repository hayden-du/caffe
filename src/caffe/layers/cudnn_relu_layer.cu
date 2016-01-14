#ifdef USE_CUDNN
#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype, typename Mtype>
void CuDNNReLULayer<Dtype,Mtype>::Forward_gpu(const vector<BlobBase*>& bottom,
    const vector<BlobBase*>& top) {
  // Fallback to standard Caffe for leaky ReLU.
  if (ReLULayer<Dtype,Mtype>::layer_param_.relu_param().negative_slope() != 0) {
    return ReLULayer<Dtype,Mtype>::Forward_gpu(bottom, top);
  }

  const Dtype* bottom_data = bottom[0]->gpu_data_base<Dtype>();
  Dtype* top_data = top[0]->mutable_gpu_data_base<Dtype>();
  CUDNN_CHECK(cudnnActivationForward(Caffe::cudnn_handle(),
        CUDNN_ACTIVATION_RELU,
        cudnn::dataType<Dtype>::one,
        this->bottom_desc_, bottom_data,
        cudnn::dataType<Dtype>::zero,
        this->top_desc_, top_data));
}

template <typename Dtype, typename Mtype>
void CuDNNReLULayer<Dtype,Mtype>::Backward_gpu(const vector<BlobBase*>& top,
    const vector<bool>& propagate_down,
    const vector<BlobBase*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }

  // Fallback to standard Caffe for leaky ReLU.
  if (ReLULayer<Dtype,Mtype>::layer_param_.relu_param().negative_slope() != 0) {
    return ReLULayer<Dtype,Mtype>::Backward_gpu(top, propagate_down, bottom);
  }

  const Dtype* top_data = top[0]->gpu_data_base<Dtype>();
  const Dtype* top_diff = top[0]->gpu_diff_base<Dtype>();
  const Dtype* bottom_data = bottom[0]->gpu_data_base<Dtype>();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff_base<Dtype>();
  CUDNN_CHECK(cudnnActivationBackward(Caffe::cudnn_handle(),
        CUDNN_ACTIVATION_RELU,
        cudnn::dataType<Dtype>::one,
        this->top_desc_, top_data, this->top_desc_, top_diff,
        this->bottom_desc_, bottom_data,
        cudnn::dataType<Dtype>::zero,
        this->bottom_desc_, bottom_diff));
}

INSTANTIATE_LAYER_GPU_FUNCS(CuDNNReLULayer);

}  // namespace caffe
#endif
