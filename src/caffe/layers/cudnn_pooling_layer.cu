#ifdef USE_CUDNN
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype, typename Mtype>
void CuDNNPoolingLayer<Dtype,Mtype>::Forward_gpu(const vector<BlobBase*>& bottom,
    const vector<BlobBase*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data<Dtype>();
  Dtype* top_data = top[0]->mutable_gpu_data<Dtype>();
  CUDNN_CHECK(cudnnPoolingForward(Caffe::cudnn_handle(), pooling_desc_,
        cudnn::dataType<Dtype>::one,
        bottom_desc_, bottom_data,
        cudnn::dataType<Dtype>::zero,
        top_desc_, top_data));
}

template <typename Dtype, typename Mtype>
void CuDNNPoolingLayer<Dtype,Mtype>::Backward_gpu(const vector<BlobBase*>& top,
    const vector<bool>& propagate_down, const vector<BlobBase*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->gpu_diff<Dtype>();
  const Dtype* top_data = top[0]->gpu_data<Dtype>();
  const Dtype* bottom_data = bottom[0]->gpu_data<Dtype>();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff<Dtype>();
  CUDNN_CHECK(cudnnPoolingBackward(Caffe::cudnn_handle(), pooling_desc_,
        cudnn::dataType<Dtype>::one,
        top_desc_, top_data, top_desc_, top_diff,
        bottom_desc_, bottom_data,
        cudnn::dataType<Dtype>::zero,
        bottom_desc_, bottom_diff));
}

INSTANTIATE_LAYER_GPU_FUNCS(CuDNNPoolingLayer);

}  // namespace caffe
#endif
