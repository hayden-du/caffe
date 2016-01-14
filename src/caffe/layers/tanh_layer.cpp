// TanH neuron activation function layer.
// Adapted from ReLU layer code written by Yangqing Jia

#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype, typename Mtype>
void TanHLayer<Dtype,Mtype>::Forward_cpu(const vector<BlobBase*>& bottom,
    const vector<BlobBase*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data_base<Dtype>();
  Dtype* top_data = top[0]->mutable_cpu_data_base<Dtype>();
  const int count = bottom[0]->count();
  for (int i = 0; i < count; ++i) {
    top_data[i] = tanh(bottom_data[i]);
  }
}

template <typename Dtype, typename Mtype>
void TanHLayer<Dtype,Mtype>::Backward_cpu(const vector<BlobBase*>& top,
    const vector<bool>& propagate_down,
    const vector<BlobBase*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_data = top[0]->cpu_data_base<Dtype>();
    const Dtype* top_diff = top[0]->cpu_diff_base<Dtype>();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff_base<Dtype>();
    const int count = bottom[0]->count();
    Mtype tanhx;
    for (int i = 0; i < count; ++i) {
      tanhx = top_data[i];
      bottom_diff[i] = top_diff[i] * (1 - tanhx * tanhx) ;
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(TanHLayer);
#endif

INSTANTIATE_CLASS(TanHLayer);

}  // namespace caffe
