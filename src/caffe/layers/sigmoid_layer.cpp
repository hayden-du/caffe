#include <algorithm>
#include <cmath>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
inline Dtype sigmoid(Dtype x) {
  return 1. / (1. + exp(-x));
}

template <typename Dtype, typename Mtype>
void SigmoidLayer<Dtype,Mtype>::Forward_cpu(const vector<BlobBase*>& bottom,
    const vector<BlobBase*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data_base<Dtype>();
  Dtype* top_data = top[0]->mutable_cpu_data_base<Dtype>();
  const int count = bottom[0]->count();
  for (int i = 0; i < count; ++i) {
    top_data[i] = sigmoid(bottom_data[i]);
  }
}

template <typename Dtype, typename Mtype>
void SigmoidLayer<Dtype,Mtype>::Backward_cpu(const vector<BlobBase*>& top,
    const vector<bool>& propagate_down,
    const vector<BlobBase*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_data = top[0]->cpu_data_base<Dtype>();
    const Dtype* top_diff = top[0]->cpu_diff_base<Dtype>();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff_base<Dtype>();
    const int count = bottom[0]->count();
    for (int i = 0; i < count; ++i) {
      const Mtype sigmoid_x = top_data[i];
      bottom_diff[i] = top_diff[i] * sigmoid_x * (1. - sigmoid_x) ;
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(SigmoidLayer);
#endif

INSTANTIATE_CLASS(SigmoidLayer);


}  // namespace caffe
