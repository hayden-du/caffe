#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

const float kBNLL_THRESHOLD = 50.;

template <typename Dtype, typename Mtype>
void BNLLLayer<Dtype,Mtype>::Forward_cpu(const vector<BlobBase*>& bottom,
    const vector<BlobBase*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data_base<Dtype>();
  Dtype* top_data = top[0]->mutable_cpu_data_base<Dtype>();
  const int count = bottom[0]->count();
  for (int i = 0; i < count; ++i) {
    top_data[i] = bottom_data[i] > 0.f ?
        bottom_data[i] + log(1. + exp(-bottom_data[i])) :
        log(1. + exp(bottom_data[i])) ;
  }
}

template <typename Dtype, typename Mtype>
void BNLLLayer<Dtype,Mtype>::Backward_cpu(const vector<BlobBase*>& top,
    const vector<bool>& propagate_down,
    const vector<BlobBase*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data_base<Dtype>();
    const Dtype* top_diff = top[0]->cpu_diff_base<Dtype>();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff_base<Dtype>();
    const int count = bottom[0]->count();
    Mtype expval;
    for (int i = 0; i < count; ++i) {
      expval = exp(std::min(bottom_data[i], Dtype(kBNLL_THRESHOLD)));
      bottom_diff[i] = top_diff[i] * expval / (expval + 1.) ;
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(BNLLLayer);
#endif

INSTANTIATE_CLASS(BNLLLayer);
REGISTER_LAYER_CLASS(BNLL);

}  // namespace caffe
