#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype, typename Mtype>
void ReLULayer<Dtype,Mtype>::Forward_cpu(const vector<BlobBase*>& bottom,
    const vector<BlobBase*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data<Dtype>();
  Dtype* top_data = top[0]->mutable_cpu_data<Dtype>();
  const int count = bottom[0]->count();
  Mtype negative_slope(this->layer_param_.relu_param().negative_slope());
  for (int i = 0; i < count; ++i) {
    top_data[i] = std::max(bottom_data[i], Dtype(0.))
        + negative_slope * std::min(bottom_data[i], Dtype(0.)) ;
  }
}

template <typename Dtype, typename Mtype>
void ReLULayer<Dtype,Mtype>::Backward_cpu(const vector<BlobBase*>& top,
    const vector<bool>& propagate_down,
    const vector<BlobBase*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data<Dtype>();
    const Dtype* top_diff = top[0]->cpu_diff<Dtype>();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff<Dtype>();
    const int count = bottom[0]->count();
    Mtype negative_slope(this->layer_param_.relu_param().negative_slope());
    for (int i = 0; i < count; ++i) {
      bottom_diff[i] = top_diff[i] * ((bottom_data[i] > 0)
          + negative_slope * (bottom_data[i] <= 0)) ;
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(ReLULayer);
#endif

INSTANTIATE_CLASS(ReLULayer);

}  // namespace caffe
