#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"


namespace caffe {

template <typename Dtype, typename Mtype>
void ThresholdLayer<Dtype,Mtype>::LayerSetUp(const vector<BlobBase*>& bottom,
      const vector<BlobBase*>& top) {
  NeuronLayer<Dtype,Mtype>::LayerSetUp(bottom, top);
  threshold_ = this->layer_param_.threshold_param().threshold();
}

template <typename Dtype, typename Mtype>
void ThresholdLayer<Dtype,Mtype>::Forward_cpu(const vector<BlobBase*>& bottom,
    const vector<BlobBase*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data_base<Dtype>();
  Dtype* top_data = top[0]->mutable_cpu_data_base<Dtype>();
  const int count = bottom[0]->count();
  for (int i = 0; i < count; ++i) {
    top_data[i] = (bottom_data[i] > threshold_) ? 1 : 0;
  }
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(ThresholdLayer, Forward);
#endif

INSTANTIATE_CLASS(ThresholdLayer);
REGISTER_LAYER_CLASS(Threshold);

}  // namespace caffe
