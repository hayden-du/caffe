#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype, typename Mtype>
void ConvolutionLayer<Dtype,Mtype>::compute_output_shape() {
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  this->output_shape_.clear();
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int input_dim = this->input_shape(i + 1);
    const int output_dim = (input_dim + 2 * pad_data[i] - kernel_shape_data[i])
        / stride_data[i] + 1;
    this->output_shape_.push_back(output_dim);
  }
}

template <typename Dtype, typename Mtype>
void ConvolutionLayer<Dtype,Mtype>::Forward_cpu(const vector<BlobBase*>& bottom,
      const vector<BlobBase*>& top) {
  const Dtype* weight = this->blobs_[0]->template cpu_data_base<Dtype>();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data_base<Dtype>();
    Dtype* top_data = top[i]->mutable_cpu_data_base<Dtype>();
    for (int n = 0; n < this->num_; ++n) {
      this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_, weight,
          top_data + n * this->top_dim_);
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->template cpu_data_base<Dtype>();
        this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
  }
}

template <typename Dtype, typename Mtype>
void ConvolutionLayer<Dtype,Mtype>::Backward_cpu(const vector<BlobBase*>& top,
      const vector<bool>& propagate_down, const vector<BlobBase*>& bottom) {
  const Dtype* weight = this->blobs_[0]->template cpu_data_base<Dtype>();
  Dtype* weight_diff = this->blobs_[0]->template mutable_cpu_diff_base<Dtype>();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff_base<Dtype>();
    const Dtype* bottom_data = bottom[i]->cpu_data_base<Dtype>();
    Dtype* bottom_diff = bottom[i]->mutable_cpu_diff_base<Dtype>();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->template mutable_cpu_diff_base<Dtype>();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_cpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_cpu_gemm(bottom_data + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_cpu_gemm(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_);
        }
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(ConvolutionLayer);
#endif

INSTANTIATE_CLASS(ConvolutionLayer);

}  // namespace caffe
