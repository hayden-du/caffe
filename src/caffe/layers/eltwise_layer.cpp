#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype, typename Mtype>
void EltwiseLayer<Dtype,Mtype>::LayerSetUp(const vector<BlobBase*>& bottom,
      const vector<BlobBase*>& top) {
  CHECK(this->layer_param().eltwise_param().coeff_size() == 0
      || this->layer_param().eltwise_param().coeff_size() == bottom.size()) <<
      "Eltwise Layer takes one coefficient per bottom blob.";
  CHECK(!(this->layer_param().eltwise_param().operation()
      == EltwiseParameter_EltwiseOp_PROD
      && this->layer_param().eltwise_param().coeff_size())) <<
      "Eltwise layer only takes coefficients for summation.";
  op_ = this->layer_param_.eltwise_param().operation();
  // Blob-wise coefficients for the elementwise operation.
  coeffs_ = vector<Dtype>(bottom.size(), 1);
  if (this->layer_param().eltwise_param().coeff_size()) {
    for (int i = 0; i < bottom.size(); ++i) {
      coeffs_[i] = this->layer_param().eltwise_param().coeff(i);
    }
  }
  stable_prod_grad_ = this->layer_param_.eltwise_param().stable_prod_grad();
}

template <typename Dtype, typename Mtype>
void EltwiseLayer<Dtype,Mtype>::Reshape(const vector<BlobBase*>& bottom,
      const vector<BlobBase*>& top) {
  for (int i = 1; i < bottom.size(); ++i) {
    CHECK(bottom[i]->shape() == bottom[0]->shape());
  }
  top[0]->ReshapeLike(*bottom[0]);
  // If max operation, we will initialize the vector index part.
  if (this->layer_param_.eltwise_param().operation() ==
      EltwiseParameter_EltwiseOp_MAX && top.size() == 1) {
    max_idx_.Reshape(bottom[0]->shape());
  }
}

template <typename Dtype, typename Mtype>
void EltwiseLayer<Dtype,Mtype>::Forward_cpu(
    const vector<BlobBase*>& bottom, const vector<BlobBase*>& top) {
  int* mask = NULL;
  const Dtype* bottom_data_a = NULL;
  const Dtype* bottom_data_b = NULL;
  const int count = top[0]->count();
  Dtype* top_data = top[0]->mutable_cpu_data_base<Dtype>();
  switch (op_) {
  case EltwiseParameter_EltwiseOp_PROD:
    caffe_mul(count, bottom[0]->cpu_data_base<Dtype>(), bottom[1]->cpu_data_base<Dtype>(), top_data);
    for (int i = 2; i < bottom.size(); ++i) {
      caffe_mul(count, top_data, bottom[i]->cpu_data_base<Dtype>(), top_data);
    }
    break;
  case EltwiseParameter_EltwiseOp_SUM:
    caffe_set(count, typedConsts<Dtype>::zero, top_data);
    // TODO(shelhamer) does BLAS optimize to sum for coeff = 1?
    for (int i = 0; i < bottom.size(); ++i) {
      caffe_axpy(count, coeffs_[i], bottom[i]->cpu_data_base<Dtype>(), top_data);
    }
    break;
  case EltwiseParameter_EltwiseOp_MAX:
    // Initialize
    mask = max_idx_.mutable_cpu_data();
    caffe_set(count, typedConsts<int>::minus_one, mask);
    caffe_set(count, - maxDtype<Dtype>(), top_data);
    // bottom 0 & 1
    bottom_data_a = bottom[0]->cpu_data_base<Dtype>();
    bottom_data_b = bottom[1]->cpu_data_base<Dtype>();
    for (int idx = 0; idx < count; ++idx) {
      if (bottom_data_a[idx] > bottom_data_b[idx]) {
        top_data[idx] = bottom_data_a[idx];  // maxval
        mask[idx] = 0;  // maxid
      } else {
        top_data[idx] = bottom_data_b[idx];  // maxval
        mask[idx] = 1;  // maxid
      }
    }
    // bottom 2++
    for (int blob_idx = 2; blob_idx < bottom.size(); ++blob_idx) {
      bottom_data_b = bottom[blob_idx]->cpu_data_base<Dtype>();
      for (int idx = 0; idx < count; ++idx) {
        if (bottom_data_b[idx] > top_data[idx]) {
          top_data[idx] = bottom_data_b[idx];  // maxval
          mask[idx] = blob_idx;  // maxid
        }
      }
    }
    break;
  default:
    LOG(FATAL) << "Unknown elementwise operation.";
  }
}

template <typename Dtype, typename Mtype>
void EltwiseLayer<Dtype,Mtype>::Backward_cpu(const vector<BlobBase*>& top,
    const vector<bool>& propagate_down, const vector<BlobBase*>& bottom) {
  const int* mask = NULL;
  const int count = top[0]->count();
  const Dtype* top_data = top[0]->cpu_data_base<Dtype>();
  const Dtype* top_diff = top[0]->cpu_diff_base<Dtype>();
  for (int i = 0; i < bottom.size(); ++i) {
    if (propagate_down[i]) {
      const Dtype* bottom_data = bottom[i]->cpu_data_base<Dtype>();
      Dtype* bottom_diff = bottom[i]->mutable_cpu_diff_base<Dtype>();
      switch (op_) {
      case EltwiseParameter_EltwiseOp_PROD:
        if (stable_prod_grad_) {
          bool initialized = false;
          for (int j = 0; j < bottom.size(); ++j) {
            if (i == j) { continue; }
            if (!initialized) {
              caffe_copy(count, bottom[j]->cpu_data_base<Dtype>(), bottom_diff);
              initialized = true;
            } else {
              caffe_mul(count, bottom[j]->cpu_data_base<Dtype>(), bottom_diff,
                        bottom_diff);
            }
          }
        } else {
          caffe_div(count, top_data, bottom_data, bottom_diff);
        }
        caffe_mul(count, bottom_diff, top_diff, bottom_diff);
        break;
      case EltwiseParameter_EltwiseOp_SUM:
        if (coeffs_[i] == Mtype(1)) {
          caffe_copy(count, top_diff, bottom_diff);
        } else {
          caffe_cpu_scale<Dtype,Mtype>(count, coeffs_[i], top_diff, bottom_diff);
        }
        break;
      case EltwiseParameter_EltwiseOp_MAX:
        mask = max_idx_.cpu_data();
        for (int index = 0; index < count; ++index) {
          Mtype gradient(0.f);
          if (mask[index] == i) {
            gradient += top_diff[index];
          }
          bottom_diff[index] = gradient;
        }
        break;
      default:
        LOG(FATAL) << "Unknown elementwise operation.";
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(EltwiseLayer);
#endif

INSTANTIATE_CLASS(EltwiseLayer);
REGISTER_LAYER_CLASS(Eltwise);

}  // namespace caffe
