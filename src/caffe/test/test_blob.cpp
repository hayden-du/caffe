#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename TypeParam>
class BlobSimpleTest : public ::testing::Test {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
 protected:
  BlobSimpleTest()
      : blob_(new Blob<Dtype>()),
        blob_preshaped_(new Blob<Dtype>(2, 3, 4, 5)) {}
  virtual ~BlobSimpleTest() { delete blob_; delete blob_preshaped_; }
  Blob<Dtype>* const blob_;
  Blob<Dtype>* const blob_preshaped_;
};

TYPED_TEST_CASE(BlobSimpleTest, TestDtypes);

TYPED_TEST(BlobSimpleTest, TestInitialization) {
  EXPECT_TRUE(this->blob_);
  EXPECT_TRUE(this->blob_preshaped_);
  EXPECT_EQ(this->blob_preshaped_->num(), 2);
  EXPECT_EQ(this->blob_preshaped_->channels(), 3);
  EXPECT_EQ(this->blob_preshaped_->height(), 4);
  EXPECT_EQ(this->blob_preshaped_->width(), 5);
  EXPECT_EQ(this->blob_preshaped_->count(), 120);
  EXPECT_EQ(this->blob_->num_axes(), 0);
  EXPECT_EQ(this->blob_->count(), 0);
}

#ifdef CPU_ONLY
TYPED_TEST(BlobSimpleTest, TestPointersCPUGPU) {
  EXPECT_TRUE(this->blob_preshaped_->cpu_data());
  EXPECT_TRUE(this->blob_preshaped_->mutable_cpu_data());
}
#else
TYPED_TEST(BlobSimpleTest, TestPointersCPUGPU) {
  EXPECT_TRUE(this->blob_preshaped_->gpu_data());
  EXPECT_TRUE(this->blob_preshaped_->cpu_data());
  EXPECT_TRUE(this->blob_preshaped_->mutable_gpu_data());
  EXPECT_TRUE(this->blob_preshaped_->mutable_cpu_data());
}
#endif

TYPED_TEST(BlobSimpleTest, TestReshape) {
  this->blob_->Reshape(2, 3, 4, 5);
  EXPECT_EQ(this->blob_->num(), 2);
  EXPECT_EQ(this->blob_->channels(), 3);
  EXPECT_EQ(this->blob_->height(), 4);
  EXPECT_EQ(this->blob_->width(), 5);
  EXPECT_EQ(this->blob_->count(), 120);
}

TYPED_TEST(BlobSimpleTest, TestLegacyBlobProtoShapeEquals) {
  BlobProto blob_proto;

  // Reshape to (3 x 2).
  vector<int> shape(2);
  shape[0] = 3;
  shape[1] = 2;
  this->blob_->Reshape(shape);

  // (3 x 2) blob == (1 x 1 x 3 x 2) legacy blob
  blob_proto.set_num(1);
  blob_proto.set_channels(1);
  blob_proto.set_height(3);
  blob_proto.set_width(2);
  EXPECT_TRUE(this->blob_->ShapeEquals(blob_proto));

  // (3 x 2) blob != (0 x 1 x 3 x 2) legacy blob
  blob_proto.set_num(0);
  blob_proto.set_channels(1);
  blob_proto.set_height(3);
  blob_proto.set_width(2);
  EXPECT_FALSE(this->blob_->ShapeEquals(blob_proto));

  // (3 x 2) blob != (3 x 1 x 3 x 2) legacy blob
  blob_proto.set_num(3);
  blob_proto.set_channels(1);
  blob_proto.set_height(3);
  blob_proto.set_width(2);
  EXPECT_FALSE(this->blob_->ShapeEquals(blob_proto));

  // Reshape to (1 x 3 x 2).
  shape.insert(shape.begin(), 1);
  this->blob_->Reshape(shape);

  // (1 x 3 x 2) blob == (1 x 1 x 3 x 2) legacy blob
  blob_proto.set_num(1);
  blob_proto.set_channels(1);
  blob_proto.set_height(3);
  blob_proto.set_width(2);
  EXPECT_TRUE(this->blob_->ShapeEquals(blob_proto));

  // Reshape to (2 x 3 x 2).
  shape[0] = 2;
  this->blob_->Reshape(shape);

  // (2 x 3 x 2) blob != (1 x 1 x 3 x 2) legacy blob
  blob_proto.set_num(1);
  blob_proto.set_channels(1);
  blob_proto.set_height(3);
  blob_proto.set_width(2);
  EXPECT_FALSE(this->blob_->ShapeEquals(blob_proto));
}

template <typename TypeParam>
class BlobMathTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
 protected:
  BlobMathTest()
      : blob_(new Blob<Dtype>(2, 3, 4, 5)),
        epsilon_(choose<Dtype>(1e-6,2e-3)) {}

  virtual ~BlobMathTest() { delete blob_; }
  Blob<Dtype>* const blob_;
  Mtype epsilon_;
};

TYPED_TEST_CASE(BlobMathTest, TestDtypesAndDevices);

TYPED_TEST(BlobMathTest, TestSumOfSquares) {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;

  // Uninitialized Blob should have sum of squares == 0.
  Mtype sumsq = this->blob_->template sumsq_data<Dtype,Mtype>();
  EXPECT_EQ(0, sumsq);
  sumsq = this->blob_->template sumsq_diff<Dtype,Mtype>();
  EXPECT_EQ(0, sumsq);
  FillerParameter filler_param;
  filler_param.set_min(-3);
  filler_param.set_max(3);
  UniformFiller<Dtype,Mtype> filler(filler_param);
  filler.Fill(this->blob_);
  Mtype expected_sumsq = 0, psum = 0;
  const Dtype* data = this->blob_->cpu_data();
  for (int i = 0; i < this->blob_->count(); ++i) {
    psum += data[i] * data[i];
    if (i > 0 && i % 10 == 0) {
      expected_sumsq += psum;
      psum = 0;
    }
  }
  expected_sumsq += psum;

  // Do a mutable access on the current device,
  // so that the sumsq computation is done on that device.
  // (Otherwise, this would only check the CPU sumsq implementation.)
  switch (TypeParam::device) {
  case Caffe::CPU:
    this->blob_->mutable_cpu_data();
    break;
  case Caffe::GPU:
    this->blob_->mutable_gpu_data();
    break;
  default:
    LOG(FATAL) << "Unknown device: " << TypeParam::device;
  }
  sumsq = this->blob_->template sumsq_data<Dtype,Mtype>();
  EXPECT_NEAR(expected_sumsq, sumsq, this->epsilon_ * expected_sumsq);
  sumsq = this->blob_->template sumsq_diff<Dtype,Mtype>();
  EXPECT_EQ(0, sumsq);

  // Check sumsq_diff too.
  const Mtype kDiffScaleFactor = 7;
  caffe_cpu_scale<Dtype,Mtype>(this->blob_->count(), kDiffScaleFactor, data,
                  this->blob_->mutable_cpu_diff());
  switch (TypeParam::device) {
  case Caffe::CPU:
    this->blob_->mutable_cpu_diff();
    break;
  case Caffe::GPU:
    this->blob_->mutable_gpu_diff();
    break;
  default:
    LOG(FATAL) << "Unknown device: " << TypeParam::device;
  }
  sumsq = this->blob_->template sumsq_data<Dtype,Mtype>();
  EXPECT_NEAR(expected_sumsq, sumsq,
              this->epsilon_ * expected_sumsq);
  const Mtype expected_sumsq_diff =
      expected_sumsq * kDiffScaleFactor * kDiffScaleFactor;
  sumsq = this->blob_->template sumsq_diff<Dtype,Mtype>();
  EXPECT_NEAR(expected_sumsq_diff, sumsq,
              tol<Dtype>(this->epsilon_) * expected_sumsq_diff);
}

TYPED_TEST(BlobMathTest, TestAsum) {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;

  // Uninitialized Blob should have asum == 0.
  Mtype asum = this->blob_->template asum_data<Dtype,Mtype>();
  EXPECT_EQ(0, asum);
  asum = this->blob_->template asum_diff<Dtype,Mtype>();
  EXPECT_EQ(0, asum);
  FillerParameter filler_param;
  filler_param.set_min(-3);
  filler_param.set_max(3);
  UniformFiller<Dtype,Mtype> filler(filler_param);
  filler.Fill(this->blob_);
  Mtype expected_asum = 0, psum = 0;
  const Dtype* data = this->blob_->cpu_data();
  for (int i = 0; i < this->blob_->count(); ++i) {
    psum += std::fabs(data[i]);
    if (i > 0 && i % 10 == 0) {
      expected_asum += psum;
      psum = 0;
    }
  }
  expected_asum += psum;

  // Do a mutable access on the current device,
  // so that the asum computation is done on that device.
  // (Otherwise, this would only check the CPU asum implementation.)
  switch (TypeParam::device) {
  case Caffe::CPU:
    this->blob_->mutable_cpu_data();
    break;
  case Caffe::GPU:
    this->blob_->mutable_gpu_data();
    break;
  default:
    LOG(FATAL) << "Unknown device: " << TypeParam::device;
  }
  asum = this->blob_->template asum_data<Dtype,Mtype>();
  EXPECT_NEAR(expected_asum, asum,
              this->epsilon_ * expected_asum);
  asum = this->blob_->template asum_diff<Dtype,Mtype>();
  EXPECT_EQ(0, asum);

  // Check asum_diff too.
  const Mtype kDiffScaleFactor = 7;
  caffe_cpu_scale<Dtype,Mtype>(this->blob_->count(), kDiffScaleFactor, data,
                  this->blob_->mutable_cpu_diff());
  switch (TypeParam::device) {
  case Caffe::CPU:
    this->blob_->mutable_cpu_diff();
    break;
  case Caffe::GPU:
    this->blob_->mutable_gpu_diff();
    break;
  default:
    LOG(FATAL) << "Unknown device: " << TypeParam::device;
  }
  asum = this->blob_->template asum_data<Dtype,Mtype>();
  EXPECT_NEAR(expected_asum, asum,
              this->epsilon_ * expected_asum);
  const Mtype expected_diff_asum = expected_asum * kDiffScaleFactor;
  asum = this->blob_->template asum_diff<Dtype,Mtype>();
  EXPECT_NEAR(expected_diff_asum, asum,
      tol<Dtype>(this->epsilon_) * expected_diff_asum);
}

TYPED_TEST(BlobMathTest, TestScaleData) {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;

  Mtype asum = this->blob_->template asum_data<Dtype,Mtype>();
  EXPECT_EQ(0, asum);
  asum = this->blob_->template asum_diff<Dtype,Mtype>();
  EXPECT_EQ(0, asum);
  FillerParameter filler_param;
  filler_param.set_min(-3);
  filler_param.set_max(3);
  UniformFiller<Dtype,Mtype> filler(filler_param);
  filler.Fill(this->blob_);
  const Mtype asum_before_scale = this->blob_->template asum_data<Dtype,Mtype>();
  // Do a mutable access on the current device,
  // so that the asum computation is done on that device.
  // (Otherwise, this would only check the CPU asum implementation.)
  switch (TypeParam::device) {
  case Caffe::CPU:
    this->blob_->mutable_cpu_data();
    break;
  case Caffe::GPU:
    this->blob_->mutable_gpu_data();
    break;
  default:
    LOG(FATAL) << "Unknown device: " << TypeParam::device;
  }
  const Mtype kDataScaleFactor = 3;
  this->blob_->scale_data(kDataScaleFactor);
  asum = this->blob_->template asum_data<Dtype,Mtype>();
  EXPECT_NEAR(asum_before_scale * kDataScaleFactor, asum,
              tol<Dtype>(this->epsilon_) * asum_before_scale * kDataScaleFactor);
  asum = this->blob_->template asum_diff<Dtype,Mtype>();
  EXPECT_NEAR(0, asum, choose<Dtype>(1.e-6,1.e-4));

  // Check scale_diff too.
  const Mtype kDataToDiffScaleFactor = 7;
  const Dtype* data = this->blob_->cpu_data();
  caffe_cpu_scale<Dtype,Mtype>(this->blob_->count(), kDataToDiffScaleFactor, data,
                  this->blob_->mutable_cpu_diff());
  const Mtype expected_asum_before_scale = asum_before_scale * kDataScaleFactor;
  asum = this->blob_->template asum_data<Dtype,Mtype>();
  EXPECT_NEAR(expected_asum_before_scale, asum,
      tol<Dtype>(this->epsilon_) * expected_asum_before_scale);
  const Mtype expected_diff_asum_before_scale =
      asum_before_scale * kDataScaleFactor * kDataToDiffScaleFactor;
  asum = this->blob_->template asum_diff<Dtype,Mtype>();
  EXPECT_NEAR(expected_diff_asum_before_scale, asum,
      tol<Dtype>(this->epsilon_) * expected_diff_asum_before_scale);
  switch (TypeParam::device) {
  case Caffe::CPU:
    this->blob_->mutable_cpu_diff();
    break;
  case Caffe::GPU:
    this->blob_->mutable_gpu_diff();
    break;
  default:
    LOG(FATAL) << "Unknown device: " << TypeParam::device;
  }
  const Mtype kDiffScaleFactor = 3;
  this->blob_->scale_diff(kDiffScaleFactor);
  asum = this->blob_->template asum_data<Dtype,Mtype>();
  EXPECT_NEAR(asum_before_scale * kDataScaleFactor, asum,
      tol<Dtype>(this->epsilon_) * asum_before_scale * kDataScaleFactor);
  const Mtype expected_diff_asum =
      expected_diff_asum_before_scale * kDiffScaleFactor;
  asum = this->blob_->template asum_diff<Dtype,Mtype>();
  EXPECT_NEAR(expected_diff_asum, asum,
      tol<Dtype>(this->epsilon_) * expected_diff_asum);
}

}  // namespace caffe
