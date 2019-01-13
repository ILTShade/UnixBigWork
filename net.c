#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#define __DEBUG__ 0

// 对于在某些情况下才输出DEBUG信息
#define DEBUG_OUT(x) if (__DEBUG__) x;

// 由于本机是intel处理器，所以需要进行大小端的转换
#define TRANS_32(x) \
    ((uint32_t)((((uint32_t)(x) & 0xff000000) >> 24) | \
                (((uint32_t)(x) & 0x00ff0000) >> 8)  | \
                (((uint32_t)(x) & 0x0000ff00) << 8)  | \
                (((uint32_t)(x) & 0x000000ff) << 24)   \
                ))
// 用来做CHECK的函数
#define CHECK(A, B) if ((A) != (B)) {printf("something error happen\n"); return -1;}
// 一些需要需先设定的参数
static uint32_t image_num = 10000;
static uint32_t image_channel = 1;
static uint32_t image_height = 28;
static uint32_t image_width = 28;
// 一个用来存储中间运算数据的结构体，保存参数的形状和数据的地址，每次只进行一张图片的前向
struct Tensor {
  uint32_t channel;
  uint32_t height;
  uint32_t width;
  float* data;
};
// 读取其中一幅图片和对应的label
int read_image_label(int image_file, int label_file, float *image_data, int *label_data) {
  // 由于这是单独的一个字节，所以只需要按照字节单独直接读取就可以了，需要做归一化
  uint8_t buf;
  for (int i = 0; i < image_height; i++) {
    for (int j = 0; j < image_width; j++) {
      read(image_file, &buf, sizeof(buf));
      *(image_data++) = (float)buf / 255.f;
    }
  }
  read(label_file, &buf, sizeof(buf));
  *label_data = (int)buf;
  return 0;
}
// 对Tensor按照指定的格式进行卷积运算
int conv_tensor(int model_file, int weights_file, struct Tensor *tensor) {
  // conv的参数为in_channels, out_channels, kernel_size, bias_flag
  uint32_t in_channels, out_channels, kernel_size, bias_flag;
  read(model_file, &in_channels, sizeof(in_channels));
  read(model_file, &out_channels, sizeof(out_channels));
  read(model_file, &kernel_size, sizeof(kernel_size));
  read(model_file, &bias_flag, sizeof(bias_flag));
  // 输出卷积层网络的参数作为调试信息
  DEBUG_OUT(printf("conv in_channels %d, out_channels %d, kernel_size %d, bias_flag %d\n", in_channels, out_channels, kernel_size, bias_flag));
  // 进行weights信息的读取
  uint32_t weights_len = in_channels * out_channels * kernel_size * kernel_size;
  float *weights_data = (float *)malloc(weights_len * sizeof(float));
  CHECK(read(weights_file, weights_data, weights_len * sizeof(float)), weights_len * sizeof(float));
  // 进行bias信息的读取，如果bias不存在，那么将所有的值设为0
  uint32_t bias_len = out_channels;
  float *bias_data = (float *)malloc(bias_len * sizeof(float));
  if (bias_flag) {
    CHECK(read(weights_file, bias_data, bias_len * sizeof(float)), bias_len * sizeof(float));
  }
  else {
    for (int i = 0; i < bias_len; i++) {
      *(bias_data + i) = 0.f;
    }
  }
  // 先计算输出数据的长宽，申请地址空间
  uint32_t out_height = tensor->height - kernel_size + 1;
  uint32_t out_width = tensor->width - kernel_size + 1;
  uint32_t out_len = out_channels * out_height * out_width;
  float *out_data = (float *)malloc(out_len * sizeof(float));
  // 由于只支持pad为0，stride为1的卷积，所以采用最简单的方式即可
  int data_channel_shift = tensor-> height * tensor-> width;
  int data_height_shift = tensor->width;
  int weights_batch_shift = in_channels * kernel_size * kernel_size;
  int weights_channel_shift = kernel_size * kernel_size;
  int weights_height_shift = kernel_size;
  float tmp_sum1, tmp_sum2;
  for (int out_c = 0; out_c < out_channels; out_c++) {
    for (int out_h = 0; out_h < out_height; out_h++) {
      for (int out_w = 0; out_w < out_width; out_w++) {
         // 根据bias赋初值
         *out_data = bias_data[out_c];
         // 为了减少float计算的误差，引入了部分和计算，增加了中间变量
         // 按照卷积核计算最终的结果
         for (int in_c = 0; in_c < in_channels; in_c++) {
           tmp_sum1 = 0.f;
           for (int in_h = 0; in_h < kernel_size; in_h++) {
             tmp_sum2 = 0.f;
             for (int in_w = 0; in_w < kernel_size; in_w++) {
               tmp_sum2 += (tensor->data[in_c * data_channel_shift + (out_h + in_h) * data_height_shift + (out_w + in_w)] * \
                            weights_data[out_c * weights_batch_shift + in_c * weights_channel_shift + in_h * weights_height_shift + in_w]);
             } // for (int in_w = 0; in_w < kernel_size; in_w++)
             tmp_sum1 += tmp_sum2;
           } // for (int in_h = 0; in_h < kernel_size; in_h++)
           *out_data += tmp_sum1;
         } // for (int in_c = 0; in_c < in_channels; in_c++)
         out_data++; // next value
      } // for (int out_w = 0; out_w < out_width; out_w++)
    } // for (int out_h = 0; out_h < out_height; out_h++)
  } // for (int out_c = 0; out_c < out_channels; out_c++)
  // 计算完成后，对tensor内容进行修改
  tensor->channel = out_channels;
  tensor->height = out_height;
  tensor->width = out_width;
  free(tensor->data);
  tensor->data = out_data - out_channels * out_height * out_width;
  // 消除计算的中间过程中存储的变量
  free(weights_data);
  free(bias_data);
  // 返回值
  return 0;
}
// 对Tensor按照指定的格式进行ReLU运算
int relu_tensor(int model_file, int weights_file, struct Tensor *tensor) {
  // ReLU运算不需要任何额外的操作，只需要将小于0的置为0即可
  DEBUG_OUT(printf("relu\n"));
  float *data = tensor->data;
  for (int c = 0; c < tensor->channel; c++) {
    for (int h = 0; h < tensor->height; h++) {
      for (int w = 0; w < tensor->width; w++) {
        *data = *data > 0 ? *data : 0;
        data++;
      } // for (int w = 0; w < tensor->width; w++)
    } // for (int h = 0; h < tensor->height; h++)
  } // for (int c = 0; c < tensor->channel; c++)
  return 0;
}
// 对Tensor按照指定的格式进行Pooling运算
int pooling_tensor(int model_file, int weights_file, struct Tensor *tensor) {
  // Pooling运算只支持kernel_size方形操作，且要求边长能够整除
  uint32_t kernel_size;
  uint32_t pooling_method;
  // 输出相关参数作为调试信息
  read(model_file, &kernel_size, sizeof(kernel_size));
  read(model_file, &pooling_method, sizeof(pooling_method));
  DEBUG_OUT(printf("pooling kernel_size %d, pooling method %d\n", kernel_size, pooling_method));
  // 对于pooling不需要参数，但是要更新tensor的相关参数，同时check相关参数的正确性
  CHECK(tensor->height % kernel_size, 0);
  CHECK(tensor->width % kernel_size, 0);
  // 申请新的数据空间
  uint32_t out_height = tensor->height / kernel_size;
  uint32_t out_width = tensor->width / kernel_size;
  float *out_data = (float *)malloc(tensor->channel * out_height * out_width * sizeof(float));
  // channel height shift
  uint32_t data_channel_shift = tensor->height * tensor->width;
  uint32_t data_height_shift = tensor->width;
  // 在不同的method下选择不同的分支进行计算
  if (pooling_method == 0) {
    // 0 mean MAXPOOL
    for (int out_c = 0; out_c < tensor->channel; out_c++) {
      for (int out_h = 0; out_h < out_height; out_h++) {
        for (int out_w = 0; out_w < out_width; out_w++) {
          // 按照最大值的方式查找，初始最大值为第一个值
          float max_value = tensor->data[out_c * data_channel_shift + out_h * kernel_size * data_height_shift + out_w * kernel_size];
          float tmp_value;
          for (int in_h = out_h * kernel_size; in_h < (out_h + 1) * kernel_size; in_h++) {
            for (int in_w = out_w * kernel_size; in_w < (out_w + 1) * kernel_size; in_w++) {
              tmp_value = tensor->data[out_c * data_channel_shift + in_h * data_height_shift + in_w];
              max_value = tmp_value > max_value ? tmp_value : max_value;
            }
          }
          // 找到最大值后赋给当前的data
          *out_data = max_value;
          out_data++;
        }
      }
    } // for (int out_c = 0; out_c < tensor->channel; out_c++)
  }
  if (pooling_method == 1) {
    // 0 mean AVGPOOL
    for (int out_c = 0; out_c < tensor->channel; out_c++) {
      for (int out_h = 0; out_h < out_height; out_h++) {
        for (int out_w = 0; out_w < out_width; out_w++) {
          // 按照最大值的方式查找，初始最大值为第一个值
          float tmp_value = 0;
          for (int in_h = out_h * kernel_size; in_h < (out_h + 1) * kernel_size; in_h++) {
            for (int in_w = out_w * kernel_size; in_w < (out_w + 1) * kernel_size; in_w++) {
              tmp_value += tensor->data[out_c * data_channel_shift + in_h * data_height_shift + in_w];
            }
          }
          // 找到最大值后赋给当前的data
          *out_data = tmp_value / (kernel_size * kernel_size);
          out_data++;
        }
      }
    } // for (int out_c = 0; out_c < tensor->channel; out_c++)
  }
  // 赋值完成后实现对一些数据的处理
  tensor->height = out_height;
  tensor->width = out_width;
  free(tensor->data);
  tensor->data = out_data - tensor->channel * out_height * out_width;
  // 返回值
  return 0;
}
// 对Tensor按照指定的格式进行FC运算
int fc_tensor(int model_file, int weights_file, struct Tensor *tensor) {
  // fc的参数为in_features, out_features, bias_flag
  uint32_t in_features, out_features, bias_flag;
  read(model_file, &in_features, sizeof(in_features));
  read(model_file, &out_features, sizeof(out_features));
  read(model_file, &bias_flag, sizeof(bias_flag));
  // 输出全连接层层网络的参数作为调试信息
  DEBUG_OUT(printf("fc in_features %d, out_features %d, bias_flag %d\n", in_features, out_features, bias_flag));
  // 测试相关的输入尺寸是否正确
  CHECK(tensor->channel * tensor->height * tensor->width, in_features);
  // 进行weights信息的读取
  uint32_t weights_len = in_features * out_features;
  float *weights_data = (float *)malloc(weights_len * sizeof(float));
  CHECK(read(weights_file, weights_data, weights_len * sizeof(float)), weights_len * sizeof(float));
  // 进行bias信息的读取，如果bias不存在，那么将所有的值设为0
  uint32_t bias_len = out_features;
  float *bias_data = (float *)malloc(bias_len * sizeof(float));
  if (bias_flag) {
    CHECK(read(weights_file, bias_data, bias_len * sizeof(float)), bias_len * sizeof(float));
  }
  else {
    for (int i = 0; i < bias_len; i++) {
      *(bias_data + i) = 0.f;
    }
  }
  // 先计算输出数据的长度，申请地址空间
  uint32_t out_len = out_features;
  float *out_data = (float *)malloc(out_len * sizeof(float));
  // 只支持向量矩阵乘
  for (int out_f = 0; out_f < out_features; out_f++) {
    *out_data = bias_data[out_f];
    for (int in_f = 0; in_f < in_features; in_f++) {
      (*out_data) += weights_data[out_f * in_features + in_f] * tensor->data[in_f];
    }
    out_data++;
  }
  // 计算完成后，对tensor内容进行修改
  tensor->channel = out_features;
  tensor->height = 1;
  tensor->width = 1;
  free(tensor->data);
  tensor->data = out_data - out_features;
  // 消除计算的中间过程中存储的变量
  free(weights_data);
  free(bias_data);
  // 返回值
  return 0;
}

int main(void)
{
  uint32_t buf;
  // 从/mnist/raw/t10k-images-idx3-ubytes和/mnist/raw/t10k-labels-idx1-ubytes
  // 读取图片信息，要求前几个格式描述符满足某种需求
  int image_file = open("./mnist/raw/t10k-images-idx3-ubyte", O_RDONLY, 0000);
  if (image_file == -1) return 0;
  // 确认读取的前四个4bit数组满足某种要求，分别要求满足2051, 10000, 28, 28
  read(image_file, &buf, sizeof(buf));
  CHECK(TRANS_32(buf), (0x08 << 8) + 0x03);
  read(image_file, &buf, sizeof(buf));
  CHECK(TRANS_32(buf), image_num);
  read(image_file, &buf, sizeof(buf));
  CHECK(TRANS_32(buf), image_height);
  read(image_file, &buf, sizeof(buf));
  CHECK(TRANS_32(buf), image_width);
  int label_file = open("./mnist/raw/t10k-labels-idx1-ubyte", O_RDONLY, 0000);
  if (label_file == -1) return 0;
  // 确认读取的前四个2bit数字满足某种要求，分别要求满足2049, 10000
  read(label_file, &buf, sizeof(buf));
  CHECK(TRANS_32(buf), (0x08 << 8) + 0x01);
  read(label_file, &buf, sizeof(buf));
  CHECK(TRANS_32(buf), image_num);

  // 对所有的图片进行遍历
  int test_total = 0;
  int correct_total = 0;
  for (int i = 0; i < image_num; i++) {
    // 进行文件数据的读取
    float *image_data = (float *)malloc(image_height * image_width * sizeof(float));
    int label_data;
    read_image_label(image_file, label_file, image_data, &label_data);
    // 进行网络参数的相关读取
    int model_file = open("./layer_params.bin", O_RDONLY, 0000);
    if (model_file == -1) return 0;
    // 进行网络权重的读取
    int weights_file = open("./zoo/layer_data.bin", O_RDONLY, 0000);
    if (weights_file == -1) return 0;
    // 生成记录中间结果的tensor
    struct Tensor tensor;
    tensor.channel = image_channel;
    tensor.height = image_height;
    tensor.width = image_width;
    tensor.data = image_data;
    // 每隔1000张图片对图片进行一次测试
    if ((i + 1) % 1000 == 0) {
      float *tmp_data = tensor.data;
      for (int s = 0; s < tensor.height; s++) {
        for (int t = 0; t < tensor.width; t++) {
          if (*tmp_data > 0.5f) printf("*");
          else printf(" ");
          tmp_data++;
        }
        printf("\n");
      }
    }
    // 由于是从自己生成的文件中读取，那么不需要大小端的转换
    while (read(model_file, &buf, sizeof(buf))) {
      switch (buf) {
        case 0: conv_tensor(model_file, weights_file, &tensor); break;
        case 1: relu_tensor(model_file, weights_file, &tensor); break;
        case 2: pooling_tensor(model_file, weights_file, &tensor); break;
        case 3: fc_tensor(model_file, weights_file, &tensor); break;
        default: printf("something error happened\n"); break;
      }
    }
    // 对于输出的结果，找到最大值，进而给出预测的结果，作出对height和长度的判断
    CHECK(tensor.height, 1);
    CHECK(tensor.width, 1);
    float max_value = tensor.data[0];
    int max_index = 0;
    for (int k = 1; k < tensor.channel; k++) {
      if (tensor.data[k] > max_value) {
        max_value = tensor.data[k];
        max_index = k;
      }
    }
    // 计算每张图片分类是否准确
    if (max_index == label_data) correct_total++;
    test_total++;
    // 每1000张给出一个测试结果
    if ((i + 1) % 1000 == 0) {
      printf("本张图片的预测类别为%d\n", max_index);
      printf("%d/%d\n", correct_total, test_total);
    }
    // 关闭文件
    close(model_file);
    close(weights_file);
  }
  // 关闭文件
  close(image_file);
  close(label_file);
  return 0;
}
