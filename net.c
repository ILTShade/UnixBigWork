#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>

// 由于本机是intel处理器，所以需要进行大小端的转换
#define TRANS_32(x) \
    (uint32_t)((((uint32_t)(x) & 0xff000000) >> 24) | \
               (((uint32_t)(x) & 0x00ff0000) >> 8) | \
               (((uint32_t)(x) & 0x0000ff00) << 8) | \
               (((uint32_t)(x) & 0x000000ff) << 24) \
               )
// 用来做CHECK的函数
#define CHECK(A, B) if ((A) != (B)) {printf("something error happen\n"); return -1;}

static uint32_t image_num = 10000;
static uint32_t image_height = 28;
static uint32_t image_width = 28;

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

int main(void)
{
  uint32_t buf;
  // 从/mnist/raw/t10k-images-idx3-ubytes和/mnist/raw/t10k-labels-idx1-ubytes
  // 读取图片信息，要求前几个格式描述符满足某种需求
  int image_file = open("./mnist/raw/t10k-images-idx3-ubyte", O_RDONLY, 0000);
  if (image_file == -1) return 0;
  // 确认读取的前四个4bit数组满足某种要求，分别要求满足2051, 10000, 28, 28
  read(image_file, &buf, sizeof(buf));
  CHECK(TRANS_32(buf), (0x08 << 8) + 0x03)
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
  CHECK(TRANS_32(buf), (0x08 << 8) + 0x01)
  read(label_file, &buf, sizeof(buf));
  CHECK(TRANS_32(buf), image_num);

  // 进行网络参数的相关读取
  int model_file = open("./layer_params.bin", O_RDONLY, 0000);
  if (model_file == -1) return 0;
  // 由于是从自己生成的文件中读取，那么不需要大小端的转换
  while (read(model_file, &buf, sizeof(buf))) {
  }
  // 进行文件数据的读取
  float image_data[image_height * image_width];
  int label_data;
  read_image_label(image_file, label_file, image_data, &label_data);
  printf("position [0, 26, 12] value is %f\n", image_data[26 * image_width + 12]);
  printf("label is %d\n", label_data);

  // 关闭文件
  close(image_file);
  close(label_file);
  return 0;
}
