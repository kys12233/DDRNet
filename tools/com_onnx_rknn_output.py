# 比较rknn和onnx的输出结果
# onnx_output = './onnx_output_txt/onnx_output_save_3_0_255.txt'
# rknn_output = './rknn_output_txt/ddrnet_save_3_0_255_opset_version_12_argmax_float32_02_rknn_output.txt'

# onnx_output = './onnx_output_txt/onnx_output_save_3_0_255_demo_02.txt'
# rknn_output = './rknn_output_txt/ddrnet_save_3_0_255_opset_version_12_argmax_float32_02_demo_02_rknn_output.txt'

# onnx_output = './onnx_output_txt/onnx_output_save_3_0_255_demo_03.txt'
# rknn_output = './rknn_output_txt/ddrnet_save_3_0_255_opset_version_12_argmax_float32_02_demo_03_rknn_output.txt'

onnx_output = './onnx_output_txt/onnx_output_save_3_0_255_train_qt_demo.txt'
# rknn_output = './rknn_output_txt/ddrnet_save_3_0_255_opset_version_12_argmax_float32_train_qt_train_qt____from_grassDataSets_20231207____data____mask__label__start__1__10__1____1695801069.291000_rknn_output.txt'
rknn_output = './rknn_output_txt/ddrnet_save_3_0_255_opset_version_12_argmax_float32_train_qt_train_qt_train_demo_rknn_output.txt'

# 完全相等的数据有多少（即位置一样，结果也一样）
same_count = 0

with open(onnx_output, 'r') as file:
    f_onnx = file.readlines()
    file.close()

with open(rknn_output, "r") as file_02:
    f_rknn = file_02.readlines()
    file_02.close()

for i in range(len(f_onnx)):
    f_onnx[i] = f_onnx[i].split('\n')[0]

for i in range(len(f_rknn)):
    f_rknn[i] = f_rknn[i].split('\n')[0]

print(len(f_onnx))
print(len(f_rknn))
if len(f_onnx) == len(f_rknn):
    for i in range(len(f_onnx)):
        if f_onnx[i] == f_rknn[i]:
            same_count += 1

print("相同位置的值相同的个数是：", same_count)
print("相同位置的值相同的个数占总的像素的比例是：", same_count / len(f_onnx))
