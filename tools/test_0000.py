import torch
import torch.onnx
from PIL import Image
import torchvision.transforms as transforms
import onnxruntime as rt
import numpy as np


######################################################################################################################
# pth模型转onnx模型
def pth_to_onnx(input, pth_path, onnx_path, input_names=['input'], output_names=['output'], device='cpu'):
    if not onnx_path.endswith('.onnx'):
        print('Warning! The onnx model name is not correct, please give a name that ends with \'.onnx\'!')
        return 0

    model = torch.load(pth_path)  # 并不能像tf一样直接导入，同一级目录下需要有模型的定义文件
    model.eval()  # .eval()用于通知BN层和dropout层，采用推理模式而不是训练模式
    model.to(device)

    # 指定模型的输入，以及onnx的输出路径，input的维度决定了onnx的固定输入维度，例如如果input的batch是1，那onnx的固定batch维度也是1
    # 通过dynamic_axes参数也可以设定动态维度，其中dynamic_axes可以以列表形式设置也可以以字典形式设置，以字典形式设置会同步设置不同维度的名称
    # 动态维度参考：https://blog.csdn.net/xz1308579340/article/details/124908825?utm_medium=distribute.pc_aggpage_search_result.none-task-blog-2~aggregatepage~first_rank_ecpm_v1~rank_v31_ecpm-2-124908825-null-null.pc_agg_new_rank&utm_term=torch%E8%BD%AConnx%E5%A4%9Abatch&spm=1000.2123.3001.4430
    torch.onnx.export(model, input, onnx_path, verbose=True, opset_version=11, input_names=input_names,
                      output_names=output_names)
    print("Exporting .pth model to onnx model has been successful!")


######################################################################################################################
# 采用torch进行预测
def pth_predict(image, pth_path):
    # 读取模型
    model = torch.load(pth_path)  # 并不能像tf一样直接导入，同一级目录下需要有模型的定义文件
    # print(model)

    # 单张图片推理
    model.cpu().eval()  # .eval()用于通知BN层和dropout层，采用推理模式而不是训练模式
    with torch.no_grad():  # torch.no_grad()用于整体修改模型中每一层的requires_grad属性，使得所有可训练参数不能修改，且正向计算时不保存中间过程，以节省内存
        output = torch.squeeze(model(image))
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    # 输出结果
    print('Pth Predicted:', predict.numpy())
    return predict.numpy()


######################################################################################################################
# 采用onnx进行预测
def onnx_predict(image, onnx_path):
    def softmax(x, axis=None):
        x = x - x.max(axis=axis, keepdims=True)
        y = np.exp(x)
        return y / y.sum(axis=axis, keepdims=True)

    # 读取onnx模型，安装GPUonnx，并设置providers = ['GPUExecutionProvider']，可以实现GPU运行onnx
    providers = ['CPUExecutionProvider']
    m = rt.InferenceSession(onnx_path, providers=providers)

    # 推理onnx模型
    output_names = ["output"]
    onnx_pred = m.run(output_names, {"input": image})

    # 输出结果
    print('ONNX Predicted:', softmax(onnx_pred[0][0]))
    return softmax(onnx_pred[0][0])


######################################################################################################################
# 获取图片
def get_image(image_path):
    # 等比例拉伸图片，多余部分填充value
    def resize_padding(image, target_length, value=0):
        h, w = image.size  # 获得原始尺寸
        ih, iw = target_length, target_length  # 获得目标尺寸
        scale = min(iw / w, ih / h)  # 实际拉伸比例
        nw, nh = int(scale * w), int(scale * h)  # 实际拉伸后的尺寸
        image_resized = image.resize((nh, nw), Image.ANTIALIAS)  # 实际拉伸图片
        image_paded = Image.new("RGB", (ih, iw), value)
        dw, dh = (iw - nw) // 2, (ih - nh) // 2
        image_paded.paste(image_resized, (dh, dw, nh + dh, nw + dw))  # 居中填充图片
        return image_paded

    # 变换函数
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])])

    # 读取图片并预处理
    image = resize_padding(Image.open(image_path), 128)
    image = transform(image)
    image = image.reshape(1, 3, 128, 128)

    return image


if __name__ == '__main__':
    pth_path = "xxx.pth"
    onnx_path = "xxx.onnx"
    image_path = "car.jpg"

    input = torch.randn(1, 3, 128, 128)
    pth_to_onnx(input, pth_path, onnx_path)

    image = get_image(image_path)
    pth_result = pth_predict(image, pth_path)
    onnx_result = onnx_predict(image.numpy(), onnx_path)
    np.testing.assert_allclose(pth_result, onnx_result, rtol=1e-4)