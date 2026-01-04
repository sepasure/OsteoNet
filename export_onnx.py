import torch
import torch.nn as nn
import os

# 1. 引用你的模型定义
# 确保 model.py 文件在当前目录下
from model import efficientnetv2_m as create_model


def export_onnx():

    weights_path = ".\\weights\\5FOLD\\5D\\fold_1_best_model.pth"

    my_num_classes = 2

    img_size = 224

    # 导出的文件名
    output_onnx_name = "fold_1_best_model.onnx"
    # ======================================================================

    print(f"1. 正在重建网络结构 (类别数: {my_num_classes})...")
    # 注意：这里我们告诉模型，你要构建一个只有 my_num_classes 输出的网络
    # 这样它的最后一层全连接层就会自动适配你的权重
    try:
        model = create_model(num_classes=my_num_classes)
    except TypeError:
        # 某些特殊的 model.py 实现可能不叫 num_classes，备用尝试
        print("提示：尝试使用 classes 参数...")
        model = create_model(classes=my_num_classes)

    print(f"2. 正在加载微调后的权重: {weights_path}...")
    device = torch.device('cpu')

    # 加载权重
    checkpoint = torch.load(weights_path, map_location=device)

    # 处理权重字典 (兼容只保存 state_dict 或保存了整个 checkpoint 的情况)
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
    else:
        # 极其少见的情况：直接保存了模型对象
        print("检测到直接保存了模型对象，尝试提取 state_dict...")
        state_dict = checkpoint.state_dict()

    # 处理 'module.' 前缀 (如果使用了 DataParallel 多卡训练)
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v

    try:
        # strict=True 确保每一个参数都完美匹配
        model.load_state_dict(new_state_dict, strict=True)
        print("✅ 权重加载成功！结构匹配完美。")
    except RuntimeError as e:
        print(f"\n❌ 权重加载失败！最常见的原因是类别数不匹配。")
        print(f"错误详情: {e}")
        print(f"\n请检查：你代码里的 my_num_classes={my_num_classes} 是否真的等于你训练时的类别数？")
        return

    model.eval()

    print(f"3. 开始导出 ONNX (Input: {img_size}x{img_size})...")

    # 创建虚拟输入数据
    dummy_input = torch.randn(1, 3, img_size, img_size).to(device)

    # 导出
    torch.onnx.export(
        model,
        dummy_input,
        output_onnx_name,
        opset_version=12,  # 建议使用 11 或 12
        input_names=['input'],  # 输入节点名称 (在 GUI 代码里会用到)
        output_names=['output'],  # 输出节点名称
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

    print(f"\n🎉 成功！已生成: {output_onnx_name}")
    print(f"文件大小: {os.path.getsize(output_onnx_name) / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    export_onnx()