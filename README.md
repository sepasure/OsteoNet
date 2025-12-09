# OsteoNet
## ✨ 功能特点
- 支持加载任意 `.onnx` 格式的分类模型
- 图形化界面 (GUI)，操作简单
- 也就是不需要安装 Python 环境，直接运行 exe 即可 (见 Release 下载)

## 📥 下载与使用 (针对普通用户)
请前往 [Releases 页面](https://github.com/sepasure/OsteoNet/releases/tag/classification) 下载最新版本的 `main_gui.exe` 和 `model.onnx`。

1. 双击打开 `main_gui.exe`
2. 点击“加载模型”选择 `.onnx` 文件
3. 选择图片进行预测

## 💻 开发者运行 (针对程序员)
如果你想修改代码，请按以下步骤操作：

```bash
# 1. 克隆项目
git clone [https://github.com/你的用户名/你的仓库名.git](https://github.com/你的用户名/你的仓库名.git)

# 2. 安装依赖
pip install onnxruntime numpy pillow

# 3. 运行
python main_gui.py
