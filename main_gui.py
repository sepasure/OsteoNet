import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import onnxruntime as ort
import os


class UniversalONNXApp:
    def __init__(self, root):
        self.root = root
        self.root.title("OsteoNet成骨预测 (ONNX版)")
        self.root.geometry("800x650")

        # 核心状态
        self.ort_session = None
        self.input_name = None
        self.input_shape = None
        self.classes = []

        # 初始化 UI
        self.setup_ui()

    def setup_ui(self):
        top_frame = tk.Frame(self.root, pady=15, bg="#f0f0f0")
        top_frame.pack(fill=tk.X)

        tk.Label(top_frame, text="步骤 1: ", bg="#f0f0f0", font=("bold", 12)).pack(side=tk.LEFT, padx=(20, 5))
        tk.Button(top_frame, text="加载 ONNX 模型文件", command=self.load_model, bg="white", width=20).pack(
            side=tk.LEFT)
        self.lbl_model_name = tk.Label(top_frame, text="未加载", fg="red", bg="#f0f0f0")
        self.lbl_model_name.pack(side=tk.LEFT, padx=10)

        tk.Label(top_frame, text="|  步骤 1.5 (可选): ", bg="#f0f0f0", font=("bold", 12)).pack(side=tk.LEFT, padx=5)
        tk.Button(top_frame, text="加载类别名(.txt)", command=self.load_labels, bg="white", width=15).pack(side=tk.LEFT)

        mid_frame = tk.Frame(self.root, pady=10)
        mid_frame.pack(expand=True, fill=tk.BOTH)

        self.img_label = tk.Label(mid_frame, text="[ 图片预览区域 ]\n请先加载模型，然后选择图片", bg="#e0e0e0")
        self.img_label.pack(expand=True, fill=tk.BOTH, padx=20, pady=10)

        bot_frame = tk.Frame(self.root, pady=20, bg="#f0f0f0")
        bot_frame.pack(fill=tk.X)

        tk.Button(bot_frame, text="步骤 2: 选择图片", command=self.load_image, height=2, width=15).pack(side=tk.LEFT,
                                                                                                        padx=(50, 20))
        tk.Button(bot_frame, text="步骤 3: 开始预测", command=self.predict, height=2, width=15, bg="#007ACC",
                  fg="white").pack(side=tk.LEFT, padx=20)

        self.result_label = tk.Label(bot_frame, text="", font=("微软雅黑", 16, "bold"), fg="#333", bg="#f0f0f0")
        self.result_label.pack(side=tk.LEFT, padx=30)

    def load_model(self):
        path = filedialog.askopenfilename(filetypes=[("ONNX Model", "*.onnx")])
        if not path: return

        try:
            self.ort_session = ort.InferenceSession(path)

            self.input_name = self.ort_session.get_inputs()[0].name
            self.input_shape = self.ort_session.get_inputs()[0].shape

            # 更新 UI
            self.lbl_model_name.config(text=os.path.basename(path), fg="green")
            messagebox.showinfo("成功",
                                f"模型加载成功！\n输入节点: {self.input_name}\n检测到输入尺寸: {self.input_shape}")

        except Exception as e:
            self.lbl_model_name.config(text="加载失败", fg="red")
            messagebox.showerror("错误", f"模型文件损坏或不兼容:\n{e}")

    def load_labels(self):
        """加载 txt 文件，每行一个类别名"""
        path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
        if not path: return
        try:
            with open(path, "r", encoding='utf-8') as f:
                self.classes = [line.strip() for line in f.readlines()]
            messagebox.showinfo("提示", f"已加载 {len(self.classes)} 个类别名称")
        except:
            messagebox.showerror("错误", "无法读取类别文件")

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.png *.jpeg *.bmp")])
        if not path: return

        self.curr_img_path = path

        # 显示图片
        img = Image.open(path)
        img.thumbnail((400, 400))
        photo = ImageTk.PhotoImage(img)
        self.img_label.config(image=photo, text="")
        self.img_label.image = photo  # 保持引用

        self.result_label.config(text="")  # 清空旧结果

    def predict(self):
        if not self.ort_session:
            tk.messagebox.showwarning("提示", "请先加载 ONNX 模型！")
            return
        if not hasattr(self, 'curr_img_path'):
            tk.messagebox.showwarning("提示", "请先选择图片！")
            return

        try:
            image = Image.open(self.curr_img_path).convert('RGB')

            target_h, target_w = 224, 224
            if self.input_shape and len(self.input_shape) == 4:
                # 形状通常是 [1, 3, H, W]
                h = self.input_shape[2]
                w = self.input_shape[3]
                if isinstance(h, int) and isinstance(w, int):
                    target_h, target_w = h, w

            image = image.resize((target_w, target_h))


            img_data = np.array(image).astype(np.float32) / 255.0

            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

            img_data = (img_data - mean) / std

            img_data = img_data.transpose(2, 0, 1)

            img_data = np.expand_dims(img_data, axis=0)
            img_data = img_data.astype(np.float32)



            outputs = self.ort_session.run(None, {self.input_name: img_data})
            logits = outputs[0][0]


            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / np.sum(exp_logits)

            pred_idx = np.argmax(probs)
            confidence = probs[pred_idx] * 100

            class_str = self.classes[pred_idx] if pred_idx < len(self.classes) else f"类别 {pred_idx}"
            self.result_label.config(text=f"预测: {class_str}\n置信度: {confidence:.2f}%", fg="blue")

        except Exception as e:
            tk.messagebox.showerror("预测出错", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    app = UniversalONNXApp(root)
    root.mainloop()