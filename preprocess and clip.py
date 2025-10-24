import cv2
import os
import numpy as np
import pandas as pd
from PIL import Image
import sys


def estimate_noise_std(image_cv):
    if image_cv is None:
        return 0
    if len(image_cv.shape) == 2:
        gray = image_cv
    else:
        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    std_dev = laplacian.std()
    return std_dev


def calculate_snr(image_cv):
    if image_cv is None:
        return 0

    if len(image_cv.shape) == 2:
        gray = image_cv
    else:
        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)

    mean_signal = np.mean(gray)
    std_noise = estimate_noise_std(image_cv)

    if std_noise == 0:
        return float('inf')

    snr = mean_signal / std_noise
    return snr


def batch_denoise_and_export(input_dir, output_dir, excel_path, kernel_size, sigmaX):
    if not os.path.exists(output_dir):
        print(f"输出文件夹 '{output_dir}' 不存在，正在创建...")
        os.makedirs(output_dir)

    if not os.path.exists(input_dir):
        print(f"错误：输入文件夹 '{input_dir}' 未找到。请确保路径正确。", file=sys.stderr)
        return

    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    results_data = []
    processed_count = 0

    print(f"开始处理文件夹 '{input_dir}' 中的图片...")

    try:
        file_list = os.listdir(input_dir)
    except FileNotFoundError:
        print(f"错误：无法访问输入文件夹 '{input_dir}'。", file=sys.stderr)
        return

    for filename in file_list:
        if filename.lower().endswith(supported_formats):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            try:
                original_image = cv2.imread(input_path)
                if original_image is None:
                    print(f"警告：无法读取 '{filename}'，已跳过。")
                    continue

                snr_before = calculate_snr(original_image)

                denoised_image = cv2.GaussianBlur(original_image, kernel_size, sigmaX)

                snr_after = calculate_snr(denoised_image)

                print(f"  已处理: {filename} (SNR: {snr_before:.2f} -> {snr_after:.2f})")

                results_data.append({
                    '文件名': filename,
                    '信噪比 (处理前)': snr_before,
                    '信噪比 (处理后)': snr_after
                })

                cv2.imwrite(output_path, denoised_image)
                processed_count += 1

            except Exception as e:
                print(f"处理图片 '{filename}' 时发生错误: {e}", file=sys.stderr)

    if processed_count > 0:
        print(f"\n共处理了 {processed_count} 张图片。正在将结果保存到Excel文件...")

        df = pd.DataFrame(results_data)
        df = df[['文件名', '信噪比 (处理前)', '信噪比 (处理后)']]

        try:
            excel_dir = os.path.dirname(excel_path)
            if excel_dir and not os.path.exists(excel_dir):
                print(f"正在创建Excel输出目录: {excel_dir}")
                os.makedirs(excel_dir)

            df.to_excel(excel_path, index=False, float_format='%.2f')
            print(f"成功！结果已保存到文件: '{excel_path}'")
        except Exception as e:
            print(f"保存Excel文件时发生错误: {e}", file=sys.stderr)

    else:
        print(f"在 '{input_dir}' 中未找到任何支持格式的图片进行处理。")


def split_images(input_folder, output_folder, crop_size=(500, 500), output_format="jpeg"):
    if not os.path.exists(output_folder):
        print(f"输出文件夹 '{output_folder}' 不存在，正在创建...")
        os.makedirs(output_folder)

    if not os.path.exists(input_folder):
        print(f"错误：输入文件夹 '{input_folder}' 未找到。请确保路径正确。", file=sys.stderr)
        return

    print(f"开始裁剪 '{input_folder}' 中的图片...")
    processed_count = 0
    total_crops = 0

    for file_name in os.listdir(input_folder):
        input_path = os.path.join(input_folder, file_name)
        if not os.path.isfile(input_path):
            continue

        try:
            with Image.open(input_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                width, height = img.size
                crop_width, crop_height = crop_size

                num_cols = width // crop_width
                num_rows = height // crop_height

                if num_rows == 0 or num_cols == 0:
                    print(f"  跳过: '{file_name}' 尺寸 ({width}x{height}) 小于裁剪尺寸 ({crop_width}x{crop_height})。")
                    continue

                print(f"  正在处理: {file_name} (将裁剪为 {num_rows}x{num_cols} 个图块)")

                for row in range(num_rows):
                    for col in range(num_cols):
                        left = col * crop_width
                        upper = row * crop_height
                        right = left + crop_width
                        lower = upper + crop_height

                        cropped_img = img.crop((left, upper, right, lower))

                        base_name = os.path.splitext(file_name)[0]
                        output_file_name = f"{base_name}_{row}_{col}.{output_format.lower()}"
                        output_path = os.path.join(output_folder, output_file_name)

                        cropped_img.save(output_path, format=output_format.upper())
                        total_crops += 1

                processed_count += 1

        except Exception as e:
            print(f"处理图片 '{file_name}' 时发生错误: {e}", file=sys.stderr)

    if processed_count > 0:
        print(f"\n裁剪完成。共处理了 {processed_count} 张大图，生成了 {total_crops} 个图块。")
        print(f"裁剪后的文件保存在: '{output_folder}'")
    else:
        print(f"在 '{input_folder}' 中未找到可处理的图片。")


def main():
    DO_DENOISING = True

    DENOISE_INPUT_FOLDER = './input_original'
    DENOISE_OUTPUT_FOLDER = './output_denoised'
    SNR_EXCEL_PATH = './output_denoised/snr_comparison_results.xlsx'

    GAUSSIAN_KERNEL_SIZE = (5, 5)
    GAUSSIAN_SIGMAX = 0

    DO_CROPPING = True

    CROP_INPUT_FOLDER = DENOISE_OUTPUT_FOLDER

    CROP_OUTPUT_FOLDER = './output_cropped'
    CROP_SIZE = (500, 500)
    CROP_FORMAT = "jpeg"

    print("=== 开始图像处理流水线 ===")

    if DO_DENOISING:
        print("\n--- [步骤 1: 批量去噪与SNR分析] ---")
        batch_denoise_and_export(
            input_dir=DENOISE_INPUT_FOLDER,
            output_dir=DENOISE_OUTPUT_FOLDER,
            excel_path=SNR_EXCEL_PATH,
            kernel_size=GAUSSIAN_KERNEL_SIZE,
            sigmaX=GAUSSIAN_SIGMAX
        )
        print("--- [步骤 1: 完成] ---")
    else:
        print("\n--- [步骤 1: 批量去噪与SNR分析] (已跳过) ---")

    if DO_CROPPING:
        print("\n--- [步骤 2: 图像裁剪] ---")

        if not os.path.exists(CROP_INPUT_FOLDER):
            print(f"错误：裁剪的输入文件夹 '{CROP_INPUT_FOLDER}' 未找到！", file=sys.stderr)
            if DO_DENOISING and CROP_INPUT_FOLDER == DENOISE_OUTPUT_FOLDER:
                print(" -> 可能是因为步骤1 (去噪) 失败或未产生任何输出。", file=sys.stderr)
            print("--- [步骤 2: 已终止] ---")
        else:
            split_images(
                input_folder=CROP_INPUT_FOLDER,
                output_folder=CROP_OUTPUT_FOLDER,
                crop_size=CROP_SIZE,
                output_format=CROP_FORMAT
            )
            print("--- [步骤 2: 完成] ---")
    else:
        print("\n--- [步骤 2: 图像裁剪] (已跳过) ---")

    print("\n=== 图像处理流水线全部结束 ===")


if __name__ == '__main__':
    main()