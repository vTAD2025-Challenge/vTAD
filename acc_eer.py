import json
import os
import numpy as np
from sklearn.metrics import roc_curve


# 读取配置文件
def read_config(config_path):
    with open(config_path, 'r') as file:
        config = json.load(file)
    return config

# 读取文件内容
def read_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return lines

# 计算EER的函数
def compute_eer(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)  
    fnr = 1 - tpr  # False Negative Rate is 1 - TPR
    eer_threshold = thresholds[np.nanargmin(np.abs(fpr - fnr))]  # Find threshold where FPR = FNR
    eer = fpr[np.nanargmin(np.abs(fpr - fnr))]  # EER is the value at the threshold where FPR = FNR
    return eer, eer_threshold

# 计算ACC和EER的主函数
def calculate_acc_eer(model_output, test_path):
    predictions = {}
    true_labels = {}

    # 处理模型输出，提取预测标签和预测分数
    for line in model_output:
        parts = line.strip().split('|')
        attribute = parts[0]
        predicted_score = float(parts[3])
        predicted_label = int(parts[4])

        if attribute not in predictions:
            predictions[attribute] = []
        predictions[attribute].append((predicted_score, predicted_label))

    # 处理测试数据，提取真实标签
    for line in test_path:
        parts = line.strip().split('|')
        attribute = parts[0]
        true_label = int(parts[3])

        if attribute not in true_labels:
            true_labels[attribute] = []
        true_labels[attribute].append(true_label)

    acc_results = {}
    eer_results = {}

    # 计算每个标签的ACC和EER
    for attribute in predictions:
        pred_labels = [label for _, label in predictions[attribute]]
        true_labels_for_attr = true_labels[attribute]

        # ACC 计算
        correct_predictions = sum([1 for pred, true in zip(pred_labels, true_labels_for_attr) if pred == true])
        acc = (correct_predictions / len(true_labels_for_attr)) * 100
        acc_results[attribute] = acc

        # EER 计算
        predicted_scores = [score for score, _ in predictions[attribute]]
        eer, eer_threshold = compute_eer(np.array(true_labels_for_attr), np.array(predicted_scores))
        eer_results[attribute] = eer * 100  # 转换为百分比

    # 计算总ACC和总EER
    all_pred_labels = [label for labels in predictions.values() for _, label in labels]
    all_true_labels = [label for labels in true_labels.values() for label in labels]

    total_correct_predictions = sum([1 for pred, true in zip(all_pred_labels, all_true_labels) if pred == true])
    total_acc = (total_correct_predictions / len(all_true_labels)) * 100

    total_fnr = sum([1 for pred, true in zip(all_pred_labels, all_true_labels) if pred == 0 and true == 1]) / len(all_true_labels)
    total_fpr = sum([1 for pred, true in zip(all_pred_labels, all_true_labels) if pred == 1 and true == 0]) / len(all_true_labels)
    total_eer = (total_fnr + total_fpr) / 2 * 100  # 转换为百分比

    # 计算Male和Female的平均ACC和EER
    male_acc = [acc_results[attr] for attr in acc_results if "_M" in attr]
    female_acc = [acc_results[attr] for attr in acc_results if "_F" in attr]
    male_eer = [eer_results[attr] for attr in eer_results if "_M" in attr]
    female_eer = [eer_results[attr] for attr in eer_results if "_F" in attr]

    male_acc_avg = sum(male_acc) / len(male_acc) if male_acc else 0
    female_acc_avg = sum(female_acc) / len(female_acc) if female_acc else 0
    male_eer_avg = sum(male_eer) / len(male_eer) if male_eer else 0
    female_eer_avg = sum(female_eer) / len(female_eer) if female_eer else 0

    # 保存结果
    result_lines = []
    for attribute in acc_results:
        result_lines.append(f"{attribute} Accuracy: {acc_results[attribute]:.2f}")
    for attribute in eer_results:
        result_lines.append(f"{attribute} EER: {eer_results[attribute]:.2f}%")

    result_lines.append(f"Male Average Accuracy: {male_acc_avg:.2f}")
    result_lines.append(f"Female Average Accuracy: {female_acc_avg:.2f}")
    result_lines.append(f"Total Accuracy: {total_acc:.2f}")
    result_lines.append(f"Male Average EER: {male_eer_avg:.2f}%")
    result_lines.append(f"Female Average EER: {female_eer_avg:.2f}%")
    result_lines.append(f"Total EER: {total_eer:.2f}%")

    # 打印结果
    for line in result_lines:
        print(line)

    return result_lines

# 主函数，加载配置并运行EER和ACC计算
def main(config_path):
    # 加载配置文件
    config = read_config(config_path)

    # 读取模型输出和测试数据文件
    model_output_path = config["acc_eer"]["model_output"]
    test_path_path = config["acc_eer"]["test_path"]
    result_path = config["acc_eer"]["result_path"]

    model_output = read_file(model_output_path)
    test_path = read_file(test_path_path)

    # 计算ACC和EER
    result_lines = calculate_acc_eer(model_output, test_path)

    # 保存结果到文件
    with open(result_path, 'w') as result_file:
        for line in result_lines:
            result_file.write(line + "\n")

    print("Results saved to:", result_path)


if __name__ == "__main__":
    config_path = "configs/VADV_baseline.json"  # 配置文件路径
    main(config_path)
