import matplotlib.pyplot as plt
import numpy as np

# iou_list = [0.8353571428571428, 
#             0.696077380952381,
#             0.6712976190476191,
#             0.6241011904761905,
#             0.6289226190476189,
#             0.6037083333333336,
#             0.5574702380952379,
#             0.4996011904761906,
#             0.4563333333333332,
#             0.3640952380952382,
#             0.4156547619047618]

# recall below
iou_list = [0.8705714285714286,
            0.7352023809523814,
            0.708607142857143,
            0.6616845238095237,
            0.6508750000000001,
            0.6547023809523808,
            0.5903273809523807,
            0.5319404761904765,
            0.4715952380952381,
            0.39122619047619056,
            0.43440476190476185]

iou_avg = sum(iou_list) / len(iou_list)

# Increase the figure size
plt.figure(figsize=(10, 6))
plt.rcParams["font.size"] = 15

# 棒グラフを描画
plt.bar(range(len(iou_list)), iou_list, width=0.6)

# 平均を示す横線を追加
plt.axhline(y=iou_avg, color='red', linestyle='--', label='Average IOU')

# グラフにタイトルと軸ラベルを追加
# plt.title('IOU Values')
# plt.xlabel('Overlap Ratio[%]')  # カスタムの目盛りラベル
# plt.ylabel('IoU')
plt.xlabel('Overlap Ratio[%]')  # カスタムの目盛りラベル
plt.ylabel('Recall')
# Add grid lines
# plt.grid(True, linestyle='-', alpha=0.6)

# カスタムの目盛りラベルを指定
custom_labels = []
for i in range(11):
    custom_labels.append(f"{i * 10}")
# custom_labels[-1] = "100"
# stride = 2
plt.xticks(range(len(iou_list)), custom_labels)
plt.tick_params(axis='both', which='both', direction='in')

# y軸の範囲を0から1に設定
plt.ylim(0, 1)

# y軸の目盛りを自動的に10分の1に調整
plt.yticks(np.arange(0, 1.1, 0.1))

# y軸の目盛りを非表示にする
plt.tick_params(axis='x', which='both', top=False, bottom=False)
plt.tick_params(axis='y', which='both', left=True, right=True)
# plt.xticks(range(0, len(iou_list), stride), custom_labels[::stride])
# plt.xticks([i * 2 for i in range(len(iou_list))], custom_labels)

# グラフを表示
plt.savefig("recall.png")
