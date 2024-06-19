import numpy as np

def Vrand(seg_result, ground_truth):
    # 将分割结果展平为一维数组
    seg_result = seg_result.flatten()
    ground_truth = ground_truth.flatten()
    P_ij = np.sum(np.logical_and(seg_result, ground_truth))
    P_i = np.sum(seg_result)
    P_j = np.sum(ground_truth)

    return 2*P_ij/(P_i+P_j)

def Vinfo(seg_result, ground_truth):
    # 计算分割结果和真实标签的直方图
    hist_1, _ = np.histogram(seg_result, bins=np.arange(seg_result.max() + 2))
    hist_2, _ = np.histogram(ground_truth, bins=np.arange(ground_truth.max() + 2))

    # 计算联合直方图
    joint_hist, _, _ = np.histogram2d(seg_result.flatten(), ground_truth.flatten(), bins=[hist_1.size, hist_2.size])

    # 计算互信息
    p_xy = joint_hist / float(np.sum(joint_hist))
    p_x = hist_1 / float(np.sum(hist_1))
    p_y = hist_2 / float(np.sum(hist_2))

    # 限制概率值在非零范围内
    p_xy = np.clip(p_xy, a_min=np.finfo(float).eps, a_max=None)
    p_x = np.clip(p_x, a_min=np.finfo(float).eps, a_max=None)
    p_y = np.clip(p_y, a_min=np.finfo(float).eps, a_max=None)

    mi = np.sum(np.sum(p_xy * np.log2(p_xy / (np.outer(p_x, p_y)))))

    # 计算Vinfo分数
    vinfo = 1 - (mi / np.log2(np.min([seg_result.size, ground_truth.size])))

    return vinfo
