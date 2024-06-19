import numpy as np
import matplotlib.pyplot as plt


modes = {'U-Net':'UNet', 'SS-UNet':'UNet_SNN', 'SNN-VGG':'SNN_VGG', 'Spiking UNet':'SpikingUnet', 'DMTS-UNet':'DMTSpikingUnet'}
losses = {'WCE':'Weighted_Cross_Entropy_Loss', 'DWCE':'DWCE_Loss'}
types = {'IOU':'IOU', 'pix accuracy':'pix_acc', 'train loss':'train_loss', 'test loss':'test_loss'}

def read_data(m_name, l_name, t_name):
    max_len = 100
    x = list()
    y = list()
    with open(f'visualization/ISBI_2012_EM/{modes[m_name]}_{losses[l_name]}_{types[t_name]}.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            xi, yi = line.split(' ')
            x.append(int(float(xi)))
            y.append(float(yi))

            max_len -= 1
            if max_len == 0:
                break
    return x, y

def draw(t_name):
    modes_need = ['U-Net', 'SNN-VGG', 'Spiking UNet', 'DMTS-UNet', 'SS-UNet']
    for idx in range(len(modes_need)):
        xi, yi = read_data(modes_need[idx], 'WCE', t_name)

        # 绘制函数图形
        plt.plot(xi, yi, label=modes_need[idx])
        plt.legend()
        # 添加标签和标题
        plt.xlabel('epochs')
        plt.ylabel(f'{t_name}')
        plt.title(f'{t_name} on WCE loss')
        # 显示图形
        plt.savefig(f"total/{types[t_name]}_WCE.png", bbox_inches='tight')
    plt.clf()

    for idx in range(len(modes_need)):
        xi, yi = read_data(modes_need[idx], 'DWCE', t_name)
        
        # 绘制函数图形
        plt.plot(xi, yi, label=modes_need[idx])
        plt.legend()
        # 添加标签和标题
        plt.xlabel('epochs')
        plt.ylabel(f'{t_name}')
        plt.title(f'{t_name} on DWCE loss')
        # 显示图形
        plt.savefig(f"total/{types[t_name]}_DWCE.png", bbox_inches='tight')
    plt.clf()

def draw_loss():
    modes_need = ['U-Net', 'SNN-VGG', 'Spiking UNet', 'DMTS-UNet', 'SS-UNet']
    for idx in range(len(modes_need)):
        xi, yi1 = read_data(modes_need[idx], 'WCE', 'train loss')
        _, yi2 = read_data(modes_need[idx], 'WCE', 'test loss')
        # 绘制函数图形
        plt.plot(xi, yi1, label=f"{modes_need[idx]}_train")
        plt.plot(xi, yi2, label=f"{modes_need[idx]}_test")
        plt.legend()
        # 添加标签和标题
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.title(f'train and test loss on WCE loss')
        # 显示图形
        plt.savefig(f"total/loss_WCE.png", bbox_inches='tight')
    plt.clf()
        
    for idx in range(len(modes_need)):
        xi, yi1 = read_data(modes_need[idx], 'DWCE', 'train loss')
        _, yi2 = read_data(modes_need[idx], 'DWCE', 'test loss')
        # 绘制函数图形
        plt.plot(xi, yi1, label=f"{modes_need[idx]}_train")
        plt.plot(xi, yi2, label=f"{modes_need[idx]}_test")
        plt.legend()
        # 添加标签和标题
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.title(f'train and test loss on DWCE loss')
        # 显示图形
        plt.savefig(f"total/loss_DWCE.png", bbox_inches='tight')
    plt.clf()

draw('IOU')
draw('pix accuracy')
# draw_loss()
draw('train loss')
draw('test loss')