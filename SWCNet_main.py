import torch
import torch.nn as nn
import numpy as np
import scipy.io as sio
import torchvision.transforms as transforms
from SWCNet_model import Unmixing
from utility import load_HSI, hyperVca, load_data, reconstruction_SADloss
from utility import plotAbundancesGT, plotAbundancesSimple, plotEndmembersAndGT, reconstruct
import time
import os
import pandas as pd
import pywt
from scipy.spatial.distance import cosine
start_time = time.time()


def cos_dist(x1, x2):
    return cosine(x1, x2)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

datasetnames = {'Samson': 'Samson',
                'houston': 'houston',
                'moffett': 'moffett',
                }
dataset = "houston"
hsi = load_HSI("Datasets/" + datasetnames[dataset] + ".mat")
data = hsi.array()
endmember_number = hsi.gt.shape[0]
col = hsi.cols
line = hsi.rows
band_number = data.shape[1]
batch_size = 1
EPOCH = 300
num_runs = 1
if dataset == "Samson":
    EPOCH = 400  
    drop_out = 0.1
    learning_rate = 0.03
    step_size = 35
    gamma = 0.7
    weight_decay = 2e-4
if dataset == "houston":
    drop_out = 0.1
    learning_rate = 0.005
    step_size = 40
    gamma = 0.8
    weight_decay = 1e-3
if dataset == "moffett":
    seed = 2
    drop_out = 0.1
    learning_rate = 0.02
    step_size = 35
    gamma = 0.5
    weight_decay = 5e-4
MSE = torch.nn.MSELoss(reduction='mean')

end = []
abu = []
r = []

output_path = 'Results'
method_name = 'SWC-Net'
mat_folder = output_path + '/' + method_name + '/' + datasetnames[dataset] + '/' + 'mat'
endmember_folder = output_path + '/' + method_name + '/' + datasetnames[dataset] + '/' + 'endmember'
abundance_folder = output_path + '/' + method_name + '/' + datasetnames[dataset] + '/' + 'abundance'
if not os.path.exists(mat_folder):
    os.makedirs(mat_folder)
if not os.path.exists(endmember_folder):
    os.makedirs(endmember_folder)
if not os.path.exists(abundance_folder):
    os.makedirs(abundance_folder)

for run in range(1, num_runs + 1):
    print('Start training!', 'run:', run)
    abundance_GT = torch.from_numpy(hsi.abundance_gt)
    abundance_GT = torch.reshape(abundance_GT, (col * line, endmember_number))
    original_HSI = torch.from_numpy(data)
    original_HSI = torch.reshape(original_HSI.T, (band_number, col, line))
    abundance_GT = torch.reshape(abundance_GT.T, (endmember_number, col, line))
    image = np.array(original_HSI)

    families = pywt.families()
    wavelets_in_family = pywt.wavelist('sym')

    xiaobo = image.copy()
    xiaobo2 = image.copy()
    xiaobo3 = image.copy()
    xiaobo4 = image.copy()
    for i in range(image.shape[0]):
        wavelet = 'bior3.3'  # 选择小波基函数，这里使用 bior3.3 小波
        level = 1  # 设置分解的尺度级别
        signal = image[i]
        if (image.shape[1] % 2) != 0:
            pad_width = [(0, 1), (0, 1)]
            signal = np.pad(signal, pad_width, mode='edge')
        coeffs = pywt.swt2(signal, wavelet, 1)
        coeffs2 = coeffs.copy()
        coeffs3 = coeffs.copy()
        coeffs4 = coeffs.copy()
        for j in range(len(coeffs)):
            approx, (h1, v1, d1) = coeffs[j]
            coeffs[j] = (approx, (np.zeros_like(h1), np.zeros_like(v1), np.zeros_like(d1)))

            coeffs2[j] = (np.zeros_like(approx), (h1 + np.random.normal(0, 0.1, h1.shape), np.zeros_like(v1), np.zeros_like(d1)))
            coeffs3[j] = (np.zeros_like(approx), (np.zeros_like(h1), v1 + np.random.normal(0, 0.1, v1.shape), np.zeros_like(d1)))
            coeffs4[j] = (np.zeros_like(approx), (np.zeros_like(h1), np.zeros_like(v1), d1 + np.random.normal(0, 0.1, d1.shape)))

        res = pywt.iswt2(coeffs, wavelet)
        res2 = pywt.iswt2(coeffs2, wavelet)
        res3 = pywt.iswt2(coeffs3, wavelet)
        res4 = pywt.iswt2(coeffs4, wavelet)
        if (image.shape[1] % 2) != 0:
            xiaobo[i] = res[:-1, :-1]
            xiaobo2[i] = res2[:-1, :-1]
            xiaobo3[i] = res3[:-1, :-1]
            xiaobo4[i] = res4[:-1, :-1]
        else:
            xiaobo[i] = res
            xiaobo2[i] = res2
            xiaobo3[i] = res3
            xiaobo4[i] = res4

    endmembers, _, _ = hyperVca(data.T, endmember_number, datasetnames[dataset])
    VCA_endmember = torch.from_numpy(endmembers)
    GT_endmember = hsi.gt.T
    endmember_init = VCA_endmember.unsqueeze(2).unsqueeze(3).float()

    # load data
    train_dataset = load_data(img=original_HSI, transform=transforms.ToTensor())
    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)

    net = Unmixing(band_number, endmember_number, drop_out, col).cuda()

    endmember_name = datasetnames[dataset] + '_run' + str(run)
    endmember_path = endmember_folder + '/' + endmember_name
    endmember_path2 = endmember_folder + '/' + endmember_name + 'vca'

    abundance_name = datasetnames[dataset] + '_run' + str(run)
    abundance_path = abundance_folder + '/' + abundance_name


    def weights_init(m):
        nn.init.kaiming_normal_(net.layer1[0].weight.data)
        nn.init.kaiming_normal_(net.layer1[4].weight.data)
        nn.init.kaiming_normal_(net.layer1[8].weight.data)

        nn.init.kaiming_normal_(net.layer2[0].weight.data)
        nn.init.kaiming_normal_(net.layer2[4].weight.data)
        nn.init.kaiming_normal_(net.layer2[8].weight.data)

        nn.init.kaiming_normal_(net.layer3[0].weight.data)
        nn.init.kaiming_normal_(net.layer3[4].weight.data)
        nn.init.kaiming_normal_(net.layer3[8].weight.data)

        nn.init.kaiming_normal_(net.layer4[0].weight.data)
        nn.init.kaiming_normal_(net.layer4[4].weight.data)
        nn.init.kaiming_normal_(net.layer4[8].weight.data)


    net.apply(weights_init)

    # decoder weight init by VCA

    model_dict = net.state_dict()
    model_dict["decoderlayer4.0.weight"] = endmember_init
    net.load_state_dict(model_dict)

    # optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    y = torch.from_numpy(xiaobo).unsqueeze(0).to(device)
    y2 = torch.from_numpy(xiaobo2).unsqueeze(0).to(device)
    y3 = torch.from_numpy(xiaobo3).unsqueeze(0).to(device)
    y4 = torch.from_numpy(xiaobo4).unsqueeze(0).to(device)

    for epoch in range(EPOCH):
        for i, x in enumerate(train_loader):
            x = x.cuda()
            net.train().cuda()
            torch.cuda.empty_cache()

            en_abundance, reconstruction_result = net(y, y2, y3, y4)
            abundanceLoss = reconstruction_SADloss(x, reconstruction_result)
            total_loss = abundanceLoss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if epoch % 100 == 0:
                """print(ELoss.cpu().data.numpy())"""
                print("Epoch:", epoch, "| loss: %.4f" % total_loss.cpu().data.numpy())
        scheduler.step()
    en_abundance, reconstruction_result = net(y, y2, y3, y4)
    en_abundance = torch.squeeze(en_abundance)

    en_abundance = torch.reshape(en_abundance, [endmember_number, col * line])
    en_abundance = en_abundance.T
    en_abundance = torch.reshape(en_abundance, [col, line, endmember_number])
    abundance_GT = torch.reshape(abundance_GT, [endmember_number, col * line])
    abundance_GT = abundance_GT.T
    abundance_GT = torch.reshape(abundance_GT, [line, col, endmember_number])
    en_abundance = en_abundance.cpu().detach().numpy()
    abundance_GT = abundance_GT.cpu().detach().numpy()

    endmember_hat = net.state_dict()["decoderlayer4.0.weight"].cpu().numpy()
    endmember_hat = np.squeeze(endmember_hat)
    endmember_hat = endmember_hat.T

    GT_endmember = GT_endmember.T
    y_hat = reconstruct(en_abundance, endmember_hat)
    RE = np.sqrt(np.mean(np.mean((y_hat - data) ** 2, axis=1)))
    r.append(RE)

    sio.savemat(mat_folder + '/' + method_name + '_run' + str(run) + '.mat', {'A': en_abundance,
                                                                              'E': endmember_hat,
                                                                              })

    plotAbundancesSimple(en_abundance, abundance_GT, abundance_path, abu)
    plotEndmembersAndGT(endmember_hat, GT_endmember, endmember_path, end)

    torch.cuda.empty_cache()

    print('-' * 70)
end_time = time.time()
end = np.reshape(end, (-1, endmember_number + 1))
abu = np.reshape(abu, (-1, endmember_number + 1))
dt = pd.DataFrame(end)
dt2 = pd.DataFrame(abu)
dt3 = pd.DataFrame(r)
dt.to_csv(output_path + '/' + method_name + '/' + datasetnames[dataset] + '/' + datasetnames[
    dataset] + '各端元SAD及mSAD运行结果.csv')
dt2.to_csv(output_path + '/' + method_name + '/' + datasetnames[dataset] + '/' + datasetnames[
    dataset] + '各丰度图RMSE及mRMSE运行结果.csv')
dt3.to_csv(output_path + '/' + method_name + '/' + datasetnames[dataset] + '/' + datasetnames[
    dataset] + '重构误差RE运行结果.csv')
abundanceGT_path = output_path + '/' + method_name + '/' + datasetnames[dataset] + '/' + datasetnames[
    dataset] + '参照丰度图'
plotAbundancesGT(hsi.abundance_gt, abundanceGT_path)
print('程序运行时间为:', end_time - start_time, 's')