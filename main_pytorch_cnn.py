from torch.utils.data import DataLoader
import torch.utils.data as Data
import gc
import os
import torch
import torch.nn as nn
import torch.optim as optim
import datetime
import numpy as np
from sklearn.model_selection import StratifiedKFold
# import pandas as pd
from nnModelST_pytorch import cnn
from sklearn.model_selection import KFold
from sklearn.metrics import cohen_kappa_score, f1_score, recall_score, roc_auc_score

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

is_support = torch.cuda.is_available()
if is_support:
    device = torch.device('cuda:0')
    #device = torch.device('cuda:1')
else:
    device = torch.device('cpu')


def datanorm(x):
    for i in range(np.shape(x)[0]):
        x[i] = (x[i] - np.min(x[i])) / (np.max(x[i]) - np.min(x[i]))
    return x


def normalize_adj(adj):
    d = np.diag(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0)
    a_norm = adj.dot(d).transpose().dot(d)
    return a_norm


def preprocess_adj(adj):
    adj = adj + np.eye(adj.shape[0])
    adj = normalize_adj(adj)
    return adj


# df = pd.read_excel('channel_link_1_1.xlsx')
# Abf = df.iloc[:, 1:].values
# A = preprocess_adj(Abf)
# A = np.ones((60,60))
# A = np.float32(A)
# A = torch.from_numpy(A)
label = np.array([0, 1]).squeeze()


start_time = datetime.datetime.now()
# ----------------------CNN------------------------
result = open('result-e-3_new.xls', 'w', encoding='gbk')
result.write('sub\tworst_acc\tbest_acc\tmean_acc\tstd_acc\tkappa\trecall\tf1_score\tauc-roc\n')

for p in range(23, 24):
    acc_kappa_list = list()
    Test_index = list()
    Test_index.append(p)
    dataName = 'data_'+str(p)
    labelName = 'label_' + str(p)
    datapath = r'./new_data/{}.npy'.format(dataName)
    labelpath = r'./new_data/{}.npy'.format(labelName)

    mydata = np.load(datapath)
    mydata = mydata.transpose((0, 3, 1, 2))
    # mydata = mydata[30:,:,:,:]


    Y = np.load(labelpath) - 1
    # Y = Y[30:]
    X = datanorm(mydata)

    # data_test = X[0:30,:,:,:]
    # label_test = Y[0:30]


    del mydata
    gc.collect()

    skf = StratifiedKFold(n_splits=10, shuffle=True)
    model_acc = list()
    count = 0

    for train_index, test_index in skf.split(X, Y):

        X_train, X_test = X[train_index].astype(np.float32), X[test_index].astype(np.float32)
        Y_train, Y_test = Y[train_index].astype(np.long), Y[test_index].astype(np.long)

        X_train = torch.from_numpy(X_train)
        X_test = torch.from_numpy(X_test)
        y_train = torch.from_numpy(Y_train)
        y_train = y_train.type(torch.LongTensor)
        y_test = torch.from_numpy(Y_test)
        y_test = y_test.type(torch.LongTensor)
        count = count + 1
        print("number of training examples = " + str(X_train.shape[0]))
        print("number of test examples = " + str(X_test.shape[0]))

        data_train = Data.TensorDataset(X_train, y_train)
        trainloader = DataLoader(data_train, batch_size=20, shuffle=True, num_workers=0)

        net = cnn()
        # net.apply(weights_init_normal)
        # net.load_state_dict(torch.load("save_model/model_17save/net10.pth"))
        net = net.cuda()

        # criterion = nn.BCELoss()
        criterion = nn.CrossEntropyLoss()
        criterion = criterion.cuda()

        optimizer = optim.Adam(net.parameters(), lr=1e-3)

        acc_before = 0
        acclist = list()
        for epoch in range(500):
            running_loss = 0.0
            c = 0
            correct = 0.0
            total = 0

            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                # inputs = inputs.to(device)
                inputs = inputs.cuda()
                labels = labels.cuda()
                # labels = labels.to(device)
                # labels = labels.reshape([np.shape(inputs)[0], 1])
                # A = A.to(device)
                # A = A.cuda()
                ##清楚上一次留下的梯度
                optimizer.zero_grad()

                outputs = net(inputs)

                loss = criterion(outputs, labels)

                pred = torch.argmax(outputs, 1)

                correct+=torch.eq(pred, labels).sum().float().item()
                total += labels.size(0)
                acc_tr = float(correct) / total
                ##求导并更新参数
                loss.backward()

                optimizer.step()

                running_loss += loss.item()
                c = i
            print('======>>>>>>[%d] Train Loss: %.3f  Train ACC: %.3f' %
                  (epoch + 1, running_loss / c, acc_tr))  # 输出loss的平均值

            correct = 0
            total = 0

            with torch.no_grad():
                X_test = X_test.cuda()
                y_test = y_test.cuda()
                out = net(X_test)
                _, pred = torch.max(out, 1)
                correct+=(pred == y_test).sum().item()
                total += y_test.size(0)

            acc = float(correct) / total

            print('Val Acc = {:.5f}'.format(acc))
            acclist.append(acc)
            if acc >= acc_before:
                # torch.save(net.state_dict(), "save_model/""model_"+str(p)+"save/""net"+str(count)+".pth")
                print("model has been saved")
            acc_before = max(acclist)
        accuracy = max(acclist)
        print(count, p)
        print('test accuracy: ', accuracy)
        model_acc.append(accuracy)

    model_acc = np.array(model_acc)
    acc_kappa_list.append(p)
    acc_kappa_list.append(np.min(model_acc))
    acc_kappa_list.append(np.max(model_acc))
    acc_kappa_list.append(np.mean(model_acc))
    acc_kappa_list.append(np.std(model_acc))

    for h in range(len(acc_kappa_list)):
        result.write(str(acc_kappa_list[h]))
        result.write('\t')
    result.write('\n')
    del X, Y
    gc.collect()
    print('model_acc:', model_acc)
    print('min', np.min(model_acc))
    print('max', np.max(model_acc))
    print('mean', np.mean(model_acc))
    print('std', np.std(model_acc))
result.close()
end_time = datetime.datetime.now()
print('program time:', end_time - start_time)