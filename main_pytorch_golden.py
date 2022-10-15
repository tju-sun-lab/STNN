import argparse
# from utils import *
from tools_golden_subject import *
from sklearn.model_selection import StratifiedKFold
from G_D import *
import os




os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=300, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=10, help="size of the batches")
parser.add_argument("--lr", type=float, default=1e-3, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--height", type=int, default=64, help="size of image height")
parser.add_argument("--width", type=int, default=64, help="size of image width")
parser.add_argument("--channels", type=int, default=60, help="number of image channels")
opt = parser.parse_args()
print(opt)



# def datanorm(x):
#     for i in range(np.shape(x)[0]):
#         x[i] = (x[i] - np.min(x[i])) / (np.max(x[i]) - np.min(x[i]))
#     return x

golden_data = np.load('data/data_17.npy')
transferred_data = np.load('data/data_23.npy')
golden_data = golden_data.transpose((0, 3, 1, 2))
transferred_data = transferred_data.transpose((0, 3, 1, 2))
transferred_label = np.load('data/label_23.npy') - 1

cuda = torch.cuda.is_available()

transferred_data = transferred_data.astype(np.float32)
golden_data = golden_data.astype(np.float32)

X, Y = transferred_data, transferred_label
acc_max = 0


class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (n_epochs - decay_start_epoch) > 0
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)

p=10
input_shape = (opt.channels, opt.height, opt.width)
count = 0
skf = StratifiedKFold(n_splits=10)
model_acc = list()
acc_kappa_list = list()

for train_index, test_index in skf.split(X, Y):

    X_train, X_test = X[train_index].astype(np.float32), X[test_index].astype(np.float32)
    Y_train, Y_test = Y[train_index].astype(np.long), Y[test_index].astype(np.long)

    X_train = torch.from_numpy(X_train)
    X_test = torch.from_numpy(X_test)
    y_train = torch.from_numpy(Y_train)
    y_test = torch.from_numpy(Y_test)
    X_torch = torch.from_numpy(X)
    count = count + 1

    print('the split is:', count)
    print("number of training examples = " + str(X_train.shape[0]))
    print("number of test examples = " + str(X_test.shape[0]))

    Loss_identity = torch.nn.MSELoss()
    criterion = nn.CrossEntropyLoss()

    cuda = True if torch.cuda.is_available() else False

    G_AB = Generator()
    D = cnn()
    D.load_state_dict(torch.load("model_save/sub17_cross3.pth", map_location='cuda:0'))
    for name, param in D.named_parameters():
        param.requires_grad = False

    if cuda:
        G_AB = G_AB.cuda()
        D = D.cuda()
        Loss_identity = Loss_identity.cuda()
        criterion = criterion.cuda()

    ###优化器
    optimizer_G = torch.optim.Adam(params=G_AB.parameters(),
                                   lr=opt.lr, betas=(opt.b1, opt.b2))

    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
        optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
    )

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

    trainloader = DataLoader(
        cwtDataset(root1=X_train, root2=golden_data, root3=y_train),
        batch_size=opt.batch_size,
        shuffle = True
    )

    acclist = list()

    for epoch in range(opt.n_epochs):
        running_loss = 0.0
        c = 0
        correct = 0.0
        total = 0

        for i, (batch) in enumerate(trainloader):

            A = batch['A']
            B = batch['B']
            C = batch['C']
            A = A.type(Tensor)
            B = B.type(Tensor)
            C = C.cuda()

            optimizer_G.zero_grad()

            fake_A = G_AB(A)

            D_fake_A1, D_fake_A2, D_fake_A3, D_fake_A4 = D(fake_A)

            D_real_A1, D_real_A2, D_real_A3, D_real_A4 = D(A)



            D_B1, D_B2, D_B3, D_B4 =D(B)

            style_loss = Loss_identity(D_fake_A4, D_B4) + Loss_identity(D_fake_A3, D_B3) + Loss_identity(D_fake_A2, D_B2)

            label_loss_fake = criterion(D_fake_A1, C)

            label_loss = label_loss_fake

            loss =  style_loss + label_loss

            loss.backward()

            optimizer_G.step()

            pred = torch.argmax(D_fake_A1, 1)

            correct += torch.eq(pred, C).sum().float().item()

            total += C.size(0)

            acc_tr = float(correct) / total

            running_loss += loss.item()


            c = i
        print('======>>>>>>[%d] Train Loss: %.3f  Train ACC: %.3f' %
              (epoch + 1, running_loss / c, acc_tr))

        correct = 0
        total = 0
        with torch.no_grad():
            X_test = X_test.cuda()
            y_test = y_test.cuda()

            new_X_test = G_AB(X_test)
            out, _, _, _ = D(new_X_test)
            _, pred = torch.max(out, 1)
            correct +=(pred == y_test).sum().item()
            total += y_test.size(0)

        acc = float(correct) / total

        print('Val Acc = {:.5f}'.format(acc))
        acclist.append(acc)
        if acc > acc_max:
            # torch.save(G_AB.state_dict(), str(p)+"net_WOSA"+str(count)+".pth")
            print("model has been saved")
            acc_max = acc

    accuracy = max(acclist)
    print('test accuracy: ', accuracy)
    model_acc.append(accuracy)

model_acc = np.array(model_acc)
acc_kappa_list.append(np.min(model_acc))
acc_kappa_list.append(np.max(model_acc))
acc_kappa_list.append(np.mean(model_acc))
acc_kappa_list.append(np.std(model_acc))

print('model_acc:', model_acc)
print('min', np.min(model_acc))
print('max', np.max(model_acc))
print('mean', np.mean(model_acc))
print('std', np.std(model_acc))
print("number:", p)
