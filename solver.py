import os
import numpy as np
import torch
import torchvision
from torch import optim
from evaluation import *
from network import U_Net,R2U_Net
import csv
from configuration import config
from PIL import Image
import re

train_accuracy = []
validation_accuracy = []

def reconstruct_image(self, result):
    img_size = config['resize']
    patch_size = config['patch_size']

    result = np.reshape(result, (-1, patch_size, patch_size))

    fid = open('datasets' + '/test_patch_read_order.txt', 'r', encoding='utf-8')
    read_index = []
    for line in fid.readlines():
        if line != '\n':
            read_index.append(int(re.findall(r"\d+", line)[0]))

    img = np.zeros((img_size, img_size))


    for i in range(9 * 9):
        row, col = np.unravel_index(i, [9, 9], 'C')

        img[row * patch_size: row * patch_size + patch_size,
            col * patch_size: col * patch_size + patch_size] = result[read_index.index(i)]

    img = Image.fromarray(img)
    save_path = 'results/' + '0.tif'
    img.save(save_path)

class Solver(object):
    def __init__(self, config, train_loader, valid_loader, test_loader):
        # data loader
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        # Models
        self.unet = None
        self.optimizer = None
        self.img_ch = config['img_ch']
        self.output_ch = config['output_ch']
        self.criterion = torch.nn.BCELoss()  # binary cross entropy loss

        # Hyper-parameters
        self.lr = config['lr']
        self.beta1 = config['beta1']  # momentum1 in Adam
        self.beta2 = config['beta2']  # momentum2 in Adam

        # Training settings
        self.num_epochs = config['num_epochs']
        self.num_epochs_decay = config['num_epoches_decay']
        self.batch_size = config['batch_size']

        # Path
        self.model_path = config['model_path']
        self.result_path = config['result_path']

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = config['model_type']
        self.t = config['t']
        self.unet_path = os.path.join(self.model_path, '%s-%d-%.4f-%d.pkl' % (
                            self.model_type, self.num_epochs, self.lr, self.num_epochs_decay))
        self.best_epoch = 0
        self.build_model()


    def build_model(self):
        """Build generator and discriminator."""
        if self.model_type == 'U_Net':
            self.unet = U_Net(img_ch=1, output_ch=1)
        elif self.model_type == 'R2U_Net':
            self.unet = R2U_Net(img_ch=1, output_ch=1, t=self.t)
            #init_weights(self.unet, 'normal')

        self.optimizer = optim.Adam(list(self.unet.parameters()), self.lr, (self.beta1, self.beta2))
        self.unet.to(self.device)


    def train(self):
        """Print out the network information."""
        num_params = 0
        for p in self.unet.parameters():
            num_params += p.numel()  # accumulate the number of mmodel parameters
        print("The number of parameters: {}".format(num_params))

        # ====================================== Training ===========================================#

        # network train
        if os.path.isfile(self.unet_path):
            # Load the pretrained Encoder
            self.unet.load_state_dict(torch.load(self.unet_path))
            print('%s is Successfully Loaded from %s' % (self.model_type, self.unet_path))

        else:
            lr = self.lr
            best_unet_score = 0.0
            best_epoch = 0

            for epoch in range(self.num_epochs):
                self.unet.train(True)
                epoch_loss = 0

                acc = 0.  # Accuracy
                SE = 0.  # Sensitivity (Recall)
                SP = 0.  # Specificity
                PC = 0.  # Precision
                F1 = 0.  # F1 Score
                JS = 0.  # Jaccard Similarity
                DC = 0.  # Dice Coefficient
                length = 0

                for i, (images, GT) in enumerate(self.train_loader):
                    images, GT = images.to(self.device), GT.to(self.device)

                    # forward result
                    SR = self.unet(images)
                    SR_probs = torch.sigmoid(SR)
                    SR_flat = SR_probs.view(SR_probs.size(0), -1)  # size(0) is batch_size
                    GT_flat = GT.view(GT.size(0), -1)

                    loss = self.criterion(SR_flat, GT_flat)
                    epoch_loss += loss.item()

                    # Backprop + optimize
                    self.unet.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    acc += get_accuracy(SR, GT)
                    SE += get_sensitivity(SR, GT)
                    SP += get_specificity(SR, GT)
                    PC += get_precision(SR, GT)
                    F1 += get_F1(SR, GT)
                    JS += get_JS(SR, GT)
                    DC += get_DC(SR, GT)
                    length = length + 1

                acc = acc / length
                SE = SE / length
                SP = SP / length
                PC = PC / length
                F1 = F1 / length
                JS = JS / length
                DC = DC / length

                # Print the log info
                print('Epoch [%d/%d], Loss: %.4f, \n[Training] Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f,'
                      ' F1: %.4f, JS: %.4f, DC: %.4f' % (
                        epoch + 1, self.num_epochs, epoch_loss, acc, SE, SP, PC, F1, JS, DC))
                train_accuracy.append(acc)

                # Decay learning rate
                if (epoch + 1) > (self.num_epochs - self.num_epochs_decay):
                    lr -= (self.lr / float(self.num_epochs_decay))
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = lr
                    print('Decay learning rate to lr: {}.'.format(lr))

                # ===================================== Validation ====================================#
                self.unet.train(False)
                self.unet.eval()

                acc = 0.  # Accuracy
                SE = 0.  # Sensitivity (Recall)
                SP = 0.  # Specificity
                PC = 0.  # Precision
                F1 = 0.  # F1 Score
                JS = 0.  # Jaccard Similarity
                DC = 0.  # Dice Coefficient
                length = 0
                for i, (images, GT) in enumerate(self.valid_loader):
                    images, GT = images.to(self.device), GT.to(self.device)
                    SR = torch.sigmoid(self.unet(images))
                    acc += get_accuracy(SR, GT)
                    SE += get_sensitivity(SR, GT)
                    SP += get_specificity(SR, GT)
                    PC += get_precision(SR, GT)
                    F1 += get_F1(SR, GT)
                    JS += get_JS(SR, GT)
                    DC += get_DC(SR, GT)

                    length = length + 1

                acc = acc / length
                SE = SE / length
                SP = SP / length
                PC = PC / length
                F1 = F1 / length
                JS = JS / length
                DC = DC / length
                unet_score = JS + DC

                print('[Validation] Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, '
                      'F1: %.4f, JS: %.4f, DC: %.4f' % (
                acc, SE, SP, PC, F1, JS, DC))
                validation_accuracy.append(acc)

                if unet_score > best_unet_score:
                    best_unet_score = unet_score
                    self.best_epoch = epoch
                    best_unet = self.unet.state_dict() # contain best parameters for each layer
                    print('Best %s model score : %.4f' % (self.model_type, best_unet_score))
                    torch.save(best_unet, self.unet_path)


    def test(self):
        self.unet.load_state_dict(torch.load(self.unet_path))
        self.unet.eval()

        acc = 0.  # Accuracy
        SE = 0.  # Sensitivity (Recall)
        SP = 0.  # Specificity
        PC = 0.  # Precision
        F1 = 0.  # F1 Score
        JS = 0.  # Jaccard Similarity
        DC = 0.  # Dice Coefficient
        length = 0
        result = []
        for i, (images, GT) in enumerate(self.test_loader):
            images = images.to(self.device)
            GT = GT.to(self.device)
            SR = torch.sigmoid(self.unet(images))

            acc += get_accuracy(SR, GT)
            SE += get_sensitivity(SR, GT)
            SP += get_specificity(SR, GT)
            PC += get_precision(SR, GT)
            F1 += get_F1(SR, GT)
            JS += get_JS(SR, GT)
            DC += get_DC(SR, GT)

            length = length + 1

            SR = SR.to('cpu')
            SR = SR.detach().numpy()
            result.extend(SR)

        acc = acc / length
        SE = SE / length
        SP = SP / length
        PC = PC / length
        F1 = F1 / length
        JS = JS / length
        DC = DC / length
        unet_score = JS + DC


        reconstruct_image(self, np.array(result))

        f = open(os.path.join(self.result_path, 'result.csv'), 'a', encoding='utf-8', newline='')
        wr = csv.writer(f)
        wr.writerow(
            [self.model_type, acc, SE, SP, PC, F1, JS, DC, self.lr, self.best_epoch, self.num_epochs, self.num_epochs_decay,])
        f.close()




