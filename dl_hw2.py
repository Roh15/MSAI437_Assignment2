# Imports

import argparse
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.optim as optim
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Classes

## Dataset

class BeansDataset(Dataset):

    def __init__(self, data_dict, image_dir, transform=None):
        self.data_dict = data_dict
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx):

        img_name = os.path.join(self.image_dir, self.data_dict[idx]['filename']) 
        image = Image.open(img_name)

        if self.transform:
          image = self.transform(image)

        label = torch.Tensor([self.data_dict[idx]['label']]).to(torch.int64)

        return image, label
    
## ConvNet

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 12, 5, stride=5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(12, 30, 3, padding=1)
        self.conv3 = nn.Conv2d(30, 100, 15, stride=3)
        self.pool2 = nn.MaxPool2d(4, 4)    
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool1(F.relu(self.conv2(x)))
        x = self.pool2(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

## AutoEncoder

class Autoencoder1(nn.Module):
    def __init__(self):
        super(Autoencoder1,self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, 5, stride=5),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(12, 30, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(30, 100, 15, stride=3),
            nn.Flatten()
            )
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (100, 4, 4)),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.ConvTranspose2d(100, 50, 6, stride=1),
            nn.ReLU(True),
            nn.Upsample(scale_factor=1.2, mode='bilinear'),
            nn.ConvTranspose2d(50, 20, 6, stride=1),
            nn.ReLU(True),
            nn.Upsample(scale_factor=1.2, mode='bilinear'),
            nn.ConvTranspose2d(20, 15, 6, stride=2),
            nn.ReLU(True),
            nn.Upsample(scale_factor=1.2, mode='bilinear'),
            nn.ConvTranspose2d(15, 10, 8, stride=2),
            nn.ReLU(True),
            nn.Upsample(scale_factor=1.2, mode='bilinear'),
            nn.ConvTranspose2d(10, 5, 8, stride=2),
            nn.ReLU(True),
            nn.Upsample(scale_factor=1.2, mode='bilinear'),
            nn.ConvTranspose2d(5, 3, 20),
            nn.ReLU(True),
            nn.Upsample(size=500, mode='bilinear'),
            nn.Tanh()
            )
    def forward(self, x):
        enc = self.encoder(x)
        dec = self.decoder(enc)
        return dec
    
## Feed Forward NN

class FFNNet(nn.Module):
    def __init__(self):
        super().__init__()   
        self.fc1 = nn.Linear(1600, 800)
        self.fc2 = nn.Linear(800, 200)
        self.fc3 = nn.Linear(200, 20)
        self.fc4 = nn.Linear(20, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.sigmoid(x)
        return x
    
# Funcs

def create_data_dict(image_dir, labels2int):
  data_dict = {}
  index = 0
  for name in os.listdir(image_dir):
    if name[-4:] == '.jpg':
      if 'bean_rust' in name or 'angular_leaf_spot' in name:
        label = labels2int['BAD']
      else:
        label = labels2int['GOOD']
      data_dict[index] = {'filename': name, 'label': label}
      index += 1

  return data_dict

def get_train_dict(images_dir):
    train_dict = create_data_dict('./' + images_dir + '/train', labels2int)
    return train_dict

def get_valid_dict(images_dir):
    valid_dict = create_data_dict('./' + images_dir + '/valid', labels2int)
    return valid_dict

def create_conf_mat(model, images_dir):

    # handle data
    valid_dict = get_valid_dict(images_dir)
    validsetCM = BeansDataset(valid_dict, './' + images_dir + '/valid', transform)
    validloaderCM = DataLoader(validsetCM, batch_size = len(valid_dict), shuffle = True)

    # create conf mat
    for images, labels in validloaderCM:
        pass
    out = model(images.to(device))
    predicted_labels = torch.round(out)
    predicted_labels = predicted_labels.squeeze().cpu()
    val_conf_mat = confusion_matrix(labels, predicted_labels.detach().numpy())
    tn, fp, fn, tp = val_conf_mat.ravel()
    
    disp = ConfusionMatrixDisplay(confusion_matrix=val_conf_mat, display_labels=['good', 'bad'])
    accuracy = (tn + tp)/(tn + fp + fn + tp)

    return disp, accuracy

def data_for_test(images_dir):

    image_files = []
    for name in os.listdir('./' + images_dir + '/test'):
        if name[-4:] == '.jpg':
            image_files.append(name)

    test_data = []
    for image_file in image_files:
        img_name = os.path.join('./' + images_dir + '/test', image_file) 
        image = Image.open(img_name)
        image = transform(image)
        test_data.append(image)

    inputs = torch.stack(test_data)

    return inputs, image_files

# Global

labels2int = {'GOOD': 0, 'BAD': 1}
int2labels = {0: 'GOOD', 1: 'BAD'}

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Models

## Conv Net

def trainConvNet(images_dir, device):

    # data handling
    train_dict = get_train_dict(images_dir)
    valid_dict = get_valid_dict(images_dir)
    trainset = BeansDataset(train_dict, './' + images_dir + '/train', transform)
    trainloader = DataLoader(trainset, batch_size = 4, shuffle = True)
    validset = BeansDataset(valid_dict, './' + images_dir + '/valid', transform)
    validloader = DataLoader(validset, batch_size = 4, shuffle = True)

    # initialize model, hyperparameters
    beanConvNet = ConvNet()
    beanConvNet.to(device)
    loss_function = nn.BCELoss()
    optimizer = optim.Adam(beanConvNet.parameters(), lr=0.001)

    # helper function to get accuracy
    def get_accuracy_from_out(out, labels):
        predicted_label = torch.round(out)
        acc = (predicted_label == labels).float().mean()
        return acc

    # helper function to evaluate model on dev data
    def evaluate(model, criterion, dataloader, device):
        model.eval()

        mean_acc, mean_loss = 0, 0
        count = 0

        with torch.no_grad():
            for it, (image_tensor, label_tensor) in enumerate(dataloader):
                image_tensor, label_tensor = image_tensor.to(device), label_tensor.to(device)
                out = model(image_tensor)
                mean_loss += criterion(out, label_tensor.float()).item()
                mean_acc += get_accuracy_from_out(out, label_tensor)
                count += 1

        return mean_acc / count, mean_loss / count
    
    # train and save model
    training_loss = []
    validation_loss = []
    best_acc = 0
    best_model_path = None
    mean_acc, mean_loss = 0, 0
    count = 0
    for epoch in range(20):
        print("\n--- Training model Epoch: {} ---".format(epoch))
        for it, (image_tensor, label_tensor) in enumerate(trainloader):
            image_tensor, label_tensor = image_tensor.to(device), label_tensor.to(device)

            # zero out the gradients from the old instance
            beanConvNet.zero_grad()

            # get log probabilities over next words
            out = beanConvNet(image_tensor)

            # calculate current accuracy
            acc = get_accuracy_from_out(out, label_tensor)
            mean_acc += acc

            # compute loss function
            loss = loss_function(out, label_tensor.float())
            mean_loss += loss

            # backward pass and update gradient
            loss.backward()
            optimizer.step()

            count += 1

        train_acc = mean_acc/count
        train_loss = mean_loss/count
        print("Epoch {} complete! Training Accuracy: {}; Training Loss: {}".format(epoch, train_acc, train_loss))
        training_loss.append(float(train_loss.cpu().detach()))
        print("\n--- Evaluating model on dev data ---")
        dev_acc, dev_loss = evaluate(beanConvNet, loss_function, validloader, device)
        print("Epoch {} complete! Validation Accuracy: {}; Validation Loss: {}".format(epoch, dev_acc, dev_loss))
        validation_loss.append(dev_loss)
        if dev_acc > best_acc:
            print("Best validation accuracy improved from {} to {}, saving model...".format(best_acc, dev_acc))
            best_acc = dev_acc
            # set best model path
            best_model_path = 'beanConvNet.dat'
            # saving best model
            torch.save(beanConvNet.state_dict(), best_model_path)

    # plot loss
    plt.plot(training_loss)
    plt.plot(validation_loss)
    plt.title('CNN Model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig('CNN_loss.png')
    plt.show()

    return


def validateConvNet(images_dir, model, device):

    # load model
    beanConvNet = ConvNet()
    beanConvNet.load_state_dict(torch.load(model, map_location=device))
    beanConvNet.eval()

    # display conf mat
    disp, accuracy = create_conf_mat(beanConvNet, images_dir)
    disp.plot()
    plt.savefig('CNN_valid_conf_mat.png')
    plt.show()
    print("validation accuracy: ", accuracy)

    return


def testConvNet(images_dir, model, device):

    # load model
    beanConvNet = ConvNet()
    beanConvNet.load_state_dict(torch.load(model, map_location=device))
    beanConvNet.eval()

    # predict
    input, image_files = data_for_test(images_dir)
    out = beanConvNet(input.to(device))
    predicted_labels = torch.round(out)
    predicted_labels = predicted_labels.squeeze().cpu()

    # create csv
    field_names = ['Number', 'Image', 'Prediction']
    preds = []
    for image_file, pred_label in zip(image_files, predicted_labels):
        pred_dict = {}
        pred_dict['Number'] = int(image_file[-7:-4])
        pred_dict['Image'] = image_file
        pred_dict['Prediction'] = int2labels[int(pred_label.item())]
        preds.append(pred_dict)

    with open('CNN_TestResults.csv', 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(preds)

    return

## AutoEncoder

def trainAE(images_dir, device):
    return

def validateAE(images_dir, device):
    return

def testAE(images_dir, model, device):
    return

if __name__ == "__main__":
 
    parser = argparse.ArgumentParser(description='Run HW2')
    parser.add_argument('images_dir', metavar='dir', type=str,
                        help='path to the "beans" directory')
    parser.add_argument('--model',
                        help='model for inference')
    parser.add_argument('--mac', action='store_true', default=False, 
                        help='use if running on m1 mac')
    parser.add_argument('--conv', action='store_true', default=False, 
                        help='choose the cnn')
    parser.add_argument('--ae', action='store_true', default=False,
                        help='choose the autoencoder')
    parser.add_argument('--train', action='store_true', default=False, 
                        help='train the chosen model')
    parser.add_argument('--valid', action='store_true', default=False,
                        help='validate chosen pretrained model')
    parser.add_argument('--infer', action='store_true', default=False,
                        help='infer using chosen pretrained model')
    
    args = parser.parse_args()
    arg_dict = vars(args)

    images_dir = arg_dict['images_dir']

    # set device
    if arg_dict['mac']:
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if arg_dict['conv']:
        if arg_dict['train']:
            trainConvNet(images_dir, device)
        elif arg_dict['valid']:
            if arg_dict['model']:
                validateConvNet(images_dir, arg_dict['model'], device)
            else:
                print('ERROR: No model provided')
        elif arg_dict['infer']:
            if arg_dict['model']:
                testConvNet(images_dir, arg_dict['model'], device)
            else:
                print('ERROR: No model provided')
    elif arg_dict['ae']:
        if arg_dict['train']:
            trainAE(images_dir, device)
        elif arg_dict['valid']:
            if arg_dict['model']:
                validateAE(images_dir, arg_dict['model'], device)
            else:
                print('ERROR: No model provided')
        elif arg_dict['infer']:
            if arg_dict['model']:
                testAE(images_dir, arg_dict['model'], device)
            else:
                print('ERROR: No model provided')


    


    
