import argparse
import os
import sys
import time
import re
import numpy as np
import torch
import utils
from vgg import Vgg16
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import torch.onnx
from tqdm import tqdm
from transformer_net import TransformerNet
import copy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)


def check_paths():
    try:
        if not os.path.exists('saved_models'):
            os.makedirs('saved_models')

    except OSError as e:
        print(e)
        sys.exit(1)


        
        
        
def train(style_path_par="images/style-images/mosaic.jpg", content_path_par='images/content-images/content'
                                                                               '-train/COCO.jpg'):
    epochs = 60
    batch_size = 4

    # 训练图像(输入的一张图像)路径
    content_train_path = '/'.join(content_path_par.split('/')[:-2])
    print(content_train_path)

    content_path = content_path_par

    # 风格图像(filename)
    style_path = style_path_par

    # 训练的模型保存的路径
    save_model_dir = 'saved_models'

    # 输出图像路径
    output_image_epoch = "images/output-images"

    # # path to folder where checkpoints of trained models will be saved
    # checkpoint_model_dir = None

    # size of training images, default is 256 X 256
    image_size = 256

    # size of style-image, default is the original size of style image
    style_size = None

    # random seed for training
    seed = 42

    # weight for content-loss, default is 1e5
    content_weight = 1e5

    # weight for style-loss, default is 1e10
    style_weight = 1.3e10

    # learning rate, default is 1e-3
    lr = 1e-3

    # number of images after which the training loss is logged, default is 500
    log_interval = 500

    # number of batches after which a checkpoint of the trained model will be created
    checkpoint_interval = 2000

    np.random.seed(seed)
    torch.manual_seed(seed)

    # transform = transforms.Compose([
    #     transforms.Resize(image_size),
    #     transforms.CenterCrop(image_size),
    #     transforms.ToTensor(),
    #     transforms.Lambda(lambda x: x.mul(255))
    # ])
    # train_dataset = datasets.ImageFolder(content_train_path, transform)
    # print('图像个数：', len(train_dataset))
    # print(train_dataset)
    # print(train_dataset[0])
    # print(train_dataset[0][0])
    # print(train_dataset[0][1])
    # print(train_dataset[0][0][0])
    # print(train_dataset[0][0][0][0])
    # train_loader = DataLoader(train_dataset, batch_size=batch_size)
    # print(type(train_loader))
    
    
    # 训练图像
    train_image = utils.load_image(content_path)
    train_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    train_image = train_transform(train_image)
    train_image = train_image.unsqueeze(0).to(device)
    # print(train_image.shape)
    # print(type(train_image))
    # print('单张')
    # print(train_image)
    
    
    transformer = TransformerNet().to(device)
    optimizer = Adam(transformer.parameters(), lr)
    mse_loss = torch.nn.MSELoss()

    vgg = Vgg16(requires_grad=False).to(device)
    style_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])

    
    style = utils.load_image(style_path, size=style_size)
    # style = utils.load_image(args.style_image, size=args.style_size)
    style = style_transform(style)
    style = style.repeat(batch_size, 1, 1, 1).to(device)

    features_style = vgg(utils.normalize_batch(style))
    gram_style = [utils.gram_matrix(y) for y in features_style]

    print('训练内容图像：', content_path)
    content_image = utils.load_image(content_path)
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).to(device)


    # 生成输出图像文件夹
    tim = str(time.ctime()).replace(' ', '_').replace(':', '.')
    save_dir_path = os.path.join(output_image_epoch, tim)
    os.makedirs(save_dir_path)

    STEPS_PER_EPOCH = 100

    for e in range(epochs):
        transformer.train()
        agg_content_loss = 0.
        agg_style_loss = 0.
        count = 0
        
        # 使用tqdm提示训练进度
        with tqdm(total=STEPS_PER_EPOCH, desc='Epoch {}/{}'.format(e + 1, epochs)) as pbar:
            # 每个epoch训练settings.STEPS_PER_EPOCH次
            
            for step in range(STEPS_PER_EPOCH):
                # for batch_id, (x, _) in enumerate(train_loader):
                # print('new')
                # print(x)
                # print(x.shape)
                x = copy.deepcopy(train_image)
                # print(train_image)
                n_batch = len(x)
                count += n_batch
                optimizer.zero_grad()
                x = x.to(device)
                y = transformer(x)
                y = utils.normalize_batch(y)
                x = utils.normalize_batch(x)
                # if step == STEPS_PER_EPOCH-1:
                #     save_pic_filename = 'epoch_'+ str(e+1) + '.jpg'
                #     save_pic_path = os.path.join(save_dir_path, save_pic_filename)
                #     print('保存epochs',e+1,'的训练生成图像：',save_pic_path)
                #     utils.save_image(save_pic_path, y[0])

                features_y = vgg(y)
                features_x = vgg(x)

                content_loss = content_weight * mse_loss(features_y.relu2_2, features_x.relu2_2)

                style_loss = 0.
                for ft_y, gm_s in zip(features_y, gram_style):
                    gm_y = utils.gram_matrix(ft_y)
                    style_loss += mse_loss(gm_y, gm_s[:n_batch, :, :])
                style_loss *= style_weight

                total_loss = content_loss + style_loss
                # _loss = total_loss
                total_loss.backward()
                optimizer.step()

                agg_content_loss += content_loss.item()
                agg_style_loss += style_loss.item()

                # train_image=x

                pbar.set_postfix({'loss': '%.4f' % float(total_loss)})
                pbar.update(1)

        # # 每个epoch保存一次图片
        with torch.no_grad():
            transformer.to(device)
            transformer.eval()
            output = transformer(content_image).cpu()

        save_pic_filename = 'epoch_' + str(e + 1) + '.jpg'
        save_pic_path = os.path.join(save_dir_path, save_pic_filename)
        print('保存epochs', e + 1, '的训练生成图像：', save_pic_path)
        utils.save_image(save_pic_path, output[0])

    # save model
    transformer.eval().cpu()
    save_model_filename = tim + '.pth'
    save_model_path = os.path.join(save_model_dir, save_model_filename)
    torch.save(transformer.state_dict(), save_model_path)
    print("\nDone, trained model saved at", save_model_path)


def stylize(content_image_path_par='images/content-images/content-train/COCO.jpg', model_path_par='saved_models'
                                                                                                     '/mosaic.pth'):
    # saved model to be used for stylizing the image. If file ends in .pth - PyTorch path is used, if in .onnx -
    # Caffe2 path")
    model_path = model_path_par

    # 内容图像路径
    content_image_path = content_image_path_par

    # factor for scaling down the content image
    content_scale = None

    # 输出图像路径
    output_image = 'images/output-images'

    # 生成输出图像文件夹
    save_dir_path = os.path.join(output_image, str(time.ctime()).replace(' ', '_').replace(':', '.'))
    os.makedirs(save_dir_path)

    content_image = utils.load_image(content_image_path)
    # content_image = utils.load_image(args.content_image)
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])

    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).to(device)

    with torch.no_grad():
        style_model = TransformerNet()
        state_dict = torch.load(model_path)
        # state_dict = torch.load(args.model)
        # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
        for k in list(state_dict.keys()):
            if re.search(r'in\d+\.running_(mean|var)$', k):
                del state_dict[k]
        style_model.load_state_dict(state_dict)
        style_model.to(device)
        style_model.eval()
        output = style_model(content_image).cpu()

    save_pic_filename = 'neural_picture.jpg'
    save_pic_path = os.path.join(save_dir_path, save_pic_filename)
    print('保存eval生成图像：', save_pic_path)
    utils.save_image(save_pic_path, output[0])


def main():
    # if not torch.cuda.is_available():
    #     print("ERROR: cuda is not available, try running on CPU")
    #     sys.exit(1)
    # else:
    #     check_paths()
    #     train()
    #
    #     # stylize(args)

    check_paths()
    train()

    # stylize()


if __name__ == "__main__":
    main()
