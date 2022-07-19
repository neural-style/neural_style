import argparse
import os
import sys
import time
import re
import copy
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import torch.onnx
from tqdm import tqdm
from neural_style import utils
from neural_style.transformer_net import TransformerNet
from neural_style.vgg import Vgg16

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def match(content_image='../static/images/input/input.jpg'):
    '''
    最佳风格匹配

    return .jpg with filename
    '''
    # ---------------------------------arguments---------------------------------
    style_image = '../static/images/style'
    seed = 42
    image_size = 256
    batch_size = 4
    lr = 1e-3
    content_weight = 1
    style_weight = 1e5
    model = '../static/models'
    output_image = '../static/images/output'
    # -------------------------------train-----------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    np.random.seed(seed)
    torch.manual_seed(seed)

    transformer = TransformerNet().to(device)
    optimizer = Adam(transformer.parameters(), lr)
    mse_loss = torch.nn.MSELoss()

    vgg = Vgg16(requires_grad=False).to(device)
    style_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])

    min_loss = 999999999999.  # 最小损失函数值
    min_arg = 0  # 最小损失函数值对应的风格图片

    content_image = utils.load_image(content_image)

    content_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).to(device)
    for i in range(4):  # 四个默认风格图片
        if i == 0:
            st_image = os.path.join(style_image, 'candy.jpg')
        elif i == 1:
            st_image = os.path.join(style_image, 'mosaic.jpg')
        elif i == 2:
            st_image = os.path.join(style_image, 'rain-princess.jpg')
        elif i == 3:
            st_image = os.path.join(style_image, 'udnie.jpg')
        style = utils.load_image(st_image)
        style = style_transform(style)
        style = style.repeat(batch_size, 1, 1, 1).to(device)

        features_style = vgg(utils.normalize_batch(style))
        gram_style = [utils.gram_matrix(y) for y in features_style]

        transformer = TransformerNet().to(device)
        transformer.train()
        agg_content_loss = 0.
        agg_style_loss = 0.
        count = 0
        # 使用tqdm提示训练进度
        with tqdm(desc='style {}/{}'.format(i + 1, 4)) as pbar:
            # 每个epoch训练settings.STEPS_PER_EPOCH次
            x = content_image
            n_batch = len(x)
            count += n_batch
            optimizer.zero_grad()
            x = x.to(device)
            # ---------------

            with torch.no_grad():
                style_model = TransformerNet()
                if i == 0:
                    model_path = os.path.join(model, 'candy.pth')
                elif i == 1:
                    model_path = os.path.join(model, 'mosaic.pth')
                elif i == 2:
                    model_path = os.path.join(model, 'rain_princess.pth')
                elif i == 3:
                    model_path = os.path.join(model, 'udnie.pth')
                state_dict = torch.load(model_path)
                for k in list(state_dict.keys()):
                    if re.search(r'in\d+\.running_(mean|var)$', k):
                        del state_dict[k]
                style_model.load_state_dict(state_dict)
                style_model.to(device)
                style_model.eval()

                pre_y = style_model(content_image).cpu()
            # ---------------------------------

            y = pre_y.to(device)
            y = y[0]
            y = y[:, :256, :256]
            y = y.view(1, 3, 256, 256)
            x = utils.normalize_batch(x)
            features_y = vgg(y)
            features_x = vgg(x)

            content_loss = content_weight * mse_loss(features_y.relu2_2, features_x.relu2_2)

            style_loss = 0.
            for ft_y, gm_s in zip(features_y, gram_style):
                gm_y = utils.gram_matrix(ft_y)
                style_loss += mse_loss(gm_y, gm_s[:n_batch, :, :])
            style_loss *= style_weight

            total_loss = content_loss + style_loss
            _loss = total_loss

            agg_content_loss += content_loss.item()
            agg_style_loss += style_loss.item()
            pbar.set_postfix({'loss': '%.4f' % float(_loss)})
            pbar.update(1)
        # print('content loss of style_' + str(i) + ': ' + str(agg_content_loss))  #style loss的大小关系固定，因此看content loss
        # print('style loss of style_' + str(i) + ': ' + str(agg_style_loss))  # style loss的大小关系固定，因此看content loss
        # print('loss of style_' + str(i) + ': ' + str(agg_content_loss+agg_style_loss))  # style loss的大小关系固定，因此看content loss
        if agg_content_loss < min_loss:
            min_loss = agg_content_loss
            min_arg = i
            output = pre_y
            style_path = st_image

    # -----------------------------result----------------------------------

    print("the best match style(0~3): " + str(min_arg))
    save_path = os.path.join(output_image, 'match')
    save_path = os.path.join(save_path, str(time.ctime()).replace(' ', '_').replace(':', '.'))
    os.makedirs(save_path)
    save_path = os.path.join(save_path, '1.jpg')
    utils.save_image(save_path, output[0])
    return style_path, save_path  # 返回风格图片路径、融合图片路径


def random_train(style_path_par="static/images/style/mosaic.jpg", content_path_par='static/images/input/input.jpg'):
    epochs = 30
    batch_size = 4

    # 训练图像(输入的一张图像)路径
    content_path = content_path_par

    # 风格图像(filename)
    style_path = style_path_par

    # 输出图像路径
    output_image_epoch = "static/images/output/random"

    # size of training images, default is 256 X 256
    image_size = 256

    # size of style-image, default is the original size of style image
    style_size = None

    # random seed for training
    seed = 42

    # weight for content-loss, default is 1e5
    content_weight = 1e5

    # weight for style-loss, default is 1e10
    style_weight = 1e10

    # learning rate, default is 1e-3
    lr = 1e-3

    np.random.seed(seed)
    torch.manual_seed(seed)

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
                x = copy.deepcopy(train_image)
                n_batch = len(x)
                count += n_batch
                optimizer.zero_grad()
                x = x.to(device)
                y = transformer(x)
                y = utils.normalize_batch(y)
                x = utils.normalize_batch(x)

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

                pbar.set_postfix({'loss': '%.4f' % float(total_loss)})
                pbar.update(1)

        # # 每个epoch保存一次图片
        with torch.no_grad():
            transformer.to(device)
            transformer.eval()
            output = transformer(content_image).cpu()

        save_pic_filename = str(e + 1) + '.jpg'
        save_pic_path = os.path.join(save_dir_path, save_pic_filename)
        print('保存epochs', e + 1, '的训练生成图像：', save_pic_path)
        utils.save_image(save_pic_path, output[0])

    return save_dir_path, epochs


def pre_stylize(content_image_path_par='static/images/input/input.jpg', model_path_par='static/models/mosaic.pth'):
    # 调用模型路径
    model_path = model_path_par

    # 内容图像路径
    content_image_path = content_image_path_par

    # factor for scaling down the content image
    content_scale = None

    # 输出图像路径
    output_image = 'static/images/output/random'

    # 生成输出图像文件夹
    save_dir_path = os.path.join(output_image, str(time.ctime()).replace(' ', '_').replace(':', '.'))
    os.makedirs(save_dir_path)

    # 内容图像
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
        for k in list(state_dict.keys()):
            if re.search(r'in\d+\.running_(mean|var)$', k):
                del state_dict[k]
        style_model.load_state_dict(state_dict)
        style_model.to(device)
        style_model.eval()
        output = style_model(content_image).cpu()

    # 保存风格迁移图像
    save_pic_filename = '1.jpg'
    save_pic_path = os.path.join(save_dir_path, save_pic_filename)
    print('保存eval生成图像：', save_pic_path)
    utils.save_image(save_pic_path, output[0])
    return save_pic_path
