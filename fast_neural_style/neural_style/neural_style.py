import argparse
import os
import sys
import time
import re

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import torch.onnx
from tqdm import tqdm
import utils
from transformer_net import TransformerNet
from vgg import Vgg16


def check_paths(args):
    try:
        if not os.path.exists(args.save_model_dir):
            os.makedirs(args.save_model_dir)
        if args.checkpoint_model_dir is not None and not (os.path.exists(args.checkpoint_model_dir)):
            os.makedirs(args.checkpoint_model_dir)
    except OSError as e:
        print(e)
        sys.exit(1)


def train(args):
    """
    模型训练过程 使用给定的数据集 对于指定的风格图片进行训练
    """
    # 指定设备 判断是否能够使用GPU进行加速
    device = torch.device("cuda" if args.cuda else "cpu")

    # 手动设定随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 数据集预处理 transform
    transform = transforms.Compose([
        transforms.Resize(args.image_size),             # 重新指定大小
        transforms.CenterCrop(args.image_size),         # 中心裁剪
        transforms.ToTensor(),                          # 转 Tensor
        transforms.Lambda(lambda x: x.mul(255))         # 调整像素点至 0-255
    ])

    # 加载数据集 并进行数据预处理
    train_dataset = datasets.ImageFolder(args.dataset, transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
    steps_per_epoch = len(train_loader)

    # 风格图片预处理 transform
    style_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    # 加载风格图片 进行预处理 并构造batch
    style = utils.load_image(args.style_image, size=args.style_size)
    style = style_transform(style)
    style = style.repeat(args.batch_size, 1, 1, 1).to(device)

    # 构造 Image Transform Net
    transformer = TransformerNet().to(device)

    # 加载 Vgg16 预训练模型
    vgg = Vgg16(requires_grad=False).to(device)

    # 构造优化器 损失函数
    optimizer = Adam(transformer.parameters(), args.lr)
    mse_loss = torch.nn.MSELoss()

    # 计算风格图片在 VGG 网络中每一模块的输出 便于计算 loss 值
    features_style = vgg(utils.normalize_batch(style))
    # 根据风格图片在 VGG 网络中每一模块的输出构造 Gram 矩阵
    gram_style = [utils.gram_matrix(y) for y in features_style]

    # 训练过程
    for e in range(args.epochs):
        # 训练模式
        transformer.train()
        agg_content_loss = 0.
        agg_style_loss = 0.
        count = 0
        with tqdm(total=steps_per_epoch, desc='Epoch {}/{}'.format(e + 1, args.epochs)) as pbar:
            for batch_id, (x, _) in enumerate(train_loader):
                n_batch = len(x)
                count += n_batch
                
                # 梯度归零
                optimizer.zero_grad()

                x = x.to(device)
                # 使用 Image Transform Net 生成风格和内容混合图片
                y = transformer(x)

                # 进行 batch_normalization 防止梯度爆炸/消失 加速收敛
                y = utils.normalize_batch(y)
                x = utils.normalize_batch(x)

                # 将 原始内容图片 和 生成混合图片 通过 VGG 获得各层输出结果 方便进行 loss 计算
                features_y = vgg(y)
                features_x = vgg(x)

                # 计算 内容损失 使用 VGG 的 relu3_3 来计算 论文中指出 VGG 的高层输出能够更好的代表内容特征
                content_loss = args.content_weight * mse_loss(features_y.relu3_3, features_x.relu3_3)

                # 计算风格特征
                style_loss = 0.
                # 使用 VGG 每一个模块的输出 论文中指出 风格特征可以从每一个模块的不同卷积通道之间的相似度来提取
                for ft_y, gm_s in zip(features_y, gram_style):
                    gm_y = utils.gram_matrix(ft_y)
                    style_loss += mse_loss(gm_y, gm_s[:n_batch, :, :])
                style_loss *= args.style_weight

                # 计算总的 loss
                total_loss = content_loss + style_loss

                # 反向传播 
                total_loss.backward()
                optimizer.step()

                agg_content_loss += content_loss.item()
                agg_style_loss += style_loss.item()

                # if (batch_id + 1) % args.log_interval == 0:
                #     mesg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.6f}\tstyle: {:.6f}\ttotal: {:.6f}".format(
                #         time.ctime(), e + 1, count, len(train_dataset),
                #                       agg_content_loss / (batch_id + 1),
                #                       agg_style_loss / (batch_id + 1),
                #                       (agg_content_loss + agg_style_loss) / (batch_id + 1)
                #     )
                #     print(mesg)

                # 保存训练过程中的 check_point
                if args.checkpoint_model_dir is not None and (batch_id + 1) % args.checkpoint_interval == 0:
                    transformer.eval().cpu()
                    ckpt_model_filename = "ckpt_epoch_" + str(e) + "_batch_id_" + str(batch_id + 1) + ".pth"
                    ckpt_model_path = os.path.join(args.checkpoint_model_dir, ckpt_model_filename)
                    torch.save(transformer.state_dict(), ckpt_model_path)
                    transformer.to(device).train()

                pbar.set_postfix({'total_loss': '%.4f' % float((agg_content_loss + agg_style_loss) / (batch_id + 1))})
                pbar.update(1)

    # 保存最终模型
    transformer.eval().cpu()
    save_model_filename = "epoch_" + str(args.epochs) + ".pth"
    save_model_path = os.path.join(args.save_model_dir, save_model_filename)
    torch.save(transformer.state_dict(), save_model_path)

    print("\nDone, trained model saved at", save_model_path)


def stylize(args):
    """
    使用预训练好的风格模型 和指定的内容图片 合成最终结果图片
    """
    # 指定设备 判断是否能够使用GPU进行加速
    device = torch.device("cuda" if args.cuda else "cpu")

    # 读取内容图片 并对图片进行预处理
    content_image = utils.load_image(args.content_image, scale=args.content_scale)
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(content_image)
    # 转成二维张量
    content_image = content_image.unsqueeze(0).to(device)

    # 设置 tensor.requires_grad 为 False 不计算梯度
    with torch.no_grad():
        # 构造 Image Transform Net
        style_model = TransformerNet()
        # 加载与训练模型
        state_dict = torch.load(args.model)
        for k in list(state_dict.keys()):
            if re.search(r'in\d+\.running_(mean|var)$', k):
                del state_dict[k]
        style_model.load_state_dict(state_dict)
        style_model.to(device)
        style_model.eval()

        # 获得输出图像
        output = style_model(content_image).cpu()

    # 保存输出图像
    utils.save_image(args.output_image, output[0])


def main():
    main_arg_parser = argparse.ArgumentParser(description="parser for fast-neural-style")
    subparsers = main_arg_parser.add_subparsers(title="subcommands", dest="subcommand")

    train_arg_parser = subparsers.add_parser("train", help="parser for training arguments")
    train_arg_parser.add_argument("--epochs", type=int, default=2,
                                  help="number of training epochs, default is 20")
    train_arg_parser.add_argument("--batch-size", type=int, default=4,
                                  help="batch size for training, default is 4")
    train_arg_parser.add_argument("--dataset", type=str, required=True,
                                  help="path to training dataset, the path should point to a folder "
                                       "containing another folder with all the training images")
    train_arg_parser.add_argument("--style-image", type=str, default="images/style-images/mosaic.jpg",
                                  help="path to style-image")
    train_arg_parser.add_argument("--save-model-dir", type=str, required=True,
                                  help="path to folder where trained model will be saved.")
    train_arg_parser.add_argument("--checkpoint-model-dir", type=str, default=None,
                                  help="path to folder where checkpoints of trained models will be saved")
    train_arg_parser.add_argument("--image-size", type=int, default=256,
                                  help="size of training images, default is 256 X 256")
    train_arg_parser.add_argument("--style-size", type=int, default=None,
                                  help="size of style-image, default is the original size of style image")
    train_arg_parser.add_argument("--cuda", type=int, required=True,
                                  help="set it to 1 for running on GPU, 0 for CPU")
    train_arg_parser.add_argument("--seed", type=int, default=42,
                                  help="random seed for training")
    train_arg_parser.add_argument("--content-weight", type=float, default=1e5,
                                  help="weight for content-loss, default is 1e5")
    train_arg_parser.add_argument("--style-weight", type=float, default=1e10,
                                  help="weight for style-loss, default is 1e10")
    train_arg_parser.add_argument("--lr", type=float, default=1e-3,
                                  help="learning rate, default is 1e-3")
    train_arg_parser.add_argument("--log-interval", type=int, default=500,
                                  help="number of images after which the training loss is logged, default is 500")
    train_arg_parser.add_argument("--checkpoint-interval", type=int, default=2000,
                                  help="number of batches after which a checkpoint of the trained model will be created")

    eval_arg_parser = subparsers.add_parser("eval", help="parser for evaluation/stylizing arguments")
    eval_arg_parser.add_argument("--content-image", type=str, required=True,
                                 help="path to content image you want to stylize")
    eval_arg_parser.add_argument("--content-scale", type=float, default=None,
                                 help="factor for scaling down the content image")
    eval_arg_parser.add_argument("--output-image", type=str, required=True,
                                 help="path for saving the output image")
    eval_arg_parser.add_argument("--model", type=str, required=True,
                                 help="saved model to be used for stylizing the image. File ends in .pth - PyTorch path")
    eval_arg_parser.add_argument("--cuda", type=int, required=True,
                                 help="set it to 1 for running on GPU, 0 for CPU")

    args = main_arg_parser.parse_args()

    if args.subcommand is None:
        print("ERROR: specify either train or eval")
        sys.exit(1)
    if args.cuda and not torch.cuda.is_available():
        print("ERROR: cuda is not available, try running on CPU")
        sys.exit(1)

    if args.subcommand == "train":
        check_paths(args)
        train(args)
    elif args.subcommand == "eval":
        stylize(args)
    else:
        check_paths(args)
        match(args)

if __name__ == "__main__":
    main()
