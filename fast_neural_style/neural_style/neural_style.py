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

def match(content_image = 'images\content-images\\amber.jpg'):
    '''
    最佳风格匹配

    return .jpg with filename
    '''
    #---------------------------------arguments---------------------------------
    style_image = 'E:\edgedownload\\fast_neural_style\images\style-images'
    seed = 42
    image_size = 256
    batch_size = 4
    lr = 1e-3
    dataset = 'match'
    content_weight = 1
    style_weight = 1e5
    model = 'models'
    output_image = 'images\output-images'
    #-------------------------------train-----------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    np.random.seed(seed)
    torch.manual_seed(seed)

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    train_dataset = datasets.ImageFolder(dataset, transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size)

    transformer = TransformerNet().to(device)
    optimizer = Adam(transformer.parameters(), lr)
    mse_loss = torch.nn.MSELoss()

    vgg = Vgg16(requires_grad=False).to(device)
    style_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])

    min_loss = 999999999999.   #最小损失函数值
    min_arg = 0     #最小损失函数值对应的风格图片

    content_image = utils.load_image(content_image,size=1080)   #size很重要
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).to(device)
    for i in range(4):  #四个默认风格图片
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
            for batch_id, (x, _) in enumerate(train_loader):
                n_batch = len(x)
                count += n_batch
                optimizer.zero_grad()
                x = x.to(device)
                #---------------

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
                #---------------------------------

                y = pre_y.to(device)
                y = y[0]
                y=y[:,:256,:256]
                y=y.view(1,3,256,256)
                #y = utils.normalize_batch(y)
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
        print('content loss of style_' + str(i) + ': ' + str(agg_content_loss))  #style loss的大小关系固定，因此看content loss
        print('style loss of style_' + str(i) + ': ' + str(agg_style_loss))  # style loss的大小关系固定，因此看content loss
        print('loss of style_' + str(i) + ': ' + str(agg_content_loss+agg_style_loss))  # style loss的大小关系固定，因此看content loss
        if agg_content_loss < min_loss:
            min_loss = agg_content_loss
            min_arg = i
            output = pre_y

    #-----------------------------result----------------------------------

    print("the best match style(0~3): "+str(min_arg))

    save_path = os.path.join(output_image,'match')
    save_path = os.path.join(save_path,str(time.ctime()).replace(' ', '_').replace(':', '.'))
    os.makedirs(save_path)
    save_path=os.path.join(save_path,'res.jpg')
    utils.save_image(save_path, output[0])


def train(args):
    device = torch.device("cuda" if args.cuda else "cpu")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    train_dataset = datasets.ImageFolder(args.dataset, transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)

    transformer = TransformerNet().to(device)
    optimizer = Adam(transformer.parameters(), args.lr)
    mse_loss = torch.nn.MSELoss()

    vgg = Vgg16(requires_grad=False).to(device)
    style_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    style = utils.load_image(args.style_image, size=args.style_size)
    style = style_transform(style)
    style = style.repeat(args.batch_size, 1, 1, 1).to(device)

    features_style = vgg(utils.normalize_batch(style))
    gram_style = [utils.gram_matrix(y) for y in features_style]

    for e in range(args.epochs):
        transformer.train()
        agg_content_loss = 0.
        agg_style_loss = 0.
        count = 0
        with tqdm(desc='Epoch {}/{}'.format(e + 1, args.epochs)) as pbar:
            for batch_id, (x, _) in enumerate(train_loader):
                n_batch = len(x)
                count += n_batch
                optimizer.zero_grad()

                x = x.to(device)
                y = transformer(x)

                y = utils.normalize_batch(y)
                x = utils.normalize_batch(x)

                features_y = vgg(y)
                features_x = vgg(x)

                content_loss = args.content_weight * mse_loss(features_y.relu2_2, features_x.relu2_2)

                style_loss = 0.
                for ft_y, gm_s in zip(features_y, gram_style):
                    gm_y = utils.gram_matrix(ft_y)
                    style_loss += mse_loss(gm_y, gm_s[:n_batch, :, :])
                style_loss *= args.style_weight

                total_loss = content_loss + style_loss
                total_loss.backward()
                optimizer.step()

                agg_content_loss += content_loss.item()
                agg_style_loss += style_loss.item()

                if (batch_id + 1) % args.log_interval == 0:
                    mesg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.6f}\tstyle: {:.6f}\ttotal: {:.6f}".format(
                        time.ctime(), e + 1, count, len(train_dataset),
                                      agg_content_loss / (batch_id + 1),
                                      agg_style_loss / (batch_id + 1),
                                      (agg_content_loss + agg_style_loss) / (batch_id + 1)
                    )
                    print(mesg)

                if args.checkpoint_model_dir is not None and (batch_id + 1) % args.checkpoint_interval == 0:
                    transformer.eval().cpu()
                    ckpt_model_filename = "ckpt_epoch_" + str(e) + "_batch_id_" + str(batch_id + 1) + ".pth"
                    ckpt_model_path = os.path.join(args.checkpoint_model_dir, ckpt_model_filename)
                    torch.save(transformer.state_dict(), ckpt_model_path)
                    transformer.to(device).train()

    # save model
    transformer.eval().cpu()
    save_model_filename = "epoch_" + str(args.epochs) + ".pth"
    save_model_path = os.path.join(args.save_model_dir, save_model_filename)
    torch.save(transformer.state_dict(), save_model_path)

    print("\nDone, trained model saved at", save_model_path)


def stylize(args):
    device = torch.device("cuda" if args.cuda else "cpu")

    content_image = utils.load_image(args.content_image, scale=args.content_scale)
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).to(device)

    if args.model.endswith(".onnx"):
        output = stylize_onnx(content_image, args)
    else:
        with torch.no_grad():
            style_model = TransformerNet()
            state_dict = torch.load(args.model)
            # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
            for k in list(state_dict.keys()):
                if re.search(r'in\d+\.running_(mean|var)$', k):
                    del state_dict[k]
            style_model.load_state_dict(state_dict)
            style_model.to(device)
            style_model.eval()
            if args.export_onnx:
                assert args.export_onnx.endswith(".onnx"), "Export model file should end with .onnx"
                output = torch.onnx._export(
                    style_model, content_image, args.export_onnx, opset_version=11,
                ).cpu()            
            else:
                output = style_model(content_image).cpu()
    utils.save_image(args.output_image, output[0])


def stylize_onnx(content_image, args):
    """
    Read ONNX model and run it using onnxruntime
    """

    assert not args.export_onnx

    import onnxruntime

    ort_session = onnxruntime.InferenceSession(args.model)

    def to_numpy(tensor):
        return (
            tensor.detach().cpu().numpy()
            if tensor.requires_grad
            else tensor.cpu().numpy()
        )

    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(content_image)}
    ort_outs = ort_session.run(None, ort_inputs)
    img_out_y = ort_outs[0]

    return torch.from_numpy(img_out_y)


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
                                 help="saved model to be used for stylizing the image. If file ends in .pth - PyTorch path is used, if in .onnx - Caffe2 path")
    eval_arg_parser.add_argument("--cuda", type=int, required=True,
                                 help="set it to 1 for running on GPU, 0 for CPU")
    eval_arg_parser.add_argument("--export_onnx", type=str,
                                 help="export ONNX model to a given file")

    match_arg_parser = subparsers.add_parser("match", help="parser for matching arguments")
    # match_arg_parser.add_argument("--epochs", type=int, default=2,
    #                               help="number of matching epochs, default is 2")
    # match_arg_parser.add_argument("--batch-size", type=int, default=4,
    #                               help="batch size for matching, default is 4")
    # match_arg_parser.add_argument("--dataset", type=str, required=True,
    #                               help="path to matching dataset, the path should point to a folder "
    #                                    "containing another folder with all the matching images")
    # match_arg_parser.add_argument("--style-image", type=str,
    #                               help="path to content-image")
    # match_arg_parser.add_argument("--save-model-dir", type=str, required=True,
    #                               help="path to folder where natched model will be saved.")
    # match_arg_parser.add_argument("--checkpoint-model-dir", type=str, default=None,
    #                               help="path to folder where checkpoints of matched models will be saved")
    # match_arg_parser.add_argument("--image-size", type=int, default=256,
    #                               help="size of matching images, default is 256 X 256")
    # match_arg_parser.add_argument("--style-size", type=int, default=None,
    #                               help="size of content-image, default is the original size of content image")
    # match_arg_parser.add_argument("--cuda", type=int, required=True,
    #                               help="set it to 1 for running on GPU, 0 for CPU")
    # match_arg_parser.add_argument("--seed", type=int, default=42,
    #                               help="random seed for matching")
    # match_arg_parser.add_argument("--content-weight", type=float, default=1,
    #                               help="weight for content-loss, default is 1e5")
    # match_arg_parser.add_argument("--style-weight", type=float, default=1e5,
    #                               help="weight for style-loss, default is 1e10")
    # match_arg_parser.add_argument("--lr", type=float, default=1e-3,
    #                               help="learning rate, default is 1e-3")
    # match_arg_parser.add_argument("--log-interval", type=int, default=500,
    #                               help="number of images after which the matching loss is logged, default is 500")
    # match_arg_parser.add_argument("--checkpoint-interval", type=int, default=2000,
    #                               help="number of batches after which a checkpoint of the matched model will be created")
    # match_arg_parser.add_argument("--output-image", type=str, required=True,
    #                              help="path for saving the output image")
    # match_arg_parser.add_argument("--content-image", type=str, required=True,
    #                              help="path to content image you want to stylize")
    # match_arg_parser.add_argument("--content-scale", type=float, default=None,
    #                              help="factor for scaling down the content image")
    # match_arg_parser.add_argument("--model", type=str, required=True,
    #                              help="saved model to be used for stylizing the image. If file ends in .pth - PyTorch path is used, if in .onnx - Caffe2 path")
    # match_arg_parser.add_argument("--export_onnx", type=str,
    #                              help="export ONNX model to a given file")
    args = main_arg_parser.parse_args()

    # if args.subcommand is None:
    #     print("ERROR: specify either train or eval")
    #     sys.exit(1)
    # if args.cuda and not torch.cuda.is_available():
    #     print("ERROR: cuda is not available, try running on CPU")
    #     sys.exit(1)

    if args.subcommand == "train":
        check_paths(args)
        train(args)
    elif args.subcommand == "eval":
        stylize(args)
    else:
        # check_paths(args)
        match()

if __name__ == "__main__":
    main()
