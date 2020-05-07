from .models import Detector, save_model
from .utils import load_detection_data
from . import dense_transforms
import torch.utils.tensorboard as tb


def train(args):
    from os import path
    model = Detector()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    """
    Your code here, modify your HW3 code
    """
    import torch

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = Detector().to(device)
    if args.continue_training:
        model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'det.th')))

    # optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-5)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)

    import inspect
    transform = eval(args.transform, {k: v for k, v in inspect.getmembers(dense_transforms) if inspect.isclass(v)})
    valid_transform = eval(args.valid_transform,
                           {k: v for k, v in inspect.getmembers(dense_transforms) if inspect.isclass(v)})
    train_data = load_detection_data('dense_data/train', num_workers=64, transform=transform)
    valid_data = load_detection_data('dense_data/valid', num_workers=64, transform=valid_transform)

    det_loss = torch.nn.BCEWithLogitsLoss(reduction='none')
    # size_loss = torch.nn.MSELoss(reduction='none')

    global_step = 0
    global_step_val = 0
    for epoch in range(args.num_epoch):
        model.train()

        avg_train_loss = []
        for img, gt_det, gt_size in train_data:
            img, gt_det, gt_size = img.to(device), gt_det.to(device), gt_size.to(device)
            # size_w, _ = gt_det.max(dim=1, keepdim=True)
            # cls_w = gt_det * 1 + (1-gt_det) * gt_det.mean() ** 0.5

            # det, size = model(img)
            det = model(img)
            # Continuous version of focal loss
            p_det = torch.sigmoid(det * (1 - 2 * gt_det))
            det_loss_val = (det_loss(det, gt_det) * p_det).mean() / p_det.mean()
            # size_loss_val = (size_w * size_loss(size, gt_size)).mean() / size_w.mean()
            loss_val = det_loss_val  # + size_loss_val * args.size_weight
            avg_train_loss.append(loss_val.item())

            if train_logger is not None and global_step % 100 == 0:
                train_logger.add_image('image', img[0], global_step)
                train_logger.add_image('label', gt_det[0], global_step)
                train_logger.add_image('pred', torch.sigmoid(det[0]), global_step)

            if train_logger is not None:
                # train_logger.add_scalar('det_loss', det_loss_val, global_step)
                # train_logger.add_scalar('size_loss', size_loss_val, global_step)
                train_logger.add_scalar('train_loss', loss_val, global_step)

            print('[Train loss]', loss_val.item(), ' [epoch]', epoch)

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            global_step += 1

        if train_logger is not None:
            train_logger.add_scalar('epoch loss', sum(avg_train_loss) / len(avg_train_loss), epoch)

        # Scoring on the validation set
        model.eval()
        # avg_valid_loss = []
        for img, gt_det, gt_size in valid_data:
            img, gt_det, gt_size = img.to(device), gt_det.to(device), gt_size.to(device)
            det = model(img)
            p_det = torch.sigmoid(det * (1 - 2 * gt_det))
            det_loss_val = (det_loss(det, gt_det) * p_det).mean() / p_det.mean()
            loss_val_val = det_loss_val
            # avg_valid_loss.append(avg_valid_loss.item())

            if valid_logger is not None:
                valid_logger.add_scalar('valid_loss', loss_val_val, global_step_val)

            global_step_val += 1

            print('Valid loss is ', loss_val_val, ' and epoch is ', epoch)

        #         if valid_logger is not None:
        #             valid_logger.add_scalar('epoch loss', sum(avg_valid_loss)/len(avg_valid_loss), epoch)

        save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here
    parser.add_argument('-n', '--num_epoch', type=int, default=150)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('-c', '--continue_training', action='store_true')
    parser.add_argument('-t', '--transform',
                        default='Compose([ColorJitter(0.9, 0, 0, 0), RandomHorizontalFlip(), ToTensor(), ToHeatmap(2)])')
    parser.add_argument('-vt', '--valid_transform',
                        default='Compose([ToTensor(), ToHeatmap(2)])')
    parser.add_argument('-w', '--size-weight', type=float, default=0.01)
    # [ColorJitter(0.9, 0.9, 0.9, 0.1)
    args = parser.parse_args()
    train(args)
