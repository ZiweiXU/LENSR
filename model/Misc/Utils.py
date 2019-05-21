"""
All kinds of utilities that are related to this project.
"""
import os
import sys
import uuid
import random
import argparse
import copy
import functools
import inspect
import logging

import numpy as np
from numpy.linalg import norm
from scipy import spatial
from scipy.optimize import linprog, minimize
import torch
import torch.nn.functional as F
from torch.autograd import Variable


def myself():
    return inspect.stack()[1][3]


class ColorText:
    MGT = '\033[95m'  # magenta
    LCY = '\033[96m'  # light cyan
    GRN = '\033[92m'  # green
    YLW = '\033[93m'  # yellow
    RED = '\033[91m'  # red
    END = '\033[0m'  # end of color
    BLD = '\033[1m'  # bold
    UDL = '\033[4m'  # underline
    wrn = BLD + YLW + 'WARNING: ' + END
    info = BLD + LCY + 'INFO: ' + END
    ok = BLD + GRN + 'OK: ' + END
    err = BLD + RED + 'ERROR: ' + END

    @staticmethod
    def yellow(message):
        return ColorText.YLW + message + ColorText.END

    @staticmethod
    def blue(message):
        return ColorText.LCY + message + ColorText.END

    @staticmethod
    def green(message):
        return ColorText.GRN + message + ColorText.END

    @staticmethod
    def red(message):
        return ColorText.RED + message + ColorText.END

    @staticmethod
    def bold(message):
        return ColorText.BLD + message + ColorText.END

    @staticmethod
    def underline(message):
        return ColorText.UDL + message + ColorText.END

    @staticmethod
    def warn_msg(message):
        return ColorText.wrn + message

    @staticmethod
    def info_msg(message):
        return ColorText.info + message

    @staticmethod
    def ok_msg(message):
        return ColorText.ok + message

    @staticmethod
    def err_msg(message):
        return ColorText.err + message


def load_json(f):
    import json, jsoncomment

    if type(f) == str:
        f = open(f, 'r')
    jc = jsoncomment.JsonComment(json)
    r = jc.load(f)
    # ro = jc.load(f, object_hook=lambda d: namedtuple('X', d.keys())(*d.values()))
    f.close()
    return r


def rgetattr(obj, attr_str):
    return functools.reduce(lambda obj, attr: getattr(obj, attr), [obj] + attr_str.split('.'))


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def chunks(l, n):
    """Divide a list into chunks, according to a fixed size n or a list of sizes

    Arguments:
        l {list} -- The list to be divided
        n {int or list} -- If n is an int, the list l will be divided into n-sized chunks. If n is a
        list, the list l will be divide into len(l) chunks, with corresponding sizes. The last element
        of list n can be None.

    Returns:
        {generator} -- The generator of chunks.
    """
    if type(n) is not list:
        n = max(1, n)
        for i in range(0, len(l), n):
            yield l[i:i + n]
    else:
        start = 0
        for end in n:
            if end is None:
                yield l[start:]
            else:
                yield l[start:start + end]
                start += end


def stable_relu(x, eps=1e-2):
    return x if x > eps else ((x - x) + eps)


def to_cpu_ext(g, g_ref, cuda):
    if cuda:
        g = np.vstack([g[i].cpu().numpy() for i in range(0, len(g))])
        g_ref = g_ref.cpu().numpy()
    return g, g_ref


def make_pred(model, input_d, target_d, label_modifier, test_dataset='mnist'):
    model.eval()
    v_input = torch.from_numpy(input_d) if len(input_d.shape) == 4 else torch.from_numpy(
        input_d).unsqueeze(0)
    v_target = torch.from_numpy(target_d - label_modifier)
    correct = 0
    if test_dataset == 'mnist':
        data, target = \
            Variable(v_input.repeat(1, 3, 1, 1).type(torch.cuda.DoubleTensor), volatile=True), \
            Variable(v_target.type(torch.cuda.LongTensor), volatile=True)
    else:
        data, target = \
            Variable(v_input.type(torch.cuda.DoubleTensor), volatile=True), \
            Variable(v_target.type(torch.cuda.LongTensor), volatile=True)
    output = model(data)
    pred = output.data.max(1, keepdim=True)[1]
    correct = pred.eq(target.data.view_as(pred)).long().cpu()
    return list(map(int, pred.cpu().numpy())), list(map(int, correct.numpy()))


def test(model, dataset, args=None, test_dataset='mnist', **kwargs):
    model = copy.deepcopy(model)
    model = model.eval()
    test_count = 0
    test_loss = 0
    correct = 0
    # initially set end_of_epoch to false
    while True:
        data, target, eoe = dataset.get_batch(epoch_sensitive=True)
        if eoe:
            break
        test_count += target.shape[0]
        output = F.log_softmax(kwargs['fn_branch_select'](model(data)), dim=1)
        test_loss += F.nll_loss(output, target.view(-1),
                                size_average=False).data[0]
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
    test_loss /= test_count
    return correct, test_count, test_loss


def evaluate(model, dataloaders, **kwargs) -> (float, float):
    """
    Validate the model on validation dataset
    :param model: the model to be validated
    :param dataloaders: dict of dataloaders which correspond to validation data
    :return: dict of validation information
    """
    correct, test_count, test_loss = test(model, dataloaders, **kwargs)
    return {'acc': correct * 1.0 / test_count, 'correct': correct, 'count': test_count,
            'loss': test_loss}


def modify_lr(optimizer, iter_count, init_lr=0.001, all_iter=10000, decay=0.999931,
              prompt_every=1000):
    new_lr = init_lr * (decay ** float(iter_count))
    if (iter_count + 1) % prompt_every == 0:
        print('INFO: Current learning rate is: {:.6f}'.format(new_lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    return optimizer


def get_batch_ready(data, dataset, batchsize, args, **options):
    batch_ref = data.get_batch(
        batch_size=batchsize, dataset=dataset, normalize=True)
    data_ref = Variable(torch.from_numpy(batch_ref['input']).repeat(*options['dim_repeat']).type(
        torch.DoubleTensor))
    target_ref = Variable(torch.from_numpy(
        batch_ref['label']).type(torch.LongTensor))
    if args.cuda:
        data_ref, target_ref = (data_ref.cuda(), target_ref.cuda())
    return data_ref, target_ref


def get_batch(data, data_name, phase, model_options, args):
    mt_data_batch_size = model_options['dataset_phases'][phase][data_name]['batch_size']
    mt_data_dim_repeat = model_options['dataset_phases'][phase][data_name]['dim_repeat']
    data_ref, target_ref = get_batch_ready(
        data, data_name, mt_data_batch_size, args, dim_repeat=mt_data_dim_repeat)
    target_ref -= model_options['dataset_phases'][phase][data_name]['label_modifier']
    return data_ref, target_ref


def get_grad(model, args):
    grad = dict()
    for layer in args.all_layers:
        if not args.cuda:
            grad[layer] = {
                'gw': np.copy(getattr(model, layer).weight.grad.data.numpy()),
                'gb': np.copy(getattr(model, layer).bias.grad.data.numpy())
            }
        else:
            grad[layer] = {
                'gw': copy.deepcopy(getattr(model, layer).weight.grad.data),
                'gb': copy.deepcopy(getattr(model, layer).bias.grad.data)
            }
    return grad


def flatten_grad(grad, layer, args):
    def flatten(t):
        return t.view(t.numel())

    if not args.cuda:
        return np.concatenate([grad[layer]['gw'].flatten(), grad[layer]['gb'].flatten()])
    else:
        return torch.cat([flatten(grad[layer]['gw']), flatten(grad[layer]['gb'])])


def flatten_layer_grad(grad, args, layers=None, **options):
    def flatten(t):
        return t.view(t.numel())

    layers = grad.keys() if layers is None else layers
    ref_gw_flatten = torch.cat([flatten(grad[x]['gw']) for x in layers if x in grad.keys()])
    ref_gb_flatten = torch.cat([flatten(grad[x]['gb']) for x in layers if x in grad.keys()])
    ref_flatten_all = torch.cat([ref_gw_flatten, ref_gb_flatten])
    return ref_gw_flatten, ref_gb_flatten, ref_flatten_all


def should_save_model(loss, acc):
    try:
        should_save_model.lowest_loss += 0
        should_save_model.highest_acc += 0
    except AttributeError:
        should_save_model.lowest_loss = 99999.999
        should_save_model.highest_acc = 0.0

    better_loss = loss <= should_save_model.lowest_loss
    better_acc = acc >= should_save_model.highest_acc
    if better_loss:
        should_save_model.lowest_loss = loss
    if better_acc:
        should_save_model.highest_acc = acc
    if better_loss or better_acc:
        return True
    else:
        logging.getLogger(myself()).warning(
            f'Loss={loss}, acc={acc}, model not saved.')
        return False


class EarlyStop:
    def __init__(self, patience: int, verbose: bool = True):
        self.patience = patience
        self.verbose = verbose
        self.lowest_loss = 99999.999

    def step(self, loss: float):
        if loss < self.lowest_loss:
            self.lowest_loss = loss
            return False
        else:
            self.patience -= 1
            if self.verbose:
                logging.getLogger(myself()).debug('Remaining patience: {}'.format(self.patience))
            if self.patience < 0:
                if self.verbose:
                    logging.getLogger(myself()).warning('Ran out of patience.')
                return True


def early_stop(loss: float, patience: int, verbose: bool = True) -> bool:
    try:
        early_stop.patience += 0
    except AttributeError:
        early_stop.lowest_loss = 99999.999
        early_stop.patience = patience
    if loss < early_stop.lowest_loss:
        early_stop.lowest_loss = loss
        early_stop.patience = patience
        return False
    else:
        early_stop.patience -= 1
        if verbose:
            logging.getLogger(myself()).debug('Remaining patience: {}'.format(early_stop.patience))
        if early_stop.patience < 0:
            if verbose:
                logging.getLogger(myself()).warning('Ran out of patience.')
            return True


def write_log(path, content, end='\n'):
    with open(path, 'a') as f:
        f.write(content + end)


def final_test(model, dataset, model_id, val_acc_record, args, **others):
    if len(val_acc_record) < 1:
        return

    acc_sorted = sorted(val_acc_record.items(),
                        key=lambda x: (-x[1][1], x[1][0]))
    best_model = acc_sorted[-1][0]
    logging.getLogger(myself()).info(f'Lowest val loss: {best_model}')

    test_model = torch.load(best_model)

    correct_ev, test_count_ev, _ = test(test_model, dataset)
    logging.getLogger(myself()).info(
        f'Test result of {best_model}: acc={correct_ev*1.0 / test_count_ev}')


def config_logger(log_file='./log.log'):
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s [%(levelname)-8.8s] (%(name)-8.8s %(filename)15.15s:%(lineno)5.5d) %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename=log_file,
                        filemode='w')
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)-8.8s] (%(name)-8.8s %(filename)15.15s:%(lineno)5.5d) %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)

    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logging.getLogger('').critical("Uncaught exception",
                                       exc_info=(exc_type, exc_value, exc_traceback))

    sys.excepthook = handle_exception


def create_parser():
    parser = argparse.ArgumentParser(description='input arguments.')

    """#################### File/data ####################"""
    parser.add_argument('--root_path', required=True)
    parser.add_argument('--file_name', type=str, default='data.hdf5')
    parser.add_argument('--save_model_to', default='./snapshots')

    """#################### Network ####################"""
    # initialization
    parser.add_argument('--from_model', default=None)
    parser.add_argument('--from_gpu', type=int, default=0)
    parser.add_argument('--start_iter', type=int, default=0)
    parser.add_argument('--kaiming_init', type=int, default=1)

    # architecture
    parser.add_argument('--model_config', type=str, required=True)

    """#################### Training ####################"""
    # general
    parser.add_argument('--max_iter', type=int, default=1e+4)
    parser.add_argument('--val_interval', type=int, default=500)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--snapshot_interval', type=int, default=5000)

    # optimizer
    parser.add_argument('--optim', type=str, required=True, help='Optimizer to use.')
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--lr_decay', type=float, default=0.99985)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--opt_activation', type=str)
    parser.add_argument('--opt_activation_args', nargs='+')
    parser.add_argument('--nesterov', type=int, default=0)

    # weight scaler
    parser.add_argument('--opt_method', type=str, default='adaptive_square')
    parser.add_argument('--opt_max_iter', type=int, default=100)
    parser.add_argument('--opt_lr', type=float, default=1e-2)
    parser.add_argument('--opt_step', type=float, default=1e-2)
    parser.add_argument('--opt_lower_bound', type=float, default=1e-3)
    parser.add_argument('--opt_upper_bound', type=float, default=1.0)
    parser.add_argument('--opt_scale', type=int, default=0)

    """#################### Miscellaneous ####################"""
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--uuid', default=None)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--report_info', default='')
    parser.add_argument('--cuda', type=int, default=None,
                        help='1 to enable CUDA. If unspecified, use CUDA if available.')

    return parser


def set_randomness(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init():
    """Parse and return arguments
    
    Returns:
        args -- Namespace of arguments
    """

    # parse input arguments
    parser = create_parser()
    config = parser.parse_args()

    # detect CUDA
    config.cuda = torch.cuda.is_available() if config.cuda is None else config.cuda

    # randomness control
    set_randomness(config.seed)

    # model ID
    config.model_id = config.uuid if config.uuid is not None else str(uuid.uuid4().hex)[:8]

    # load dataset options
    options = load_json(config.model_config)

    # create logger
    config_logger(os.path.join('log', config.model_id + '.log'))

    # create model save path
    if not os.path.exists(config.save_model_to):
        os.mkdir(config.save_model_to)
    if not os.path.exists(os.path.join(config.save_model_to, config.model_id)):
        os.mkdir(os.path.join(config.save_model_to, config.model_id))

    # log
    logging.getLogger(myself()).info('Initialization completed.')
    return config, options
