import argparse


def set_config():
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument('--dataset', type=str, default='DEAP')
    parser.add_argument('--data-path', type=str, default='/home/xiaominghao/eeg-1000-shuffled/')
    parser.add_argument('--subjects', type=int, default=19)
    parser.add_argument('--num-class', type=int, default=2, choices=[2, 3, 4])
    parser.add_argument('--label-type', type=str, default='V', choices=['A', 'V', 'D', 'L'])
    parser.add_argument('--segment', type=int, default=4)  # segment length in seconds
    parser.add_argument('--overlap', type=float, default=0)
    parser.add_argument('--sampling-rate', type=int, default=1000)
    parser.add_argument('--target-rate', type=int, default=1000)
    parser.add_argument('--trial-duration', type=int, default=59, help='trial duration in seconds')
    parser.add_argument('--scale-coefficient', type=float, default=1)  # 比例系数
    parser.add_argument('--input-shape', type=tuple, default=(1, 32, 4000))  # 输入形状 (1, 32, 512)
    parser.add_argument('--data-format', type=str, default='eeg')
    # Training Process
    parser.add_argument('--fold', type=int, default=10)
    parser.add_argument('--random-seed', type=int, default=2023)
    parser.add_argument('--max-epoch', type=int, default=200)
    parser.add_argument('--patient', type=int, default=8)  # 早停 最开始为20
    parser.add_argument('--patient-cmb', type=int, default=2)  # 原始值为8
    parser.add_argument('--max-epoch-cmb', type=int, default=10)  # 最大迭代次数 原始值为20
    parser.add_argument('--batch-size', type=int, default=25)
    parser.add_argument('--learning-rate', type=float, default=1e-5)  # 学习率 原始值为1e-3
    parser.add_argument('--training-rate', type=float, default=0.8)
    parser.add_argument('--weight-decay', type=float, default=0.001)  # 权重衰减
    parser.add_argument('--step-size', type=int, default=5)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--LS', type=bool, default=True, help="Label smoothing")
    parser.add_argument('--LS-rate', type=float, default=0.1)
    parser.add_argument('--gpu', default='0')

    parser.add_argument('--save-path', default='./save/')
    parser.add_argument('--load-path', default='./save/max-acc.pth')
    parser.add_argument('--load-path-final', default='./save/final_model.pth')
    parser.add_argument('--save-model', type=bool, default=True)
    # Model Parameters
    parser.add_argument('--model', type=str, default='LGGNet')
    parser.add_argument('--pool', type=int, default=16)
    parser.add_argument('--pool-step-rate', type=float, default=0.25)
    parser.add_argument('--T', type=int, default=64)
    parser.add_argument('--graph-type', type=str, default='gen', choices=['fro', 'gen', 'hem', 'BL'])
    parser.add_argument('--hidden', type=int, default=32) # 隐藏层

    # Reproduce the result using the saved model
    parser.add_argument('--reproduce', action='store_true', default=False)

    args = parser.parse_args()
    gpu = args.gpu

    return args, gpu
