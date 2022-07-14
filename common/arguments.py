import argparse 
from common.utils import str2bool

def read_args():

    parser = argparse.ArgumentParser(description='track_car')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='target smoothing coefficient(tao) (default: 0.005)')
    parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                        help='learning rate (default: 0.0003)')
    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='batch size (default: 256)')
    parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                        help='hidden size (default: 256)')
    parser.add_argument('--reward_scale', type=int, default=15, metavar='N',
                        help='reward_scale (default:15)')
    parser.add_argument('--train', type=str2bool, default=True, metavar='N',
                        help='if train (default:True)')
    parser.add_argument('--episode', type=int, default=1000, metavar='N',
                        help='episode (default:1000)')
    parser.add_argument('--warmup',type=str2bool, default=False, metavar='N',
                        help='if warmup (default:False')
    parser.add_argument('--RL',type=str2bool, default=True,metavar='N',
                        help = 'if use RL(defaul:True)')
    parser.add_argument('--seed',type=int, default=123456, metavar='N',
                    help='random seed(default=123456)')
    parser.add_argument('--agent_num',type=int, default=3, metavar='N',
                        help='agent number(default=4)')
    parser.add_argument('--pref_speed', type=float, default=2.0, metavar='N',
                        help='speed of agent(default=2)')                
    args = parser.parse_args()

    return args