#!/usr/bin/env python3
import gym 
import rospy 
from mrobot.srv import StartUp  # service call
from common.arguments import read_args
from common.utils import set_seed
import envs
from run.runner import Runner
from run.build import build_agents
import wandb
from common.log import init_log


def shutdown():
    """
    shutdown properly
    """
    print("shutdown!")
    env.stop()
    service_call(False)

# desired formation
neighbor_info = [
                {0:[0, 0]},
                {0:[-1, 1]},
                {0:[-1, -1]},
                ]

if __name__ == '__main__':
               
    # 激活环境检测节点
    rospy.wait_for_service('/Activate')
    service_call = rospy.ServiceProxy('/Activate', StartUp)
    response = service_call(True)
    print(response)
    
    logger, save_path, log_file = init_log('data')
    args = read_args()
    set_seed(args.seed)
    agents = build_agents(args, save_path)
    env = gym.make('LineFollower-v0')
    env.seed(args.seed)
    env.set_agents(agents, neighbor_info)
    rospy.on_shutdown(shutdown)
    runner = Runner(env, agents, args, save_path, logger)
    wandb.init(project="Formation", entity="the-one", group='Train')
    runner.train(args.episode)
    runner.test(10)
    env.stop()

    







