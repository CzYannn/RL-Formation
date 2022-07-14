from agents.leader import Leader
from agents.follower import Follower
from algos.sac.sac_torch import SAC
import os

def build_agents(args, save_path):

    """
    build inital policy for agents
    """
    
    policy_list = []
    agents = []
    leader_policy = None
    policy_list.append(leader_policy)
    save_dir = os.path.join(save_path, 'models')
    for i in range(1, args.agent_num):
        policy_list.append(SAC(alpha=args.lr,beta=args.lr,gamma=args.gamma, input_dims=[7],
                reward_scale=args.reward_scale, layer1_size=args.hidden_size,n_actions=2,
                layer2_size=args.hidden_size, tau=args.tau,batch_size=args.batch_size, id=i, chkpt_dir=save_dir))

    leader = Leader('Leader', args.pref_speed, leader_policy, id=0, RL=False)
    agents.append(leader)
    for i in range(1, args.agent_num):
        follower = Follower('Follower'+str(i), args.pref_speed, policy_list[i],
                            id=i, RL=True)
        agents.append(follower)
    
    return agents