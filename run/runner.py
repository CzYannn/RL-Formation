import numpy as np 
import wandb
import os

class Runner(object):

    def __init__(self, env, agents, args, save_path, logger):

        self.env = env 
        self.agents = agents
        self.args = args
        self.save_path = save_path
        self.logger = logger
        
    def train(self, episode_num):
        """
        train followers
        """
        train_step = 0
        score_history = [[] for _ in range(len(self.agents))]
        best_score = [0 for _ in range(len(self.agents))]
        avg_score = [0 for _ in range(len(self.agents))]
        for i in range(episode_num):
            obs = self.env.reset()
            scores = np.zeros(len(self.agents))
            step = 0 
            while not self.env.game_over:
                all_actions = []
                for idx, ag in enumerate(self.agents):
                    if ag.group == 0 :
                        action = ag.act(obs[idx], warmup = self.args.warmup, 
                                        eval = False, action_num = 6)
                    else:
                        action = ag.act(obs[idx], warmup = self.args.warmup, 
                                    eval = False, action_num = 2)
                    all_actions.append(action)
                obs_, rewards, dones, _ = self.env.step(all_actions)
                for idx, ag in enumerate(self.agents):
                    if ag.group==1:
                        if not ag.quit :
                            scores[idx] += rewards[idx]
                            ag.policy.remember(obs[idx], all_actions[idx], rewards[idx], 
                                                obs_[idx], dones[idx])
                            ag.policy.learn()
                obs = obs_
                step +=1
                train_step +=1

            for idx, ag in enumerate(self.agents): 
                if ag.group==1:
                    score_history[idx].append(scores[idx])
                    avg_score[idx] = np.mean(score_history[idx][-50:])
                    if best_score[idx] <= avg_score[idx] and i>=50:
                        best_score[idx] = avg_score[idx]
                    ag.policy.save_models()
                    self.logger.info(f'{ag.name}, episode ={i}, score = {scores[idx]:.2f}, avg_score = {avg_score[idx]:.2f}, best_score = {best_score[idx]:.2f}, step: {step}')
                    wandb.log({f'{ag.name}_score':scores[idx]}, step=train_step)
                    if train_step > self.args.batch_size:
                        avg_pi_loss, avg_q_loss, avg_v_loss = ag.policy.get_avg_loss()
                        wandb.log({f'{ag.name}_actor_loss':avg_pi_loss}, step=train_step)
                        wandb.log({f'{ag.name}_critic_loss':avg_q_loss}, step=train_step)
                        wandb.log({f'{ag.name}_value_loss':avg_v_loss}, step=train_step)
                    if i > 0:
                        wandb.log({f'success_rate':self.env.get_avg_success_rate()}, step=train_step)
        
    def test(self, episode_num):
        """
        test followers
        """
        score_history = [[] for _ in range(len(self.agents))]
        best_score = [0 for _ in range(len(self.agents))]
        avg_score = [0 for _ in range(len(self.agents))]
        os.mkdir(os.path.join(self.save_path, 'test'))
        for i in range(episode_num):
            obs = self.env.reset()
            scores = np.zeros(len(self.agents))
            step = 0 
            while not self.env.game_over:
                all_actions = []
                for idx, ag in enumerate(self.agents):
                    if ag.group == 0 :
                        action = ag.act(obs[idx], warmup = self.args.warmup, 
                                        eval = False, action_num = 6)
                    else:
                        action = ag.act(obs[idx], warmup = self.args.warmup, 
                                    eval = True, action_num = 2)
                    all_actions.append(action)
                obs_, rewards, _, _ = self.env.step(all_actions)
                for idx, ag in enumerate(self.agents):
                     if ag.group==1:
                        if not ag.quit :
                            scores[idx] += rewards[idx]
                obs = obs_
                step +=1
            for idx, ag in enumerate(self.agents): 
                if ag.group==1:
                    score_history[idx].append(scores[idx])
                    avg_score[idx] = np.mean(score_history[idx][-50:])
                    if best_score[idx] <= avg_score[idx]:
                        best_score[idx] = avg_score[idx]
                    self.logger.info(f'{ag.name}, episode ={i}, score = {scores[idx]:.2f}, avg_score = {avg_score[idx]:.2f}, best_score = {best_score[idx]:.2f}, step: {step}')
        for idx, ag in enumerate(self.agents):
            if ag.group == 1:
                with open(self.save_path+f'/test/score_{ag.name}.txt','w') as file_object:
                                        file_object.write(str(score_history[idx])+'\n')
                with open(self.save_path+f'/test/error_{ag.name}.txt','w') as file_object:
                                        file_object.write(str(ag.error_list)+'\n')
            with open(self.save_path+f'/test/point_{ag.name}.txt', 'w') as file_object:
                                        file_object.write(str(ag.pose_list)+'\n')


    