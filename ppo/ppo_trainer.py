# This script implements PPO training for an OpenVLA-OFT Actor

# author: Abhinandan, Ishita, Sreeharsha

import torch
import random


class PPOTrainer(torch.nn.Module):
    def __init__(self, state_dim, action_dim, clip_param=0.2, gamma=0.99, lr=3e-4, lam = 0.98, critic_type="vla", entropy_coef=0.01):
        super(PPOTrainer, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.clip_param = clip_param
        self.gamma = gamma # discount factor
        self.lr = lr
        self.lam = lam # lambda for GAE
        self.entropy_coef = entropy_coef

        # Initialize actor and critic networks
        self.actor = None
        self.critic = None
        self.actor_optimizer = None
        self.critic_optimizer = None


    def set_vla_actor(self):
        # set actor to VLA actor
        # also create optimizer for actor
        pass

    def set_critic(self, critic_type):
        # set critic based on type (FiLM, VLA, etc)
        # also create optimizer for critic
        pass

    def compute_gae(self, minibatch):
        states = minibatch['states']
        rewards = minibatch['rewards']
        next_states = minibatch['next_states']
        dones = minibatch['dones']

        # get values for states and next states
        values = self.critic(states).detach()
        next_values = self.critic(next_states).detach()

        # compute TD residuals
        deltas = rewards + self.gamma * next_values * (1 - dones) - values

        advantages = torch.zeros_like(rewards)
        for t in reversed(range(len(rewards))):
            advantage = deltas[t] + self.gamma * self.lam * (1 - dones[t]) * advantage
            advantages[t] = advantage
        
        return values, advantages
    
    def ppo_loss(self, minibatch):
        states = minibatch['states']
        actions = minibatch['actions']
        old_log_probs = minibatch['log_probs']
        
        # get GAEs
        values, advantages = self.compute_gae(minibatch)

        # compute log probs for current version of actor
        action_tokens = self.actor(states)
        action_dists = torch.distributions.Categorical(logits=action_tokens)
        new_log_probs = action_dists.log_prob(actions)

        # importance sampling ratio
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()

        # entropy bonus
        entropy = action_dists.entropy().mean() 
        actor_loss -= self.entropy_coef * entropy

        # value loss
        returns = advantages + values
        value_loss = torch.nn.functional.mse_loss(values, returns)

        return actor_loss, value_loss
    
    def update(self, epochs, batch, minibatch_size):
        if len(batch['states']) == 0:
            return

        if self.actor is None or self.critic is None or self.actor_optimizer is None or self.critic_optimizer is None:
            raise ValueError("Actor and Critic networks must be initialized before training.")
        
        num_minibatches = len(batch['states']) // minibatch_size

        for _ in range(epochs):
            # shuffle the batch
            random.shuffle(batch)

            for i in range(num_minibatches):
                start = i * minibatch_size
                end = start + minibatch_size
                minibatch = {k: v[start:end] for k, v in batch.items()}

                actor_loss, value_loss = self.ppo_loss(minibatch)

                # update actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # update critic
                self.critic_optimizer.zero_grad()
                value_loss.backward()
                self.critic_optimizer.step()


                
        





