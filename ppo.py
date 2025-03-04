import torch
import torch.nn.functional as F

def ppo_update(policy, states, actions, rewards, next_states, old_probs, gamma=0.99, clip_epsilon=0.2):
    # Calculate the advantage (TD error)
    values = policy(states)
    next_values = policy(next_states)
    td_error = rewards + gamma * next_values - values

    # Calculate the ratio between the new and old policies
    probs = policy(states)
    ratio = torch.exp(probs - old_probs)

    # Calculate the policy loss
    loss = torch.min(ratio * td_error, torch.clamp(ratio, 1-clip_epsilon, 1+clip_epsilon) * td_error)
    return loss.mean()
 