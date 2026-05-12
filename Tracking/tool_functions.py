import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import random
import rl_utils

def onehot_from_logits(logits, eps=0.01):
    '''
    生成最优动作的独热（one-hot）形式，使用 epsilon-贪婪策略
    Args:
        logits: [batch_size, num_actions] 的动作 logits
        eps: 随机动作的概率阈值
    Returns:
        actions: [batch_size, num_actions] 的独热编码动作
    '''
    # 1. 最优动作（argmax）
    argmax_acs = (logits == logits.max(dim=1, keepdim=True)[0]).float()

    # 2. 随机动作（修正索引方式）
    batch_size, num_actions = logits.shape
    rand_indices = torch.randint(0, num_actions, (batch_size,))  # 直接生成随机索引
    rand_acs = torch.eye(num_actions, device=logits.device)[rand_indices]  # 正确索引方式

    # 3. Epsilon-贪婪选择
    mask = (torch.rand(batch_size, device=logits.device) > eps).float().unsqueeze(1)  # [batch_size, 1]
    return mask * argmax_acs + (1 - mask) * rand_acs


def sample_gumbel(shape, eps=1e-20, tens_type=torch.FloatTensor):
    """从Gumbel(0,1)分布中采样"""
    U = torch.autograd.Variable(tens_type(*shape).uniform_(), requires_grad=False)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    """ 从Gumbel-Softmax分布中采样"""
    y = logits + sample_gumbel(logits.shape, tens_type=type(logits.data)).to(logits.device)
    return F.softmax(y / temperature, dim=1)


def gumbel_softmax(logits, temperature=1.0):
    """从Gumbel-Softmax分布中采样,并进行离散化"""
    y = gumbel_softmax_sample(logits, temperature)
    y_hard = onehot_from_logits(y)
    y = (y_hard.to(logits.device) - y).detach() + y
    # 返回一个y_hard的独热量,但是它的梯度是y,我们既能够得到一个与环境交互的离散动作,又可以
    # 正确地反传梯度
    return y

def explore_with_noise(action,episode,total_episodes, noise_scale=0.2,explore=True):
    """
    为连续动作空间添加探索噪声episodes
    """
    # 保持tensor在原来的设备上
    if not isinstance(action, torch.Tensor):
        action = torch.FloatTensor(action)
    min_noise_scale = 0.01
    if explore:
        progress = min(1.0, episode / total_episodes)
        current_noise_scale = max(min_noise_scale, noise_scale - (noise_scale - min_noise_scale) * progress)
        # print(current_noise_scale)
        noise = torch.randn_like(action) * current_noise_scale
        action = action + noise
        action = torch.clamp(action, -1, 1)
    return action

def convert_to_numpy(data):
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    elif isinstance(data, list):
        # 处理列表中可能包含张量的情况
        converted = []
        for item in data:
            if isinstance(item, torch.Tensor):
                converted.append(item.detach().cpu().numpy())
            else:
                converted.append(item)
        return np.array(converted)
    else:
        return np.array(data)