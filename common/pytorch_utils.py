import torch

import numpy as np

USE_CUDA = torch.cuda.is_available()
DEVICE =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def compute_td_error(model,target_model,optimizer,batch_size,replay_buffer,device):
    if batch_size > len(replay_buffer):
        print("replay buffer to small ")
        return None


    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state = torch.FloatTensor(state).to(device)

    reward = torch.FloatTensor(reward).unsqueeze(1). to(device)  # assume state are array of numpy
    action = torch.LongTensor(action).unsqueeze(1).to(device)  # assume action are array on integer
    mask = torch.FloatTensor(1 - np.float32(done)).unsqueeze(1).to(device)

    next_state = torch.FloatTensor(next_state).to(device)


    #state = state.unsqueeze(1).to(device)
    #print(" shape ",state[0])
    q_values = model.forward(state)
    q_value = q_values.gather(1, action)

    #print("Q values",q_values)
    #print("Q value_action ",q_value)
    next_q_values = target_model.forward(next_state)
    target_action = next_q_values.max(1)[1].unsqueeze(1)
    next_q_value = target_model(next_state).gather(1, target_action)

    expected_q_value = 0.8 * (reward + 0.99 * next_q_value * mask)

    loss = (q_value - expected_q_value.detach()).pow(2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss
