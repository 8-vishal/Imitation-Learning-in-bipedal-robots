from abc import ABC
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import gym
import bipedal
import sys

sys.path.append("./"), sys.path.append("../")


def write_data(path, d):
    with open(path, 'a') as file:
        file.write(str(d) + "\n")


class DQNFNN(nn.Module, ABC):
    def __init__(self, L_R, In_Dims, Layer1_dims, Layer2_dims,
                 N_Action):
        super(DQNFNN, self).__init__()
        self.In_Dims = In_Dims
        self.Layer1_dims = Layer1_dims
        self.Layer2_dims = Layer2_dims
        self.N_Action = N_Action
        self.layer_1 = nn.Conv1d(*self.In_Dims, self.Layer1_dims)
        self.layer_2 = nn.Conv1d(self.Layer1_dims, self.Layer2_dims, 3)
        self.layer_3 = nn.Conv1d(self.Layer2_dims, 256, 3)
        self.layer_4 = nn.Conv1d(256, 512, 3)
        self.layer_5 = nn.Conv1d(512, self.N_Action, 3)

        self.optimizer = optim.Adam(self.parameters(), lr=L_R)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.layer_1(state))
        x = F.relu(self.layer_2(x))
        x = F.relu(self.layer_3(x))
        x = F.relu(self.layer_4(x))
        actions = self.layer_5(x)
        return actions


class FNNAgent():
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions,
                 max_mem_size=100000, eps_end=0.05, eps_dec=5e-4):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0
        self.iter_cntr = 0
        self.replace_target = 100

        self.Q_eval = DQNFNN(lr, N_Action=n_actions, In_Dims=input_dims,
                             Layer1_dims=256, Layer2_dims=256)
        self.Q_next = DQNFNN(lr, N_Action=n_actions, In_Dims=input_dims,
                             Layer1_dims=64, Layer2_dims=64)

        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, terminal):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = terminal

        self.mem_cntr += 1

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor([observation]).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def learn(self):
        if self.mem_cntr < self.batch_size:
            return

        self.Q_eval.optimizer.zero_grad()

        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        action_batch = self.action_memory[batch]
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)

        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next = self.Q_eval.forward(new_state_batch)
        q_next[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0]

        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        write_data("./data.txt", loss.item())
        loss.backward()
        self.Q_eval.optimizer.step()

        self.iter_cntr += 1
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min \
            else self.eps_min
        # if self.iter_cntr % self.replace_target == 0:
        #   self.Q_next.load_state_dict(self.Q_eval.state_dict())


def plotLearning(x, scores, epsilons, filename, lines=None):
    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t - 20):(t + 1)])

    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel('Number of Iterations')
    ax1.set_ylabel('Epsilon', color=color)
    ax1.plot(x, eps_history, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    # Adding Twin Axes to plot using dataset_2
    ax2 = ax1.twinx()

    color = 'tab:green'
    ax2.set_ylabel('Scores', color=color)
    ax2.plot(x, running_avg, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    # Adding title
    plt.title('Epsilon and scores for training DQN')

    plt.savefig(filename)
    plt.show()


def loss_data(path):
    return [float(j) for j in list(open(path))]


def plot_loss(loss):
    loss = loss_data("./data.txt")
    no_none = []
    for val in loss:
        if val is not None:
            no_none.append(val)
    mse = []
    num = int(len(no_none) / 500)
    for i in range(int(len(no_none) / num)):
        mse.append(np.mean(no_none[i:i + num]))
    rmse = [np.sqrt(j) for j in mse]
    x = range(len(mse))
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Number of Iterations')
    ax1.set_ylabel('MSE Loss', color=color)
    ax1.plot(x, mse, color=color, label="MSE Loss")
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend()
    # Adding Twin Axes to plot using dataset_2
    ax2 = ax1.twinx()

    color = 'tab:green'
    ax2.set_ylabel('RMSE Loss', color=color)
    ax2.plot(x, rmse, color=color, label="RMSE Loss")
    ax2.tick_params(axis='y', labelcolor=color)

    # Adding title
    plt.title('Epsilon and scores for training DQN')
    plt.legend()
    # Show plot
    plt.savefig("loss.png")
    plt.show()


if __name__ == '__main__':
    env = gym.make('Bipedal-v0')
    agent = FNNAgent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=4, eps_end=0.01,
                     input_dims=[8], lr=0.001)
    scores, eps_history, total_loss = [], [], []
    n_games = 500

    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.store_transition(observation, action, reward,
                                   observation_, done)
            agent.learn()
            observation = observation_
        scores.append(score)
        eps_history.append(agent.epsilon)

        avg_score = np.mean(scores[-100:])

        print('episode ', i, 'score %.2f' % score,
              'average score %.2f' % avg_score,
              'epsilon %.2f' % agent.epsilon)
    x = [i + 1 for i in range(n_games)]
    filename = 'lunar_lander.png'
    plotLearning(x, scores, eps_history, filename)
    plot_loss(total_loss)
