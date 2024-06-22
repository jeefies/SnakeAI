import tensorflow as tf
import numpy as np
from tensorflow import keras
from matplotlib import pyplot as plt

from collections import deque
import random
import time
from random import Random

# import tkinter as tk

RAND_SEED = 27 + int(time.time())
tf.random.set_seed(RAND_SEED)
random.seed(RAND_SEED)
np.random.seed(RAND_SEED)

class Snake:
    def __init__(self, n = 5, showcase = True, _nearDis = 4):
        self.n = n
        self.showcase = showcase
        self.apple = [-n, -n]
        self._nearDis = _nearDis
        self.map = np.zeros((n, n))
        self.rand = Random(RAND_SEED)
        self.stateLen = 8 + len(self._nearHead(0, 0, self._nearDis))
        if showcase:
            plt.ion()

    def reset(self):
        n = self.n
        self.map = np.zeros((n, n))
        (x, y) = self.head = self._randPlace()
        self.map[x, y] = 1
        self.time = n * 2
        self.apple = self.genApple()
        return self.state(), None
    
    def _nearHead(self, i, j, k):
        isIn = lambda x, y: 0 <= x < self.n and 0 <= y < self.n
        return [
            (self.map[x, y] if isIn(x, y) else (-1 if x == self.apple[0] and y == self.apple[1] else 0))
                for x in range(i - k, i + k + 1)
                for y in range(j - k + abs(i - x), j + k - abs(i - x) + 1)
        ]
    
    def state(self):
        # return np.hstack([self.map.flatten(), np.array(self.apple)])
        return np.array([
            *self.head,
            self.n - self.head[0] - 1,
            self.n - self.head[1] - 1,
            self.time,
            *self.apple,
            abs(self.apple[0] - self.head[0]) + abs(self.apple[1] - self.head[1]),
            *self._nearHead(*self.head, self._nearDis)
        ])
    
    def stateShape(self):
        return (self.stateLen,)

    def actionN(self):
        return 4

    def _randPlace(self):
        return [
            self.rand.randint(0, self.n - 1),
            self.rand.randint(0, self.n - 1)
            ]

    def genApple(self):
        x, y = self._randPlace()
        while self.map[x, y] > 0:
            x, y = self._randPlace()
        return x, y
    
    def selectAction(self, P):
        x, y = self.head
        inf = 1e9
        if x == 0:
            P[2] = -inf
        elif x == self.n - 1:
            P[3] = -inf
        if y == 0:
            P[0] = -inf
        elif x == self.n - 1:
            P[1] = -inf
        return np.argmax(P)
    
    def sample(self):
        return self.selectAction(np.random.rand(4))

    def render(self):
        if not self.showcase:
            return
        n = self.n
        # print("Rendering...")
        
        self.map[self.apple[0], self.apple[1]] = -1
        plt.clf()
        plt.imshow(self.map)
        plt.show(block=False)
        plt.pause(0.1)
        # print(self.map)
        self.map[self.apple[0], self.apple[1]] = 0

    
    def step(self, action):
        """
            return state, reward, done
        """
        n = self.n
        length = sum(sum(self.map > 0))

        # 0 for left 1 for right 2 for up 3 for down
        x, y = self.head
        if action == 0:
            y -= 1
        elif action == 1:
            y += 1
        elif action == 2:
            x -= 1
        elif action == 3:
            x += 1
        else:
            raise "FUCK YOU, bad action !"
        self.head = [x, y]

        bodyMap = np.maximum(self.map - 1, np.zeros((n, n)))
        
        # edge Crash or body Crash
        if x < 0 or x >= self.n or y < 0 or y >= self.n or bodyMap[x, y] > 0:
            self.map = - np.ones((n, n))
            return self.state(), 0, 1, None, None
        
        if x == self.apple[0] and y == self.apple[1]:
            self.map[x, y] = length + 1
            self.apple = self.genApple()
            self.time += 2 * n + length
            self.render()
            return self.state(), (n + length) * n, 0, None, None
        
        self.map = bodyMap
        self.map[x, y] = length
        
        self.time -= 1
        if self.time < 0:
            return self.state(), 0, 1, None, None
            
        reward = n - (abs(x - self.apple[0]) + abs(y - self.apple[1])) / 2
        self.render()
        return self.state(), reward, 0, None, None
    
    def close(self):
        pass


N = 5
env = Snake(N, False)


def agent(state_shape, action_shape):
    print(state_shape, action_shape)
    learningRate = 0.001
    init = keras.initializers.HeUniform(seed=RAND_SEED)
    model = keras.Sequential()
    model.add(keras.layers.Dense(
        64 + 16, input_shape=state_shape, activation="relu", kernel_initializer=init
    ))
    model.add(keras.layers.Dense(
        16 + 4, activation='relu', kernel_initializer=init
    ))
    model.add(keras.layers.Dense(
        8, "relu", kernel_initializer=init
    ))
    model.add(keras.layers.Dense(
        action_shape, activation='linear', kernel_initializer=init
    ))
    model.compile(
        loss=keras.losses.Huber(),
        optimizer=keras.optimizers.Adam(learning_rate=learningRate),
        metrics=[keras.metrics.BinaryAccuracy()]
    )
    return model

def train(env, mem, Qnet, Tnet, done, batchSize = 200, MIN_REPLAY_SIZE = 1000):
    learningRate = 0.7
    discountFactor = 0.94

    if len(mem) < MIN_REPLAY_SIZE:
        return 0
    
    batch = random.sample(mem, batchSize)
    curStates = np.array([pack[0] for pack in batch])
    newStates = np.array([pack[1] for pack in batch])

    curQs, newQs = Qnet.predict(curStates, batch_size=batchSize, verbose = False), Tnet.predict(newStates, batch_size=batchSize, verbose = False)

    X, y = deque(), deque()
    for i, (cur, nex, act, reward, done) in enumerate(batch):
        maxQ = reward + discountFactor * np.max(newQs[i]) if not done else reward
        curQs[i][act] = (1 - learningRate) * curQs[i][act] + learningRate * maxQ

    print("Fitting...")
    Qnet.fit(curStates, curQs, batch_size = batchSize, verbose = False, shuffle = True, epochs = 2)
    return 1

def play(rounds = 20):
    Qnet = keras.models.load_model("Qnet.keras")
    env.showcase = True

    for round in range(rounds):
        print("Start Round: ", round)
        state, info = env.reset()
        totalReward = 0

        done = False
        while not done:
            predict = Qnet.predict(state.reshape((1, env.stateLen)), verbose = False)
            action = env.selectAction(predict[0])

            state, reward, done, _, _ = env.step(action)
            totalReward += reward
        
        print("Total Reward: ", totalReward) 

        time.sleep(1)

def mainTraining():
    eps, minEps, decay = 1, 0.2, 0.0006
    trainRounds, trainCounts = 10000, 0

    isFirstTime = False

    if isFirstTime:
        Tnet = agent(env.stateShape(), env.actionN())
        Qnet = agent(env.stateShape(), env.actionN())
        Qnet.save("Qnet.keras")
    else:
        Qnet = keras.models.load_model("Qnet.keras")
        Tnet = keras.models.load_model("Qnet.keras")

    replayMem = deque(maxlen = 500000)

    print("Start Training...")
    rewardMem = deque()

    for t in range(trainRounds):
        if t > trainRounds - 1:
            env.showcase = True

        totalReward, steps = 0, 0
        state, _ = env.reset()
        # print(state, env.action_space.sample())

        done = False
        while not done:
            if False:
                env.render()

            # print("cur state: ", state)

            steps += 1
            randPlace = np.random.rand()
            if randPlace <= eps and steps > 5:
                action = env.sample()
            else:
                # print("pred state: ", state)
                predict = Qnet.predict(state.reshape((1, env.stateLen)), verbose = False)
                # print(predict)
                action = env.selectAction(predict[0])

            newState, reward, done, _, _ = env.step(action)
            
            replayMem.append(
                [state, newState, action, reward, done]
            )

            if steps % 8 == 0 or done:
                trainCounts += train(env, replayMem, Qnet, Tnet, done)

            state = newState
            totalReward += reward

            if done:
                print("Round %d Total Rewards: %d, after %d steps\n" % (t, totalReward, steps))
                rewardMem.append(totalReward)

                if trainCounts > 100:
                    print("Copying...")
                    Tnet.set_weights(Qnet.get_weights())
                    Qnet.save("Qnet.keras")
                    trainCounts = 0
                
                if t and t % 1000 == 0:
                    plt.plot(np.array(rewardMem))
                    plt.show()
        
        eps = minEps + (eps - minEps) * np.exp(-decay * eps)
        print("Eps to ", eps)

    env.close()

if __name__ == "__main__":
    mainTraining()
    # play()