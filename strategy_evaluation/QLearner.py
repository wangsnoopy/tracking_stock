import random
import numpy as np

class QLearner(object):
    def __init__(
        self,
        num_states=100,
        num_actions=4,
        alpha=0.2,
        gamma=0.9,
        rar=0.5,
        radr=0.99,
        dyna=0,
        verbose=False,
    ):
        self.verbose = verbose
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna
        self.s = 0
        self.a = 0

        self.Q = np.zeros((num_states, num_actions))
        self.T_c = np.ones((num_states, num_actions, num_states)) * 0.00001
        self.T = self.T_c / np.sum(self.T_c, axis=2, keepdims=True)
        self.R = np.zeros((num_states, num_actions))
        self.experiences = []

    def querysetstate(self, s):
        self.s = s
        if np.random.random() < self.rar:
            action = random.randint(0, self.num_actions - 1)
        else:
            action = np.argmax(self.Q[s])
        if self.verbose:
            print(f"s = {s}, a = {action}")
        return action

    def query(self, s_prime, r):
        s = self.s
        a = self.a

        self.Q[s, a] = (1 - self.alpha) * self.Q[s, a] + self.alpha * (r + self.gamma * np.max(self.Q[s_prime]))

        if self.dyna > 0:
            self.T_c[s, a, s_prime] += 1
            self.T = self.T_c / np.sum(self.T_c, axis=2, keepdims=True)
            self.R[s, a] = (1 - self.alpha) * self.R[s, a] + self.alpha * r
            self.experiences.append((s, a, s_prime, r))

            for _ in range(self.dyna):
                if self.experiences:
                    thiss, thisa, thiss_prime, thisr = random.choice(self.experiences)
                    thisa_prime = np.argmax(self.Q[thiss_prime])
                    self.Q[thiss, thisa] = (1 - self.alpha) * self.Q[thiss, thisa] + self.alpha * (
                        thisr + self.gamma * self.Q[thiss_prime, thisa_prime]
                    )

        if np.random.random() < self.rar:
            action = random.randint(0, self.num_actions - 1)
        else:
            action = np.argmax(self.Q[s_prime])

        self.s = s_prime
        self.a = action
        self.rar *= self.radr

        if self.verbose:
            print(f"s = {s_prime}, a = {action}, r={r}")
        return action

    def author(self):
        return "awang758"

if __name__ == "__main__":
    print("Remember Q from Star Trek? Well, this isn't him")	