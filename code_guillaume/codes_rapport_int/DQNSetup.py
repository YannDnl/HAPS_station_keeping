import numpy as np
import random as rd
from collections import deque
import sys
from policies import max_action

class DQNSetup():
    def __init__(self, env, model, policy, gamma=.9, alpha=.1, batch_size=100, buffer_limit=10000):
        '''
        Crée un objet permettant d'entraîner le NN model sur l'environnement env, avec la stratégie policy
        gamma correspond au paramètre dans la formule d'actualisation des rewards : R(s,a) = R + gamma*Q(S', A')
        alpha correspond au paramètre dans la formule d'actualisation de Q : Q'(S,A) = (1-alpha)*Q(S,A) + alpha*(R + gamma*Q(S', A'))
        batch_size est la taille des batchs utilisés à chaque fitting du NN, après chaque pas
        '''

    def fit(self, nb_episodes=100, visualize=None, verbose=1):
        '''
        Entraîne le modèle grâce aux techniques de DQN
        '''
        for episode in range(nb_episodes):
            state = self.env.reset()
            done = False
            while not done:
                state = np.array([self.env.currentstate()])
                # On choisit une action en fonction du modèle, de l'état actuel et de la stratégie
                action = self.policy(self.env, self.model)
                next_state, reward, done, _ = self.env.step(action)
                
                # On mémorise ce qui s'est passé
                self.buffer.append((state, action, reward, next_state, done))
                
                # On sélectionne un sous-ensemble du buffer
                if len(self.buffer) > self.batch_size:

                    samples = rd.sample(self.buffer, self.batch_size)

                    sample_states = [] # Contiendra les états des samples
                    sample_fs = [] # Contiendra les valeurs target des Q(S,A) pour les états S

                    for curr_state, action, reward, next_state, curr_done in samples:
                        sample_states.append(curr_state[0])
                        target = reward
                        if not curr_done:
                            # On calcule R'(S,A) = R + gamma*max(Q(S',A'))
                            target = reward + self.gamma * np.amax(self.model.predict(np.array([next_state]))[0])
                        target_f = self.model.predict(curr_state)
                        # On calcule Q'(S,A) = (1-alpha)*Q(S,A) + alpha*R'(S,A)
                        target_f[0][action] = (1-self.alpha)*target_f[0][action] + self.alpha*target

                        sample_fs.append(target_f[0])

                    # On fit le modèle
                    sample_states = np.array(sample_states)
                    sample_fs = np.array(sample_fs)
                    self.model.fit(sample_states, sample_fs, epochs=1, verbose=0)

                state = next_state

    def test(self, nb_episodes=10, visualize=None):
        '''
        Fonction de tests et visualisation
        Renvoie une liste des rewards accumulés à chaque épisode
        '''

    def export_model(self, n=100):
        '''
        Fonction pour exporter le modèle
        '''
