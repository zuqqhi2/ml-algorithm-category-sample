#!/usr/bin/python

from numpy.random import normal, rand
from hmmlearn import hmm
import numpy as np

def generate_series(num_data):
  state_centers = [[1.0, 8.0], [5.0, 3.0], [8.0, 5.0]]
  transition_prob = [[0.5, 0.4, 0.1], [0.05, 0.3, 0.65], [0.1, 0.4, 0.5]]
  labels = ['error', 'wait', 'run']

  noise_mean = [0.0, 0.0, 0.0]
  noise_std = [0.001, 0.001, 0.001]

  current_state = 1
  result_series = []
  result_states = []
  for cnt in range(num_data):
    result_states.append(current_state)
    result_series.append([normal(noise_mean[current_state], noise_std[current_state])])

    boundary = 0.0
    rval = rand()
    for state_id, prob in enumerate(transition_prob[current_state]):
      boundary += prob
      if rval < boundary:
        current_state = state_id
        break

  return result_states, result_series

print(generate_series(10))

model = hmm.GaussianHMM(n_components=3, covariance_type="full")
states, X = generate_series(100000)
model.fit(np.array(X))

states, X = generate_series(100)
predicted_states = model.predict(np.array(X))
#print(states)
#print(predicted_states)

accuracy = 0.0
for idx, state_id in enumerate(states):
  if state_id == predicted_states[idx]:
    accuracy += 1.0
accuracy /= float(len(states))
print(accuracy)
