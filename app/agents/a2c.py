import gym
import logging
import argparse
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl
import tensorflow.keras.losses as kls
import tensorflow.keras.optimizers as ko
import const

class ProbabilityDistribution(tf.keras.Model):
  def call(self, logits, **kwargs):
    # Sample a random categorical action from the given logits.
    return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)


class Model(tf.keras.Model):
  def __init__(self, num_actions):
    super().__init__('mlp_policy')
    # Note: no tf.get_variable(), just simple Keras API!
    self.model1 = tf.keras.Sequential()
    self.model2 = tf.keras.Sequential()

    for i in range(0, const.NUM_LAYERS):
      self.model1.add(kl.Dense(const.HIDDEN_LAYER_SIZE, activation='relu'))
      self.model2.add(kl.Dense(const.HIDDEN_LAYER_SIZE, activation='relu'))

    self.value = kl.Dense(1, name='value')
    # Logits are unnormalized log probabilities.
    self.logits = kl.Dense(num_actions, name='policy_logits')
    self.dist = ProbabilityDistribution()

  def call(self, inputs, **kwargs):
    # Inputs is a numpy array, convert to a tensor.
    x = tf.convert_to_tensor(inputs)
    # Separate hidden layers from the same input tensor.
    hidden_logs = self.model1(x)
    hidden_vals = self.model2(x)
    return self.logits(hidden_logs), self.value(hidden_vals)

  def action_value(self, obs, actions):
    # Executes `call()` under the hood.
    logits, value = self.predict_on_batch(obs)
    action = self.dist.predict_on_batch(logits)
    action_number = np.asarray(action)[0]
    while not(actions[action_number]):
      arr = np.array(logits)
      arr[0][action_number] -= 1
      logits = tf.convert_to_tensor(arr)
      action = self.dist.predict_on_batch(logits)
      action_number = np.asarray(action)[0]
    # Another way to sample actions:
    #   action = tf.random.categorical(logits, 1)
    # Will become clearer later why we don't use it.
    return np.squeeze(action, axis=-1), np.squeeze(value, axis=-1)


class A2CAgent:
  def __init__(self, 
    model, 
    lr=const.LEARNING_RATE, 
    gamma=const.GAMMA, 
    value_c=const.VALUE_C, 
    entropy_c=const.ENTROPY_C):

    # `gamma` is the discount factor; coefficients are used for the loss terms.
    self.gamma = gamma
    self.value_c = value_c
    self.entropy_c = entropy_c

    self.model = model
    self.model.compile(
      optimizer=ko.RMSprop(lr=lr),
      # Define separate losses for policy logits and value estimate.
      loss=[self._logits_loss, self._value_loss])

  def train(self, env, batch_sz=const.BATCH_SIZE, updates=const.UPDATES):
    # Storage helpers for a single batch of data.
    actions = np.empty((batch_sz,), dtype=np.int32)
    rewards, dones, values = np.empty((3, batch_sz))
    observations = np.empty((batch_sz,) + env.observation_space.shape)
    # Training loop: collect samples, send to optimizer, repeat updates times.
    ep_rewards = [0.0]
    next_obs = env.reset()
    for update in range(updates):
      for step in range(batch_sz):
        observations[step] = next_obs.copy()
        actions[step], values[step] = self.model.action_value(next_obs[None, :], env.get_actions())
        next_obs, rewards[step], dones[step], _ = env.step(actions[step])

        ep_rewards[-1] += rewards[step]
        if dones[step]:
          ep_rewards.append(0.0)
          next_obs = env.reset()
          logging.info(f'Episode: {len(ep_rewards)-1}, Reward: {round(ep_rewards[-2], 4)}')

      _, next_value = self.model.action_value(next_obs[None, :], env.get_actions())
      returns, advs = self._returns_advantages(rewards, dones, values, next_value)
      # A trick to input actions and advantages through same API.
      acts_and_advs = np.concatenate([actions[:, None], advs[:, None]], axis=-1)
      # Performs a full training step on the collected batch.
      # Note: no need to mess around with gradients, Keras API handles it.
      losses = self.model.train_on_batch(observations, [acts_and_advs, returns])
      logging.debug("[%d/%d] Losses: %s" % (update + 1, updates, losses))

    return ep_rewards

  def test(self, env):
    obs, done, ep_reward = env.reset(), False, 0
    while not done:
      action, _ = self.model.action_value(obs[None, :], env.get_actions())
      obs, reward, done, _ = env.step(action)
      ep_reward += reward
    return ep_reward

  def _returns_advantages(self, rewards, dones, values, next_value):
    # `next_value` is the bootstrap value estimate of the future state (critic).
    returns = np.append(np.zeros_like(rewards), next_value, axis=-1)
    # Returns are calculated as discounted sum of future rewards.
    for t in reversed(range(rewards.shape[0])):
      returns[t] = rewards[t] + self.gamma * returns[t + 1] * (1 - dones[t])
    returns = returns[:-1]
    # Advantages are equal to returns - baseline (value estimates in our case).
    advantages = returns - values
    return returns, advantages

  def _value_loss(self, returns, value):
    # Value loss is typically MSE between value estimates and returns.
    return self.value_c * kls.mean_squared_error(returns, value)

  def _logits_loss(self, actions_and_advantages, logits):
    # A trick to input actions and advantages through the same API.
    actions, advantages = tf.split(actions_and_advantages, 2, axis=-1)
    # Sparse categorical CE loss obj that supports sample_weight arg on `call()`.
    # `from_logits` argument ensures transformation into normalized probabilities.
    weighted_sparse_ce = kls.SparseCategoricalCrossentropy(from_logits=True)
    # Policy loss is defined by policy gradients, weighted by advantages.
    # Note: we only calculate the loss on the actions we've actually taken.
    actions = tf.cast(actions, tf.int32)
    policy_loss = weighted_sparse_ce(actions, logits, sample_weight=advantages)
    # Entropy loss can be calculated as cross-entropy over itself.
    probs = tf.nn.softmax(logits)
    entropy_loss = kls.categorical_crossentropy(probs, probs)
    # We want to minimize policy and maximize entropy losses.
    # Here signs are flipped because the optimizer minimizes.
    return policy_loss - self.entropy_c * entropy_loss
