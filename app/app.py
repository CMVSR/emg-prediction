
# standard
import logging
import argparse
import datetime
import json
import random

# debug
import ptvsd

# packages
import gym
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

# internal
import const
from dataset import DataSet
from env import DataEnv
from agents.a2c import A2CAgent, Model


def main(
    test_size: float=const.TEST_SIZE,
    random_state: int=const.RANDOM_STATE,
    illegal_action_punish: int=const.ILLEGAL_ACTION_PUNISH,
    feature_action_punish: int=const.FEATURE_ACTION_PUNISH,
    classification_reward: int=const.CLASSIFICATION_REWARD,
    classification_punishment: int=const.CLASSIFICATION_PUNISHMENT,
    hidden_layer_size: int=const.HIDDEN_LAYER_SIZE,
    learning_rate: float=const.LEARNING_RATE,
    gamma: float=const.GAMMA,
    value_c: float=const.VALUE_C,
    entropy_c: float=const.ENTROPY_C,
    batch_size: int=const.BATCH_SIZE,
    updates: int=const.UPDATES):

    logging.getLogger().setLevel(logging.INFO)

    # load dataset
    dataset = DataSet()

    x_train, x_test, y_train, y_test = train_test_split(
        dataset.data,
        dataset.target,
        test_size=const.TEST_SIZE,
        random_state=random_state)

    # initialize environments
    train_env = DataEnv(features=x_train, labels=y_train, random=True)
    test_env = DataEnv(features=x_test, labels=y_test, random=False)

    # rl training
    model = Model(train_env.action_space.n)
    agent = A2CAgent(model, learning_rate)
    rewards_history = agent.train(train_env)
    print("Finished training. Testing...")
    print(f'Test-train split: {const.TEST_SIZE}')

    # reset test env, create variable placeholders to store results
    obs, done, ep_reward = test_env.reset(), False, 0
    num_correct = 0
    num_labels = 0
    true_labels = []
    generated_labels = []

    # environment will generate episodes until the agent has seen each one
    while not test_env.done_testing:
        # agent will run on the episode until done
        print(f"Total Episode Reward: {agent.test(test_env)}")

        # load the generated and correct labels for analysis
        agent_label, correct_label = test_env.render()
        generated_labels.append(str(agent_label))
        true_labels.append(str(correct_label))

        if agent_label == correct_label:
            num_correct += 1
        num_labels += 1

    # calculate accuracy
    label_accuracy = 100*num_correct/num_labels
    print(f'Accuracy on test data: {num_correct} out of {num_labels} for a score of \
        {label_accuracy}')

    time = datetime.datetime.now().strftime('%d%m%y_%H_%M_%S')

    # create result object
    result = {
        'MODEL_TYPE': 'A2C',
        'TIME': time,
        'PARAMS': {
            'TEST_SIZE': test_size,
            'RANDOM_STATE': random_state,
            'ILLEGAL_ACTION_PUNISH': illegal_action_punish,
            'FEATURE_ACTION_PUNISH': feature_action_punish,
            'CLASSIFICATION_REWARD': classification_reward,
            'CLASSIFICATION_PUNISHMENT': classification_punishment,
            'HIDDEN_LAYER_SIZE': hidden_layer_size,
            'LEARNING_RATE': learning_rate,
            'GAMMA': gamma,
            'VALUE_C': value_c,
            'ENTROPY_C': entropy_c,
            'BATCH_SIZE': batch_size,
            'UPDATES': updates
        },
        'RESULTS': {
            'ACCURACY': round(label_accuracy, 2),
            'TRUE_LABELS': true_labels,
            'GENERATED_LABELS': generated_labels,
        }
    }

    # average reward history for every 100 episodes
    n = 100
    rewards_history = [sum(rewards_history[i:i+n])//n for i in range(0, len(rewards_history), n)]

    rewards = {
        'REWARD_HISTORY': rewards_history,
        'TIME': time
    }

    with open(f'/home/emg-prediction/logs/rewards.json', 'a') as f:
        json.dump(rewards, f)

    with open(f'/home/emg-prediction/logs/result.json', 'a') as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()
