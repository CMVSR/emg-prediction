# /app/agents

The A2C agent in this directory uses the following implementation: http://inoryy.com/post/tensorflow2-deep-reinforcement-learning/

There is one small change to the `action_value` function for the `Model` class. The model is prevented from choosing an action not in the list of actions input to the `action_value` call. This allows the program to send allowed actions to the model and not allow illegal actions.