# ReinforcementLearning-1



---

# DQN Agent for CartPole-v1 Environment

This project demonstrates the implementation of a Deep Q-Network (DQN) agent to solve the CartPole-v1 environment in OpenAI Gym. The agent is designed using a neural network to approximate the Q-value function, enabling it to learn and act in the environment to achieve the goal of balancing the pole on the cart.

## Project Structure

- `DQNAgent` Class:
  - **Initialization**: Initializes the environment, action space, and model parameters like gamma, learning rate, epsilon, etc.
  - **build_model()**: Defines and compiles the neural network model used to predict Q-values for a given state.
  - **remember()**: Stores the experience tuple `(state, action, reward, next_state, done)` in memory for replay.
  - **act()**: Chooses an action based on the epsilon-greedy policy.
  - **replay()**: Trains the neural network on a randomly sampled mini-batch from memory to update the Q-values.
  - **adaptiveEGreedy()**: Adjusts the epsilon value to reduce exploration over time.

- **Training Loop**:
  - **Episodes**: Runs for a specified number of episodes, with each episode consisting of the agent interacting with the environment.
  - **State Management**: Reshapes the state and next state to be compatible with the neural network input.
  - **Replay and Learning**: The agent trains and updates its network based on the experiences stored in memory.

## Requirements

Before running the code, install the following dependencies:

```bash
pip install numpy pandas gym keras tensorflow
```

## Running the Code

To train the DQN agent, simply run the script:

```bash
python dqn_cartpole.py
```

The code will output the number of episodes and the time steps the agent managed to balance the pole in each episode.

## Hyperparameters

- `gamma` (Discount Factor): 0.95
- `learning_rate`: 0.001
- `epsilon` (Exploration Rate): Starts at 1, decreases to a minimum of 0.01
- `epsilon_decay`: 0.995
- `memory` (Replay Buffer Size): 1000
- `batch_size`: 16
- `episodes`: 50

## Environment

The environment used is CartPole-v1 from OpenAI Gym. The goal is to keep the pole balanced on the cart for as long as possible.

- **State**: A 4-dimensional vector representing the cart's position, velocity, pole angle, and pole velocity.
- **Action Space**: 2 discrete actions - moving the cart left or right.

## Future Improvements

- Implementing experience replay prioritization to improve sample efficiency.
- Adding target networks to stabilize learning.
- Experimenting with different network architectures and hyperparameters.

## License

This project is licensed under the MIT License.

---

