==========
QuickStart
==========

Install the latest CityLearn version from PyPi with the :code:`pip` command:

.. code-block:: console

   pip install CityLearn

Decentralized-Independent RBC
*****************************
Run the following to simulate an environment controlled by decentralized-independent RBC agents for a single episode:

.. code:: python

    from citylearn.citylearn import CityLearnEnv
    from citylearn.agents.rbc import BasicRBC as RBCAgent

    dataset_name = 'citylearn_challenge_2022_phase_1'
    env = CityLearnEnv(dataset_name)
    agents = RBCAgent(
        action_space=env.action_space,
        observation_space=env.observation_space,
        building_information=env.get_building_information(),
        observation_names=env.observation_names,
    )
    observations = env.reset()

    while not env.done:
        actions = agents.select_actions(observations)

        # apply actions to env
        observations, rewards, _, _ = env.step(actions)

    # print cost functions at the end of episode
    print(env.evaluate())

Decentralized-Independent SAC
*****************************
Run the following to simulate an environment controlled by decentralized-independent SAC agents for a 10 episodes:

.. code:: python

    from citylearn.citylearn import CityLearnEnv
    from citylearn.agents.sac import SAC as SACAgent

    dataset_name = 'citylearn_challenge_2022_phase_1'
    env = CityLearnEnv(dataset_name)
    agents = SACAgent(
        action_space=env.action_space,
        observation_space=env.observation_space,
        building_information=env.get_building_information(),
        observation_names=env.observation_names,
    )
    episodes = 10 # number of training episodes

    # train agents
    for e in range(episodes):
        observations = env.reset()

        while not env.done:
            actions = agents.select_actions(observations)

            # apply actions to env
            next_observations, rewards, _, _ = env.step(actions)

            # update policies
            agents.add_to_buffer(observations, actions, rewards, next_observations, done=env.done)
            observations = [o for o in next_observations]

    # print cost functions at the end of episode
    print(env.evaluate())