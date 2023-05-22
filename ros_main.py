#!/usr/bin/env python3

import pickle

import numpy as np
import rospy

import agents
import ros_environment
import simulation


def main():
  if not rospy.is_shutdown():
    rospy.init_node('ros_adapter')
    rospy.set_param('~model_file', '')
    env_creator = lambda render: ros_environment.RosEnvironment(render)
    if rospy.get_param('~model_file', '') == '':
      simulation.Simulation(
        env_creator=env_creator,
        fitter=agents.CrossEntropyFitter(
          env_type=agents.CrossEntropyFitter.EnvType.ROS,
          n_sessions=200,
          n_elites=50,
          policy_hidden_layers=(25, 30)
        ),
        parallel=True,
        save_options=(
          simulation.SaveOption.MODEL,
          simulation.SaveOption.MEAN_REWARDS,
          simulation.SaveOption.MEDIAN_REWARDS,
          simulation.SaveOption.MOVING_AVERAGES,
          simulation.SaveOption.MAX_REWARDS,
        ),
      ).run()
    else:
      with open(rospy.get_param('~model_file'), 'rb') as file:
        model = pickle.load(file)

      dummy_env = env_creator(False)
      np.random.seed(42)
      agent = agents.CrossEntropyAgent(dummy_env.N_ACTIONS, model, dummy_env.reset())
      simulation.Simulation(
        env_creator=env_creator,
        agent=agent,
        save_options=(),
      ).run()


if __name__ == '__main__':
    main()
