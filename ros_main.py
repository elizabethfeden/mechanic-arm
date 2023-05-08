#!/usr/bin/env python3

import rospy

import agents
import ros_environment
import simulation


def main():
  if not rospy.is_shutdown():
    rospy.init_node('ros_adapter')
    simulation.Simulation(
      env_creator=lambda render: ros_environment.RosEnvironment(render),
      fitter=agents.CrossEntropyFitter(
        env_type=agents.CrossEntropyFitter.EnvType.ROS,
        n_sessions=50,
        n_elites=10,
        policy_hidden_layers=(5,)
      ),
      parallel=True,
      save_options=(
        simulation.SaveOption.MODEL,
        simulation.SaveOption.MEAN_REWARDS,
        simulation.SaveOption.MEDIAN_REWARDS,
      ),
    ).run()


if __name__ == '__main__':
    main()
