This code works with Ego-planner or similarly structured planners, replace 'traj_server.cpp' with this file for easy use. A few parameters need modifying. For other planners, please modify it to fit in your own structure. For those planner that add yaw angle after optimization, it won't be a complex work.

This code includes two main components: anchor selection and utilization(which includes adding yaw to the trajectory.). You can find anchor selection part in Function AnchorReplan().
