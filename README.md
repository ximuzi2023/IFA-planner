## IFA-planner
Image-Feature-Aware (IFA) planner focuses on adjusting the quadrotor’s orientation to garantee enough image features for the visual–inertial odometry (VIO).

### structure
The IFA-planner introdued a Image Feature Perception Mechanism model compared to previous work. The figure below explains how it works. For more details, please refer to our essay.

Our code is in the folder 'image feature perception mechanism'. This code includes two main components: anchor selection and utilization(which includes adding yaw to the trajectory). You can find anchor selection part in Function 'AnchorReplan()'. 

### how to use?
This code works with Ego-Planner or similarly structured planners, replace 'traj_server.cpp' with 'traj_server_anchor.cpp' for easy use. A few parameters need modifying. 

For other planners, please modify it to fit in your own structure. For planners that add yaw angle after optimization, it won't be a complex work. For other planners, a modification on the loss function is needed to incorporate penalty on image feature perception.


