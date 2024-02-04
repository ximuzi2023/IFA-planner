#include <nav_msgs/Odometry.h>
#include <traj_utils/PolyTraj.h>
#include <optimizer/poly_traj_utils.hpp>
#include <quadrotor_msgs/PositionCommand.h>
#include <std_msgs/Empty.h>
#include <visualization_msgs/Marker.h>
#include <ros/ros.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/distances.h>
#include <pcl/common/common.h>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <algorithm>
#include <numeric>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud_conversion.h>
#include "std_msgs/Float64.h"

using namespace Eigen;

// enabling
bool anchor_enable_;
// parameters
int test_time_;
int density_threshold_;
double exclude_distance_;

ros::Publisher pos_cmd_pub, cmd_vel_enu_pub, pose_cmd_pub;
ros::Publisher arch_point_pub, way_point_pub, local_position_pud, state_pub, cast_point_pub, visible_point_pub, average_distance_pub, featurenum_pub;
quadrotor_msgs::PositionCommand cmd;
geometry_msgs::Twist cmdvel;
geometry_msgs::Pose pose_cmd;
// double pos_gain[3] = {0, 0, 0};
// double vel_gain[3] = {0, 0, 0};

#define FLIP_YAW_AT_END 0
#define TURN_YAW_TO_CENTER_AT_END 0

Eigen::Vector3d originpoint(0, 0, 0);
bool receive_traj_ = false;
bool have_anchor = false;
bool have_new_anchor = false;
bool record_start = false;
bool first_roll = true;
boost::shared_ptr<poly_traj::Trajectory> traj_;
double traj_duration_;
double last_possible;
ros::Time start_time_;
int traj_id_;
int exit_prediction, switch_time, switch_duration;
ros::Time heartbeat_time_(0);
Eigen::Vector3d last_pos_;
Eigen::Vector3d last_anchor_point, new_anchor_point;
Eigen::Vector3d current_pos;
Eigen::Vector3d last_pos;
std::vector<Eigen::Vector3d> feature_clouds; // 最新特征云
std::vector<Eigen::Vector3d> last_map;       // 最新地图
std::vector<Eigen::Vector3d> cast_clouds;    // 特征点投影
std::vector<Eigen::Vector3d> merge_clouds;   // 特征点合并
std::vector<Eigen::Vector3d> all_points;
std::vector<int> cast_sliding_num, all_sliding_num;
std::vector<int> feature_num_record;
int cast_sliding_count, all_sliding_count;
sensor_msgs::PointCloud way_points, local_position;
// yaw control
double last_yaw_, last_yawdot_, slowly_flip_yaw_target_, slowly_turn_to_center_target_;
double time_forward_;
// visualization
sensor_msgs::PointCloud anchor_point_cloud;

void mapCallback(const sensor_msgs::PointCloud2ConstPtr &img);
void featureCallback(const sensor_msgs::PointCloudConstPtr &img);

struct Cluster
{
  Eigen::Vector3d center;
  int density;
};

cv::Mat calculatePointMat(const std::vector<Eigen::Vector3d> &pointCloud)
{
  cv::Mat points(pointCloud.size(), 3, CV_32F);
  for (size_t i = 0; i < pointCloud.size(); ++i)
  {
    points.at<float>(i, 0) = pointCloud[i].x();
    points.at<float>(i, 1) = pointCloud[i].y();
    points.at<float>(i, 2) = pointCloud[i].z();
  }
  return points;
}

bool compareClusters(const Cluster &a, const Cluster &b)
{
  return a.density > b.density; // 按照密度从高到低排序
}

std::vector<Eigen::Vector3d> findDenseClusters(const std::vector<Eigen::Vector3d> &pointCloud, int maxClusters)
{
  std::vector<Eigen::Vector3d> results;
  std::vector<Eigen::Vector3d> datas = pointCloud;
  if (pointCloud.size() == 0)
  {
    return results;
  }
  if (pointCloud.size() < 3)
  {
    for (const auto &point : pointCloud)
    {
      datas.push_back(point);
      datas.push_back(point);
    }
  }

  cv::Mat points = calculatePointMat(datas);
  std::vector<Cluster> clusters;
  std::vector<double> distortions; // 存储每个聚类数量对应的误差
  int optimalClusterCount = 1;
  double maxDistortionChange = 0.0;
  cv::Mat labels;
  std::vector<cv::Point3f> centers;
  int Clusterslimit = points.rows > maxClusters ? maxClusters : points.rows;

  for (int clusterCount = 1; clusterCount <= maxClusters; ++clusterCount)
  {
    cv::kmeans(points, clusterCount, labels, cv::TermCriteria(), 3, cv::KMEANS_RANDOM_CENTERS, centers);

    // 计算误差（聚类内的点到中心的距离的平方和）
    double distortion = 0.0;
    for (int i = 0; i < points.rows; ++i)
    {
      int clusterIndex = labels.at<int>(i);
      Eigen::Vector3d point_tmp;
      point_tmp << points.at<float>(i, 0) - centers[clusterIndex].x, points.at<float>(i, 1) - centers[clusterIndex].y, points.at<float>(i, 2) - centers[clusterIndex].z;
      double distance = point_tmp.norm();
      distortion += distance;
    }

    distortions.push_back(distortion);

    if (clusterCount > 1)
    {
      double distortionChange = distortions[clusterCount - 2] - distortions[clusterCount - 1];
      if (distortionChange > maxDistortionChange)
      {
        maxDistortionChange = distortionChange;
        optimalClusterCount = clusterCount;
      }
      else
      {
        optimalClusterCount = clusterCount - 1;
        break;
      }
    }
  }
  cv::kmeans(points, optimalClusterCount, labels, cv::TermCriteria(), 3, cv::KMEANS_RANDOM_CENTERS, centers);
  ROS_INFO("clustered %d points in %d classes!", points.rows, optimalClusterCount);
  std::vector<int> densitys(optimalClusterCount);
  for (int i = 0; i < points.rows; ++i)
  {
    int clusterIndex = labels.at<int>(i);
    densitys[clusterIndex]++;
  }

  // 计算每个簇的平均中心和密度
  for (int i = 0; i < optimalClusterCount; ++i)
  {
    if (densitys[i] > density_threshold_)
    {
      Cluster cluster;
      Eigen::Vector3d point_tmp;
      point_tmp << centers[i].x, centers[i].y, centers[i].z;
      cluster.center = point_tmp;
      cluster.density = densitys[i]; // 在这里使用聚类数量作为密度，也可以使用其他方式计算密度
      clusters.push_back(cluster);
    }
  }

  // 按照密度排序
  std::sort(clusters.begin(), clusters.end(), compareClusters);
  for (const auto &cluster : clusters)
  {
    ROS_INFO("clustering density: %d !", cluster.density);
    results.push_back(cluster.center);
  }

  return results;
}

void heartbeatCallback(std_msgs::EmptyPtr msg)
{
  heartbeat_time_ = ros::Time::now();
}

double calculate_dir(const Eigen::Vector3d &pos1, const Eigen::Vector3d &pos2)
{
  Eigen::Vector3d dir = pos1 - pos2;
  double direction = atan2(dir(1), dir(0));
  return direction;
}

double calculate_heading(double t_cur, Eigen::Vector3d &pos) // 计算当前位置到未来某点的方向向量
{
  std::pair<double, double> yaw_yawdot(0, 0);

  Eigen::Vector3d dir = t_cur + time_forward_ <= traj_duration_
                            ? traj_->getPos(t_cur + time_forward_) - pos
                            : traj_->getPos(traj_duration_) - pos;
  return atan2(dir(1), dir(0));
}

std::pair<double, double> calculate_yaw(double arch, double dt)
{
  constexpr double YAW_DOT_MAX_PER_SEC = 3 * M_PI;
  constexpr double YAW_DOT_DOT_MAX_PER_SEC = 5 * M_PI;
  std::pair<double, double> yaw_yawdot(0, 0);

  double yaw_temp = arch;

  double yawdot = 0;
  double d_yaw = yaw_temp - last_yaw_;
  if (d_yaw >= M_PI)
  {
    d_yaw -= 2 * M_PI;
  }
  if (d_yaw <= -M_PI)
  {
    d_yaw += 2 * M_PI;
  }

  const double YDM = d_yaw >= 0 ? YAW_DOT_MAX_PER_SEC : -YAW_DOT_MAX_PER_SEC;
  const double YDDM = d_yaw >= 0 ? YAW_DOT_DOT_MAX_PER_SEC : -YAW_DOT_DOT_MAX_PER_SEC;
  double d_yaw_max;
  if (fabs(last_yawdot_ + dt * YDDM) <= fabs(YDM))
  {
    // yawdot = last_yawdot_ + dt * YDDM;
    d_yaw_max = last_yawdot_ * dt + 0.5 * YDDM * dt * dt;
  }
  else
  {
    // yawdot = YDM;
    double t1 = (YDM - last_yawdot_) / YDDM;
    d_yaw_max = ((dt - t1) + dt) * (YDM - last_yawdot_) / 2.0;
  }

  if (fabs(d_yaw) > fabs(d_yaw_max))
  {
    d_yaw = d_yaw_max;
  }
  yawdot = d_yaw / dt;

  double yaw = last_yaw_ + d_yaw;
  if (yaw > M_PI)
    yaw -= 2 * M_PI;
  if (yaw < -M_PI)
    yaw += 2 * M_PI;
  yaw_yawdot.first = yaw;
  yaw_yawdot.second = yawdot;

  last_yaw_ = yaw_yawdot.first;
  last_yawdot_ = yaw_yawdot.second;

  return yaw_yawdot;
}

void publish_cmd(Vector3d p, Vector3d v, Vector3d a, Vector3d j, double y, double yd)
{
  cmd.header.stamp = ros::Time::now();
  cmd.header.frame_id = "world";
  cmd.trajectory_flag = quadrotor_msgs::PositionCommand::TRAJECTORY_STATUS_READY;
  cmd.trajectory_id = traj_id_;

  cmd.position.x = p(0);
  cmd.position.y = p(1);
  cmd.position.z = p(2);
  cmd.velocity.x = v(0);
  cmd.velocity.y = v(1);
  cmd.velocity.z = v(2);
  cmd.acceleration.x = a(0);
  cmd.acceleration.y = a(1);
  cmd.acceleration.z = a(2);
  cmd.jerk.x = j(0);
  cmd.jerk.y = j(1);
  cmd.jerk.z = j(2);
  cmd.yaw = y;
  cmd.yaw_dot = yd;
  pos_cmd_pub.publish(cmd);
  cmdvel.linear.x = v(0);
  cmdvel.linear.y = v(1);
  cmdvel.linear.z = v(2);
  cmdvel.angular.z = yd;
  cmd_vel_enu_pub.publish(cmdvel);

  last_pos_ = p;
  pose_cmd.position.x = p(0);
  pose_cmd.position.y = p(1);
  pose_cmd.position.z = p(2);

  pose_cmd.orientation.x = 0.0;
  pose_cmd.orientation.y = 0.0;
  pose_cmd.orientation.z = sin(y / 2);
  pose_cmd.orientation.w = cos(y / 2);
  pose_cmd_pub.publish(pose_cmd);
}

double calculateAverageDistance(const std::vector<Eigen::Vector3d> &points, bool iflocal = true)
{
  double totalDistance = 0.0;
  Eigen::Vector3d pt;

  for (const auto &point : points)
  {
    if (iflocal)
    {
      pt << point.x(), point.y(), 0;
    }
    else
    {
      pt << point.x() - current_pos.x(), point.y() - current_pos.y(), 0;
    }
    double distance = pt.norm();

    totalDistance += distance;
  }

  if (!points.empty())
  {
    // 计算平均值
    return totalDistance / points.size();
  }
  else
  {
    std::cerr << "Error: Empty vector, cannot calculate average distance." << std::endl;
    return 0.0;
  }
}

bool compareNorm(const Eigen::Vector3d &v1, const Eigen::Vector3d &v2)
{
  double dis1 = v1.x() * v1.x() + v1.y() * v1.y();
  double dis2 = v2.x() * v2.x() + v2.y() * v2.y();
  return dis1 > dis2;
}

Eigen::Vector3d isPathClear(const Eigen::Vector3d &target, const Eigen::Vector3d &position, int mode,
                            const std::vector<Eigen::Vector3d> &obstacles, bool &success, double maxDistance = 8.5)
// mode 1 = checking occlusion
// mode 2 = casting points
{
  // 计算光线方向和长度
  Eigen::Vector3d result = target;
  Eigen::Vector3d direction = target - position;
  double distance = direction.norm();

  // 确保光线长度不为零，避免除以零错误
  if (distance == 0.0)
  {
    success = true;
    return target;
  }
  // 归一化光线方向
  direction << direction.x() / distance, direction.y() / distance, direction.z() / distance;
  distance = distance > maxDistance ? maxDistance : distance;
  if (mode == 1)
  {
    distance -= 0.5;
  }

  result << direction.x() * distance, direction.y() * distance, direction.z() * distance; // too far point is not reliable
  result = position + result;

  // 步长
  double step = 0.10; // 调整步长以提高准确性

  if (obstacles.size() != 0)
  {
    // 沿光线进行遍历
    for (double t = 0.3; t <= distance; t += step)
    {
      // 计算当前点的坐标
      double currentX = position.x() + t * direction.x();
      double currentY = position.y() + t * direction.y();
      double currentZ = position.z() + t * direction.z();
      Eigen::Vector3d current(currentX, currentY, currentZ);

      for (const auto &obstacle : obstacles) // 检查当前点是否在障碍物中
      {
        double distanceToObstacle = (current - obstacle).norm();

        double obstacleRadius = 0.26;            // 假设障碍物分辨率为0.15(0.15*1.73)
        if (distanceToObstacle < obstacleRadius) // 如果当前点在障碍物范围内，则不可见
        {
          success = false;
          return obstacle;
        }
      }
    }
  }
  success = true; // 没有遇到障碍物，路径可见
  return result;
}

void check_exit(int &exit_pre, bool &ifsuccess, int &exit_flag, Eigen::Vector3d check_pt, int mode,
                const double t_current, const std::vector<Eigen::Vector3d> &map,
                const std::vector<Eigen::Vector3d> &local_cloud_origin, const int test_range,
                double dt = 0.01)
// mode 1 == no distance check
// mode 2 == no angle
// mode 3 == no check
// mode 4 == full check
{
  ifsuccess = true;
  exit_pre = 0; // 轨迹上通过检验的点数
  for (size_t i = 0; i < test_range; i++)
  {
    int t = t_current + i * dt;
    t = t > traj_duration_ ? traj_duration_ : t; // time limit
    Eigen::Vector3d pos(Eigen::Vector3d::Zero());
    pos = traj_->getPos(t);
    double last_direction = calculate_dir(check_pt, pos); // 弧度
    double heading_direction = calculate_heading(t, pos);
    double angle_limit = M_PI / 4;

    // judgement
    if ((mode == 1) || (mode == 3))
    {
      ifsuccess = true;
    }
    else
    {
      Eigen::Vector3d local_pt;
      std::vector<Eigen::Vector3d> local_clouds;
      for (const auto &point : local_cloud_origin)
      {
        local_pt = point + current_pos - pos;
        local_clouds.push_back(local_pt); // local
      }
      double averageDistance;
      if (local_clouds.size() > 0)
      {
        averageDistance = calculateAverageDistance(local_clouds) * 0.5;
      }
      else
      {
        averageDistance = 0;
      }
      Eigen::Vector3d check_dis_tmp = check_pt - pos;
      double last_arch_dis = sqrt(check_dis_tmp.x() * check_dis_tmp.x() + check_dis_tmp.y() * check_dis_tmp.y());
      if ((last_arch_dis < averageDistance))
      {
        exit_flag = 2;
        ifsuccess = false;
        break;
      }
    }

    if ((mode == 2) || (mode == 3))
    {
      ifsuccess = true;
    }
    else
    {
      if (std::abs(heading_direction - last_direction) > angle_limit)
      {
        exit_flag = 3;
        ifsuccess = false;
        break;
      }
    }

    if ((mode == 5) || (mode == 3))
    {
      ifsuccess = true;
    }
    else
    {
      bool PathClear;
      Eigen::Vector3d check_path_tmp = isPathClear(check_pt, pos, 1, map, PathClear);
      if (!PathClear)
      {
        exit_flag = 1;
        ifsuccess = false;
        break;
      }
    }

    exit_pre++;
  }
}

// 锚点选取函数
double archreplan(const std::vector<Eigen::Vector3d> &map, const double t_cur, const double heading, double dt = 0.01)
{
  double result = heading;
  std::vector<Eigen::Vector3d> local_clouds; // 可视特征
  std::vector<Eigen::Vector3d> all_locals;
  // visualization
  // sensor_msgs::PointCloud visible_point_cloud;
  for (const auto &point : merge_clouds)
  {
    int check_time;
    bool check_success;
    int check_reason;
    check_exit(check_time, check_success, check_reason, point, 1, t_cur, map, local_clouds, 1); // 检查是否可视
    if (check_success)
    {
      geometry_msgs::Point32 p;
      p.x = point.x();
      p.y = point.y();
      p.z = point.z();
      // visible_point_cloud.points.push_back(p);
      Eigen::Vector3d local_pt = point - current_pos;
      local_clouds.push_back(local_pt); // local
    }
  }
  if (all_points.size() > 10)
  {
    for (const auto &point : all_points)
    {
      int check_time;
      bool check_success;
      int check_reason;
      check_exit(check_time, check_success, check_reason, point, 1, t_cur, map, local_clouds, 1); // 检查是否可视
      if (check_success)
      {
        Eigen::Vector3d local_pt = point - current_pos;
        all_locals.push_back(local_pt); // local}
      }
    }
  }
  // visible_point_cloud.header.stamp = ros::Time::now();
  // visible_point_cloud.header.frame_id = "world";
  // visible_point_pub.publish(visible_point_cloud);

  // if exit needed or not
  if (have_anchor)
  {
    int exit_reason;
    check_exit(exit_prediction, have_anchor, exit_reason, last_anchor_point, 4, t_cur, map, local_clouds, test_time_ / 2);
    if (!have_anchor)
    {
      switch (exit_reason)
      {
      case 1:
        ROS_WARN("<clearance exit>");
        break;
      case 2:
        ROS_WARN("<distance exit>");
        break;
      case 3:
        ROS_WARN("<angle exit>");
        break;
      default:
        break;
      }
    }
  }

  if (!have_anchor) // enter switching
  {
    ROS_WARN("switching");
    int new_exit_time;
    int exit_reason_new = 0;
    if (!have_new_anchor) // find new anchor
    {
      std::vector<Eigen::Vector3d> candidate_clouds;
      if (local_clouds.size() > density_threshold_)
      {
        ROS_WARN("LOCAL_CLOUDS_SIZE:%ld", local_clouds.size());
        // std::sort(local_clouds.begin(), local_clouds.end(), compareNorm);
        // 找到候选点 candidate_clouds
        // 最大聚类数量
        int maxClusters = 5;
        candidate_clouds = findDenseClusters(local_clouds, maxClusters);

        int new_exit_time_max = 0;
        Eigen::Vector3d new_anchor_tmp;
        Eigen::Vector3d pt_global;
        for (size_t i = 0; i < candidate_clouds.size(); ++i)
        {
          pt_global = candidate_clouds[i] + current_pos;
          check_exit(new_exit_time, have_new_anchor, exit_reason_new, pt_global, 4, t_cur, map, local_clouds, test_time_);
          if (new_exit_time > new_exit_time_max)
          {
            new_exit_time_max = new_exit_time;
            new_anchor_tmp = pt_global;
          }
          if (have_new_anchor)
          {
            switch_time = 0;
            new_anchor_point = pt_global;
            switch_duration = test_time_;
            ROS_WARN("switch_duration = test_time_ : %d", switch_duration);
            break;
          }
        }
        if (!have_new_anchor)
        {
          if (new_exit_time_max > 0)
          {
            switch_time = 0;
            new_anchor_point = new_anchor_tmp;
            have_new_anchor = true;
            switch_duration = int(new_exit_time_max / 2);
            ROS_WARN("switch_duration = new_exit_time_max : %d", switch_duration);
          }
        }
      }

      if ((all_locals.size() > density_threshold_) && (!have_new_anchor))
      {
        ROS_WARN("ALL_LOCAL_SIZE:%ld", all_locals.size());
        // std::sort(all_locals.begin(), all_locals.end(), compareNorm);
        // 找到候选点 candidate_clouds
        // 最大聚类数量
        int maxClusters = 5;
        candidate_clouds = findDenseClusters(all_locals, maxClusters);

        int new_exit_time_max = 0;
        Eigen::Vector3d new_anchor_tmp;
        Eigen::Vector3d pt_global;
        for (size_t i = 0; i < candidate_clouds.size(); ++i)
        {
          pt_global = candidate_clouds[i] + current_pos;
          check_exit(new_exit_time, have_new_anchor, exit_reason_new, pt_global, 1, t_cur, map, all_locals, test_time_);
          if (new_exit_time > new_exit_time_max)
          {
            new_exit_time_max = new_exit_time;
            new_anchor_tmp = pt_global;
          }
          if (have_new_anchor)
          {
            switch_time = 0;
            new_anchor_point = pt_global;
            switch_duration = test_time_;
            ROS_WARN("switch_duration = test_time_ : %d", switch_duration);
            // visualization
            geometry_msgs::Point32 p;
            p.x = new_anchor_point.x();
            p.y = new_anchor_point.y();
            p.z = new_anchor_point.z();
            anchor_point_cloud.points.push_back(p);
            arch_point_pub.publish(anchor_point_cloud);
            break;
          }
        }
        if (!have_new_anchor)
        {
          if (new_exit_time_max > 0)
          {
            switch_time = 0;
            new_anchor_point = new_anchor_tmp;
            have_new_anchor = true;
            switch_duration = int(new_exit_time_max / 2);
            ROS_WARN("switch_duration = new_exit_time_max : %d", switch_duration);
            // visualization
            geometry_msgs::Point32 p;
            p.x = new_anchor_point.x();
            p.y = new_anchor_point.y();
            p.z = new_anchor_point.z();
            anchor_point_cloud.points.push_back(p);
            arch_point_pub.publish(anchor_point_cloud);
          }
        }
      }

      if (!have_new_anchor)
      {
        switch (exit_reason_new)
        {
        case 1:
          ROS_WARN("FAIL REASON: clearance exit");
          break;
        case 2:
          ROS_WARN("FAIL REASON: distance exit");
          break;
        case 3:
          ROS_WARN("FAIL REASON: angle exit");
          break;
        default:
          ROS_WARN("FAIL REASON: no available point");
          break;
        }
      }
    }

    if (have_new_anchor) // perform switching
    {
      double new_anchor_point_yaw;
      new_anchor_point_yaw = calculate_dir(new_anchor_point, current_pos);
      double last_anchor_point_yaw;
      if (first_roll) // first perform
      {
        ROS_WARN("<first_roll>");
        last_anchor_point_yaw = heading;
        switch_duration = 2;
      }
      else
      {
        last_anchor_point_yaw = calculate_dir(last_anchor_point, current_pos);
      }
      switch_time++;
      if (switch_time < switch_duration)
      {
        if ((last_anchor_point_yaw - new_anchor_point_yaw) > M_PI)
        {
          new_anchor_point_yaw += 2 * M_PI;
        }
        if ((new_anchor_point_yaw - last_anchor_point_yaw) > M_PI)
        {
          new_anchor_point_yaw -= 2 * M_PI;
        }
        double radio = double(switch_time) / double(switch_duration); // 比例
        result = new_anchor_point_yaw * radio + last_anchor_point_yaw * (1.0 - radio);
      }
      else // switch done!
      {
        if (first_roll)
        {
          first_roll = false;
        }
        last_anchor_point = new_anchor_point;
        result = calculate_dir(last_anchor_point, current_pos);
        have_new_anchor = false;
        have_anchor = true; // exit switching
      }
    }
    else
    {
      // return result; // no anchor found
      if (exit_prediction > 0)
      {
        have_anchor = true;
        result = calculate_dir(last_anchor_point, current_pos);
      }
      else
      {
        return result; // no anchor found
      }
    }
  }
  else // not switching (normal condition)
  {
    result = calculate_dir(last_anchor_point, current_pos);
  }

  if (result > M_PI)
    result -= 2 * M_PI;
  if (result < -M_PI)
    result += 2 * M_PI;
  return result;
}

void cmdCallback(const ros::TimerEvent &e)
{
  static ros::Time time_last = ros::Time::now();
  /* no publishing before receive traj_ and have heartbeat */
  if (heartbeat_time_.toSec() <= 1e-5)
  {
    // ROS_ERROR_ONCE("[traj_server] No heartbeat from the planner received");
    return;
  }
  if (!receive_traj_)
    return;

  ros::Time time_now = ros::Time::now();

  if ((time_now - heartbeat_time_).toSec() > 0.5)
  {
    ROS_ERROR("[traj_server] Lost heartbeat from the planner, is it dead?");

    receive_traj_ = false;
    publish_cmd(last_pos_, Vector3d::Zero(), Vector3d::Zero(), Vector3d::Zero(), last_yaw_, 0);
  }

  double t_cur = (time_now - start_time_).toSec();

  Eigen::Vector3d pos(Eigen::Vector3d::Zero()), vel(Eigen::Vector3d::Zero()), acc(Eigen::Vector3d::Zero()), jer(Eigen::Vector3d::Zero());
  std::pair<double, double> yaw_yawdot(0, 0);

#if FLIP_YAW_AT_END or TURN_YAW_TO_CENTER_AT_END
  static bool finished = false;
#endif
  // start from here
  if (t_cur < traj_duration_ && t_cur >= 0.0)
  {
    pos = traj_->getPos(t_cur);
    vel = traj_->getVel(t_cur);
    acc = traj_->getAcc(t_cur);
    jer = traj_->getJer(t_cur);

    way_points.header.stamp = ros::Time::now();
    geometry_msgs::Point32 p;
    p.x = pos.x();
    p.y = pos.y();
    p.z = pos.z();
    way_points.points.push_back(p);
    way_point_pub.publish(way_points);
    if (!record_start)
    {
      record_start = true;
    }

    /*** calculate yaw ***/
    double heading = calculate_heading(t_cur, pos);
    double arch = heading;
    if (anchor_enable_)
    {
      arch = archreplan(last_map, t_cur, heading);
    }
    std_msgs::Float64 state;
    if (have_anchor || have_new_anchor)
    {
      state.data = 1;
      ROS_INFO("arch_direction: %f.", arch);
      if ((arch - heading) > M_PI)
      {
        heading += 2 * M_PI;
      }
      if ((heading - arch) > M_PI)
      {
        heading -= 2 * M_PI;
      }
      // arch = (arch + heading) / 2.0;
      // ROS_INFO("arch_point: %f,%f,%f.", new_anchor_point.x(), new_anchor_point.y(), new_anchor_point.z());
      yaw_yawdot = calculate_yaw(arch, (time_now - time_last).toSec());
    }
    else
    {
      first_roll = true;
      state.data = 0;
      ROS_WARN("NO ANCHOR!");
      yaw_yawdot = calculate_yaw(heading, (time_now - time_last).toSec());
    }
    state_pub.publish(state);

    /*** calculate yaw ***/

    time_last = time_now;
    last_yaw_ = yaw_yawdot.first;
    last_pos_ = pos;

    slowly_flip_yaw_target_ = yaw_yawdot.first + M_PI;
    if (slowly_flip_yaw_target_ > M_PI)
      slowly_flip_yaw_target_ -= 2 * M_PI;
    if (slowly_flip_yaw_target_ < -M_PI)
      slowly_flip_yaw_target_ += 2 * M_PI;
    constexpr double CENTER[2] = {0.0, 0.0};
    slowly_turn_to_center_target_ = atan2(CENTER[1] - pos(1), CENTER[0] - pos(0));

    // publish
    publish_cmd(pos, vel, acc, jer, yaw_yawdot.first, yaw_yawdot.second);
#if FLIP_YAW_AT_END or TURN_YAW_TO_CENTER_AT_END
    finished = false;
#endif
  }

#if FLIP_YAW_AT_END
  else if (t_cur >= traj_duration_)
  {
    if (finished)
      return;

    /* hover when finished traj_ */
    pos = traj_->getPos(traj_duration_);
    vel.setZero();
    acc.setZero();
    jer.setZero();

    if (slowly_flip_yaw_target_ > 0)
    {
      last_yaw_ += (time_now - time_last).toSec() * M_PI / 2;
      yaw_yawdot.second = M_PI / 2;
      if (last_yaw_ >= slowly_flip_yaw_target_)
      {
        finished = true;
      }
    }
    else
    {
      last_yaw_ -= (time_now - time_last).toSec() * M_PI / 2;
      yaw_yawdot.second = -M_PI / 2;
      if (last_yaw_ <= slowly_flip_yaw_target_)
      {
        finished = true;
      }
    }

    yaw_yawdot.first = last_yaw_;
    time_last = time_now;

    publish_cmd(pos, vel, acc, jer, yaw_yawdot.first, yaw_yawdot.second);
  }
#endif

#if TURN_YAW_TO_CENTER_AT_END
  else if (t_cur >= traj_duration_)
  {
    if (finished)
      return;

    /* hover when finished traj_ */
    pos = traj_->getPos(traj_duration_);
    vel.setZero();
    acc.setZero();
    jer.setZero();

    double d_yaw = last_yaw_ - slowly_turn_to_center_target_;
    if (d_yaw >= M_PI)
    {
      last_yaw_ += (time_now - time_last).toSec() * M_PI / 2;
      yaw_yawdot.second = M_PI / 2;
      if (last_yaw_ > M_PI)
        last_yaw_ -= 2 * M_PI;
    }
    else if (d_yaw <= -M_PI)
    {
      last_yaw_ -= (time_now - time_last).toSec() * M_PI / 2;
      yaw_yawdot.second = -M_PI / 2;
      if (last_yaw_ < -M_PI)
        last_yaw_ += 2 * M_PI;
    }
    else if (d_yaw >= 0)
    {
      last_yaw_ -= (time_now - time_last).toSec() * M_PI / 2;
      yaw_yawdot.second = -M_PI / 2;
      if (last_yaw_ <= slowly_turn_to_center_target_)
        finished = true;
    }
    else
    {
      last_yaw_ += (time_now - time_last).toSec() * M_PI / 2;
      yaw_yawdot.second = M_PI / 2;
      if (last_yaw_ >= slowly_turn_to_center_target_)
        finished = true;
    }

    yaw_yawdot.first = last_yaw_;
    time_last = time_now;

    publish_cmd(pos, vel, acc, jer, yaw_yawdot.first, yaw_yawdot.second);
  }
#endif
}

void polyTrajCallback(traj_utils::PolyTrajPtr msg)
{
  if (msg->order != 5)
  {
    ROS_ERROR("[traj_server] Only support trajectory order equals 5 now!");
    return;
  }
  if (msg->duration.size() * (msg->order + 1) != msg->coef_x.size())
  {
    ROS_ERROR("[traj_server] WRONG trajectory parameters, ");
    return;
  }

  int piece_nums = msg->duration.size();
  std::vector<double> dura(piece_nums);
  std::vector<poly_traj::CoefficientMat> cMats(piece_nums);
  for (int i = 0; i < piece_nums; ++i)
  {
    int i6 = i * 6;
    cMats[i].row(0) << msg->coef_x[i6 + 0], msg->coef_x[i6 + 1], msg->coef_x[i6 + 2],
        msg->coef_x[i6 + 3], msg->coef_x[i6 + 4], msg->coef_x[i6 + 5];
    cMats[i].row(1) << msg->coef_y[i6 + 0], msg->coef_y[i6 + 1], msg->coef_y[i6 + 2],
        msg->coef_y[i6 + 3], msg->coef_y[i6 + 4], msg->coef_y[i6 + 5];
    cMats[i].row(2) << msg->coef_z[i6 + 0], msg->coef_z[i6 + 1], msg->coef_z[i6 + 2],
        msg->coef_z[i6 + 3], msg->coef_z[i6 + 4], msg->coef_z[i6 + 5];

    dura[i] = msg->duration[i];
  }

  traj_.reset(new poly_traj::Trajectory(dura, cMats));

  start_time_ = msg->start_time;
  traj_duration_ = traj_->getTotalDuration();
  traj_id_ = msg->traj_id;

  receive_traj_ = true;
}

void odomCallback(const nav_msgs::OdometryConstPtr &msg)
{
  last_pos = current_pos;
  local_position.header.stamp = ros::Time::now();
  geometry_msgs::Point32 pos;
  pos.x = msg->pose.pose.position.x;
  pos.y = msg->pose.pose.position.y;
  pos.z = msg->pose.pose.position.z;
  current_pos << pos.x, pos.y, pos.z;
  local_position.points.push_back(pos);
  local_position_pud.publish(local_position);
}

void mapCallback(const sensor_msgs::PointCloud2ConstPtr &img)
{
  pcl::PointCloud<pcl::PointXYZ> map_tmp;
  Eigen::Vector3d pt;
  pcl::fromROSMsg(*img, map_tmp);
  last_map.clear();
  last_map.shrink_to_fit();
  for (size_t i = 0; i < map_tmp.points.size(); ++i)
  {
    pt << map_tmp.points[i].x, map_tmp.points[i].y, map_tmp.points[i].z;
    last_map.push_back(pt); // global
  }
  // visualization
  if (last_map.size() > 0)
  {
    std_msgs::Float64 average_distance;
    average_distance.data = calculateAverageDistance(last_map, false);
    average_distance_pub.publish(average_distance);
  }
}

double compute_average(const std::vector<int> &v)
{
  if (v.empty())
  {
    return 0.0; // 如果向量为空，返回0
  }
  double sum = std::accumulate(v.begin(), v.end(), 0.0); // 计算向量的和
  double average = sum / v.size();                       // 计算平均值
  return average;
}

void featureCallback(const sensor_msgs::PointCloudConstPtr &img)
{
  sensor_msgs::PointCloud2 cloud2_msg;
  pcl::PointCloud<pcl::PointXYZ> feature_latest_cloud;
  Eigen::Vector3d pt;
  sensor_msgs::convertPointCloudToPointCloud2(*img, cloud2_msg);
  pcl::fromROSMsg(cloud2_msg, feature_latest_cloud);
  if (record_start)
  {
    std_msgs::Float64 featurenum;
    featurenum.data = feature_latest_cloud.points.size();
    featurenum_pub.publish(featurenum);
    feature_num_record.push_back(feature_latest_cloud.points.size());
    double num_average = compute_average(feature_num_record);
    ROS_INFO("num_average:%f", num_average);
  }
  feature_clouds.clear();
  feature_clouds.shrink_to_fit();
  for (size_t i = 0; i < feature_latest_cloud.points.size(); ++i)
  {
    pt << feature_latest_cloud.points[i].x, feature_latest_cloud.points[i].y, feature_latest_cloud.points[i].z;
    double distance_check = (pt - current_pos).norm();
    if (distance_check < exclude_distance_)
    {
      feature_clouds.push_back(pt); // global
    }
  }

  int cast_num = 0;
  for (const auto &feature_point : feature_clouds)
  {
    bool success_flag = true;
    pt = isPathClear(feature_point, current_pos, 2, last_map, success_flag);
    all_points.push_back(pt);
    if (!success_flag)
    {
      cast_clouds.push_back(pt); // global
      cast_num++;
    }
  }
  // 滑动窗口存储
  cast_sliding_num.push_back(cast_num);
  all_sliding_num.push_back(feature_clouds.size());
  cast_sliding_count++;
  all_sliding_count++;
  if (cast_sliding_count > 50)
  {
    if (cast_sliding_num[0] > 0)
    {
      cast_clouds.erase(cast_clouds.begin(), cast_clouds.begin() + cast_sliding_num[0]);
    }
    cast_sliding_num.erase(cast_sliding_num.begin(), cast_sliding_num.begin() + 1);
  }
  if (all_sliding_count > 1)
  {
    if (all_sliding_num[0] > 0)
    {
      all_points.erase(all_points.begin(), all_points.begin() + all_sliding_num[0]);
    }
    all_sliding_num.erase(all_sliding_num.begin(), all_sliding_num.begin() + 1);
  }

  // 基于体素的融合
  merge_clouds = cast_clouds;
  std::sort(merge_clouds.begin(), merge_clouds.end(), [](const Eigen::Vector3d &a, const Eigen::Vector3d &b)
            {
        // 定义Eigen::Vector3d的比较方式
        if (a[0] != b[0]) return a[0] < b[0];
        if (a[1] != b[1]) return a[1] < b[1];
        return a[2] < b[2]; });
  // 使用std::unique移除重复元素
  merge_clouds.erase(std::unique(merge_clouds.begin(), merge_clouds.end(), [](const Eigen::Vector3d &a, const Eigen::Vector3d &b)
                                 {
                       // 定义Eigen::Vector3d的相等方式
                       return a[0] == b[0] && a[1] == b[1] && a[2] == b[2]; }),
                     merge_clouds.end());
  // 可视化
  ROS_WARN("CAST_SIZE:%ld,MERGE_SIZE:%ld", cast_clouds.size(), merge_clouds.size());
  sensor_msgs::PointCloud cast_point_cloud;
  geometry_msgs::Point32 p;
  for (const auto &cast_point : cast_clouds)
  {
    p.x = cast_point.x();
    p.y = cast_point.y();
    p.z = cast_point.z();
    cast_point_cloud.points.push_back(p);
  }
  cast_point_cloud.header.stamp = ros::Time::now();
  cast_point_cloud.header.frame_id = "world";
  cast_point_pub.publish(cast_point_cloud);
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "traj_server");
  // ros::NodeHandle node;
  ros::NodeHandle nh("~");

  ros::Subscriber poly_traj_sub = nh.subscribe("planning/trajectory", 10, polyTrajCallback);
  ros::Subscriber heartbeat_sub = nh.subscribe("heartbeat", 10, heartbeatCallback);
  ros::Subscriber feature_cloud_sub_ = nh.subscribe("/vins_estimator/point_cloud", 10, featureCallback);
  ros::Subscriber map_sub_ = nh.subscribe("/grid_map/occupancy", 10, mapCallback);
  ros::Subscriber odom_sub_ = nh.subscribe("/iris_0/mavros/local_position/odom", 10, odomCallback);

  pos_cmd_pub = nh.advertise<quadrotor_msgs::PositionCommand>("/position_cmd", 50);
  cmd_vel_enu_pub = nh.advertise<geometry_msgs::Twist>("/cmd_vel_enu", 50);
  pose_cmd_pub = nh.advertise<geometry_msgs::Pose>("/pose_cmd", 50);
  arch_point_pub = nh.advertise<sensor_msgs::PointCloud>("arch_point", 1000);
  cast_point_pub = nh.advertise<sensor_msgs::PointCloud>("/cast_point", 1000);
  way_point_pub = nh.advertise<sensor_msgs::PointCloud>("way_point", 1000);
  local_position_pud = nh.advertise<sensor_msgs::PointCloud>("local_position", 1000);
  state_pub = nh.advertise<std_msgs::Float64>("/state", 10);
  average_distance_pub = nh.advertise<std_msgs::Float64>("/average_distance", 10); // average distance to obstacles
  featurenum_pub = nh.advertise<std_msgs::Float64>("/featurenum", 10);
  visible_point_pub = nh.advertise<sensor_msgs::PointCloud>("/visible_point", 1000);

  ros::Timer cmd_timer = nh.createTimer(ros::Duration(0.01), cmdCallback);

  nh.param("traj_server/time_forward", time_forward_, -1.0);
  nh.param("traj_server/anchor_enable", anchor_enable_, true);
  nh.param("traj_server/test_time", test_time_, 60);
  nh.param("traj_server/density_threshold", density_threshold_, 5);
  nh.param("traj_server/exclude_distance", exclude_distance_, 15.0);

  // visualization
  anchor_point_cloud.header.stamp = ros::Time::now();
  anchor_point_cloud.header.frame_id = "world";
  local_position.header.frame_id = "world";
  way_points.header.frame_id = "world";
  switch_time = 0;
  last_yaw_ = 0.0;
  last_yawdot_ = 0.0;
  cast_sliding_count = 0;
  all_sliding_count = 0;
  last_anchor_point << 1, 1, 1;
  new_anchor_point << 1, 1, 1;
  current_pos << 0, 0, 0;
  last_pos << 0, 0, 0;
  ros::Duration(1.0).sleep();

  ROS_INFO("[Traj server]: ready.");

  ros::spin();

  return 0;
}