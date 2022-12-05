import numpy as np
import open3d as o3d
from ampcl.io import load_pointcloud
from lidar_apollo_instance_segmentation_pyb import LidarApolloInstanceSegmentation
from sensor_msgs.msg import PointCloud2
from rclpy.node import Node
from ampcl.ros_utils import (pointcloud2_to_xyzi_array,
                             xyzirgb_numpy_to_pointcloud2)
from std_msgs.msg import Header
import rclpy
from ampcl.visualization import intensity_to_color_pcd


class InstanceSegmentation(Node):
    def __init__(self):
        super().__init__("apollo_instance_segmentation")
        lidar_topic_name = "/sensing/lidar/top/rslidar_points"
        self.pc_sub = self.create_subscription(PointCloud2, lidar_topic_name, self.pc_callback, 100)
        self.background_pc_pub = self.create_publisher(PointCloud2, "/background_pointcloud", 10)
        self.foreground_pc_pub = self.create_publisher(PointCloud2, "/foreground_pointcloud", 10)
        self.lidar_apollo_instance_segmentation = LidarApolloInstanceSegmentation()

        self.red = np.uint32((255 << 16) | (0 << 8) | (0 << 0)).view(np.float32)
        self.grey = np.uint32((155 << 16) | (155 << 8) | (155 << 0)).view(np.float32)

    def pc_callback(self, pc_msg):
        pointcloud_np = pointcloud2_to_xyzi_array(pc_msg).astype(np.float32)

        # 需要进行高度的补偿
        pointcloud_np[:, 2] = pointcloud_np[:, 2] - 1.73
        mask = self.lidar_apollo_instance_segmentation.segmentation(pointcloud_np)
        pointcloud_np[:, 2] = pointcloud_np[:, 2] + 1.73

        mask = np.array(mask)
        foreground_mask = mask > 0
        background_mask = mask < 0

        foreground_pc_np = pointcloud_np[foreground_mask]
        background_pc_np = pointcloud_np[background_mask]

        foreground_pc_np = np.hstack((foreground_pc_np, np.ones((foreground_pc_np.shape[0], 1)) * self.red))
        background_pc_np = np.hstack((background_pc_np, np.ones((background_pc_np.shape[0], 1)) * self.grey))

        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "lidar"
        foreground_pc_ros = xyzirgb_numpy_to_pointcloud2(foreground_pc_np, header)
        background_pc_ros = xyzirgb_numpy_to_pointcloud2(background_pc_np, header)
        self.foreground_pc_pub.publish(foreground_pc_ros)
        self.background_pc_pub.publish(background_pc_ros)


if __name__ == '__main__':
    rclpy.init()
    isinstance_segmentation = InstanceSegmentation()
    rclpy.spin(isinstance_segmentation)

    # Destroy the node explicitly
    isinstance_segmentation.destroy_node()
    rclpy.shutdown()
