import numpy as np
import open3d as o3d
from ampcl.io import load_pointcloud
from lidar_apollo_instance_segmentation_pyb import LidarApolloInstanceSegmentation


if __name__ == '__main__':
    pointcloud = load_pointcloud("../demo/rslidar16.pcd").copy()
    lidar_apollo_instance_segmentation = LidarApolloInstanceSegmentation()

    # 需要进行高度的补偿
    pointcloud[:, 2] = pointcloud[:, 2] - 1.73
    mask = lidar_apollo_instance_segmentation.segmentation(pointcloud)
    pointcloud[:, 2] = pointcloud[:, 2] + 1.73
    mask = np.array(mask)
    foreground_mask = mask > 0
    background_mask = mask < 0

    foreground_pc_o3d = o3d.geometry.PointCloud()
    foreground_pc_o3d.points = o3d.utility.Vector3dVector(pointcloud[foreground_mask][:, 0:3])
    foreground_pc_o3d.paint_uniform_color([0.5, 0.5, 0.5])

    background_pc_o3d = o3d.geometry.PointCloud()
    background_pc_o3d.points = o3d.utility.Vector3dVector(pointcloud[background_mask][:, 0:3])
    background_pc_o3d.paint_uniform_color([0, 1, 0])

    o3d.visualization.draw_geometries([foreground_pc_o3d, background_pc_o3d])
