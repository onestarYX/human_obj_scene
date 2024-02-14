import open3d as o3d
import numpy as np
import os
from matplotlib import pyplot as plt

# Create a point cloud
points = np.random.rand(1000, 3)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
mesh2 = o3d.geometry.TriangleMesh.create_sphere()
bunny = o3d.data.BunnyMesh()
mesh3 = o3d.io.read_triangle_mesh(bunny.path)
mesh3.compute_vertex_normals()
mesh3_pcd = mesh3.sample_points_uniformly(number_of_points=500)
R = mesh.get_rotation_matrix_from_xyz((np.pi / 50, 0, 0))

# # Create a visualizer
# vis = o3d.visualization.Visualizer()
# vis.create_window()
#
# # Add point cloud to visualizer
# vis.add_geometry(pcd)
#
# # Get the view control
# view_control = vis.get_view_control()
#
# # Custom loop to continuously update the camera view
# rot_angle = 0
# while True:
#     # Rotate camera by a small angle
#     view_control.rotate(0.1, rot_angle)
#
#     # Update the visualization
#     vis.update_geometry(pcd)
#     vis.poll_events()
#     vis.update_renderer()
#
#     # Increase rotation angle
#     rot_angle += 0.1
#
#     # Check for user input to break the loop
#     if vis.poll_events():
#         break
#
# # Keep the window open
# vis.run()
# vis.destroy_window()

def custom_draw_geometry_with_camera_trajectory(pcd):
    custom_draw_geometry_with_camera_trajectory.index = -1
    custom_draw_geometry_with_camera_trajectory.vis = o3d.visualization.Visualizer(
    )
    image_path = os.path.join('temp/test_out', 'image')
    if not os.path.exists(image_path):
        os.makedirs(image_path)

    def move_forward(vis):
        # This function is called within the o3d.visualization.Visualizer::run() loop
        # The run loop calls the function, then re-render
        # So the sequence in this function is to:
        # 1. Capture frame
        # 2. index++, check ending criteria
        # 3. Set camera
        # 4. (Re-render)
        ctr = vis.get_view_control()
        glb = custom_draw_geometry_with_camera_trajectory
        if glb.index >= 0:
            print("Capture image {:05d}".format(glb.index))
            image = vis.capture_screen_float_buffer(False)
            plt.imsave(os.path.join(image_path, '{:05d}.png'.format(glb.index)),
                       np.asarray(image),
                       dpi=1)
        glb.index = glb.index + 1
        if glb.index < 10:
            cam_params = ctr.convert_to_pinhole_camera_parameters()
            old_ext = np.copy(cam_params.extrinsic)
            old_ext[0, 3] = 10000
            cam_params.extrinsic = old_ext
            ctr.convert_from_pinhole_camera_parameters(cam_params)
            vis.poll_events()
            vis.update_renderer()
            print("here")
            # ctr.rotate(1000 * glb.index, 10)
        else:
            custom_draw_geometry_with_camera_trajectory.vis.\
                register_animation_callback(None)
        return False

    vis = custom_draw_geometry_with_camera_trajectory.vis
    vis.create_window()
    vis.add_geometry(pcd)
    vis.register_animation_callback(move_forward)
    vis.run()
    vis.destroy_window()



def custom_draw_geometry_with_rotation(pcd):
    def rotate_view(vis):
        ctr = vis.get_view_control()
        # ctr.rotate(10.0, 0.0)
        # ctr.set_lookat(np.random.rand(3))
        cam_params = ctr.convert_to_pinhole_camera_parameters()
        ctr.camera_local_translate(10, 0, 0)
        cam_params = ctr.convert_to_pinhole_camera_parameters()
        vis.poll_events()
        vis.update_renderer()

        # pcd.translate((.002, .002, .002))
        # # pcd.rotate(R, center=(0, 0, 0))
        # # mesh3.rotate(R, center=(1, 0, 0))
        # # mesh.rotate(R, center=(0, 0, 0))
        # # mesh3_pcd.translate((.002, .002, .002))
        # mesh3_pcd.rotate(R, center=(0, 0, 0))
        # # Update the geometry
        # vis.update_geometry(pcd)
        # vis.update_geometry(mesh3_pcd)
        # opt = vis.get_render_option()
        # opt.background_color = np.random.rand(3)
        return False

    o3d.visualization.draw_geometries_with_animation_callback([pcd, mesh, mesh3_pcd],
                                                              rotate_view)

if __name__ == '__main__':
    custom_draw_geometry_with_camera_trajectory(pcd)
    # custom_draw_geometry_with_rotation(pcd)
