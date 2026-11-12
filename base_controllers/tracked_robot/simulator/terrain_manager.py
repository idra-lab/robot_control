# import open3d as o3d
# import sys
# # Load mesh and convert to open3d.t.geometry.TriangleMesh
# # Load mesh and convert to open3d.t.geometry.TriangleMesh
# cube = o3d.geometry.TriangleMesh.create_box().translate([0, 0, 0])
# cube = o3d.t.geometry.TriangleMesh.from_legacy(cube)
#
# scene = o3d.t.geometry.RaycastingScene()
# cube_id = scene.add_triangles(cube)
#
# print(cube_id)
#
#
# # We create two rays:
# # The first ray starts at (0.5,0.5,10) and has direction (0,0,-1).
# # The second ray start at (-1,-1,-1) and has direction (0,0,-1).
# rays = o3d.core.Tensor([[0.5, 0.5, 10, 0, 0, -1], [-1, -1, -1, 0, 0, -1]],
#                        dtype=o3d.core.Dtype.Float32)
#
# ans = scene.cast_rays(rays)
# print(ans.keys())
# # t_hit is the distance to the intersection. The unit is defined by the length of the ray direction. If there is no intersection this is inf
# # geometry_ids gives the id of the geometry hit by the ray. If no geometry was hit this is RaycastingScene.INVALID_ID=4294967295
# # primitive_ids is the triangle index of the triangle that was hit or RaycastingScene.INVALID_ID
# # primitive_uvs is the barycentric coordinates of the intersection point within the triangle.
# # primitive_normals is the normal of the hit triangle.
# #print(ans['t_hit'].numpy())
# sys.exit()


import numpy as np
import math
import rospkg
import open3d as o3d
class TerrainManager:
    def __init__(self, mesh_path = None):
        if mesh_path is None:
            mesh_path = rospkg.RosPack().get_path('tractor_description') + "/meshes/terrain.stl"
        mesh =  o3d.io.read_triangle_mesh(mesh_path)
        self.set_mesh(mesh)
        self.baseline = -10. # is the Z level from which we cast rays

    def set_mesh(self, mesh):
        self.mesh = mesh
        # Cleanup the mesh
        self.mesh.remove_degenerate_triangles()  # Remove zero-area triangles
        self.mesh.remove_duplicated_triangles()  # Remove duplicate faces
        self.mesh.remove_duplicated_vertices()  # Remove duplicate vertices
        self.mesh.remove_non_manifold_edges()  # Fix non-manifold edges
        self.mesh.orient_triangles()  # If the mesh is orientable this function orients all triangles such that all normals point towards the same direction.
        self.mesh.compute_vertex_normals()  # Recompute normals
        self.mesh.compute_triangle_normals()
        # self.mesh.normalize_normals()

        # Scale the mesh by a factor of 100 (does not help)
        # self.scale_factor = 0.01
        # self.mesh.scale(self.scale_factor, center=self.mesh.get_center())

        # Create a LineSet for visualizing normals
        # vertices = np.asarray(self.mesh.vertices)
        # triangles = np.asarray(self.mesh.triangles)
        # normals = np.asarray(self.mesh.triangle_normals)  # Use triangle normals
        # lines = self.visualize_normals(triangles, vertices, normals,2.)
        # # Visualize the mesh and the normals
        # self.visualize([self.mesh ,lines])

        self.triangle_mesh = o3d.t.geometry.TriangleMesh.from_legacy(self.mesh)

        # define scene
        self.scene = o3d.t.geometry.RaycastingScene()
        # returns the ID for the added geometry
        self.scene.add_triangles(self.triangle_mesh)

    def visualize_normals(self, triangles, vertices, normals, length= 1.):
        lines = []
        colors = []
        new_vertices = []
        for i, triangle in enumerate(triangles):
            v0, v1, v2 = triangle

            # Compute the center of the triangle (face)
            triangle_center = (vertices[v0] + vertices[v1] + vertices[v2]) / 3.0
            normal_start = triangle_center
            normal_end = triangle_center + normals[i] * length

            new_vertices.append(normal_start)
            new_vertices.append(normal_end)

            lines.append([2 * i, 2 * i + 1])
            colors.extend([[1, 0, 0]])  # Red color for normals
        new_vertices = np.array(new_vertices)
        lines = np.array(lines)

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(new_vertices)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)

        return line_set

    def project_points_on_mesh(self, points, direction):
        # SUGGESTED BY https://github.com/matteodv99tn
        # use open3d ray casting to compute distance from point to surface
        
        #np.concatenate: Combines the original array with a new array containing the scalar, which is fast and avoids some overhead compared to np.append
        #array_to_append = np.concatenate(([self.baseline], direction)) #adds the z coord as baseline and the direction of the ray

        # Append using broadcasting and concatenate This approach is faster and memory-efficient for large arrays compared to using np.hstack or np.tile.
        tensor_set = np.concatenate((points, direction[None, :].repeat(points.shape[0], axis=0)), axis=1)

        rays = o3d.core.Tensor([tensor_set], dtype=o3d.core.Dtype.Float32)
        #The result contains information about a possible intersection with the geometry in the scene.
        ans = self.scene.cast_rays(rays)

        #t_hit is the distance to the intersection from the baseline. The unit is defined by the length of the ray direction. If there is no intersection this is inf
        distances_from_baseline = ans['t_hit'][0].numpy()

        # check missed rays
        #intersect_mask = np.isfinite(distances_from_baseline)   # True if the ray hits something, False otherwise
        #print("Number of failed  intersections:", np.count_nonzero(~intersect_mask))
        #TODO debug this
        #missed_rays = rays[~intersect_mask]  # Extract missed rays
        # missed_points = o3d.geometry.PointCloud()
        # missed_points.points = o3d.utility.Vector3dVector(missed_rays)
        # missed_points.paint_uniform_color([1, 0, 0])  # Red for missed intersections
        # o3d.visualization.draw_geometries([self.mesh, missed_points])

        z_coords = distances_from_baseline[:, None] * direction
        intersection_points = points + z_coords

        return intersection_points

    def project_on_mesh(self, point, direction, base_yaw = None, debug=False):
        # returns the point on the mesh along direction from point, and the roll/pitch (ZYX euler) of a robot with heading base_yaw lying on the mesh
        if base_yaw is None:
            base_yaw = 0.
        # SUGGESTED BY https://github.com/matteodv99tn
        point = np.append(point, self.baseline)
        # use open3d ray casting to compute distance from point to surface
        ray = o3d.core.Tensor([np.concatenate((point, direction))], dtype=o3d.core.Dtype.Float32)

        #The result contains information about a possible intersection with the geometry in the scene.
        ans = self.scene.cast_rays(ray)

        # self.logger.info(f"Distance from pelvis joint to surface: {ans}")
        # self.logger.info(f"Vertex ID: {ans['primitive_ids']}")

        #t_hit is the distance to the intersection. The unit is defined by the length of the ray direction. If there is no intersection this is inf
        distance = ans['t_hit'][0].cpu().numpy()
        z_coord = distance*direction
        eval_point = point + z_coord
        normal = ans['primitive_normals'][0].cpu().numpy().astype(float)
        # triangle normals can point downwards depending on the vertex order, make it point upwards
        if normal[2] < 0.:
            normal = -normal

        # old  way (wrong)
        # #first compute yaw to solve ambiguity, because we are reasoning in the world frame, we consider yaw the angle between the projection
        # #of the normal vector onto the XY plane n_xy =(nx, ny, 0) and the base X axis (rx,ry,0)
        # yaw = math.atan2(normal[1], normal[0])
        #
        # #then I rotate n to undo the yaw and get n_ = [sin(pitch)*cos(roll); -sint(roll), cos(pitch)*cos(roll)])
        # Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0],[math.sin(yaw), math.cos(yaw), 0],[0, 0, 1] ])
        # n_ =Rz.T.dot(normal)
        #
        # #compute roll/pitch
        # pitch = math.atan2(n_[0], n_[2])
        # roll = math.atan2(-n_[1]*np.sin(pitch), n_[0])
        # #as an alternative
        # #roll = math.asin(-n_[1])

        # First remove robot yaw from the normal (assume ZYX euler)
        Rz = np.array([[math.cos(base_yaw), -math.sin(base_yaw), 0], [math.sin(base_yaw), math.cos(base_yaw), 0], [0, 0, 1]])
        n_ = Rz.T.dot(normal)
        # now n_ will be Ry(pitch)Rx(roll)*e_z which leads to n_ = [sin(pitch)cos(roll), - sin(roll), cos(pitch)cos(roll)]
        # then we compute
        pitch = math.atan2(n_[0], n_[2])
        roll = math.atan2(-n_[1], math.sqrt(n_[0] * n_[0] + n_[2] * n_[2]))

        #to have it in the range between  -pi/2 and pi/2
        if roll > np.pi/2:
            roll-=np.pi
        if roll <-np.pi / 2:
            roll += np.pi

        # Visualize
        if debug:
            mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2, origin=[0, 0, 0])
            line_ray = self.draw_line(point, direction, self.baseline*2, colors=[[1,0,0]])
            line_normal = self.draw_line(eval_point, normal,5., colors=[[0,1,0]])
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.2).translate(eval_point)
            sphere.paint_uniform_color([1, 0, 0])
            # draw_geometries is high level function - it create a window and add geometries. If you put draw_geometries inside the lop, this will make many windows simultaneously and may cause problem.
            #o3d.visualization.draw_geometries([self.mesh, line_ray, line_normal,mesh_frame, sphere])
            self.visualize([self.mesh, line_ray, line_normal, mesh_frame, sphere])
            print("normal is: ", normal)
            print("eval point is ",eval_point)

        return eval_point, roll, pitch




    def draw_line(self, start, length, direction, colors =  [[1,0,0]]):
        end = start + length * direction
        points = [start, end]
        lines = [[0, 1]]
        line = o3d.geometry.LineSet()
        line.points = o3d.utility.Vector3dVector(points)
        line.lines = o3d.utility.Vector2iVector(lines)
        line.colors = o3d.utility.Vector3dVector(colors)
        return line

    def visualize(self, meshes):
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        #only in 0.17
        # vis.get_render_option().line_width = 5
        # vis.get_render_option().point_size = 20
        for mesh in meshes:
            vis.add_geometry(mesh)
        vis.run()
        vis.destroy_window()

        # # Create an OffscreenRenderer
        # renderer = o3d.visualization.rendering.OffscreenRenderer(800, 600)
        # # Access the scene
        # scene = renderer.scene
        # # Set up camera
        # camera = scene.camera
        # camera.look_at([0, 0, 0],  # Center of the scene
        #                eye=[3, 3, 3],  # Camera position
        #                up=[0, 1, 0])  # Up direction
        # # Add a directional light to cast shadows and create shading
        # scene.add_directional_light(
        #     direction=[-1, -1, -1],  # Direction of the light source
        #     color=[1.0, 1.0, 1.0],  # White light
        #     intensity=2.0,  # Intensity of the light
        #     cast_shadows=True  # Enable shadows
        # )
        # # Add ambient light (optional) to brighten up the scene slightly
        # scene.set_ambient_light([0.1, 0.1, 0.1])  # Soft ambient light for overall illumination
        # # Render the scene to an image
        # image = renderer.render_to_image()
        # # Save or visualize the rendered image
        # o3d.io.write_image("shaded_mesh.png", image)

    #non blocking
    # def visualize(self, mesh):
    #     o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
    #     vis = o3d.visualization.Visualizer()
    #     vis.create_window()
    #     vis.update_geometry(mesh)
    #     vis.poll_events()
    #     vis.update_renderer()
    #     vis.destroy_window()
    #     return

def create_ramp_mesh(length, width, inclination=0., origin=np.array([0,0,0])):
    """
    Create a pitch ramp mesh with the specified length, width, and height.

    Parameters:
        length (float): The horizontal length of the ramp.
        width (float): The width of the ramp.
        height (float): The height of the ramp's inclined plane.

    Returns:
        open3d.geometry.TriangleMesh: The ramp mesh.
    """
    # Define the vertices of the pitch ramp
    vertices = [
        [ -length/2,width/2,  length/2*math.tan(inclination)],  # Bottom-left corner
        [ -length/2, -width/2, length/2*math.tan(inclination)],  # Bottom-right corner
        [length/2, -width/2, -length/2*math.tan(inclination)],  # Top-right corner
        [length/2, width/2,  -length/2*math.tan(inclination)]  # Top-left triangle
    ]


    # Define the triangles (faces) of the ramp
    triangles = [
        [0, 1, 2],  # Bottom-right triangle
        [0, 2, 3]  # Top-left triangle
    ]

    # Create the mesh
    ramp_mesh = o3d.geometry.TriangleMesh()
    ramp_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    ramp_mesh.triangles = o3d.utility.Vector3iVector(triangles)

    # Compute normals for better visualization
    ramp_mesh.compute_vertex_normals()

    return ramp_mesh


if __name__ == '__main__':

    scaling_factor = 1
    point = np.array([22., 0.])
    direction = np.array([0., 0., 1.])

    mesh_path = rospkg.RosPack().get_path('tractor_description') + "/meshes/terrain.stl"
    terrainManager = TerrainManager(mesh_path)

    # Create the ramp mesh
    # ramp_mesh = create_ramp_mesh(length=50., width=50., inclination=-0.1, origin=np.array([0, 0, 0]))
    # terrainManager.set_mesh(ramp_mesh)

    # mesh = o3d.io.read_triangle_mesh(mesh_path)
    # terrainManager.visualize(mesh)
    #if you want to rotate the mesh
    # R = mesh.get_rotation_matrix_from_xyz((np.pi / 2, 0, 0 ))
    # mesh.rotate(R, center=(0, 0, 0))#rotate pi/2 roll
    # scale uniformly
    # mesh.scale(0.001, center=(0, 0, 0))
    #scale not uniformly
    #mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices) * np.array([0.5, 0.5, 0.2]))

    yaw = 0.
    eval_point, roll, pitch = terrainManager.project_on_mesh(point=point, direction=direction, base_yaw=yaw, debug=True)

    # print(scaling_factor * np.array([1, 1, 0.5]))
    print(eval_point)
    print(roll)
    print(pitch)
    print(yaw)

    #
