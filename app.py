import streamlit as st
import cv2
import numpy as np
import open3d as o3d
import plotly.graph_objects as go
from PIL import Image
import torch
from transformers import GLPNImageProcessor, GLPNForDepthEstimation
import time

# Title of the Streamlit app
st.title("3D Reconstruction from 2D Image 🚀")

# Sidebar for user inputs
st.sidebar.header("Upload Image")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Function to measure execution time
def timed_execution(task_name, func, *args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    elapsed_time = end_time - start_time
    st.sidebar.success(f"✅ {task_name} completed in {elapsed_time:.2f} sec")
    return result, elapsed_time

# Load the model
@st.cache_resource()
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_extractor = GLPNImageProcessor.from_pretrained("vinvino02/glpn-nyu")
    model = GLPNForDepthEstimation.from_pretrained("vinvino02/glpn-nyu").to(device)
    return feature_extractor, model, device

feature_extractor, model, device = load_model()

total_start_time = time.time()  # Start overall timer

# Process the uploaded image
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Resize the image
    new_height = min(image.height, 480)
    new_height -= new_height % 32
    new_width = int(new_height * image.width / image.height)
    new_width -= new_width % 32
    image = image.resize((new_width, new_height))

    # Prepare input for the model
    inputs = feature_extractor(images=image, return_tensors="pt").to(device)
    
    # Get depth estimation
    with torch.no_grad():
        outputs, depth_time = timed_execution("Depth Estimation", model, **inputs)
        predicted_depth = outputs.predicted_depth.squeeze().detach().cpu().numpy() * 1000.0

    # Normalize depth map
    normalized_output = cv2.normalize(predicted_depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_colored = cv2.applyColorMap(normalized_output, cv2.COLORMAP_PLASMA)
    
    # Display depth map
    st.image(depth_colored, caption="Estimated Depth Map", use_column_width=True)

    # Convert images to Open3D format
    depth_image = (predicted_depth * 255 / np.max(predicted_depth)).astype('uint8')
    image_np = np.array(image)

    depth_o3d = o3d.geometry.Image(depth_image)
    image_o3d = o3d.geometry.Image(image_np)

    # Create RGBD image
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(image_o3d, depth_o3d, convert_rgb_to_intensity=False)

    # Create Camera intrinsic matrix
    width, height = image.size
    camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
    camera_intrinsic.set_intrinsics(width, height, 500, 500, width / 2, height / 2)

    # Generate point cloud
    pcd_raw, point_cloud_time = timed_execution("Point Cloud Generation", 
        o3d.geometry.PointCloud.create_from_rgbd_image, rgbd_image, camera_intrinsic)

    # Convert Point Cloud to Numpy for Plotly
    pcd_points = np.asarray(pcd_raw.points)
    fig = go.Figure(data=[go.Scatter3d(
        x=pcd_points[:, 0], y=pcd_points[:, 1], z=pcd_points[:, 2],
        mode='markers',
        marker=dict(size=2, color=pcd_points[:, 2], colorscale='viridis', opacity=0.8)
    )])
    fig.update_layout(title="3D Point Cloud", scene=dict(aspectmode='data'))
    st.plotly_chart(fig)

    # Noise removal
    cl, ind = pcd_raw.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd = pcd_raw.select_by_index(ind)

    # Estimate normals
    pcd.estimate_normals()
    pcd.orient_normals_consistent_tangent_plane(30)

    # Surface reconstruction
    (mesh, densities), mesh_time = timed_execution(
    "Surface Reconstruction", o3d.geometry.TriangleMesh.create_from_point_cloud_poisson, pcd, depth=8
)

    # Convert Mesh to Numpy for Plotly
    mesh_vertices = np.asarray(mesh.vertices)
    mesh_triangles = np.asarray(mesh.triangles)
    mesh_fig = go.Figure(data=[go.Mesh3d(
        x=mesh_vertices[:, 0], y=mesh_vertices[:, 1], z=mesh_vertices[:, 2],
        i=mesh_triangles[:, 0], j=mesh_triangles[:, 1], k=mesh_triangles[:, 2],
        color='lightblue', opacity=0.5
    )])
    mesh_fig.update_layout(title="3D Mesh Reconstruction", scene=dict(aspectmode='data'))
    st.plotly_chart(mesh_fig)

    # Save the mesh
    output_mesh_path = "reconstructed_model.ply"
    o3d.io.write_triangle_mesh(output_mesh_path, mesh)
    st.sidebar.download_button("Download 3D Model", open(output_mesh_path, "rb"), file_name="3D_model.ply")
    
    total_end_time = time.time()  # End overall timer
    total_time = total_end_time - total_start_time
    
    st.sidebar.success(f"🕒 Total time taken: {total_time:.2f} sec")
    st.success("✅ 3D Reconstruction Complete! Download the model above.")
