"""
Colin Catlin, 2025, GNU GENERAL PUBLIC LICENSE Version 3
Generate a louvered bicycle fender (mudguard) as a 3D mesh and export it to an STL file.

Extending symmetrically from both sides of the spine are closely-spaced louver
"wings" or "blades". These louvers are oriented parallel to the wheel's hub
axis (the bike width direction), effectively extending sideways from the fender
spine.

In this model, airflow moves along the axis down the diameter that extends from the -90 degree (front) to +90 degree (back) angles of the arc.
The 0 degree position is the top of the wheel.

The spacing of the louvers (controlled by louver_spacing parameter) allows them
to effectively block water spray coming off the wheel, while the gaps between
the blades permit front-to-back airflow with minimal aerodynamic drag.

This implementation is a simplified design without hardware mounting points or secondary structural details.

Potential Future Improvements:
- Round the corners of the spine and/or add a gutter to the spine
- Add an option for reduced louver angle (parallel to ground) with longer chord lengths
- Add a slight twist to the louver wings to create outwash
"""
import math
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable, Union, Set
from collections import Counter
from scipy.spatial import Delaunay


try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    print("YOU REALLY WILL WANT PYVISTA FOR A PROPER FRONT CAP! Install it via 'pip install pyvista'")
    pv = None
    HAS_PYVISTA = False


def gaussian_smooth_5point(points: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Apply 5-point Gaussian smoothing to a series of points.
    
    Uses a 5-point Gaussian kernel [1, 4, 6, 4, 1] / 16 for smooth filtering
    that preserves overall shape while reducing high-frequency noise.
    
    Parameters
    ----------
    points : np.ndarray
        Array of points to smooth, shape (N, D) where N is number of points
        and D is dimensionality (2 or 3).
    iterations : int
        Number of smoothing passes (default 1).
    
    Returns
    -------
    np.ndarray
        Smoothed points with same shape as input.
    """
    if len(points) < 5:
        return points.copy()
    
    # Gaussian kernel: [1, 4, 6, 4, 1] / 16
    kernel = np.array([1, 4, 6, 4, 1]) / 16.0
    
    smoothed = points.copy()
    for _ in range(iterations):
        result = smoothed.copy()
        # Apply kernel to interior points
        for i in range(2, len(points) - 2):
            weighted_sum = np.zeros_like(smoothed[i])
            for j, weight in enumerate(kernel):
                weighted_sum += weight * smoothed[i - 2 + j]
            result[i] = weighted_sum
        smoothed = result
    
    return smoothed


def naca_4digit_camber(x: float, m: float = 0.02, p: float = 0.4) -> float:
    """Calculate camber line y-coordinate for a NACA 4-digit airfoil."""
    if x < p:
        return m * (2 * p * x - x * x) / (p * p)
    else:
        return m * ((1 - 2 * p) + 2 * p * x - x * x) / ((1 - p) * (1 - p))


def naca_4digit_thickness(x: float, t: float = 0.12) -> float:
    """Calculate half-thickness for a NACA 4-digit airfoil."""
    # NACA 4-digit thickness distribution with modified trailing edge for better printability
    a0, a1, a2, a3 = 0.2969, -0.126, -0.3516, 0.2843
    a4 = -0.1015  # Modified from -0.1036 to avoid exactly zero at x=1
    
    return 5 * t * (a0 * math.sqrt(x) + a1 * x + a2 * x**2 + a3 * x**3 + a4 * x**4)


def create_airfoil_profile(
    chord_length: float,
    max_thickness: float = 0.12,
    max_camber: float = 0.02,
    camber_position: float = 0.4,
    n_points: int = 20,
    kammback_start: float = 0.85,
    smooth_profile: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create an airfoil profile with Kammback trailing edge for printability.
    
    The airfoil is oriented with the rounded leading edge at x=0 (facing incoming
    airflow from the wheel) and the blunt Kammback trailing edge at x=chord_length.
    
    Parameters
    ----------
    chord_length : float
        Length of the airfoil chord in mm.
    max_thickness : float
        Maximum thickness as fraction of chord (e.g., 0.12 = 12%).
    max_camber : float
        Maximum camber as fraction of chord (e.g., 0.02 = 2%).
    camber_position : float
        Position of maximum camber (0 to 1, typically 0.4).
    n_points : int
        Number of points to use for the profile.
    kammback_start : float
        Position (0 to 1) where Kammback trailing edge truncation begins.
        The airfoil is cut off at this point to create a blunt trailing edge
        that's easy to print and aerodynamically efficient.
    smooth_profile : bool
        If True, apply 5-point Gaussian smoothing to the profile for better
        surface quality (default True).
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Upper and lower surface coordinates as (n_points, 2) arrays.
        Coordinates start at leading edge (x=0) and go to trailing edge (x=chord).
    """
    # Use cosine spacing for better resolution at leading/trailing edges
    beta = np.linspace(0, math.pi, n_points)
    x = (1 - np.cos(beta)) / 2  # Cosine spacing from 0 to 1
    
    upper_points = []
    lower_points = []
    
    for xi in x:
        # Apply Kammback truncation: airfoil stops at kammback_start
        if xi > kammback_start:
            continue
        
        # Calculate camber line and thickness using NACA equations
        yc = naca_4digit_camber(xi, max_camber, camber_position)
        yt = naca_4digit_thickness(xi, max_thickness)
        
        # Calculate camber line slope for perpendicular thickness distribution
        if xi < camber_position:
            dyc_dx = 2 * max_camber * (camber_position - xi) / (camber_position**2)
        else:
            dyc_dx = 2 * max_camber * (camber_position - xi) / ((1 - camber_position)**2)
        
        theta = math.atan(dyc_dx)
        
        # Upper and lower surface points perpendicular to camber line
        xu = xi * chord_length - yt * chord_length * math.sin(theta)
        yu = yc * chord_length + yt * chord_length * math.cos(theta)
        xl = xi * chord_length + yt * chord_length * math.sin(theta)
        yl = yc * chord_length - yt * chord_length * math.cos(theta)
        
        upper_points.append([xu, yu])
        lower_points.append([xl, yl])
    
    # Add explicit Kammback trailing edge points at the truncation location
    if len(upper_points) > 0 and len(lower_points) > 0:
        x_kammback = kammback_start * chord_length
        yc_kammback = naca_4digit_camber(kammback_start, max_camber, camber_position)
        yt_kammback = naca_4digit_thickness(kammback_start, max_thickness)
        
        # Camber line slope at kammback position
        if kammback_start < camber_position:
            dyc_dx_kb = 2 * max_camber * (camber_position - kammback_start) / (camber_position**2)
        else:
            dyc_dx_kb = 2 * max_camber * (camber_position - kammback_start) / ((1 - camber_position)**2)
        
        theta_kb = math.atan(dyc_dx_kb)
        
        # For Kammback blunt trailing edge, force both upper and lower points to same X position
        # This ensures a perfectly vertical trailing edge regardless of camber
        x_kb_fixed = kammback_start * chord_length
        
        # Update last points to ensure clean blunt edge with vertical cutoff
        xu_kb = x_kb_fixed
        yu_kb = yc_kammback * chord_length + yt_kammback * chord_length * math.cos(theta_kb)
        xl_kb = x_kb_fixed
        yl_kb = yc_kammback * chord_length - yt_kammback * chord_length * math.cos(theta_kb)
        
        upper_points[-1] = [xu_kb, yu_kb]
        lower_points[-1] = [xl_kb, yl_kb]
    
    upper_array = np.array(upper_points)
    lower_array = np.array(lower_points)
    
    # Apply Gaussian smoothing if requested
    if smooth_profile and len(upper_array) >= 5 and len(lower_array) >= 5:
        # Preserve leading edge (first point) and trailing edge (last point)
        # by only smoothing interior points
        upper_array = gaussian_smooth_5point(upper_array, iterations=1)
        lower_array = gaussian_smooth_5point(lower_array, iterations=1)
    
    return upper_array, lower_array


def rotate_vector(v: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotate vector v around axis by angle (radians) using Rodrigues' formula."""
    # Ensure axis is unit length
    axis = axis / np.linalg.norm(axis)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    return (
        v * cos_a
        + np.cross(axis, v) * sin_a
        + axis * np.dot(axis, v) * (1.0 - cos_a)
    )


def create_spine(
    spine_inner_radius: float,
    spine_width: float,
    spine_thickness: float,
    coverage_angle_deg: float,
    spine_segments: int,
    forward_extension_deg: float = 0.0,
    smooth_spine: bool = True,
) -> Tuple[np.ndarray, List[Tuple[int, int, int]]]:
    """Construct a curved rectangular spine along an arc.

    Parameters
    ----------
    spine_inner_radius : float
        Radius from the wheel centre to the inner surface of the spine, in millimetres.
    spine_width : float
        Width of the spine cross‐section in the radial direction (mm).
    spine_thickness : float
        Thickness (height) of the spine cross‐section in the vertical
        direction (mm).
    coverage_angle_deg : float
        Angular coverage of the fender in degrees measured from the top of the
        wheel towards the back (positive direction rotates from +Y towards -X
        in the XY plane).  For example, 100 degrees covers a little more than
        a quarter circle.
    spine_segments : int
        Number of segments used to approximate the curved spine.  More
        segments produce a smoother curve but increase polygon count.
    forward_extension_deg : float, optional
        Additional angular extension forward of the wheel top (towards the
        bike's front). Use 0 for the legacy behaviour (no forward extension).
    smooth_spine : bool
        If True, apply Gaussian smoothing to spine vertex positions for
        smoother curvature (default True).

    Returns
    -------
    Tuple[np.ndarray, List[Tuple[int, int, int]]]
        A tuple containing the array of vertices (shape (N, 3)) and a list
        of triangular faces (triplets of vertex indices) representing the
        spine geometry.
    """
    # Calculate the radius to the centre of the spine cross‐section
    spine_centre_radius = spine_inner_radius + spine_width / 2.0
    total_coverage_deg = coverage_angle_deg + forward_extension_deg
    if total_coverage_deg <= 0.0:
        raise ValueError("Total coverage must be positive.")
    coverage_angle_rad = math.radians(total_coverage_deg)
    theta_start = math.pi / 2.0 - math.radians(forward_extension_deg)
    theta_end = theta_start + coverage_angle_rad

    # Parameterise the arc into segments
    thetas = np.linspace(theta_start, theta_end, spine_segments + 1)
    vertices: List[List[float]] = []
    faces: List[Tuple[int, int, int]] = []
    edge_vertex_cache: Dict[Tuple[int, int], int] = {}

    # Compute cross‐section vertices for each segment
    section_indices = []
    section_centers = []  # Track center positions for smoothing
    
    for theta in thetas:
        n_dir = np.array([math.cos(theta), math.sin(theta), 0.0])
        z_dir = np.array([0.0, 0.0, 1.0])
        centre = spine_centre_radius * n_dir
        section_centers.append(centre)

        # Four corners of the rectangular cross‐section
        v0 = centre - (spine_width / 2.0) * n_dir - (spine_thickness / 2.0) * z_dir
        v1 = centre - (spine_width / 2.0) * n_dir + (spine_thickness / 2.0) * z_dir
        v2 = centre + (spine_width / 2.0) * n_dir + (spine_thickness / 2.0) * z_dir
        v3 = centre + (spine_width / 2.0) * n_dir - (spine_thickness / 2.0) * z_dir
        base_idx = len(vertices)
        vertices.extend([v0.tolist(), v1.tolist(), v2.tolist(), v3.tolist()])
        section_indices.append((base_idx, base_idx + 1, base_idx + 2, base_idx + 3))

    # Apply smoothing to vertex positions if requested
    if smooth_spine and len(section_centers) >= 5:
        # Smooth the center positions
        centers_array = np.array(section_centers)
        smoothed_centers = gaussian_smooth_5point(centers_array, iterations=1)
        
        # Rebuild vertices with smoothed centers
        vertices_array = np.array(vertices)
        for i, (theta, smooth_center) in enumerate(zip(thetas, smoothed_centers)):
            n_dir = np.array([math.cos(theta), math.sin(theta), 0.0])
            z_dir = np.array([0.0, 0.0, 1.0])
            
            # Rebuild the four corners with smoothed center
            base_idx = i * 4
            vertices_array[base_idx] = smooth_center - (spine_width / 2.0) * n_dir - (spine_thickness / 2.0) * z_dir
            vertices_array[base_idx + 1] = smooth_center - (spine_width / 2.0) * n_dir + (spine_thickness / 2.0) * z_dir
            vertices_array[base_idx + 2] = smooth_center + (spine_width / 2.0) * n_dir + (spine_thickness / 2.0) * z_dir
            vertices_array[base_idx + 3] = smooth_center + (spine_width / 2.0) * n_dir - (spine_thickness / 2.0) * z_dir
        
        vertices = vertices_array.tolist()

    # Create faces between consecutive segments
    for i in range(len(section_indices) - 1):
        v0_i, v1_i, v2_i, v3_i = section_indices[i]
        v0_j, v1_j, v2_j, v3_j = section_indices[i + 1]
        # Inside, top, outside, and bottom faces
        faces.extend([
            (v0_i, v0_j, v1_j), (v0_i, v1_j, v1_i),
            (v1_i, v1_j, v2_j), (v1_i, v2_j, v2_i),
            (v2_i, v2_j, v3_j), (v2_i, v3_j, v3_i),
            (v3_i, v3_j, v0_j), (v3_i, v0_j, v0_i),
        ])
    # Add end caps
    def add_cap(section_idx: Tuple[int, int, int, int], theta: float, outward_sign: float) -> None:
        """Triangulate a rectangular end cap with outward normal control."""
        t_dir = np.array([-math.sin(theta), math.cos(theta), 0.0])
        target_dir = t_dir * outward_sign
        cap_tris = [
            (section_idx[0], section_idx[1], section_idx[2]),
            (section_idx[0], section_idx[2], section_idx[3]),
        ]
        for tri in cap_tris:
            p0, p1, p2 = np.array(vertices[tri[0]]), np.array(vertices[tri[1]]), np.array(vertices[tri[2]])
            normal = np.cross(p1 - p0, p2 - p0)
            if np.linalg.norm(normal) == 0:
                continue
            if np.dot(normal, target_dir) < 0:
                faces.append((tri[0], tri[2], tri[1]))
            else:
                faces.append(tri)

    if section_indices:
        add_cap(section_indices[0], thetas[0], outward_sign=-1.0)
        add_cap(section_indices[-1], thetas[-1], outward_sign=1.0)

    return np.array(vertices), faces


def compute_plane_intersection(p1: np.ndarray, p2: np.ndarray, y_plane: float, eps: float = 1e-9) -> Optional[np.ndarray]:
    """
    Compute the exact intersection point where edge (p1, p2) crosses the horizontal plane y = y_plane.
    
    Uses parametric interpolation: P = p1 + t*(p2 - p1) where y(P) = y_plane
    
    Parameters
    ----------
    p1, p2 : np.ndarray
        3D points defining the edge
    y_plane : float
        Y-coordinate of the clipping plane
    eps : float
        Tolerance for detecting crossing edges
        
    Returns
    -------
    Optional[np.ndarray]
        The 3D intersection point, or None if the edge doesn't cross the plane
    """
    y1, y2 = p1[1], p2[1]
    dy = y2 - y1
    
    # Check if edge crosses the plane
    if abs(dy) < eps:
        return None  # Edge is parallel to plane
    
    # Check if edge actually straddles the plane
    if (y1 - y_plane) * (y2 - y_plane) > eps:
        return None  # Both points on same side
    
    # Compute parametric position of intersection
    t = (y_plane - y1) / dy
    t = np.clip(t, 0.0, 1.0)  # Clamp to [0, 1] for safety
    
    # Compute 3D intersection point
    return p1 + t * (p2 - p1)


def slice_tri_against_yplane(
    v0: int, v1: int, v2: int,
    V: List[np.ndarray],
    y_plane: float,
    keep_above: bool = True, 
    eps: float = 1e-8,
    edge_cache: Optional[Dict[Tuple[int, int], int]] = None,
) -> List[Tuple[int, int, int]]:
    """
    Clip a triangle against a horizontal plane, returning 0-2 clipped triangles.
    
    Adds any new intersection vertices into V (with Y snapped exactly to y_plane).
    
    Parameters
    ----------
    v0, v1, v2 : int
        Vertex indices into V forming the triangle
    V : List[np.ndarray]
        Mutable list of vertices (new intersection points will be appended)
    y_plane : float
        Y-coordinate of clipping plane
    keep_above : bool
        If True, keep portion above/on plane; if False, keep below
    eps : float
        Tolerance for plane comparison
    edge_cache : Optional[Dict[Tuple[int, int], int]]
        Optional cache for edge-plane intersection vertices to keep shared
        edges watertight.
        
    Returns
    -------
    List[Tuple[int, int, int]]
        List of 0-2 triangles (as vertex index tuples) representing the clipped geometry
    """
    idxs = [v0, v1, v2]
    P = [V[v0], V[v1], V[v2]]
    side = [(p[1] - y_plane) >= -eps for p in P]  # True = above/on
    
    if not keep_above:
        side = [not s for s in side]
    
    # Quick exits
    if all(side):  # Whole triangle kept
        return [(v0, v1, v2)]
    if not any(side):  # Whole triangle discarded
        return []
    
    # Build polygon by walking edges and inserting intersections
    new_poly = []
    for i in range(3):
        i2 = (i + 1) % 3
        A, B = P[i], P[i2]
        keepA, keepB = side[i], side[i2]
        
        if keepA:
            new_poly.append((idxs[i], A))
        
        # Edge AB crosses plane?
        dy = B[1] - A[1]
        if (keepA != keepB) and abs(dy) > eps:
            idxQ = None
            Q = None
            # If either endpoint is on the plane, treat that existing vertex as the intersection.
            # This avoids dropping the intersection when the kept vertex is B (and A is discarded),
            # which can leave open boundaries and glitchy caps.
            if abs(A[1] - y_plane) <= eps:
                idxQ = idxs[i]
                Q = A
            elif abs(B[1] - y_plane) <= eps:
                idxQ = idxs[i2]
                Q = B
            else:
                t = (y_plane - A[1]) / dy
                Q = A + t * (B - A)
                Q[1] = y_plane  # Snap exactly to plane

            if edge_cache is not None:
                key = (idxs[i], idxs[i2])
                if key[0] > key[1]:
                    key = (key[1], key[0])
                cached = edge_cache.get(key)
                if cached is not None:
                    idxQ = cached
                    Q = V[idxQ]
                else:
                    edge_cache[key] = idxQ if idxQ is not None else len(V)

            if idxQ is None:
                idxQ = len(V)
                V.append(Q.copy())
            new_poly.append((idxQ, Q))
    
    # Triangulate the clipped polygon (it has 3 or 4 verts)
    out = []
    if len(new_poly) == 3:
        out.append((new_poly[0][0], new_poly[1][0], new_poly[2][0]))
    elif len(new_poly) == 4:
        a, b, c, d = [t[0] for t in new_poly]
        out.append((a, b, c))
        out.append((a, c, d))

    return out


def clip_mesh_against_yplane(
    vertices: np.ndarray,
    faces: List[Tuple[int, int, int]],
    y_plane: float,
    keep_above: bool = True,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, List[Tuple[int, int, int]], Dict[Tuple[int, int], int]]:
    """Clip an entire triangle mesh against y=y_plane using shared edge caching."""
    if vertices.size == 0 or not faces:
        return vertices.copy(), list(faces), {}

    V: List[np.ndarray] = [v.copy() for v in vertices]
    edge_cache: Dict[Tuple[int, int], int] = {}
    out_faces: List[Tuple[int, int, int]] = []

    # Snap any vertices very close to the plane exactly onto it.
    for i in range(len(V)):
        if abs(V[i][1] - y_plane) <= eps:
            V[i][1] = y_plane

    for a, b, c in faces:
        clipped_tris = slice_tri_against_yplane(
            a,
            b,
            c,
            V,
            y_plane,
            keep_above=keep_above,
            eps=eps,
            edge_cache=edge_cache,
        )
        for tri in clipped_tris:
            if tri[0] == tri[1] or tri[1] == tri[2] or tri[0] == tri[2]:
                continue
            out_faces.append(tri)

    return np.array(V), out_faces, edge_cache


def clip_mesh_prefix_against_yplane(
    vertices: np.ndarray,
    faces: List[Tuple[int, int, int]],
    y_plane: float,
    max_vertex_index_exclusive: int,
    keep_above: bool = True,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, List[Tuple[int, int, int]], Dict[Tuple[int, int], int]]:
    """
    Clip only faces whose vertices are all < max_vertex_index_exclusive.

    This is used to trim just the forward-most spine region. A constant-y plane can
    intersect the spine arc twice; clipping the entire spine would also trim the rear.
    """
    if vertices.size == 0 or not faces:
        return vertices.copy(), list(faces), {}

    V: List[np.ndarray] = [v.copy() for v in vertices]
    edge_cache: Dict[Tuple[int, int], int] = {}
    out_faces: List[Tuple[int, int, int]] = []

    for i in range(len(V)):
        if abs(V[i][1] - y_plane) <= eps:
            V[i][1] = y_plane

    for a, b, c in faces:
        if a < max_vertex_index_exclusive and b < max_vertex_index_exclusive and c < max_vertex_index_exclusive:
            clipped_tris = slice_tri_against_yplane(
                a,
                b,
                c,
                V,
                y_plane,
                keep_above=keep_above,
                eps=eps,
                edge_cache=edge_cache,
            )
            for tri in clipped_tris:
                if tri[0] == tri[1] or tri[1] == tri[2] or tri[0] == tri[2]:
                    continue
                out_faces.append(tri)
        else:
            out_faces.append((a, b, c))

    return np.array(V), out_faces, edge_cache


def infer_clip_plane_y(vertices: np.ndarray, faces: List[Tuple[int, int, int]]) -> Optional[float]:
    """Infer the clip plane as the minimum Y among vertices referenced by faces."""
    if vertices.size == 0 or not faces:
        return None
    used = np.unique(np.array(faces, dtype=np.int64).ravel())
    if used.size == 0:
        return None
    return float(np.min(vertices[used, 1]))


def compute_loop_intersections(
    loop_indices: List[int],
    vertices: np.ndarray,
    y_plane: float,
    eps: float = 1e-9,
    on_plane_tolerance: float = 0.01
) -> List[np.ndarray]:
    """
    March through a closed loop and collect all points that define the intersection with the plane.
    This includes:
    1. Vertices already at or very close to the plane
    2. Exact intersection points where edges cross the plane
    
    Parameters
    ----------
    loop_indices : List[int]
        Ordered vertex indices forming a closed loop
    vertices : np.ndarray
        Array of all vertices
    y_plane : float
        Y-coordinate of clipping plane
    eps : float
        Tolerance for edge crossing detection
    on_plane_tolerance : float
        Distance threshold for considering a vertex "on" the plane (mm)
        
    Returns
    -------
    List[np.ndarray]
        Ordered list of 3D intersection points around the boundary
    """
    intersections = []
    n = len(loop_indices)
    
    for i in range(n):
        v_idx1 = loop_indices[i]
        v_idx2 = loop_indices[(i + 1) % n]
        
        p1 = vertices[v_idx1]
        p2 = vertices[v_idx2]
        
        y1 = p1[1]
        y2 = p2[1]
        
        # Check if p1 is at/on the plane
        if abs(y1 - y_plane) <= on_plane_tolerance:
            # Snap p1 to plane and add it
            p1_on_plane = np.array([p1[0], y_plane, p1[2]])
            intersections.append(p1_on_plane)
        
        # Check if edge crosses the plane (and p1 wasn't already on it)
        elif abs(y1 - y_plane) > on_plane_tolerance:
            p_intersect = compute_plane_intersection(p1, p2, y_plane, eps)
            if p_intersect is not None:
                intersections.append(p_intersect)
    
    return intersections


def polygon_area_2d(points: np.ndarray) -> float:
    """Compute signed area of a 2D polygon (positive for CCW)."""
    if len(points) < 3:
        return 0.0
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    """Ray casting test for point-in-polygon (works for CW/CCW)."""
    x, y = point
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def triangulate_with_holes(
    outer_pts: np.ndarray,
    hole_pts: List[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Triangulate a 2D polygon with holes using constrained Delaunay."""
    if Delaunay is None:
        raise RuntimeError("scipy is required for front cap triangulation")
    all_pts = np.vstack([outer_pts] + hole_pts) if hole_pts else outer_pts.copy()
    tri = Delaunay(all_pts)
    valid_faces = []
    for simplex in tri.simplices:
        centroid = all_pts[simplex].mean(axis=0)
        if not point_in_polygon(centroid, outer_pts):
            continue
        if any(point_in_polygon(centroid, hole) for hole in hole_pts):
            continue
        valid_faces.append(simplex)
    return all_pts, np.array(valid_faces)


def extract_plane_boundary_loops(
    vertices: np.ndarray,
    faces: List[Tuple[int, int, int]],
    y_plane: float,
    eps: float = 1e-6,
) -> List[List[int]]:
    """Extract boundary loops that lie on a given Y plane."""
    edge_counts: Counter = Counter()
    for a, b, c in faces:
        for u, v in ((a, b), (b, c), (c, a)):
            if u == v:
                continue
            key = (u, v) if u < v else (v, u)
            edge_counts[key] += 1

    plane_edges = []
    for (u, v), count in edge_counts.items():
        if count != 1:
            continue
        if abs(vertices[u][1] - y_plane) <= eps and abs(vertices[v][1] - y_plane) <= eps:
            plane_edges.append((u, v))

    if not plane_edges:
        return []

    adjacency: Dict[int, List[int]] = {}
    for u, v in plane_edges:
        adjacency.setdefault(u, []).append(v)
        adjacency.setdefault(v, []).append(u)

    unused = {((u, v) if u < v else (v, u)) for u, v in plane_edges}
    loops = []

    while unused:
        u, v = unused.pop()
        loop = [u, v]
        prev, curr = u, v
        while True:
            neighbors = adjacency.get(curr, [])
            nxt = None
            for cand in neighbors:
                if cand == prev:
                    continue
                edge_key = (cand, curr) if cand < curr else (curr, cand)
                if edge_key in unused:
                    nxt = cand
                    break
            if nxt is None:
                if loop[0] in neighbors:
                    loop.append(loop[0])
                break
            edge_key = (nxt, curr) if nxt < curr else (curr, nxt)
            if edge_key in unused:
                unused.remove(edge_key)
            if nxt == loop[0]:
                loop.append(nxt)
                break
            loop.append(nxt)
            prev, curr = curr, nxt

        if len(loop) >= 4 and loop[0] == loop[-1]:
            loops.append(loop[:-1])

    return loops


def extract_plane_boundary_chains(
    vertices: np.ndarray,
    faces: List[Tuple[int, int, int]],
    y_plane: float,
    eps: float = 1e-6,
) -> List[List[int]]:
    """Extract ordered boundary chains (open or closed) that lie on a given Y plane."""
    edge_counts: Counter = Counter()
    for a, b, c in faces:
        for u, v in ((a, b), (b, c), (c, a)):
            if u == v:
                continue
            key = (u, v) if u < v else (v, u)
            edge_counts[key] += 1

    plane_boundary_edges: List[Tuple[int, int]] = []
    for (u, v), count in edge_counts.items():
        if count != 1:
            continue
        if abs(vertices[u][1] - y_plane) <= eps and abs(vertices[v][1] - y_plane) <= eps:
            plane_boundary_edges.append((u, v))

    if not plane_boundary_edges:
        return []

    adjacency: Dict[int, List[int]] = {}
    for u, v in plane_boundary_edges:
        adjacency.setdefault(u, []).append(v)
        adjacency.setdefault(v, []).append(u)

    # Build connected components and order them into chains.
    seen: set[int] = set()
    chains: List[List[int]] = []

    for start in adjacency:
        if start in seen:
            continue
        stack = [start]
        component: List[int] = []
        seen.add(start)
        while stack:
            n = stack.pop()
            component.append(n)
            for nb in adjacency.get(n, []):
                if nb not in seen:
                    seen.add(nb)
                    stack.append(nb)

        comp_set = set(component)
        degree = {n: len([nb for nb in adjacency[n] if nb in comp_set]) for n in component}
        endpoints = [n for n, deg in degree.items() if deg == 1]

        if endpoints:
            cur = endpoints[0]
        else:
            cur = component[0]

        prev = None
        chain = [cur]
        visited_local: set[int] = {cur}
        while True:
            nxt_candidates = [nb for nb in adjacency[cur] if nb in comp_set and nb != prev]
            if not nxt_candidates:
                break
            # Most boundary graphs here are degree-2; if ambiguous, pick an unvisited neighbor if possible.
            nxt = None
            for cand in nxt_candidates:
                if cand not in visited_local:
                    nxt = cand
                    break
            if nxt is None:
                nxt = nxt_candidates[0]
            prev, cur = cur, nxt
            if cur == chain[0]:
                chain.append(cur)
                break
            chain.append(cur)
            visited_local.add(cur)
            if endpoints and cur in endpoints[1:]:
                break

        if len(chain) >= 2 and chain[0] == chain[-1]:
            chains.append(chain[:-1])
        elif len(chain) >= 2:
            chains.append(chain)

    return chains


def _merge_chains_by_endpoint_proximity(
    vertices: np.ndarray,
    chains: List[List[int]],
    tol_mm: float = 0.5,
) -> List[List[int]]:
    """Merge chains whose endpoints nearly coincide in XZ, to heal tiny graph breaks."""
    if len(chains) < 2:
        return chains

    def xz(idx: int) -> np.ndarray:
        return vertices[idx][[0, 2]]

    out = [c[:] for c in chains]
    tol2 = float(tol_mm * tol_mm)
    changed = True
    while changed:
        changed = False
        best = None  # (dist2, i, j, mode)
        for i in range(len(out)):
            a = out[i]
            a0, a1 = a[0], a[-1]
            for j in range(i + 1, len(out)):
                b = out[j]
                b0, b1 = b[0], b[-1]
                candidates = [
                    ("a1_b0", a1, b0),
                    ("a1_b1", a1, b1),
                    ("a0_b1", a0, b1),
                    ("a0_b0", a0, b0),
                ]
                for mode, u, v in candidates:
                    d2 = float(np.sum((xz(u) - xz(v)) ** 2))
                    if d2 <= tol2 and (best is None or d2 < best[0]):
                        best = (d2, i, j, mode)
        if best is None:
            break
        _, i, j, mode = best
        a = out[i]
        b = out[j]
        if mode == "a1_b0":
            merged = a + (b[1:] if a[-1] == b[0] else b)
        elif mode == "a1_b1":
            b_rev = list(reversed(b))
            merged = a + (b_rev[1:] if a[-1] == b_rev[0] else b_rev)
        elif mode == "a0_b1":
            merged = b + (a[1:] if b[-1] == a[0] else a)
        else:  # a0_b0
            a_rev = list(reversed(a))
            merged = b + (a_rev[1:] if b[-1] == a_rev[0] else a_rev)
        out[i] = merged
        out.pop(j)
        changed = True

    return out


def build_plane_rim_cap_faces(
    vertices: np.ndarray,
    faces: List[Tuple[int, int, int]],
    y_plane: float,
    base_vertex_count: int,
    intersection_edge_cache: Dict[Tuple[int, int], int],
    keep_above: bool = True,
    eps: float = 1e-6,
    endpoint_merge_tol_mm: float = 0.5,
) -> List[Tuple[int, int, int]]:
    """
    Build a planar 'rim' cap on y=y_plane that stitches the inner/outer shells.

    Unlike a full planar polygon cap (which requires closed loops), this supports
    open, horseshoe-like rims where the profile is intentionally open (e.g. near
    a trailing-edge slit) and we only want to close the wall thickness.
    
    KEY FIX: Only use edges that contain intersection vertices (vertices created
    during clipping). This ensures we only cap the front section, not the entire
    airfoil boundary.
    """
    target_dir = np.array([0.0, -1.0 if keep_above else 1.0, 0.0])
    
    # Collect all intersection vertex indices - these are the vertices created during clipping
    intersection_vertex_set: Set[int] = set(intersection_edge_cache.values())
    
    if not intersection_vertex_set:
        print("  No intersection vertices found in cache")
        return []
    
    # Build a map from intersection vertex to its source edge (for classifying as outer/inner)
    intersection_sources: Dict[int, Tuple[int, int]] = {}
    for edge_key, idx in intersection_edge_cache.items():
        intersection_sources[idx] = edge_key

    def vertex_kind(idx: int) -> Optional[str]:
        """Classify a vertex as 'outer' or 'inner' based on its index pattern."""
        if idx < base_vertex_count:
            return "outer" if (idx % 4) in (0, 2) else "inner"
        src = intersection_sources.get(idx)
        if src is None:
            return None
        u, v = src
        ku = "outer" if (u % 4) in (0, 2) else "inner"
        kv = "outer" if (v % 4) in (0, 2) else "inner"
        return ku if ku == kv else None

    # Find boundary edges on the clip plane
    edge_counts: Counter = Counter()
    for a, b, c in faces:
        for u, v in ((a, b), (b, c), (c, a)):
            if u == v:
                continue
            key = (u, v) if u < v else (v, u)
            edge_counts[key] += 1

    # Only consider boundary edges that:
    # 1. Are on the clip plane
    # 2. Include at least one intersection vertex (from the clipped front section)
    plane_boundary_edges: List[Tuple[int, int]] = []
    for (u, v), count in edge_counts.items():
        if count != 1:
            continue
        if abs(vertices[u][1] - y_plane) <= eps and abs(vertices[v][1] - y_plane) <= eps:
            # KEY FILTER: At least one vertex must be an intersection vertex
            if u in intersection_vertex_set or v in intersection_vertex_set:
                plane_boundary_edges.append((u, v))

    if not plane_boundary_edges:
        print("  No plane boundary edges found with intersection vertices")
        return []
    
    print(f"  Rim cap: {len(plane_boundary_edges)} boundary edges with intersection vertices")

    outer_edges: List[Tuple[int, int]] = []
    inner_edges: List[Tuple[int, int]] = []
    for u, v in plane_boundary_edges:
        ku = vertex_kind(u)
        kv = vertex_kind(v)
        if ku is None or kv is None:
            continue
        if ku == kv == "outer":
            outer_edges.append((u, v))
        elif ku == kv == "inner":
            inner_edges.append((u, v))

    print(f"  Rim cap: {len(outer_edges)} outer edges, {len(inner_edges)} inner edges")

    def chains_from_edges(edges: List[Tuple[int, int]]) -> List[List[int]]:
        if not edges:
            return []
        # Build minimal face list to reuse existing chain extractor logic.
        # We'll directly build adjacency and walk here to avoid allocating fake triangles.
        adjacency: Dict[int, List[int]] = {}
        for u, v in edges:
            adjacency.setdefault(u, []).append(v)
            adjacency.setdefault(v, []).append(u)

        seen: set[int] = set()
        chains_local: List[List[int]] = []
        for start in adjacency:
            if start in seen:
                continue
            stack = [start]
            component: List[int] = []
            seen.add(start)
            while stack:
                n = stack.pop()
                component.append(n)
                for nb in adjacency.get(n, []):
                    if nb not in seen:
                        seen.add(nb)
                        stack.append(nb)

            comp_set = set(component)
            degree = {n: len([nb for nb in adjacency[n] if nb in comp_set]) for n in component}
            endpoints = [n for n, deg in degree.items() if deg == 1]
            cur = endpoints[0] if endpoints else component[0]
            prev = None
            chain = [cur]
            visited_local: set[int] = {cur}
            while True:
                nxt_candidates = [nb for nb in adjacency[cur] if nb in comp_set and nb != prev]
                if not nxt_candidates:
                    break
                nxt = None
                for cand in nxt_candidates:
                    if cand not in visited_local:
                        nxt = cand
                        break
                if nxt is None:
                    nxt = nxt_candidates[0]
                prev, cur = cur, nxt
                if cur == chain[0]:
                    chain.append(cur)
                    break
                chain.append(cur)
                visited_local.add(cur)
                if endpoints and cur in endpoints[1:]:
                    break

            if len(chain) >= 2 and chain[0] == chain[-1]:
                chains_local.append(chain[:-1])
            elif len(chain) >= 2:
                chains_local.append(chain)

        return chains_local

    outer_chains = _merge_chains_by_endpoint_proximity(
        vertices,
        chains_from_edges(outer_edges),
        tol_mm=endpoint_merge_tol_mm,
    )
    inner_chains = _merge_chains_by_endpoint_proximity(
        vertices,
        chains_from_edges(inner_edges),
        tol_mm=endpoint_merge_tol_mm,
    )

    if not outer_chains or not inner_chains:
        return []

    def chain_endpoints(chain: List[int]) -> Tuple[int, int]:
        return chain[0], chain[-1]

    def endpoint_cost(a: List[int], b: List[int]) -> Tuple[float, bool]:
        a0, a1 = chain_endpoints(a)
        b0, b1 = chain_endpoints(b)
        a0p = vertices[a0][[0, 2]]
        a1p = vertices[a1][[0, 2]]
        b0p = vertices[b0][[0, 2]]
        b1p = vertices[b1][[0, 2]]
        cost_same = float(np.sum((a0p - b0p) ** 2) + np.sum((a1p - b1p) ** 2))
        cost_flip = float(np.sum((a0p - b1p) ** 2) + np.sum((a1p - b0p) ** 2))
        return (min(cost_same, cost_flip), cost_flip < cost_same)

    # Greedy matching of outer to inner chains by endpoint proximity.
    pairs: List[Tuple[List[int], List[int]]] = []
    unused_inner = set(range(len(inner_chains)))
    for outer in sorted(outer_chains, key=len, reverse=True):
        best_j = None
        best_cost = None
        best_flip = False
        for j in unused_inner:
            cost, flip = endpoint_cost(outer, inner_chains[j])
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_j = j
                best_flip = flip
        if best_j is None:
            continue
        inner = inner_chains[best_j]
        if best_flip:
            inner = list(reversed(inner))
        pairs.append((outer, inner))
        unused_inner.remove(best_j)

    if not pairs:
        return []

    def add_oriented_face(out_faces: List[Tuple[int, int, int]], tri: Tuple[int, int, int]) -> None:
        a, b, c = tri
        if a == b or b == c or a == c:
            return
        p0, p1, p2 = vertices[a], vertices[b], vertices[c]
        nrm = np.cross(p1 - p0, p2 - p0)
        if np.linalg.norm(nrm) < 1e-12:
            return
        if np.dot(nrm, target_dir) < 0:
            out_faces.append((a, c, b))
        else:
            out_faces.append((a, b, c))

    cap_faces: List[Tuple[int, int, int]] = []
    for outer, inner in pairs:
        if len(outer) < 2 or len(inner) < 2:
            continue
        i = 0
        j = 0
        while i < len(outer) - 1 or j < len(inner) - 1:
            if i == len(outer) - 1:
                tri = (outer[i], inner[j + 1], inner[j])
                j += 1
                add_oriented_face(cap_faces, tri)
                continue
            if j == len(inner) - 1:
                tri = (outer[i], outer[i + 1], inner[j])
                i += 1
                add_oriented_face(cap_faces, tri)
                continue

            o0 = outer[i]
            o1 = outer[i + 1]
            i0 = inner[j]
            i1 = inner[j + 1]
            d_outer = float(np.sum((vertices[o1][[0, 2]] - vertices[i0][[0, 2]]) ** 2))
            d_inner = float(np.sum((vertices[o0][[0, 2]] - vertices[i1][[0, 2]]) ** 2))
            if d_outer <= d_inner:
                tri = (o0, o1, i0)
                i += 1
                add_oriented_face(cap_faces, tri)
            else:
                tri = (o0, i1, i0)
                j += 1
                add_oriented_face(cap_faces, tri)

    return cap_faces


def build_plane_cap_faces(
    vertices: np.ndarray,
    faces: List[Tuple[int, int, int]],
    y_plane: float,
    keep_above: bool = True,
    eps: float = 1e-6,
) -> List[Tuple[int, int, int]]:
    """Triangulate a planar cap between outer and inner boundary loops."""
    loops = extract_plane_boundary_loops(vertices, faces, y_plane, eps=eps)
    if not loops:
        return []

    loop_areas = []
    for loop in loops:
        pts_2d = np.array([[vertices[idx][0], vertices[idx][2]] for idx in loop])
        loop_areas.append(polygon_area_2d(pts_2d))

    outer_idx = int(np.argmax([abs(a) for a in loop_areas]))
    outer_loop = loops[outer_idx]
    hole_loops = [loops[i] for i in range(len(loops)) if i != outer_idx]

    outer_pts = np.array([[vertices[idx][0], vertices[idx][2]] for idx in outer_loop])
    hole_pts = [np.array([[vertices[idx][0], vertices[idx][2]] for idx in loop]) for loop in hole_loops]

    _, cap_faces = triangulate_with_holes(outer_pts, hole_pts)
    if len(cap_faces) == 0:
        return []

    cap_vertex_indices = outer_loop + [idx for loop in hole_loops for idx in loop]
    target_dir = np.array([0.0, -1.0 if keep_above else 1.0, 0.0])
    out_faces: List[Tuple[int, int, int]] = []
    for tri in cap_faces:
        a, b, c = (cap_vertex_indices[int(tri[0])],
                   cap_vertex_indices[int(tri[1])],
                   cap_vertex_indices[int(tri[2])])
        p0, p1, p2 = vertices[a], vertices[b], vertices[c]
        nrm = np.cross(p1 - p0, p2 - p0)
        if np.dot(nrm, target_dir) < 0:
            out_faces.append((a, c, b))
        else:
            out_faces.append((a, b, c))
    return out_faces


def create_front_airfoil(
    section_indices: List[Tuple[int, int, int, int]],
    spine_vertices: np.ndarray,
    thetas: np.ndarray,
    theta_start: float,
    front_start_deg: float,
    front_end_deg: float,
    fender_depth: float,
    fender_half_width: float,
    spine_thickness: float,
    louver_length: float,
    forward_extension_deg: float = 0.0,
    airfoil_thickness_ratio: float = 0.18,
    airfoil_camber: float = 0.02,
    airfoil_camber_position: float = 0.35,
    kammback_start: float = 0.75,
    airfoil_points: int = 20,
    boost_front_edge_points: bool = True,
    front_edge_points_min: int = 96,
    airfoil_thickness_mm: Optional[float] = None,
    chord_length_mm: Optional[float] = None,
    wall_thickness_mm: float = 3.0,
) -> Tuple[np.ndarray, List[Tuple[int, int, int]]]:
    """
    Create a hollow Kammback airfoil, with the front section truncated parallel
    to the ground plane using exact geometric plane intersection for a perfectly
    flat, watertight front cap.
    """
    if fender_depth <= 1e-6 or fender_half_width <= 0.0:
        return np.zeros((0, 3)), []

    # Section setup code
    front_start = max(0.0, front_start_deg)
    front_end = max(front_start, front_end_deg)
    arc_degrees = np.degrees(thetas - theta_start)

    valid_indices = [
        idx for idx, deg in enumerate(arc_degrees) if front_start - 1e-6 <= deg <= front_end + 1e-6
    ]
    if len(valid_indices) < 2:
        return np.zeros((0, 3)), []

    actual_chord_length = chord_length_mm if chord_length_mm is not None else fender_depth
    effective_thickness_ratio = airfoil_thickness_ratio
    if airfoil_thickness_mm is not None and actual_chord_length > 1e-6:
        effective_thickness_ratio = max(airfoil_thickness_mm / actual_chord_length, 1e-4)

    thickness_half_extent = fender_half_width
    thickness_inner_extent = max(fender_half_width - wall_thickness_mm, 0.0)
    section_points = max(airfoil_points, front_edge_points_min) if boost_front_edge_points else airfoil_points

    # --- Part 1: Generate all vertices for the unclipped shape ---
    unclipped_vertices: List[List[float]] = []
    unclipped_loops: List[Dict[str, List[int]]] = []
    section_trailing_indices: List[int] = []

    for idx in valid_indices:
        theta = thetas[idx]
        v0_idx, v1_idx, v2_idx, v3_idx = section_indices[idx]
        
        # Calculate the actual radial direction from the smoothed spine center
        # Get the spine center from all four vertices
        spine_center = (spine_vertices[v0_idx] + spine_vertices[v1_idx] + 
                       spine_vertices[v2_idx] + spine_vertices[v3_idx]) / 4.0
        
        # Radial direction (outward) derived from actual smoothed spine geometry
        # Take it from the 2D (XY) projection of the center point
        n_dir_xy = spine_center[:2]  # Just X, Y components
        n_norm = np.linalg.norm(n_dir_xy)
        if n_norm > 1e-9:
            n_dir = np.array([n_dir_xy[0] / n_norm, n_dir_xy[1] / n_norm, 0.0])
        else:
            # Fallback to theoretical direction if center is at origin
            n_dir = np.array([math.cos(theta), math.sin(theta), 0.0])
        
        # Calculate base position from the outer spine vertices (v2, v3)
        # Deepen the connection: sink the airfoil base INTO the spine.
        # This increases the mechanical overlap for better structural strength
        # while ensuring the airfoil profile does not protrude above the outer spine face.
        spine_outer_mid = (spine_vertices[v2_idx] + spine_vertices[v3_idx]) / 2.0
        base_mid = spine_outer_mid - n_dir * (wall_thickness_mm / 2.0)
        
        # Tangent direction (perpendicular to radial, in XY plane)
        t_dir = np.array([-n_dir[1], n_dir[0], 0.0])
        span_dir = -t_dir / np.linalg.norm(t_dir)
        chord_dir = -n_dir
        thickness_dir = np.array([0.0, 0.0, 1.0])

        upper, lower = create_airfoil_profile(
            chord_length=actual_chord_length, max_thickness=effective_thickness_ratio,
            max_camber=airfoil_camber, camber_position=airfoil_camber_position,
            n_points=section_points, kammback_start=kammback_start,
        )
        profile_outer = np.vstack([upper, lower[::-1]])[:-1]
        n_profile = profile_outer.shape[0]
        normals_raw = np.zeros_like(profile_outer)
        trailing_idx = len(upper) - 1
        
        for i in range(n_profile):
            tangent = profile_outer[(i + 1) % n_profile] - profile_outer[(i - 1) % n_profile]
            norm_tangent = np.linalg.norm(tangent)
            if norm_tangent > 1e-9:
                 normals_raw[i] = np.array([-tangent[1], tangent[0]]) / norm_tangent
        
        profile_inner = profile_outer - normals_raw * wall_thickness_mm
        
        # Force inner trailing edge to have same X-coordinate as outer trailing edge
        # to maintain perfectly vertical Kammback edge
        # Profile structure: [upper[0..trailing_idx], lower[reversed][0..end]][:-1]
        # - upper trailing edge: index = len(upper) - 1
        # - lower trailing edge: index = len(upper) (first point of reversed lower)
        upper_te_idx = trailing_idx
        lower_te_idx = len(upper)  # First point of reversed lower array
        te_x = profile_outer[upper_te_idx, 0]  # X-coordinate of trailing edge
        
        # Lock inner trailing edge X-coordinates while preserving Y offset
        profile_inner[upper_te_idx, 0] = te_x
        profile_inner[lower_te_idx, 0] = te_x
        
        section_trailing_indices.append(trailing_idx)
        loops: Dict[str, List[int]] = {"outer_neg": [], "outer_pos": [], "inner_neg": [], "inner_pos": []}
        
        base_v_idx = len(unclipped_vertices)
        
        # Generate all points for this section
        section_points_data = []
        for pt_idx in range(n_profile):
            outer_offset = chord_dir * profile_outer[pt_idx, 0] + thickness_dir * profile_outer[pt_idx, 1]
            inner_offset = chord_dir * profile_inner[pt_idx, 0] + thickness_dir * profile_inner[pt_idx, 1]
            for span_sign, label in ((-1.0, "neg"), (1.0, "pos")):
                # Both inner and outer vertices must use the same Z-coordinate (span position)
                # The offset is only in the radial/vertical plane (XY plane of the airfoil)
                v_outer = base_mid + outer_offset + span_dir * (span_sign * thickness_half_extent)
                v_inner = base_mid + inner_offset + span_dir * (span_sign * thickness_half_extent)
                section_points_data.append((v_outer.tolist(), v_inner.tolist(), label))
        
        unclipped_vertices.extend([p for v_outer, v_inner, _ in section_points_data for p in (v_outer, v_inner)])

        # Populate loop indices
        for i in range(n_profile):
            _, _, label = section_points_data[i * 2] # neg
            loops[f"outer_{label}"].append(base_v_idx + i*4)
            loops[f"inner_{label}"].append(base_v_idx + i*4 + 1)
            _, _, label = section_points_data[i * 2 + 1] # pos
            loops[f"outer_{label}"].append(base_v_idx + i*4 + 2)
            loops[f"inner_{label}"].append(base_v_idx + i*4 + 3)

        unclipped_loops.append(loops)
    
    if not unclipped_loops: 
        return np.zeros((0, 3)), []
    
    unclipped_vertices_np = np.array(unclipped_vertices)
    base_vertex_count = int(unclipped_vertices_np.shape[0])

    # --- Part 2: Determine clipping plane from the HIGHEST leading edge point ---
    front_loops = unclipped_loops[0]
    leading_edge_y_values = []
    leading_edge_labels = ["outer_neg", "outer_pos", "inner_neg", "inner_pos"]
    for key in leading_edge_labels:
        le_idx = front_loops[key][0]  # First point is leading edge
        leading_edge_y_values.append(unclipped_vertices_np[le_idx, 1])
    
    # Use MAXIMUM Y-value of the leading edge. This ensures the clip plane passes
    # through the airfoil on BOTH span sides, creating a complete horseshoe cap.
    # (Using minimum only clips one side when the airfoil is tilted.)
    y_clip_plane = max(leading_edge_y_values)
    
    print(f"\n=== GEOMETRIC PLANE INTERSECTION ===")
    print(f"  Leading edge Y values:")
    for label, y_val in zip(leading_edge_labels, leading_edge_y_values):
        print(f"    {label}: {y_val:.4f} mm")
    print(f"  Clip plane Y: {y_clip_plane:.4f} mm (based on HIGHEST leading edge point)")
    print(f"  Strategy: Triangle clipping with exact parametric edge intersection")
    print("=== END GEOMETRIC PLANE INTERSECTION ===\n")
    
    # --- Part 3: Build final vertex list and clip geometry ---
    # Start with all unclipped vertices as a mutable list
    final_vertices_list: List[np.ndarray] = [v.copy() for v in unclipped_vertices_np]
    
    # Snap any vertices very close to the plane exactly onto it
    PLANE_SNAP_EPS = 1e-6
    for i in range(len(final_vertices_list)):
        if abs(final_vertices_list[i][1] - y_clip_plane) <= PLANE_SNAP_EPS:
            final_vertices_list[i][1] = y_clip_plane
    
    # Helper to validate and add faces
    def _add_face_if_valid(v_indices: Tuple[int, int, int], face_list: List[Tuple[int, int, int]]):
        if v_indices[0] == v_indices[1] or v_indices[1] == v_indices[2] or v_indices[0] == v_indices[2]: 
            return
        # Will validate normal after vertices are finalized
        face_list.append(v_indices)
    
    faces: List[Tuple[int, int, int]] = []
    
    # Cache for edge-plane intersection vertices to keep shared edges watertight
    edge_vertex_cache: Dict[Tuple[int, int], int] = {}
    
    # Separate cache for ONLY the front section (section 0) intersections
    # This is used to build the horseshoe cap without picking up edges from other sections
    front_section_edge_cache: Dict[Tuple[int, int], int] = {}

    # --- Part 4: Build wall faces between sections (with triangle clipping) ---
    print(f"  Clipping wall faces...")
    clipped_face_count = 0
    total_face_count = 0
    
    for sec in range(len(unclipped_loops) - 1):
        # Track cache size before this section to identify new intersections
        cache_size_before = len(edge_vertex_cache)
        
        loop_a, loop_b = unclipped_loops[sec], unclipped_loops[sec + 1]
        n_points = len(loop_a["outer_neg"])
        trailing_idx = section_trailing_indices[sec]
        
        for p in range(n_points):
            if p == trailing_idx: 
                continue
            p_next = (p + 1) % n_points
            
            for key, flip in [("outer_neg", False), ("outer_pos", False), ("inner_neg", True), ("inner_pos", True)]:
                a, b, c, d = loop_a[key][p], loop_a[key][p_next], loop_b[key][p_next], loop_b[key][p]
                
                # Split quad into two triangles (respect winding)
                tris = [(a, d, c), (a, c, b)] if not flip else [(a, b, c), (a, c, d)]
                
                for t0, t1, t2 in tris:
                    total_face_count += 1
                    clipped = slice_tri_against_yplane(
                        t0,
                        t1,
                        t2,
                        final_vertices_list,
                        y_clip_plane,
                        keep_above=True,
                        edge_cache=edge_vertex_cache,
                    )
                    if len(clipped) < len([1]):  # Some clipping occurred
                        clipped_face_count += 1
                    for tri in clipped:
                        _add_face_if_valid(tri, faces)
        
        # Capture intersection vertices from section 0 (front section) only
        if sec == 0:
            for edge_key, v_idx in edge_vertex_cache.items():
                front_section_edge_cache[edge_key] = v_idx
    
    print(f"    Total faces processed: {total_face_count}, clipped: {clipped_face_count}")
    
    # IMPORTANT: Rebuild final_vertices array after adding intersection vertices
    final_vertices = np.array(final_vertices_list)
    print(f"    Final vertex count: {len(final_vertices)}")
    
    # --- Part 6: Add rear cap and wall between inner/outer surfaces ---
    if unclipped_loops:
        loops = unclipped_loops[-1]
        theta = thetas[valid_indices[-1]]
        target_dir = np.array([-math.sin(theta), math.cos(theta), 0.0])
        trailing = section_trailing_indices[-1]
        n_pts = len(loops["outer_neg"])
        
        for label in ("neg", "pos"):
            outer, inner = loops[f"outer_{label}"], loops[f"inner_{label}"]
            for p in range(n_pts):
                if p == trailing: 
                    continue
                p_next = (p + 1) % n_pts
                
                v0, v1, v2, v3 = outer[p], outer[p_next], inner[p_next], inner[p]
                p0, p1, p2 = final_vertices[v0], final_vertices[v1], final_vertices[v2]
                nrm = np.cross(p1 - p0, p2 - p0)
                
                if np.dot(nrm, target_dir) < 0:
                    _add_face_if_valid((v0, v2, v1), faces)
                    _add_face_if_valid((v0, v3, v2), faces)
                else:
                    _add_face_if_valid((v0, v1, v2), faces)
                    _add_face_if_valid((v0, v2, v3), faces)

    # --- Part 7: Add front cap ring between outer and inner shells ---
    # Now that the clip plane is at the MAXIMUM leading edge Y, both span sides
    # are clipped, creating a complete horseshoe of intersection vertices.
    print(f"  Front section edge cache: {len(front_section_edge_cache)} intersection vertices")
    
    front_cap_faces: List[Tuple[int, int, int]] = []
    try:
        front_cap_faces = build_plane_rim_cap_faces(
            final_vertices,
            faces,
            y_clip_plane,
            base_vertex_count=base_vertex_count,
            intersection_edge_cache=front_section_edge_cache,  # Use ONLY front section intersections
            keep_above=True,
            eps=PLANE_SNAP_EPS,
        )
        if not front_cap_faces:
            # Fallback: if the boundary forms closed loops, use polygon+holes capping.
            front_cap_faces = build_plane_cap_faces(
                final_vertices,
                faces,
                y_clip_plane,
                keep_above=True,
                eps=PLANE_SNAP_EPS,
            )
    except RuntimeError as exc:
        print(f"  Front cap skipped: {exc}")

    if front_cap_faces:
        faces.extend(front_cap_faces)
        print(f"  Front cap faces: {len(front_cap_faces)}")

    print(f"  Total faces generated: {len(faces)}")
    return final_vertices, faces


def create_louvers_pair(
    section_indices: List[Tuple[int, int, int, int]],
    spine_vertices: np.ndarray,
    thetas: np.ndarray,
    spine_width: float,
    spine_thickness: float,
    spine_inner_radius: float,
    coverage_angle_deg: float,
    louver_length: float,
    louver_depth: float,
    louver_spacing: float = 0.5,
    tip_truncation: float = 0.70,  # Maintain 70% chord at tip (vs 0% for elliptical)
    tip_fraction: float = 0.3,
    tilt_angle_deg: float = 5.0,
    rake_angle_deg: float = 1.0,
    min_active_angle_deg: float = 0.0,
    max_active_angle_deg: Optional[float] = None,
    both_sides: bool = True,
    louver_thickness: Optional[float] = None,
    airfoil_mode: bool = True,
    airfoil_thickness_ratio: float = 0.12,
    airfoil_camber: float = 0.03,
    airfoil_camber_position: float = 0.4,
    kammback_start: float = 0.85,
    airfoil_points: int = 16,
    wing_fence: bool = False,
    fence_position: float = 0.65,
    fence_height: float = 1.0,
    fence_chord_fraction: float = 0.65,
    min_airfoil_thickness_mm: Optional[float] = 2.0,
    tilt_profile: Optional[Callable[[float], float]] = None,
    rake_profile: Optional[Callable[[float], float]] = None,
    airfoil_thickness_profile: Optional[Callable[[float], float]] = None,
    airfoil_camber_profile: Optional[Callable[[float], float]] = None,
    airfoil_camber_position_profile: Optional[Callable[[float], float]] = None,
    kammback_profile: Optional[Callable[[float], float]] = None,
) -> Tuple[np.ndarray, List[Tuple[int, int, int]]]:
    """
    Construct louver blades attached along the spine with optional overlapping and
    symmetrical placement on both sides of the spine. Each louver is attached at a
    discrete position along the curved spine, extends outward along the width
    direction, and may overlap the preceding louver to form a continuous
    barrier.  When ``both_sides`` is true, identical louvers are produced
    symmetrically on the left and right of the spine.

    Parameters
    ----------
    section_indices : list of tuple
        Indices (v0, v1, v2, v3) into the spine vertex array for each
        segment.  Only v2 and v3 (outer top and bottom) are used to derive
        attachment points.
    spine_vertices : np.ndarray
        Array of spine vertices (shape (N,3)).
    thetas : np.ndarray
        Array of angular positions corresponding to each spine segment.
    spine_width : float
        Radial width of the spine cross‑section, in mm.
    spine_thickness : float
        Vertical thickness of the spine cross‑section, in mm.  Used to
        determine attachment points; louver thickness is set independently.
    spine_inner_radius : float
        Radius from the wheel centre to the inner surface of the spine, in mm.
    coverage_angle_deg : float
        Total angular coverage of the spine in degrees (including any forward
        extension).  Measured from the forward-most point of the spine.
    min_active_angle_deg : float, optional
        Minimum arc angle (degrees from the forward-most spine point) at which
        louvers should begin.  Louvers positioned before this angle are skipped.
    max_active_angle_deg : float, optional
        Optional arc angle limit (degrees) beyond which louvers are omitted.
        Use ``None`` to allow louvers along the entire remaining coverage.
    louver_length : float
        Total length of each louver measured along the extrude direction (bike width), in mm.
    louver_depth : float
        Radial depth of each louver at its base (distance from the spine
        towards the wheel), in mm.
    louver_spacing : float, optional
        Spacing between louvers as a fraction of louver_depth (default 0.5).
        Negative values create overlap (e.g., -0.5 means 50% overlap).
        Zero means edge-to-edge contact.
        Positive values create gaps (e.g., 0.2 means 20% of depth gap).
    tip_fraction : float, optional
        Fraction of the louver length used for the tapered tip (default 0.3).
    tilt_angle_deg : float, optional
        Downward tilt angle from horizontal (positive tilts downward toward wheel), 
        in degrees (default 25.0).
    rake_angle_deg : float, optional
        Backward rake angle (positive leans backward), in degrees (default 10.0).
    both_sides : bool, optional
        If true, louvers are generated on both sides of the spine; if false,
        only the left side is produced.
    louver_thickness : float, optional
        Vertical thickness of each louver blade at its base, in mm.  If not
        specified, defaults to the spine_thickness for backwards compatibility.
    airfoil_mode : bool, optional
        If True (default), use airfoil-shaped cross-sections for aerodynamic
        performance. If False, use simple rectangular cross-sections.
    airfoil_thickness_ratio : float, optional
        Maximum thickness of airfoil as fraction of depth (default 0.12 = 12%).
        This creates a streamlined profile for low drag.
    airfoil_camber : float, optional
        Maximum camber (curvature) as fraction of depth (default 0.03 = 3%).
        Positive camber helps direct airflow downward and reduces separation.
    airfoil_camber_position : float, optional
        Position of maximum camber along the chord (default 0.4 = 40%).
        Forward positions give gentler pressure gradients.
    kammback_start : float, optional
        Position (0-1) where Kammback trailing edge truncation begins (default 0.85).
        The airfoil is cut at this point to create a blunt, printable trailing edge.
        Values of 0.80-0.90 work well - earlier cuts are more printable, later cuts
        are slightly more aerodynamic.
    airfoil_points : int, optional
        Number of points defining each airfoil cross-section (default 16).
        More points give smoother surfaces but increase polygon count.
    wing_fence : bool, optional
        If True, add wing fences to block span-wise flow from tips toward spine.
        Fences are small vertical fins that prevent flow migration and reduce
        induced drag in the inward-pressure environment (default False).
    fence_position : float, optional
        Position of wing fence along span as fraction from spine to tip (default 0.65).
        Typical values 0.60-0.70 block span-wise flow while allowing attachment.
    fence_height : float, optional
        Height of wing fence extending above/below airfoil surface in mm (default 1.0).
        Should be 1.5-3mm for effective flow blocking without excessive drag.
    fence_chord_fraction : float, optional
        Fraction of chord covered by fence (default 0.65 = 65%).
        Fence typically starts at leading edge and extends 30-50% of chord.
    min_airfoil_thickness_mm : float, optional
        Minimum target thickness (in mm) for airfoil sections along the louver span.
        The local thickness ratio is increased as needed to maintain at least this
        thickness at the current chord length.  Set to None to disable.
    tilt_profile : Callable[[float], float], optional
        Optional function mapping arc angle in degrees (0 at top of coverage) to
        local downward tilt angle in degrees. When provided it overrides the
        constant ``tilt_angle_deg``.
    rake_profile : Callable[[float], float], optional
        Optional function mapping arc angle to local backward rake angle in degrees.
    airfoil_thickness_profile : Callable[[float], float], optional
        Optional function providing the local airfoil thickness ratio as a function
        of arc angle.
    airfoil_camber_profile : Callable[[float], float], optional
        Optional function providing the local airfoil camber ratio.
    airfoil_camber_position_profile : Callable[[float], float], optional
        Optional function providing the local camber position fraction.
    kammback_profile : Callable[[float], float], optional
        Optional function providing the local Kammback start fraction.

    Returns
    -------
    Tuple[np.ndarray, List[Tuple[int,int,int]]]
        A tuple containing all louver vertices and faces.  These indices are
        local and should be offset appropriately when combining with the
        spine geometry.
    """
    # Use spine_thickness as default louver thickness if not specified
    actual_louver_thickness = louver_thickness if louver_thickness is not None else spine_thickness
    
    # Compute the radius to the outside face of the spine where louvers attach
    spine_centre_radius = spine_inner_radius + spine_width / 2.0
    total_coverage_rad = math.radians(coverage_angle_deg)
    arc_length = spine_centre_radius * total_coverage_rad

    # Angular mapping along the spine
    theta_start = thetas[0]
    theta_end = thetas[-1]
    theta_range = theta_end - theta_start

    # Determine louver centre positions along the arc
    # louver_spacing is a fraction of louver_depth:
    #   Negative: overlap (e.g., -0.5 means 50% overlap)
    #   Zero: edge-to-edge contact
    #   Positive: gaps (e.g., 0.2 means 20% gap)
    effective_spacing = louver_depth * louver_spacing
    if effective_spacing <= 1e-6 and louver_spacing >= 0:
        effective_spacing = louver_depth * 0.5 if louver_depth > 0 else 1.0

    total_arc_deg = math.degrees(theta_range) if theta_range != 0.0 else 0.0
    start_angle_clamped = max(0.0, min(min_active_angle_deg, total_arc_deg)) if total_arc_deg > 0.0 else 0.0
    if max_active_angle_deg is None:
        end_angle_clamped = total_arc_deg
    else:
        end_angle_clamped = max(start_angle_clamped, min(max_active_angle_deg, total_arc_deg)) if total_arc_deg > 0.0 else 0.0

    # Account for radial extension of louver leading edge
    if spine_centre_radius > 0:
        # Calculate angular offset caused by radial extension
        inner_radius = spine_centre_radius - louver_depth
        if inner_radius > 0:
            # Small angle approximation for typical fender geometries
            angular_offset_rad = louver_depth / spine_centre_radius
            angular_offset_deg = math.degrees(angular_offset_rad)
        else:
            # Louver depth exceeds spine radius - use conservative estimate
            angular_offset_deg = math.degrees(louver_depth / max(spine_centre_radius, 1.0))
        
        # Add safety margin of 50% to account for airfoil shape effects
        angular_offset_deg *= 1.5
        
        # Apply the correction to prevent overlap
        start_angle_clamped = start_angle_clamped + angular_offset_deg

    if arc_length <= 1e-6 or theta_range == 0.0:
        centre_positions = [0.0]
    else:
        start_offset = louver_depth / 2.0
        end_offset = louver_depth / 2.0
        start_position = (start_angle_clamped / total_arc_deg) * arc_length if total_arc_deg > 0 else 0.0
        end_position = (end_angle_clamped / total_arc_deg) * arc_length if total_arc_deg > 0 else arc_length
        limit = min(arc_length - end_offset, end_position)
        pos = max(start_position + start_offset, start_offset)
        centre_positions = []
        min_step = max(1e-3, louver_depth * 0.25)
        high_tilt_ref = max(abs(tilt_angle_deg), 18.0)

        while pos <= limit + 1e-6:
            centre_positions.append(pos)
            theta_here = theta_start + (pos / arc_length) * theta_range
            angle_here_deg = math.degrees(theta_here - theta_start)
            local_tilt_val = tilt_profile(angle_here_deg) if tilt_profile else tilt_angle_deg
            tilt_norm = min(abs(local_tilt_val) / max(high_tilt_ref, 1e-3), 1.0)
            spacing_scale = 0.7 + 0.3 * tilt_norm  # tighter spacing near zero tilt
            step = max(effective_spacing * spacing_scale, min_step)
            pos += step

        if not centre_positions:
            mid_angle = (start_angle_clamped + end_angle_clamped) * 0.5
            centre_positions = [
                (mid_angle / total_arc_deg) * arc_length if total_arc_deg > 0 else arc_length / 2.0
            ]

    vertices: List[List[float]] = []
    faces: List[Tuple[int, int, int]] = []
    taper_segments = 10
    z_dir = np.array([0.0, 0.0, 1.0])
    
    # Reference tangent direction at 0° (top of wheel, theta = pi/2)
    # This is the "parallel to airflow" reference for all louvers when tilt=0
    theta_reference = math.pi / 2.0  # Top of wheel
    t_dir_reference = np.array([-math.sin(theta_reference), math.cos(theta_reference), 0.0])
    t_dir_reference = t_dir_reference / np.linalg.norm(t_dir_reference)  # = [-1, 0, 0] pointing forward

    # Loop over each louver position
    for s_pos in centre_positions:
        # Map s_pos (arc length) to angle theta on the spine
        theta = theta_start + (s_pos / arc_length) * theta_range if arc_length > 0 else theta_start
        theta = max(theta_start, min(theta_end, theta))
        
        # Determine closest segment index
        frac = (theta - theta_start) / theta_range if theta_range != 0.0 else 0.0
        frac = max(0.0, min(1.0, frac))
        seg_index = int(round(frac * (len(section_indices) - 1)))
        v0_idx, v1_idx, v2_idx, v3_idx = section_indices[seg_index]
        
        # Use inner spine vertices (v0, v1) as louver attachment point
        # v0, v1 are at the inner curve (closer to wheel)
        # v2, v3 are at the outer curve (farther from wheel)
        v0 = spine_vertices[v0_idx]
        v1 = spine_vertices[v1_idx]
        base_mid = (v0 + v1) / 2.0
        
        # Vertical vector (spine thickness direction)
        vertical_vec = v1 - v0
        vertical_len = np.linalg.norm(vertical_vec)
        if vertical_len == 0:
            continue
        vertical_dir = vertical_vec / vertical_len
        
        # Local radial and tangential directions for this segment
        n_dir = np.array([math.cos(theta), math.sin(theta), 0.0])
        t_dir = np.array([-math.sin(theta), math.cos(theta), 0.0])
        n_dir = n_dir / np.linalg.norm(n_dir)
        t_dir = t_dir / np.linalg.norm(t_dir)

        # Determine which sides to generate (±Z direction, parallel to hub axis)
        side_signs = (+1.0, -1.0) if both_sides else (+1.0,)

        arc_deg = math.degrees(theta - theta_start)
        if arc_deg < start_angle_clamped - 1e-6 or arc_deg > end_angle_clamped + 1e-6:
            continue
        tilt_angle_local = tilt_profile(arc_deg) if tilt_profile else tilt_angle_deg
        rake_angle_local = rake_profile(arc_deg) if rake_profile else rake_angle_deg
        thickness_ratio_local = (
            airfoil_thickness_profile(arc_deg) if airfoil_thickness_profile else airfoil_thickness_ratio
        )
        camber_local = airfoil_camber_profile(arc_deg) if airfoil_camber_profile else airfoil_camber
        camber_pos_local = (
            airfoil_camber_position_profile(arc_deg)
            if airfoil_camber_position_profile
            else airfoil_camber_position
        )
        kammback_local = kammback_profile(arc_deg) if kammback_profile else kammback_start
        tilt_angle_rad = math.radians(tilt_angle_local)
        rake_angle_rad = math.radians(rake_angle_local)

        for width_sign in side_signs:
            # Base extrusion direction: along the Z axis (bike width, parallel to hub)
            base_extrude_dir = z_dir * width_sign
            
            # Apply tilt and rake rotations
            extrude_dir = rotate_vector(base_extrude_dir, t_dir, -tilt_angle_rad * width_sign)
            extrude_dir = rotate_vector(extrude_dir, n_dir, -rake_angle_rad * width_sign)
            
            # Normalize
            norm = np.linalg.norm(extrude_dir)
            if norm == 0.0:
                continue
            extrude_dir = extrude_dir / norm
            
            # The louver chord direction determines airfoil alignment
            # When tilt is 0, we want ALL louvers parallel to the reference tangent
            # at the top of the wheel (0° position), not following the local arc tangent
            if abs(tilt_angle_rad) < 1e-6:
                # No tilt: use the fixed reference tangent direction (parallel to airflow at top)
                # This ensures all louvers point the same direction regardless of arc position
                radial_dir = t_dir_reference.copy()
            else:
                # With tilt: rotate from the reference tangent direction
                # Use a consistent rotation axis (z_dir) for both sides to maintain symmetry
                base_radial = t_dir_reference.copy()
                radial_dir = rotate_vector(base_radial, z_dir, -tilt_angle_rad)
                rad_norm = np.linalg.norm(radial_dir)
                if rad_norm > 1e-6:
                    radial_dir = radial_dir / rad_norm
                else:
                    radial_dir = t_dir_reference
            
            # Precompute tip and base lengths for taper
            tip_len = louver_length * tip_fraction
            base_len = louver_length - tip_len
            
            # Sample along the louver length
            ts = np.linspace(0.0, louver_length, taper_segments + 1)
            
            # Calculate scale factors for taper
            if airfoil_mode:
                # Blunt truncated tip: constant chord with truncation at tip
                scale_factors = [1.0 if t <= base_len or tip_len == 0.0 else 
                                1.0 - ((t - base_len) / tip_len) * (1.0 - tip_truncation) 
                                for t in ts]
            else:
                # Linear taper for rectangular cross-sections
                scale_factors = [1.0 if t <= base_len or tip_len == 0.0 else 
                                max(0.0, 1.0 - (t - base_len) / tip_len) 
                                for t in ts]
            
            # Keep track of cross‑section vertices
            cs_idx: List[List[int]] = []
            
            if airfoil_mode:
                for t_val, sc in zip(ts, scale_factors):
                    centre = base_mid + extrude_dir * t_val
                    # Scale the louver depth (chord) for taper
                    scaled_depth = louver_depth * sc
                    thickness_ratio_effective = thickness_ratio_local
                    if (
                        min_airfoil_thickness_mm is not None
                        and scaled_depth > 1e-6
                    ):
                        target_ratio = min_airfoil_thickness_mm / scaled_depth
                        thickness_ratio_effective = max(thickness_ratio_effective, target_ratio)
                        thickness_ratio_effective = min(thickness_ratio_effective, 0.6)
                    
                    if scaled_depth > 0.1:
                        # Generate full airfoil profile with scaled chord
                        upper, lower = create_airfoil_profile(
                            chord_length=scaled_depth,
                            max_thickness=thickness_ratio_effective,
                            max_camber=camber_local,
                            camber_position=camber_pos_local,
                            n_points=airfoil_points,
                            kammback_start=kammback_local,
                        )
                        
                        # Transform airfoil points to 3D louver position
                        section_vertices = []
                        
                        # Upper surface points
                        for pt in upper:
                            pos_3d = centre + radial_dir * (scaled_depth - pt[0]) + vertical_dir * pt[1]
                            vertices.append(pos_3d.tolist())
                            section_vertices.append(len(vertices) - 1)
                        
                        # Lower surface points in reverse
                        for pt in reversed(lower):
                            pos_3d = centre + radial_dir * (scaled_depth - pt[0]) + vertical_dir * pt[1]
                            vertices.append(pos_3d.tolist())
                            section_vertices.append(len(vertices) - 1)
                        
                        cs_idx.append(section_vertices)
                    else:
                        # Very small chord at extreme tip - create minimal blunt cap
                        extra_half = min_airfoil_thickness_mm * 0.5 if min_airfoil_thickness_mm is not None else 0.0
                        half_thick = max(scaled_depth * thickness_ratio_effective * 0.5, extra_half, 0.4)
                        base_idx = len(vertices)
                        vertices.extend([
                            (centre + vertical_dir * half_thick).tolist(),
                            (centre - vertical_dir * half_thick).tolist(),
                        ])
                        cs_idx.append([base_idx, base_idx + 1])
            else:
                # Rectangular cross-sections
                for t_val, sc in zip(ts, scale_factors):
                    centre = base_mid + extrude_dir * t_val
                    v_scale = actual_louver_thickness * sc
                    if min_airfoil_thickness_mm is not None:
                        v_scale = max(v_scale, min_airfoil_thickness_mm)
                    r_scale = louver_depth * sc
                    v_vec = vertical_dir * v_scale
                    r_vec = radial_dir * r_scale
                    
                    # Four corners of the louver cross-section
                    outer_bottom = centre - 0.5 * v_vec
                    outer_top = centre + 0.5 * v_vec
                    inner_top = outer_top + r_vec
                    inner_bottom = outer_bottom + r_vec
                    
                    base_idx = len(vertices)
                    vertices.extend([
                        outer_bottom.tolist(),
                        outer_top.tolist(),
                        inner_top.tolist(),
                        inner_bottom.tolist(),
                    ])
                    cs_idx.append([base_idx, base_idx + 1, base_idx + 2, base_idx + 3])
            
            # Build faces between consecutive cross‑sections
            for i in range(len(cs_idx) - 1):
                verts_i = cs_idx[i]
                verts_j = cs_idx[i + 1]
                
                # Handle degenerate cases
                if len(verts_i) == 1 or len(verts_j) == 1:
                    if len(verts_i) == 1 and len(verts_j) > 1:
                        tip = verts_i[0]
                        for k in range(len(verts_j)):
                            k_next = (k + 1) % len(verts_j)
                            faces.append((tip, verts_j[k], verts_j[k_next]))
                    elif len(verts_j) == 1 and len(verts_i) > 1:
                        tip = verts_j[0]
                        for k in range(len(verts_i)):
                            k_next = (k + 1) % len(verts_i)
                            faces.append((verts_i[k], tip, verts_i[k_next]))
                    continue
                
                # Both sections have multiple vertices
                n_i = len(verts_i)
                n_j = len(verts_j)
                
                if n_i == n_j:
                    for k in range(n_i):
                        k_next = (k + 1) % n_i
                        v0, v1 = verts_i[k], verts_i[k_next]
                        v2, v3 = verts_j[k_next], verts_j[k]
                        faces.append((v0, v3, v2))
                        faces.append((v0, v2, v1))
                else:
                    # Different number of vertices - interpolate
                    max_n = max(n_i, n_j)
                    for k in range(max_n):
                        frac_i = k / max_n
                        frac_j = (k + 1) / max_n
                        idx_i0 = int(frac_i * n_i) % n_i
                        idx_i1 = int(frac_j * n_i) % n_i
                        idx_j0 = int(frac_i * n_j) % n_j
                        idx_j1 = int(frac_j * n_j) % n_j
                        
                        v0, v1 = verts_i[idx_i0], verts_i[idx_i1]
                        v2, v3 = verts_j[idx_j1], verts_j[idx_j0]
                        
                        if v0 != v1 and v2 != v3:
                            faces.append((v0, v3, v2))
                            if v1 != v2:
                                faces.append((v0, v2, v1))
            
            # Cap the base and tip
            if cs_idx and len(cs_idx[0]) > 2:
                verts = cs_idx[0]
                for k in range(1, len(verts) - 1):
                    faces.append((verts[0], verts[k], verts[k + 1]))
            
            if cs_idx and len(cs_idx[-1]) > 2:
                verts = cs_idx[-1]
                for k in range(1, len(verts) - 1):
                    faces.append((verts[0], verts[k + 1], verts[k]))
            
            # Add wing fence if enabled
            if wing_fence and len(cs_idx) > 2:
                fence_idx = int(fence_position * (len(cs_idx) - 1))
                fence_idx = max(1, min(fence_idx, len(cs_idx) - 2))
                fence_verts = cs_idx[fence_idx]
                
                if airfoil_mode and len(fence_verts) > 2:
                    num_fence_pts = max(2, int(len(fence_verts) * fence_chord_fraction / 2))
                    
                    # Upper and lower surface fence points
                    upper_start, upper_end = 0, min(num_fence_pts, len(fence_verts) // 2)
                    lower_start, lower_end = len(fence_verts) - 1, max(len(fence_verts) // 2, len(fence_verts) - num_fence_pts - 1)
                    
                    # Calculate fence offset
                    fence_normal = -extrude_dir * width_sign
                    fence_offset = fence_normal * fence_height
                    
                    # Upper fence edge
                    upper_fence_verts = []
                    for vi in range(upper_start, upper_end):
                        v_orig = np.array(vertices[fence_verts[vi]])
                        v_fence = v_orig + fence_offset
                        fence_v_idx = len(vertices)
                        vertices.append(v_fence.tolist())
                        upper_fence_verts.append(fence_v_idx)
                    
                    # Lower fence edge
                    lower_fence_verts = []
                    for vi in range(lower_start, lower_end, -1):
                        v_orig = np.array(vertices[fence_verts[vi]])
                        v_fence = v_orig + fence_offset
                        fence_v_idx = len(vertices)
                        vertices.append(v_fence.tolist())
                        lower_fence_verts.append(fence_v_idx)
                    
                    # Create fence faces
                    for i in range(len(upper_fence_verts) - 1):
                        v0, v1 = fence_verts[upper_start + i], fence_verts[upper_start + i + 1]
                        v2, v3 = upper_fence_verts[i + 1], upper_fence_verts[i]
                        faces.append((v0, v1, v2))
                        faces.append((v0, v2, v3))
                    
                    for i in range(len(lower_fence_verts) - 1):
                        v0, v1 = fence_verts[lower_start - i], fence_verts[lower_start - i - 1]
                        v2, v3 = lower_fence_verts[i + 1], lower_fence_verts[i]
                        faces.append((v0, v1, v2))
                        faces.append((v0, v2, v3))
                    
                    # Cap the fence edge
                    if len(upper_fence_verts) > 0 and len(lower_fence_verts) > 0:
                        all_fence_edge = upper_fence_verts + lower_fence_verts
                        for i in range(1, len(all_fence_edge) - 1):
                            faces.append((all_fence_edge[0], all_fence_edge[i], all_fence_edge[i + 1]))
                
                elif not airfoil_mode and len(fence_verts) == 4:
                    # Rectangular fence
                    v0, v1, v2, v3 = fence_verts
                    fence_normal = -extrude_dir * width_sign
                    fence_offset = fence_normal * fence_height
                    
                    v2_orig, v3_orig = np.array(vertices[v2]), np.array(vertices[v3])
                    v2_fence, v3_fence = v2_orig + fence_offset, v3_orig + fence_offset
                    
                    v2_fence_idx = len(vertices)
                    v3_fence_idx = len(vertices) + 1
                    vertices.extend([v2_fence.tolist(), v3_fence.tolist()])
                    
                    faces.append((v2, v3, v3_fence_idx))
                    faces.append((v2, v3_fence_idx, v2_fence_idx))

    return np.array(vertices), faces


def write_stl(
    filename: str,
    vertices: np.ndarray,
    faces: List[Tuple[int, int, int]],
    validate_and_repair: bool = True,
) -> None:
    """Write vertices and faces to a binary STL file.
    
    Parameters
    ----------
    filename : str
        Output STL filename.
    vertices : np.ndarray
        Array of vertex coordinates, shape (N, 3).
    faces : List[Tuple[int, int, int]]
        List of triangle faces as vertex index tuples.
    validate_and_repair : bool
        If True and PyVista is available, validate mesh quality and
        attempt to repair any issues (fill holes, clean duplicates).
    """
    if validate_and_repair and HAS_PYVISTA:
        # Build PyVista mesh from faces
        pv_faces = []
        for tri in faces:
            pv_faces.extend([3, int(tri[0]), int(tri[1]), int(tri[2])])
        pv_faces = np.array(pv_faces, dtype=np.int64)
        mesh = pv.PolyData(vertices.astype(np.float64), faces=pv_faces)
        
        # Clean mesh (merge duplicate points, remove degenerate cells)
        mesh = mesh.clean(tolerance=1e-6)
        
        # Check initial state
        print(f"\n=== MESH VALIDATION ===")
        print(f"  Initial: {mesh.n_points} points, {mesh.n_cells} cells")
        
        if mesh.n_points == 0 or mesh.n_cells == 0:
            print(f"  WARNING: Empty mesh, skipping validation")
            print(f"=== END MESH VALIDATION ===\n")
            mesh.save(filename)
            return
        
        print(f"  Is manifold: {mesh.is_manifold}")
        print(f"  Open edges: {mesh.n_open_edges}")
        
        # Check connectivity
        conn = mesh.connectivity()
        n_regions = len(set(conn.point_data['RegionId']))
        print(f"  Connected regions: {n_regions}")
        
        # Attempt to fill holes if not watertight
        if mesh.n_open_edges > 0:
            print(f"  Attempting hole fill...")
            mesh = mesh.fill_holes(hole_size=1000.0)
            mesh = mesh.clean(tolerance=1e-6)
            print(f"  After repair: {mesh.n_open_edges} open edges")
        
        # Final triangulation and cleanup
        mesh = mesh.triangulate().clean()
        
        # Final validation
        is_watertight = mesh.n_open_edges == 0
        print(f"  Final: {mesh.n_points} points, {mesh.n_cells} cells")
        print(f"  Watertight: {is_watertight}")
        if not is_watertight:
            print(f"  WARNING: Mesh still has {mesh.n_open_edges} open edges")
        print(f"=== END MESH VALIDATION ===\n")
        
        # Save using PyVista
        mesh.save(filename)
    else:
        # Original binary STL writer
        with open(filename, "wb") as f:
            # 80 byte header
            header_text = "Bicycle fender mesh generated by script"
            f.write(header_text.encode('ascii') + b" " * (80 - len(header_text)))
            # Number of triangles
            f.write(len(faces).to_bytes(4, byteorder="little"))
            for tri in faces:
                i0, i1, i2 = tri
                p0 = vertices[i0]
                p1 = vertices[i1]
                p2 = vertices[i2]
                # Compute normal vector for the triangle
                edge1 = p1 - p0
                edge2 = p2 - p0
                normal = np.cross(edge1, edge2)
                norm_len = np.linalg.norm(normal)
                if norm_len == 0:
                    normal = np.array([0.0, 0.0, 0.0])
                else:
                    normal = normal / norm_len
                # Write normal and vertices as 32-bit floats (little endian)
                f.write(normal.astype(np.float32).tobytes())
                f.write(p0.astype(np.float32).tobytes())
                f.write(p1.astype(np.float32).tobytes())
                f.write(p2.astype(np.float32).tobytes())
                # Attribute byte count (2 bytes)
                f.write((0).to_bytes(2, byteorder="little"))


def build_single_louver(
    louver_length: float = 40.0,
    louver_depth: float = 8.0,
    louver_thickness: float = 2.5,
    tip_fraction: float = 0.3,
    tilt_angle_deg: float = 25.0,
    airfoil_mode: bool = True,
    airfoil_thickness_ratio: float = 0.12,
    airfoil_camber: float = 0.03,
    airfoil_camber_position: float = 0.4,
    kammback_start: float = 0.85,
    airfoil_points: int = 16,
    output_filename: str = "single_louver.stl",
    min_airfoil_thickness_mm: Optional[float] = 2.0,
) -> None:
    """Build a single isolated louver blade pair for profile examination.
    
    This creates a pair of louver blades (left and right) positioned horizontally 
    for easy viewing of the airfoil cross-section profile. Useful for verifying 
    the aerodynamic shape before printing the full fender.
    
    Parameters
    ----------
    louver_length : float
        Total length of the louver in mm (bike width direction).
    louver_depth : float
        Radial depth (chord length) in mm.
    louver_thickness : float
        Thickness at base in mm (only used if airfoil_mode=False).
    tip_fraction : float
        Fraction of length used for tapered tip.
    tilt_angle_deg : float
        Downward tilt angle in degrees.
    airfoil_mode : bool
        Whether to use airfoil cross-sections.
    airfoil_thickness_ratio : float
        Airfoil max thickness as fraction of chord.
    airfoil_camber : float
        Airfoil camber as fraction of chord.
    airfoil_camber_position : float
        Position of max camber (0-1).
    kammback_start : float
        Position (0-1) where Kammback trailing edge begins.
    airfoil_points : int
        Number of points per airfoil section.
    output_filename : str
        Name of the output STL file.
    min_airfoil_thickness_mm : float, optional
        Minimum target thickness (mm) to maintain through the airfoil depth.
    """
    # Create minimal dummy spine with single segment at horizontal orientation
    spine_radius = 100.0  # Arbitrary radius for positioning
    spine_width = 5.0
    theta = 0.0  # Position at top (horizontal)
    
    # Create simple rectangular spine cross-section vertices
    n_dir = np.array([1.0, 0.0, 0.0])  # Radial outward
    base_mid = n_dir * spine_radius
    vertical_vec = np.array([0.0, 0.0, louver_thickness])
    
    v2 = base_mid + 0.5 * vertical_vec  # Outer top
    v3 = base_mid - 0.5 * vertical_vec  # Outer bottom
    
    spine_vertices = np.array([base_mid.tolist(), base_mid.tolist(), v2.tolist(), v3.tolist()])
    section_indices = [(0, 1, 2, 3)]
    thetas = np.array([theta])
    
    # Generate louvers using create_louvers_pair with single position
    louver_verts, louver_faces = create_louvers_pair(
        section_indices=section_indices,
        spine_vertices=spine_vertices,
        thetas=thetas,
        spine_width=spine_width,
        spine_thickness=louver_thickness,
        spine_inner_radius=spine_radius,
        coverage_angle_deg=0.1,  # Minimal coverage for single position
        louver_length=louver_length,
        louver_depth=louver_depth,
        louver_spacing=999.0,  # Large spacing ensures only one louver
        tip_truncation=0.60 if airfoil_mode else 0.0,
        tip_fraction=tip_fraction,
        tilt_angle_deg=tilt_angle_deg,
        rake_angle_deg=0.0,
        min_active_angle_deg=0.0,
        max_active_angle_deg=0.1,
        both_sides=True,
        louver_thickness=louver_thickness,
        airfoil_mode=airfoil_mode,
        airfoil_thickness_ratio=airfoil_thickness_ratio,
        airfoil_camber=airfoil_camber,
        airfoil_camber_position=airfoil_camber_position,
        kammback_start=kammback_start,
        airfoil_points=airfoil_points,
        wing_fence=False,
        min_airfoil_thickness_mm=min_airfoil_thickness_mm,
    )
    
    # Write to STL
    write_stl(output_filename, louver_verts, louver_faces)
    print(f"Single louver pair STL written to {output_filename}")


def calculate_front_airfoil_thickness(
    tire_width_mm: float,
    front_airfoil_tire_clearance_mm: float,
    front_airfoil_chord_mm: float,
    front_wall_thickness_mm: float,
    kammback_start: float = 0.75,
    target_thickness_ratio: float = 0.18,
    design_speed_kmh: float = 50.0,
    max_yaw_deg: float = 10.0,
) -> Tuple[float, float, Dict[str, float]]:
    """Calculate optimal front airfoil thickness based on tire and aerodynamic constraints.
    
    This function automatically determines the full thickness of the front airfoil
    cross-section to create a proper NACA-style airfoil profile that:
    1. Provides sufficient inner clearance for the tire to pass through even at the 
       narrowest trailing edge intersection (Kammback point).
    2. Creates an aerodynamically optimal shape for the design speed/yaw conditions
    
    For bicycle fenders at ~50 km/h with ±10° yaw, NACA 4-digit airfoils with
    15-20% thickness ratio provide excellent performance: thick enough to avoid
    flow separation at yaw angles, thin enough to minimize pressure drag.
    
    Parameters
    ----------
    tire_width_mm : float
        Nominal tire width in mm.
    front_airfoil_tire_clearance_mm : float
        Minimum clearance between tire and inner wall on each side (mm).
    front_airfoil_chord_mm : float
        Chord length of the airfoil (radial depth toward wheel) in mm.
    front_wall_thickness_mm : float
        Wall thickness of the hollow airfoil shell (mm).
    kammback_start : float, optional
        Position (0-1) where Kammback trailing edge truncation begins (default 0.75).
        Since the airfoil is narrower here, this point often determines tire clearance.
    target_thickness_ratio : float, optional
        Target thickness ratio (max thickness / chord). Default 0.18 (18%).
        For 50 km/h with yaw: 0.15-0.20 recommended.
    design_speed_kmh : float, optional
        Design airspeed in km/h. Higher speeds favor thinner profiles.
    max_yaw_deg : float, optional
        Maximum expected crosswind yaw angle in degrees. Larger yaw angles
        favor thicker profiles to delay flow separation.
    
    Returns
    -------
    Tuple[float, float, Dict[str, float]]
        - full_thickness_mm: Total thickness of the airfoil (outer wall to outer wall)
        - half_thickness_mm: Half-thickness (used for fender_half_width parameter)
        - details: Dictionary with calculation breakdown for debugging
    """
    # 1. Calculate minimum required inner half-gap for tire clearance
    # This is the narrowest point inside the hollow airfoil where the tire must fit
    required_inner_half = (tire_width_mm / 2.0) + front_airfoil_tire_clearance_mm
    
    # 2. Adjust thickness ratio based on operating conditions
    # At ~50 km/h (Re ~150,000 for 70mm chord), NACA airfoils work well at 15-20%
    # Crosswind yaw increases effective AoA, so thicker profiles delay stall
    # Rule of thumb: add ~0.5% thickness per degree of max yaw above 5°
    yaw_adjustment = max(0.0, (max_yaw_deg - 5.0) * 0.005)
    
    # Speed adjustment: higher speeds allow thinner profiles (better Re behavior)
    # Below 40 km/h, use thicker; above 60 km/h, can go thinner
    speed_adjustment = 0.0
    if design_speed_kmh < 40.0:
        speed_adjustment = 0.02 * (40.0 - design_speed_kmh) / 20.0
    elif design_speed_kmh > 60.0:
        speed_adjustment = -0.02 * (design_speed_kmh - 60.0) / 40.0
    
    effective_thickness_ratio = target_thickness_ratio + yaw_adjustment + speed_adjustment
    effective_thickness_ratio = max(0.12, min(0.25, effective_thickness_ratio))  # Clamp to sane range
    
    # 3. Calculate the ideal thickness from the aerodynamic profile
    # For a NACA 4-digit airfoil, max thickness = chord × thickness_ratio
    # But the airfoil is hollow, so the outer thickness includes:
    #   - Inner half-gap (for tire): required_inner_half
    #   - Wall thickness on each side: front_wall_thickness_mm
    #   - Additional bulge from airfoil curvature (max at ~30% chord for NACA)
    ideal_aero_thickness = front_airfoil_chord_mm * effective_thickness_ratio
    ideal_half_thickness = ideal_aero_thickness / 2.0
    
    # 4. The airfoil half-width must accommodate both constraints:
    #    - Aerodynamic: at least ideal_half_thickness for the profile shape
    #    - Clearance: at least required_inner_half + wall_thickness for tire passage
    # IMPORTANT: The tire must slide through the NARROWEST part of the airfoil,
    # which is the lagging edge (Kammback point). We must size the max thickness
    # such that the thickness at the Kammback truncation point still clears the tire.
    
    # Calculate the normalized thickness ratio at the Kammback truncation point
    # Compared to maximum thickness (which is at approx x=0.297 for NACA 4-digit)
    norm_thickness_kb = naca_4digit_thickness(kammback_start, 1.0)
    norm_thickness_max = naca_4digit_thickness(0.297, 1.0) # approx 0.5
    thickness_reduction_ratio = norm_thickness_kb / norm_thickness_max
    
    # Required half-thickness at the Kammback point to clear tire
    min_half_at_kb = required_inner_half + front_wall_thickness_mm
    
    # Scale this up to find the required maximum half-thickness
    min_structural_half = min_half_at_kb / thickness_reduction_ratio
    
    # Use the larger of the two constraints
    # This ensures the tire fits AND we have a proper airfoil shape
    final_half_thickness = max(ideal_half_thickness, min_structural_half)
    full_thickness_mm = 2.0 * final_half_thickness
    
    # 5. Build details dictionary for debugging and transparency
    details = {
        "tire_width_mm": tire_width_mm,
        "clearance_mm": front_airfoil_tire_clearance_mm,
        "chord_mm": front_airfoil_chord_mm,
        "wall_thickness_mm": front_wall_thickness_mm,
        "required_inner_half_mm": required_inner_half,
        "required_inner_gap_mm": required_inner_half * 2.0,
        "target_thickness_ratio": target_thickness_ratio,
        "effective_thickness_ratio": effective_thickness_ratio,
        "yaw_adjustment": yaw_adjustment,
        "speed_adjustment": speed_adjustment,
        "ideal_aero_thickness_mm": ideal_aero_thickness,
        "kammback_reduction_ratio": thickness_reduction_ratio,
        "min_structural_half_mm": min_structural_half,
        "final_half_thickness_mm": final_half_thickness,
        "full_thickness_mm": full_thickness_mm,
        "constraint": "aerodynamic" if ideal_half_thickness >= min_structural_half else "clearance",
        "actual_thickness_ratio": full_thickness_mm / front_airfoil_chord_mm if front_airfoil_chord_mm > 0 else 0,
    }
    
    return full_thickness_mm, final_half_thickness, details


def build_fender(
    wheel_diameter_mm: float = 622.0,
    tire_width_mm: float = 32.0,
    tire_radius_addition_mm: Union[float, str] = "auto",
    coverage_angle_deg: float = 100.0,
    forward_extension_deg: float = -40.0,
    front_airfoil_thickness_ratio: float = 0.18,
    front_airfoil_thickness_mm: Optional[float] = None,
    front_wall_thickness_mm: float = 3.0,
    front_airfoil_tire_clearance_mm: float = 4.0,
    front_airfoil_end_deg: float = 10.0,
    front_airfoil_camber: float = 0.02,
    front_airfoil_camber_position: float = 0.35,
    front_airfoil_kammback_start: float = 0.75,
    front_airfoil_chord_mm: Optional[float] = None,
    front_airfoil_points: int = 32,
    front_louver_gap_deg: float = 2.0,
    front_airfoil_design_speed_kmh: float = 50.0,
    front_airfoil_max_yaw_deg: float = 10.0,
    radial_clearance_mm: float = 5.0,
    spine_width: float = 5.0,
    spine_thickness: float = 3.0,
    spine_segments: int = 50,
    louver_length: float = 40.0,
    louver_depth: float = 15.0,
    louver_spacing: float = -0.5,
    tip_fraction: float = 0.2,
    output_filename: str = "louver_fender.stl",
    both_sides: bool = True,
    louver_thickness: Optional[float] = None,
    airfoil_mode: bool = True,
    airfoil_thickness_ratio: float = 0.12,
    airfoil_camber: float = 0.03,
    airfoil_camber_position: float = 0.4,
    kammback_start: float = 0.85,
    airfoil_points: int = 16,
    wing_fence: bool = False,
    fence_position: float = 0.65,
    fence_height: float = 1.0,
    fence_chord_fraction: float = 0.65,
    min_airfoil_thickness_mm: Optional[float] = 2.0,
    louver_tilt_enabled: bool = True,
):
    """High level convenience function to build the fender and save it to STL.

    Parameters
    ----------
    wheel_diameter_mm : float, optional
        Diameter of the wheel's bead seat in millimetres. A 700C wheel has a
        nominal bead seat diameter of 622 mm.
    tire_width_mm : float, optional
        Nominal tyre width (mm). Used when estimating tyre radius.
    tire_radius_addition_mm : Union[float, str], optional
        Additional radial height contributed by the inflated tyre. Use the
        string ``"auto"`` (default) to estimate this as ``1.125 *  tire_width_mm``
    coverage_angle_deg : float, optional
        Angular coverage of the fender extending REARWARD from the top of the
        wheel (0°). For example, 100° extends from the top (12 o'clock) to about
        4:30 position on a clock, covering slightly more than a quarter circle
        toward the back of the wheel. Positive values extend rearward.
    forward_extension_deg : float, optional
        Angular extension FORWARD from the top of the wheel (0°), specified as
        a NEGATIVE angle. For example, -40° places the forward-most point of the
        spine 40° ahead of the top at the 10:30 clock position. The total spine
        length will be (abs(forward_extension_deg) + coverage_angle_deg).
    front_airfoil_end_deg : float, optional
        Ending position of the front airfoil section, measured in degrees from wheel
        top (0°)
        - Negative values extend AHEAD of wheel top (forward): -10° is 10° before crown
        - Zero means the airfoil ends exactly at wheel top (12 o'clock position)
        - Positive values extend BEHIND wheel top (rearward): +10° is 10° past crown
    front_airfoil_thickness_ratio : float, optional
        Thickness ratio for the front fender airfoil (default 0.18 = 18% for
        robust printing and effective water deflection).
    front_airfoil_thickness_mm : Optional[float], optional
        Physical thickness override for the front fender profile. When provided,
        it supersedes ``front_airfoil_thickness_ratio`` so the compact airfoil
        follows an explicit thickness in millimetres. Defaults to ``2 *
        louver_length`` to keep the airfoil twice as thick as the louver span.
    front_wall_thickness_mm : float, optional
        Wall thickness of the hollow front airfoil channel (default 3 mm). Set
        to zero to revert to a solid airfoil volume.
    front_airfoil_tire_clearance_mm : float, optional
        Minimum lateral clearance to maintain between the tyre and each inner wall.
        The inner channel width is guaranteed to be at least the tyre width plus
        twice this clearance.
    front_airfoil_camber : float, optional
        Camber ratio for the front fender (default 0.02 = 2% for gentle downward
        flow deflection).
    front_airfoil_camber_position : float, optional
        Camber position fraction for the front fender (default 0.35 = 35%).
    front_airfoil_kammback_start : float, optional
        Fraction along the chord where the front fender is truncated (default 0.75
        for a blunt, printable trailing edge).
    front_airfoil_points : int, optional
        Resolution of the airfoil profile used for the front fender (default 20).
    front_airfoil_chord_mm : Optional[float], optional
        Legacy alias for ``front_airfoil_chord_mm``. When supplied it specifies
        the radial chord length of the front airfoil in millimetres.
    front_airfoil_chord_mm : Optional[float], optional
        Directly controls the radial chord length of the front airfoil (mm). When
        neither chord parameter is provided the chord defaults to 70 mm.
    front_airfoil_design_speed_kmh : float, optional
        Design airspeed for the front airfoil (default 50 km/h). Used to optimize
        the thickness ratio: higher speeds allow thinner profiles.
    front_airfoil_max_yaw_deg : float, optional
        Maximum expected crosswind yaw angle (default 10°). Larger yaw angles
        favor thicker profiles to delay flow separation on the leeward side.
    front_louver_gap_deg : float, optional
        Angular gap (degrees) between where the front airfoil ends and where the
        first louver begins (in the relative coordinate system from forward-most
        point). This prevents geometry overlap between the two sections.
        For example, if the front airfoil ends at relative position 60° and 
        front_louver_gap_deg=2°, the first louver starts at relative angle 62°.
        Default is 2°.
    radial_clearance_mm : float, optional
        Extra radial clearance between the tyre and the inside of the spine.
    spine_width : float, optional
        Radial width of the spine cross‐section.
    spine_thickness : float, optional
        Vertical thickness of the spine cross‐section.
    spine_segments : int, optional
        Number of segments used to approximate the spine arc.
    louver_length : float, optional
        Length of each louver measured from the spine to the tip (default 40mm).
    louver_depth : float, optional
        Depth of each louver along the tangential direction at the base (default 15mm).
    louver_spacing : float, optional
        Spacing between louvers as a fraction of louver_depth (default -0.5).
        Negative values create overlap (e.g., -0.5 means 50% overlap for tight spacing).
        Zero means edge-to-edge contact.
        Positive values create gaps (e.g., 0.2 means 20% of depth gap between louvers).
    tip_fraction : float, optional
        Fraction of louver_length used for the tapered tip (default 0.3).
    output_filename : str, optional
        Name of the STL file to write (default "louver_fender.stl").
    both_sides : bool, optional
        When true (default), generate louvers on both sides of the spine.
        Set to false to generate louvers on a single side only.
    louver_thickness : float, optional
        Vertical thickness of each louver blade at its base, in mm.  If not
        specified, defaults to spine_thickness for backwards compatibility.
        Use a smaller value (e.g., 2-3mm) for thinner, more aerodynamic louvers.
    airfoil_mode : bool, optional
        If True (default), use airfoil-shaped cross-sections for superior
        aerodynamic performance. If False, use simple rectangular cross-sections.
    airfoil_thickness_ratio : float, optional
        Maximum thickness of airfoil as fraction of depth (default 0.12 = 12%).
        Thinner airfoils (0.08-0.10) have less drag; thicker ones (0.12-0.15)
        are more robust and easier to print.
    airfoil_camber : float, optional
        Maximum camber as fraction of depth (default 0.03 = 3%). Positive
        camber creates a curved surface that gently deflects air downward,
        reducing lift and directing spray away from the rider.
    airfoil_camber_position : float, optional
        Position of maximum camber along the chord (default 0.4 = 40%).
        Forward positions (0.3-0.4) provide gentler pressure gradients and
        better flow attachment.
    kammback_start : float, optional
        Position (0-1) where Kammback trailing edge truncation begins (default 0.85).
        The airfoil is cut at this position to create a blunt trailing edge that's
        easy to print and aerodynamically efficient. Values of 0.80-0.90 are typical.
    airfoil_points : int, optional
        Number of points per airfoil cross-section (default 16). More points
        create smoother surfaces but increase file size. 12-20 is typical.
    wing_fence : bool, optional
        If True, add wing fences at mid-span to block span-wise flow (default False).
        Wing fences are small vertical fins that prevent airflow from migrating
        along the span toward the spine, reducing induced drag in the bicycle
        fender's inward-pressure environment where flow converges toward center.
    fence_position : float, optional
        Spanwise position of wing fence from spine to tip (default 0.65 = 65%).
        Typical values 0.60-0.70. Positioned mid-span to block flow migration
        while allowing proper attachment at leading edge.
    fence_height : float, optional
        Height of wing fence in mm, extending perpendicular to louver surface
        (default 1.0mm). Thin fence provides effective flow blocking without
        excessive drag or print difficulty.
    fence_chord_fraction : float, optional
        Chordwise extent of fence as fraction of louver depth (default 0.65 = 65%).
        Fence extends from leading edge (toward wheel) to this position along chord.
        Values of 0.60-0.70 extend fence close to trailing edge for maximum effectiveness.
    min_airfoil_thickness_mm : float, optional
        Minimum physical thickness for the airfoil sections in mm. The local thickness
        ratio is increased as needed to stay above this value for structural integrity.
    louver_tilt_enabled : bool, optional
        If True (default), apply varying tilt profile that increases downward tilt toward
        the rear of the fender. If False, keep all louvers at 0° tilt (parallel to airflow)
        for uniform aerodynamic performance along the entire fender length.
    """
    # Total coverage is the sum of forward extension (as absolute value since it's negative)
    # and rear coverage (positive)
    total_coverage_deg = coverage_angle_deg + abs(forward_extension_deg)

    if wheel_diameter_mm <= 0.0:
        raise ValueError("wheel_diameter_mm must be positive.")
    if tire_width_mm < 0.0:
        raise ValueError("tire_width_mm cannot be negative.")
    if isinstance(tire_radius_addition_mm, str):
        if tire_radius_addition_mm.lower() != "auto":
            raise ValueError(
                'tire_radius_addition_mm must be a float value or the string "auto".'
            )
        tire_radius_mm = 1.125 * tire_width_mm
    else:
        tire_radius_mm = float(tire_radius_addition_mm)
    if tire_radius_mm < 0.0:
        raise ValueError("tire_radius_addition_mm cannot be negative.")
    if radial_clearance_mm < 0.0:
        raise ValueError("radial_clearance_mm cannot be negative.")

    rim_radius_mm = wheel_diameter_mm / 2.0
    # Inside radius of the spine equals rim radius + tire growth + clearance
    spine_inner_radius_mm = rim_radius_mm + tire_radius_mm + radial_clearance_mm

    print(f"\n=== WHEEL AND TIRE GEOMETRY ===")
    print(f"  Rim diameter (BSD): {wheel_diameter_mm:.1f} mm")
    print(f"  Tire height: {tire_radius_mm:.1f} mm")
    print(f"  Full wheel diameter (with tire): {wheel_diameter_mm + 2 * tire_radius_mm:.1f} mm")
    print(f"  Radial clearance: {radial_clearance_mm:.1f} mm")
    print(f"=== END WHEEL GEOMETRY ===\n")

    # Build spine
    # Note: create_spine expects forward_extension_deg as a positive value
    spine_vertices, spine_faces = create_spine(
        spine_inner_radius=spine_inner_radius_mm,
        spine_width=spine_width,
        spine_thickness=spine_thickness,
        coverage_angle_deg=coverage_angle_deg,
        spine_segments=spine_segments,
        forward_extension_deg=abs(forward_extension_deg),
    )
    # Recompute thetas consistent with create_spine
    theta_start = math.pi / 2.0 - math.radians(abs(forward_extension_deg))
    theta_end = theta_start + math.radians(total_coverage_deg)
    thetas = np.linspace(theta_start, theta_end, spine_segments + 1)
    seg_indices = [
        (i * 4, i * 4 + 1, i * 4 + 2, i * 4 + 3) for i in range(spine_segments + 1)
    ]

    def clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    # Choose the front chord from the most explicit argument available.
    chord_override = front_airfoil_chord_mm
    fender_chord = chord_override if chord_override is not None else 70.0
    
    # Calculate front airfoil thickness automatically based on tire/aero constraints
    # This is now INDEPENDENT of louver_length
    if front_airfoil_thickness_mm is not None:
        # User explicitly specified thickness - use it directly
        front_thickness_mm = front_airfoil_thickness_mm
        fender_thickness_half = front_thickness_mm / 2.0
        print(f"\n=== FRONT AIRFOIL THICKNESS (User Override) ===")
        print(f"  Explicit thickness: {front_thickness_mm:.1f} mm")
        print(f"  Half-thickness: {fender_thickness_half:.1f} mm")
    else:
        # Automatically calculate optimal thickness from tire width, clearance, and chord
        front_thickness_mm, fender_thickness_half, thickness_details = calculate_front_airfoil_thickness(
            tire_width_mm=tire_width_mm,
            front_airfoil_tire_clearance_mm=front_airfoil_tire_clearance_mm,
            front_airfoil_chord_mm=fender_chord,
            front_wall_thickness_mm=front_wall_thickness_mm,
            kammback_start=front_airfoil_kammback_start,
            target_thickness_ratio=front_airfoil_thickness_ratio,
            design_speed_kmh=front_airfoil_design_speed_kmh,
            max_yaw_deg=front_airfoil_max_yaw_deg,
        )
        print(f"\n=== FRONT AIRFOIL THICKNESS (Auto-Calculated) ===")
        print(f"  Inputs:")
        print(f"    Tire width: {tire_width_mm:.1f} mm")
        print(f"    Tire clearance: {front_airfoil_tire_clearance_mm:.1f} mm (each side)")
        print(f"    Chord length: {fender_chord:.1f} mm")
        print(f"    Kammback start: {front_airfoil_kammback_start:.1%}")
        print(f"    Wall thickness: {front_wall_thickness_mm:.1f} mm")
        print(f"    Design speed: {front_airfoil_design_speed_kmh:.0f} km/h")
        print(f"    Max yaw angle: {front_airfoil_max_yaw_deg:.0f}°")
        print(f"  Calculation:")
        print(f"    Required inner half (tire+clearance): {thickness_details['required_inner_half_mm']:.1f} mm")
        print(f"    Required inner gap (full): {thickness_details['required_inner_gap_mm']:.1f} mm")
        print(f"    Kammback thickness ratio: {thickness_details['kammback_reduction_ratio']:.1%}")
        print(f"    Target thickness ratio: {thickness_details['target_thickness_ratio']:.1%}")
        print(f"    Effective thickness ratio: {thickness_details['effective_thickness_ratio']:.1%}")
        print(f"    Ideal aero thickness: {thickness_details['ideal_aero_thickness_mm']:.1f} mm")
        print(f"    Min structural half (scaled for KB): {thickness_details['min_structural_half_mm']:.1f} mm")
        print(f"  Result:")
        print(f"    Constraint: {thickness_details['constraint'].upper()}")
        print(f"    Full thickness: {front_thickness_mm:.1f} mm")
        print(f"    Actual t/c ratio: {thickness_details['actual_thickness_ratio']:.1%} (Aero improves as this drops < 30%)")
        print(f"=== END FRONT AIRFOIL THICKNESS ===")

    front_airfoil_end_relative = abs(forward_extension_deg) + front_airfoil_end_deg
    
    gap_deg = max(front_louver_gap_deg, 0.0)
    front_airfoil_limit = clamp(front_airfoil_end_relative, 0.0, total_coverage_deg)
    front_airfoil_end_for_geometry = front_airfoil_limit
    louver_activation_start = clamp(front_airfoil_limit + gap_deg, 0.0, total_coverage_deg)

    front_airfoil_vertices, front_airfoil_faces = create_front_airfoil(
        section_indices=seg_indices,
        spine_vertices=spine_vertices,
        thetas=thetas,
        theta_start=theta_start,
        front_start_deg=0.0,
        front_end_deg=front_airfoil_end_for_geometry,
        fender_depth=fender_chord,
        fender_half_width=fender_thickness_half,
        spine_thickness=spine_thickness,
        louver_length=louver_length,
        forward_extension_deg=forward_extension_deg,
        airfoil_thickness_ratio=front_airfoil_thickness_ratio,
        airfoil_thickness_mm=front_thickness_mm,
        chord_length_mm=chord_override,
        wall_thickness_mm=front_wall_thickness_mm,
        airfoil_camber=front_airfoil_camber,
        airfoil_camber_position=front_airfoil_camber_position,
        kammback_start=front_airfoil_kammback_start,
        airfoil_points=front_airfoil_points,
    )

    def ramp(value: float, start: float, end: float, start_val: float, end_val: float) -> float:
        if end <= start:
            return end_val
        t = clamp((value - start) / (end - start), 0.0, 1.0)
        return start_val + (end_val - start_val) * t

    calm_end = front_airfoil_limit
    transition_end = min(total_coverage_deg, calm_end + 40.0)
    bottom_angle = total_coverage_deg

    def local_tilt(angle_deg: float) -> float:
        if not louver_tilt_enabled:
            return 0.0  # Keep all louvers parallel to airflow
        if angle_deg <= calm_end:
            return 0.0
        if angle_deg <= transition_end:
            return ramp(angle_deg, calm_end, transition_end, 0.0, 12.0)
        return ramp(angle_deg, transition_end, bottom_angle, 12.0, 20.0)

    def local_rake(angle_deg: float) -> float:
        # Negative rake creates dihedral (tips higher than spine junction for water flow inwards)
        # With louver_length=25mm, -5 degrees gives ~2mm tip rise
        if angle_deg <= calm_end:
            return -5.0
        if angle_deg <= transition_end:
            return ramp(angle_deg, calm_end, transition_end, -5.0, -4.0)
        return ramp(angle_deg, transition_end, bottom_angle, -4.0, -3.0)

    def local_thickness(angle_deg: float) -> float:
        if angle_deg <= calm_end:
            return 0.145
        if angle_deg <= transition_end:
            return ramp(angle_deg, calm_end, transition_end, 0.145, 0.150)
        return ramp(angle_deg, transition_end, bottom_angle, 0.150, 0.155)

    def local_camber(angle_deg: float) -> float:
        if angle_deg <= calm_end:
            return 0.015
        if angle_deg <= transition_end:
            return ramp(angle_deg, calm_end, transition_end, 0.015, 0.018)
        return ramp(angle_deg, transition_end, bottom_angle, 0.018, 0.022)

    def local_camber_position(angle_deg: float) -> float:
        if angle_deg <= calm_end:
            return 0.34
        if angle_deg <= transition_end:
            return ramp(angle_deg, calm_end, transition_end, 0.34, 0.31)
        return ramp(angle_deg, transition_end, bottom_angle, 0.31, 0.28)

    def local_kammback(angle_deg: float) -> float:
        if angle_deg <= calm_end:
            return 0.84
        if angle_deg <= transition_end:
            return ramp(angle_deg, calm_end, transition_end, 0.84, 0.82)
        return ramp(angle_deg, transition_end, bottom_angle, 0.82, 0.80)

    louver_vertices, louver_faces = create_louvers_pair(
        section_indices=seg_indices,
        spine_vertices=spine_vertices,
        thetas=thetas,
        spine_width=spine_width,
        spine_thickness=spine_thickness,
        spine_inner_radius=spine_inner_radius_mm,
        coverage_angle_deg=total_coverage_deg,
        min_active_angle_deg=louver_activation_start,
        louver_length=louver_length,
        louver_depth=louver_depth,
        louver_spacing=louver_spacing,
        tip_fraction=tip_fraction,
        both_sides=both_sides,
        louver_thickness=louver_thickness,
        airfoil_mode=airfoil_mode,
        airfoil_thickness_ratio=airfoil_thickness_ratio,
        airfoil_camber=airfoil_camber,
        airfoil_camber_position=airfoil_camber_position,
        kammback_start=kammback_start,
        airfoil_points=airfoil_points,
        wing_fence=wing_fence,
        fence_position=fence_position,
        fence_height=fence_height,
        fence_chord_fraction=fence_chord_fraction,
        min_airfoil_thickness_mm=min_airfoil_thickness_mm,
        tilt_profile=local_tilt,
        rake_profile=local_rake,
        airfoil_thickness_profile=local_thickness,
        airfoil_camber_profile=local_camber,
        airfoil_camber_position_profile=local_camber_position,
        kammback_profile=local_kammback,
    )

    # --- Clip and cap the spine to match the front airfoil cut plane ---
    # The front airfoil uses a clip plane chosen for rim-cap stability; to prevent the
    # spine from protruding forward of that plane, we clip the spine separately here.
    clip_plane_y = infer_clip_plane_y(front_airfoil_vertices, front_airfoil_faces)
    if clip_plane_y is not None and spine_vertices.size > 0 and spine_faces:
        print(f"\n=== SPINE FRONT CLIP ===")
        print(f"  Clipping spine at Y={clip_plane_y:.4f} mm to match front airfoil cut")
        # Clip only the forward-most portion of the spine. A constant-y plane intersects
        # the spine arc twice; clipping the entire spine would also trim the rear.
        n_sections = int(spine_vertices.shape[0] // 4)
        section_y = np.array([spine_vertices[i * 4, 1] for i in range(n_sections)])
        first_above = None
        for i, y_val in enumerate(section_y):
            if y_val >= clip_plane_y - 1e-9:
                first_above = i
                break
        if first_above is None or first_above <= 0:
            print("  Spine clip skipped (no forward crossing found)")
        else:
            max_vertex_index_exclusive = (first_above + 1) * 4
            clipped_spine_vertices, clipped_spine_faces, _ = clip_mesh_prefix_against_yplane(
                spine_vertices,
                spine_faces,
                clip_plane_y,
                max_vertex_index_exclusive=max_vertex_index_exclusive,
                keep_above=True,
                eps=1e-6,
            )
            spine_vertices = clipped_spine_vertices
            spine_faces = clipped_spine_faces
            try:
                spine_cap_faces = build_plane_cap_faces(
                    spine_vertices,
                    spine_faces,
                    clip_plane_y,
                    keep_above=True,
                    eps=1e-6,
                )
            except RuntimeError as exc:
                spine_cap_faces = []
                print(f"  Spine clip cap skipped: {exc}")
            if spine_cap_faces:
                spine_faces = list(spine_faces) + spine_cap_faces
                print(f"  Spine clip cap faces: {len(spine_cap_faces)}")
        print(f"=== END SPINE FRONT CLIP ===\n")
    # Concatenate vertices and faces, offset each component appropriately
    combined_vertices: List[np.ndarray] = [spine_vertices]
    combined_faces: List[Tuple[int, int, int]] = list(spine_faces)
    running_offset = len(spine_vertices)

    if front_airfoil_vertices.size > 0:
        combined_vertices.append(front_airfoil_vertices)
        combined_faces.extend(
            (a + running_offset, b + running_offset, c + running_offset)
            for (a, b, c) in front_airfoil_faces
        )
        running_offset += len(front_airfoil_vertices)

    if louver_vertices.size > 0:
        combined_vertices.append(louver_vertices)
        combined_faces.extend(
            (a + running_offset, b + running_offset, c + running_offset)
            for (a, b, c) in louver_faces
        )

    all_vertices = np.vstack(combined_vertices)
    all_faces = combined_faces
    # Write to STL
    write_stl(output_filename, all_vertices, all_faces)
    print(f"STL file written to {output_filename}")

if __name__ == "__main__":
    print("=" * 70)
    print("LOUVER FENDER GENERATION WITH 5-POINT GAUSSIAN SMOOTHING")
    print("=" * 70)
    print("\nSmoothing features enabled:")
    print("  • Airfoil profile smoothing (5-point Gaussian kernel)")
    print("  • Spine curve smoothing for reduced faceting")
    print("  • Improved surface quality and aerodynamic performance")
    print("")
    tire_width = 32.0
    spine_thickness = tire_width / 2.0 - 2.0
    louver_extra_margin = 5.0
    louver_length = (tire_width / 2.0) + louver_extra_margin
    print(f"Using tire width: {tire_width}, louver extension {louver_length:.1f} mm")

    build_fender(
        wheel_diameter_mm=622.0,    # mm bead seat diameter for 700C wheel
        tire_radius_addition_mm="auto",  # estimate tyre radius from width
        tire_width_mm=tire_width,         # mm tyre width
        radial_clearance_mm=5.0,  # mm clearance between tyre and spine
        coverage_angle_deg=108.0,    # Rear coverage: 0° (top) to +100°
        forward_extension_deg=-89.0,  # Front coverage: -40° means 40° ahead of wheel top
        front_airfoil_end_deg=20.0,  # End of front airfoil: 20° behind wheel top (wraps past crown)
        # Front airfoil thickness is now AUTO-CALCULATED from tire_width, clearance, and chord
        # The thickness is set to accommodate tire passage while maintaining good aerodynamics
        front_airfoil_chord_mm=98.0,  # Increased from 70mm to improve thickness ratio (aerodynamics)
        front_airfoil_thickness_ratio=0.18,  # Target 18% t/c ratio
        front_airfoil_design_speed_kmh=50.0,  # Design for 50 km/h cruise speed
        front_airfoil_max_yaw_deg=10.0,  # Allow for ±10° crosswind yaw
        front_airfoil_camber=0.0,  # 0% camber for symmetric airfoil
        front_airfoil_camber_position=0.35,
        front_airfoil_kammback_start=0.70,  # Moved forward (from 0.75) to truncate where airfoil is naturally thicker
        front_wall_thickness_mm=3.0,
        front_airfoil_tire_clearance_mm=2.0,  # Tighter 2mm clearance (each side) for reduced frontal area
        spine_width=10.0,       # mm radial width of spine cross‐section
        spine_thickness=spine_thickness,   # mm vertical thickness of spine cross‐section (with a wider expected tire this can be wider)
        spine_segments=60,     # number of segments approximating the spine
        louver_length=louver_length,    # mm length of each louver extending to sides (bike width)
        louver_depth=12.0,      # mm radial depth of each louver (toward wheel)
        louver_spacing=-0.5,   # -0.5 = 50% overlap for tight spacing (blocks spray effectively)
        tip_fraction=0.2,      # fraction of length used for tapered tip
        louver_thickness=3.2,    # mm thickness of louver blades (independent of spine)
        min_airfoil_thickness_mm=2.2,
        airfoil_mode=True,              # Use airfoil cross-sections
        airfoil_thickness_ratio=0.15,   # 15% thick - delays stall, better high-AOA performance
        airfoil_camber=0.02,            # 2% camber - reduced from 5% to minimize span-wise flow while maintaining spray deflection
        airfoil_camber_position=0.30,   # 30% position - forward camber for gentler pressure gradient
        kammback_start=0.82,            # 82% cut - earlier for thicker trailing edge, better separation control
        airfoil_points=24,              # Points per airfoil section for smooth curves
        # Wing fences block span-wise flow in the inward-pressure environment
        wing_fence=True,                # Enable wing fences to reduce induced drag
        fence_position=0.65,            # Fence at 65% span from spine to tip
        fence_height=1.0,               # 1mm fence height - thin but effective flow blocker
        fence_chord_fraction=0.8,       # Fence extends 80% along chord from leading edge
        louver_tilt_enabled=False,      # Keep all louvers parallel to airflow (0° tilt)
        output_filename="louver_fender.stl",
    )
    print("\nGenerating single louver for profile examination...")
    build_single_louver(
        output_filename="single_louver.stl",
    )
    print("\n" + "=" * 70)
    print("GENERATION COMPLETE - Smoothed geometry exported to STL files")
    print("=" * 70)
