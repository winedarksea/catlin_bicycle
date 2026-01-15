"""Seatpost-mounted aerodynamic fairing generator.

This script reuses the airfoil tooling from ``louver_fender.py`` to create a
straight fairing that can be 3D printed and slipped over a round seatpost.

Key design constraints (defaults come from the user brief):
- Seatpost diameter: 27.2 mm
- Clearance/tolerance: 0.2 mm
- Wall thickness: 3 mm (resulting outer thickness = 27.2 + 0.2 + 2*3 = 33.4 mm)
- Seat tube angle: 73°, therefore the cut plane is 17° (90° - 73°) relative to
the horizontal, pivoting at the leading edge.

The geometry is created as a solid Kammback airfoil with a cylindrical bore.
The body is extruded along the seatpost axis, then clipped against angled planes
at the top and bottom using boolean operations to ensure a watertight solid.

Uses constrained Delaunay triangulation for end caps to ensure a single
watertight, manifold mesh suitable for 3D printing.
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.spatial import Delaunay

import pyvista as pv

from louver_fender import create_airfoil_profile


def triangulate_with_hole(outer_pts: np.ndarray, inner_pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Triangulate a 2D region with a hole using constrained Delaunay.
    
    Args:
        outer_pts: Nx2 array of outer boundary points (CCW order)
        inner_pts: Mx2 array of inner hole boundary points (CW order for hole)
    
    Returns:
        (vertices, faces) where vertices is Kx2 and faces is Fx3 indices
    """
    # Combine all points
    all_pts = np.vstack([outer_pts, inner_pts])
    n_outer = len(outer_pts)
    n_inner = len(inner_pts)
    
    # Compute Delaunay triangulation
    tri = Delaunay(all_pts)
    
    # Filter out triangles that are:
    # 1. Inside the hole (centroid inside inner polygon)
    # 2. Outside the outer boundary (centroid outside outer polygon)
    valid_faces = []
    for simplex in tri.simplices:
        # Compute centroid
        centroid = all_pts[simplex].mean(axis=0)
        
        # Check if centroid is inside the outer boundary AND outside the hole
        if point_in_polygon(centroid, outer_pts) and not point_in_polygon(centroid, inner_pts):
            valid_faces.append(simplex)
    
    return all_pts, np.array(valid_faces)


def point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    """Ray casting algorithm to check if point is inside polygon."""
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


def build_tube_mesh(
    outer_profile: np.ndarray,
    inner_circle: np.ndarray,
    z_bottom: float,
    z_top: float,
    slice_normal: np.ndarray,
    bottom_origin: np.ndarray,
    top_origin: np.ndarray,
) -> pv.PolyData:
    """Build a watertight tube mesh with proper end caps using constrained Delaunay.
    
    Creates the side walls by connecting outer and inner profiles along the extrusion,
    then caps both ends with triangulated surfaces that connect outer boundary to inner hole.
    """
    n_outer = len(outer_profile)
    n_inner = len(inner_circle)
    
    # Calculate z values at each (x, y) position based on the slicing planes
    # Bottom plane: slice_normal · (p - bottom_origin) = 0
    # Top plane: (-slice_normal) · (p - top_origin) = 0
    
    def z_on_plane(xy: np.ndarray, plane_normal: np.ndarray, plane_origin: np.ndarray) -> float:
        """Calculate z coordinate where vertical line through (x,y) intersects plane."""
        # plane_normal · (x, y, z) = plane_normal · plane_origin
        # n_x * x + n_y * y + n_z * z = d
        # z = (d - n_x * x - n_y * y) / n_z
        d = np.dot(plane_normal, plane_origin)
        return (d - plane_normal[0] * xy[0] - plane_normal[1] * xy[1]) / plane_normal[2]
    
    # Build bottom and top profiles with correct z values
    outer_bottom = np.zeros((n_outer, 3))
    outer_top = np.zeros((n_outer, 3))
    inner_bottom = np.zeros((n_inner, 3))
    inner_top = np.zeros((n_inner, 3))
    
    for i, pt in enumerate(outer_profile):
        outer_bottom[i] = [pt[0], pt[1], z_on_plane(pt, slice_normal, bottom_origin)]
        outer_top[i] = [pt[0], pt[1], z_on_plane(pt, -slice_normal, top_origin)]
    
    for i, pt in enumerate(inner_circle):
        inner_bottom[i] = [pt[0], pt[1], z_on_plane(pt, slice_normal, bottom_origin)]
        inner_top[i] = [pt[0], pt[1], z_on_plane(pt, -slice_normal, top_origin)]
    
    # Build vertices array: bottom_outer, bottom_inner, top_outer, top_inner
    vertices = np.vstack([outer_bottom, inner_bottom, outer_top, inner_top])
    
    # Index offsets
    bo_start = 0                    # bottom outer
    bi_start = n_outer              # bottom inner
    to_start = n_outer + n_inner    # top outer
    ti_start = 2 * n_outer + n_inner  # top inner - WRONG, should be n_outer + n_inner + n_outer
    
    # Recalculate offsets correctly
    bo_start = 0
    bi_start = n_outer
    to_start = n_outer + n_inner
    ti_start = n_outer + n_inner + n_outer
    
    faces = []
    
    # Outer side walls (connecting bottom_outer to top_outer)
    for i in range(n_outer):
        i_next = (i + 1) % n_outer
        # Quad as two triangles (outward-facing normals)
        faces.extend([3, bo_start + i, to_start + i, to_start + i_next])
        faces.extend([3, bo_start + i, to_start + i_next, bo_start + i_next])
    
    # Inner side walls (connecting bottom_inner to top_inner)
    # Inner walls face inward (opposite winding)
    for i in range(n_inner):
        i_next = (i + 1) % n_inner
        # Quad as two triangles (inward-facing normals for the bore)
        faces.extend([3, bi_start + i, bi_start + i_next, ti_start + i_next])
        faces.extend([3, bi_start + i, ti_start + i_next, ti_start + i])
    
    # Bottom end cap (triangulate outer to inner)
    # Use 2D triangulation
    bottom_2d_outer = outer_profile.copy()
    bottom_2d_inner = inner_circle[:, :2] if inner_circle.shape[1] >= 2 else inner_circle
    
    cap_pts, cap_faces = triangulate_with_hole(bottom_2d_outer, bottom_2d_inner)
    
    # Map triangulation indices to our vertex array
    # cap_pts order: outer (0 to n_outer-1), inner (n_outer to n_outer+n_inner-1)
    # Our bottom vertices: bo_start to bo_start+n_outer-1, bi_start to bi_start+n_inner-1
    for tri in cap_faces:
        mapped = []
        for idx in tri:
            if idx < n_outer:
                mapped.append(bo_start + idx)
            else:
                mapped.append(bi_start + (idx - n_outer))
        # Bottom cap faces downward (reverse winding)
        faces.extend([3, mapped[0], mapped[2], mapped[1]])
    
    # Top end cap (same triangulation pattern, different vertices)
    for tri in cap_faces:
        mapped = []
        for idx in tri:
            if idx < n_outer:
                mapped.append(to_start + idx)
            else:
                mapped.append(ti_start + (idx - n_outer))
        # Top cap faces upward (normal winding)
        faces.extend([3, mapped[0], mapped[1], mapped[2]])
    
    mesh = pv.PolyData(vertices, faces=faces)
    return mesh




def build_seatpost_fairing(
    seatpost_diameter_mm: float = 27.2,
    clearance_mm: float = 0.2,
    wall_thickness_mm: float = 3.0,
    fairing_length_mm: float = 180.0,
    chord_length_mm: float = 150.0,
    seat_tube_angle_deg: float = 73.0,
    airfoil_camber: float = 0.00,
    airfoil_camber_position: float = 0.35,
    kammback_start: float = 0.64,
    profile_points: int = 64,
    bore_circle_points: int = 64,
    output_filename: str = "seatpost_fairing.stl",
) -> str:
    """Generate the STL for the seatpost fairing and return the filename.
    
    Uses constrained Delaunay triangulation for proper end caps that connect
    the outer airfoil boundary to the inner bore, producing a single watertight
    manifold mesh suitable for 3D printing.
    """
    total_thickness_mm = seatpost_diameter_mm + clearance_mm + 2.0 * wall_thickness_mm
    max_thickness_ratio = total_thickness_mm / chord_length_mm
    if max_thickness_ratio <= 0.0:
        raise ValueError("Chord length too short for requested thickness")

    # 1. Create the airfoil profile
    upper, lower = create_airfoil_profile(
        chord_length=chord_length_mm,
        max_thickness=max_thickness_ratio,
        max_camber=airfoil_camber,
        camber_position=airfoil_camber_position,
        n_points=profile_points,
        kammback_start=kammback_start,
    )
    upper = np.array(upper, dtype=float)
    lower = np.array(lower, dtype=float)

    # Adjust thickness if needed (scaling)
    thickness_profile = upper[:, 1] - lower[:, 1]
    max_thickness_idx = int(np.argmax(thickness_profile))
    current_thickness_mm = float(thickness_profile[max_thickness_idx])

    if not math.isclose(current_thickness_mm, total_thickness_mm, rel_tol=1e-3):
        scale = total_thickness_mm / max(current_thickness_mm, 1e-9)
        camber_line = 0.5 * (upper[:, 1] + lower[:, 1])
        upper[:, 1] = camber_line + (upper[:, 1] - camber_line) * scale
        lower[:, 1] = camber_line + (lower[:, 1] - camber_line) * scale
        thickness_profile = upper[:, 1] - lower[:, 1]
        max_thickness_idx = int(np.argmax(thickness_profile))

    # Calculate bore center at the max thickness point
    bore_center = np.array(
        [
            0.5 * (upper[max_thickness_idx, 0] + lower[max_thickness_idx, 0]),
            0.5 * (upper[max_thickness_idx, 1] + lower[max_thickness_idx, 1]),
        ]
    )

    # Create outer airfoil profile (CCW for outer boundary)
    # upper goes from leading edge (x=0) to trailing edge (x=kammback)
    # lower[::-1] goes from trailing edge to leading edge
    # 
    # The trailing edge points: upper[-1] and lower[-1] (which is lower[::-1][0])
    # have the SAME X coordinate but DIFFERENT Y coordinates - this forms the kammback face.
    # We need BOTH points in the profile to create the vertical back edge.
    #
    # The leading edge points: upper[0] and lower[0] (which is lower[::-1][-1])  
    # These are nearly identical (or the same), so we skip the duplicate.
    #
    # Profile order: upper (LE→TE), then lower reversed WITHOUT its last point (TE→near-LE)
    # This creates a closed loop: LE → TE (upper) → TE (lower, kammback) → LE (back to start)
    outer_profile = np.vstack([upper, lower[::-1][:-1]])
    
    # Create inner bore circle (CW for hole - opposite winding)
    bore_radius = (seatpost_diameter_mm + clearance_mm) / 2.0
    theta = np.linspace(0, 2 * np.pi, bore_circle_points, endpoint=False)
    # CW order for hole
    inner_circle = np.column_stack([
        bore_center[0] + bore_radius * np.cos(-theta),
        bore_center[1] + bore_radius * np.sin(-theta),
    ])

    # 2. Calculate slicing plane geometry
    slice_angle_deg = 90.0 - seat_tube_angle_deg
    slice_angle_rad = math.radians(slice_angle_deg)
    tan_angle = math.tan(slice_angle_rad)

    # Slice plane normal (pointing "up" into the kept region)
    slice_normal = np.array([-tan_angle, 0.0, 1.0])
    slice_normal /= np.linalg.norm(slice_normal)

    # Calculate z range needed to accommodate the angled cuts
    extra_length = chord_length_mm * abs(tan_angle) + 5.0
    
    # Plane origins (points on each cutting plane)
    bottom_origin = np.array([0.0, 0.0, extra_length])
    top_origin = np.array([0.0, 0.0, extra_length + fairing_length_mm])

    # 3. Build the tube mesh with proper constrained Delaunay end caps
    mesh = build_tube_mesh(
        outer_profile=outer_profile,
        inner_circle=inner_circle,
        z_bottom=0.0,
        z_top=extra_length * 2.0 + fairing_length_mm,
        slice_normal=slice_normal,
        bottom_origin=bottom_origin,
        top_origin=top_origin,
    )
    
    mesh = mesh.clean()
    print(f"Mesh after construction: {mesh.n_points} points, {mesh.n_cells} cells")
    
    # 4. Validate mesh quality
    is_manifold = mesh.is_manifold
    n_open_edges = mesh.n_open_edges
    print(f"Is manifold: {is_manifold}, Open edges: {n_open_edges}")
    
    if n_open_edges > 0:
        print("Warning: Mesh has open edges, attempting repair...")
        mesh = mesh.fill_holes(hole_size=1000.0)
        mesh = mesh.clean()
        print(f"After fill_holes: manifold={mesh.is_manifold}, open_edges={mesh.n_open_edges}")
    
    # 5. Final cleanup and export
    mesh = mesh.triangulate().clean()
    
    # Verify single connected component
    conn = mesh.connectivity()
    n_regions = len(set(conn.point_data["RegionId"]))
    print(f"Number of connected regions: {n_regions}")
    
    mesh.save(output_filename)
    print(f"Seatpost fairing written to {output_filename} (length {fairing_length_mm:.1f} mm)")
    
    return output_filename


if __name__ == "__main__":
    build_seatpost_fairing()
