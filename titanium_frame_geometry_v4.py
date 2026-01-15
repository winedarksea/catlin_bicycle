import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
import math

"""
TITANIUM FRAME GEOMETRY CALCULATOR V4 - Size 54cm
==================================================
The goal of this is an stable bike for winter road riding, while also being reasonably aerodynamic.
"""

print("=" * 85)
print("TITANIUM FRAME GEOMETRY CALCULATOR V4 - Size 54cm Custom")
print("=" * 85)

# --- Primary Specifications & Design Choices ---

# Frame Fit Targets (for a size 54 endurance/winter geometry)
TARGET_REACH = 385  # mm - A good middle-ground for reach on a size 54
TARGET_STACK = 560  # mm - Provides a comfortable, but not overly upright, position

# Core Frame Specifications
FRAME_SIZE = 54  # cm
SEAT_TUBE_LENGTH = FRAME_SIZE * 10  # mm
BB_DROP = 87  # mm
SEAT_TUBE_ANGLE = 73.0  # degrees from horizontal, slanting backward
HEAD_TUBE_ANGLE = 70.8  # degrees from horizontal (slacker for stability)
CHAINSTAY_LENGTH = 420  # mm
SEAT_STAY_DROP = 124  # mm (from seat tube/top tube junction)

# Fork Specifications
FORK_AXLE_TO_CROWN = 410  # mm
FORK_OFFSET = 55  # mm

# Wheel & Drivetrain Specifications
WHEEL_DIAMETER = 734  # mm (for 700c with large winter tires), include any desired clearance in this as well
WHEEL_RADIUS = WHEEL_DIAMETER / 2
CHAINRING_TEETH = 46
CHAINRING_RADIUS = CHAINRING_TEETH * 12.7 / (2 * math.pi)  # Approx. 85mm
CRANK_ARM_LENGTH = 165  # mm, single crank arm shown in visualization

# Tube Dimensions (outer profiles visible in side view)
SEAT_STAY_OD = 14  # mm
TOP_TUBE_VISIBLE_WIDTH = 26  # mm (flattened side visible in side view)
TOP_TUBE_LATERAL = 24  # mm (depth, documented for completeness)
DOWN_TUBE_VISIBLE_WIDTH = 52  # mm (flattened side visible in side view)
DOWN_TUBE_LATERAL = 20  # mm (depth, not shown, kept for documentation)
SEAT_POST_DIAMETER = 27.2  # mm
SEAT_TUBE_WALL_THICKNESS = 1.2  # mm per wall (assumed)
SEAT_TUBE_OD = SEAT_POST_DIAMETER + 2 * SEAT_TUBE_WALL_THICKNESS
SEAT_TUBE_RADIUS = SEAT_TUBE_OD / 2
HEAD_TUBE_OD = 44  # mm, large enough for 1.5" to 1-1/8" taper without taper depiction
CHAINSTAY_OD = 24  # mm

# Compact Geometry Option
TOP_TUBE_SEAT_TUBE_DROP = 50  # mm - lowers where the top tube connects to the seat tube (compact frame geometry)

# Head tube construction targets (measured along the head tube axis)
HEAD_TUBE_UPPER_EXTENSION = 14  # mm of tube above the top tube for headset bearing support
HEAD_TUBE_BETWEEN_TUBES = 28
  # mm of tube between the down tube and top tube welds
HEAD_TUBE_LOWER_CLEARANCE_MIN = 28  # mm of tube between the down tube weld and the fork crown
HEAD_TUBE_LABEL_X_OFFSET = 140  # mm to the right of the head tube for segment labels

# --- Calculations ---

# Convert angles to radians for trigonometric functions
seat_tube_angle_rad = math.radians(SEAT_TUBE_ANGLE)
head_tube_angle_rad = math.radians(HEAD_TUBE_ANGLE)
head_axis_vertical_component = abs(math.sin(head_tube_angle_rad))
if head_axis_vertical_component == 0:
        raise ValueError("Head tube axis vertical component is zero; cannot evaluate geometry.")
top_tube_half_vertical = TOP_TUBE_VISIBLE_WIDTH / 2
down_tube_half_vertical = DOWN_TUBE_VISIBLE_WIDTH / 2
top_tube_half_axial = top_tube_half_vertical * head_axis_vertical_component
down_tube_half_axial = down_tube_half_vertical * head_axis_vertical_component

# --- Define Coordinate System ---
# The Bottom Bracket (BB) is the origin (0, 0)
bb_x, bb_y = 0, 0

# --- Calculate Key Fixed Points ---

# Rear Wheel Axle (initial vertical placement only; horizontal position solved later)
rear_wheel_y = bb_y + BB_DROP
ground_y = rear_wheel_y - WHEEL_RADIUS

# Head Tube Top
# This is the primary fit point, determined by Stack & Reach from the BB.
head_tube_top_x = bb_x + TARGET_REACH
head_tube_top_y = bb_y + TARGET_STACK

# Seat Tube Top
# Calculated from the BB, extending up and back at the seat tube angle.
seat_tube_length_effective = SEAT_TUBE_LENGTH
seat_tube_top_x = bb_x - seat_tube_length_effective * math.cos(seat_tube_angle_rad)
seat_tube_top_y = bb_y + seat_tube_length_effective * math.sin(seat_tube_angle_rad)

# Extend the seat tube if needed so the top tube can meet the head tube target height.
required_seat_tube_length = (head_tube_top_y - bb_y) / math.sin(seat_tube_angle_rad)
if required_seat_tube_length > seat_tube_length_effective:
        seat_tube_length_effective = required_seat_tube_length
        seat_tube_top_x = bb_x - seat_tube_length_effective * math.cos(seat_tube_angle_rad)
        seat_tube_top_y = bb_y + seat_tube_length_effective * math.sin(seat_tube_angle_rad)

seat_tube_axis_dx = seat_tube_top_x - bb_x
seat_tube_axis_dy = seat_tube_top_y - bb_y
seat_tube_axis_length = math.hypot(seat_tube_axis_dx, seat_tube_axis_dy)
if seat_tube_axis_length == 0:
        raise ValueError("Seat tube axis length is zero; cannot evaluate geometry.")
seat_tube_axis_unit_x = seat_tube_axis_dx / seat_tube_axis_length
seat_tube_axis_unit_y = seat_tube_axis_dy / seat_tube_axis_length


def evaluate_chainstay_geometry(chainstay_length):
        """Return rear wheel x-position and seat tube clearance for a chainstay length."""
        rear_wheel_x_candidate = -chainstay_length
        wheel_vector_x = rear_wheel_x_candidate - bb_x
        wheel_vector_y = rear_wheel_y - bb_y
        projection_length = wheel_vector_x * seat_tube_axis_unit_x + wheel_vector_y * seat_tube_axis_unit_y
        projection_length = max(0.0, min(seat_tube_length_effective, projection_length))
        closest_axis_x = bb_x + seat_tube_axis_unit_x * projection_length
        closest_axis_y = bb_y + seat_tube_axis_unit_y * projection_length
        axis_to_wheel_dx = closest_axis_x - rear_wheel_x_candidate
        axis_to_wheel_dy = closest_axis_y - rear_wheel_y
        axis_to_wheel_distance = math.hypot(axis_to_wheel_dx, axis_to_wheel_dy)
        clearance_value = axis_to_wheel_distance - WHEEL_RADIUS - SEAT_TUBE_RADIUS
        clearance_points = None
        if axis_to_wheel_distance != 0:
                axis_to_wheel_unit_x = axis_to_wheel_dx / axis_to_wheel_distance
                axis_to_wheel_unit_y = axis_to_wheel_dy / axis_to_wheel_distance
                tire_surface_x = rear_wheel_x_candidate + axis_to_wheel_unit_x * WHEEL_RADIUS
                tire_surface_y = rear_wheel_y + axis_to_wheel_unit_y * WHEEL_RADIUS
                seat_tube_surface_x = closest_axis_x - axis_to_wheel_unit_x * SEAT_TUBE_RADIUS
                seat_tube_surface_y = closest_axis_y - axis_to_wheel_unit_y * SEAT_TUBE_RADIUS
                clearance_points = ((tire_surface_x, tire_surface_y), (seat_tube_surface_x, seat_tube_surface_y))
        return {
                'chainstay_length': chainstay_length,
                'rear_wheel_x': rear_wheel_x_candidate,
                'rear_wheel_y': rear_wheel_y,
                'clearance': clearance_value,
                'clearance_points': clearance_points
        }


def resolve_chainstay_length(base_length, required_clearance):
        """Extend the chainstay if needed to satisfy the requested rear tire clearance."""
        base_result = evaluate_chainstay_geometry(base_length)
        if base_result['clearance'] >= required_clearance:
                return base_result

        hi_length = base_length
        hi_result = base_result
        extension_step = 2.5
        max_extension = 300  # mm safety cap
        while hi_result['clearance'] < required_clearance and hi_length - base_length < max_extension:
                hi_length += extension_step
                hi_result = evaluate_chainstay_geometry(hi_length)

        if hi_result['clearance'] < required_clearance:
                raise ValueError("Unable to achieve rear tire clearance even with extended chainstays. Increase max extension or adjust geometry targets.")

        lo_length = base_length
        lo_result = base_result
        best_result = hi_result
        for _ in range(40):
                mid_length = (lo_length + hi_length) / 2
                mid_result = evaluate_chainstay_geometry(mid_length)
                if mid_result['clearance'] >= required_clearance:
                        hi_length = mid_length
                        best_result = mid_result
                else:
                        lo_length = mid_length
                        lo_result = mid_result
        return best_result


chainstay_solution = resolve_chainstay_length(CHAINSTAY_LENGTH, 0.0)
rear_wheel_x = chainstay_solution['rear_wheel_x']
chainstay_length_effective = chainstay_solution['chainstay_length']
rear_tire_clearance = chainstay_solution['clearance']
rear_tire_clearance_points = chainstay_solution['clearance_points']
chainstay_extension = chainstay_length_effective - CHAINSTAY_LENGTH

# --- Calculate Dependent Geometry ---

# Head Tube axis (steering axis)
head_axis_x = math.cos(head_tube_angle_rad)
head_axis_y = -math.sin(head_tube_angle_rad)

# Final acute angles relative to the horizontal (ignoring fore/aft direction)
seat_tube_angle_final = math.degrees(math.atan2(abs(seat_tube_axis_dy), abs(seat_tube_axis_dx)))
head_tube_angle_final = math.degrees(math.atan2(abs(head_axis_y), abs(head_axis_x)))

# Unit vector perpendicular to the steering axis, pointing "forward" from the axis toward the axle
axis_perp_x = math.sin(head_tube_angle_rad)
axis_perp_y = math.cos(head_tube_angle_rad)

if FORK_AXLE_TO_CROWN <= FORK_OFFSET:
        raise ValueError("Fork offset must be less than the axle-to-crown length to form a valid triangle.")

# Component of the axle-to-crown measurement that lies along the steering axis
fork_axis_projection = math.sqrt(FORK_AXLE_TO_CROWN**2 - FORK_OFFSET**2)

# Target position of the steerer axis point at the axle height
axis_point_target_y = rear_wheel_y - axis_perp_y * FORK_OFFSET


def compute_front_end(head_tube_top_x_val, head_tube_top_y_val):
        """Compute dependent front-end geometry for a given head tube top location."""
        desired_top_tube_axis_distance = HEAD_TUBE_UPPER_EXTENSION + top_tube_half_axial
        desired_top_tube_center_y = head_tube_top_y_val - head_axis_vertical_component * desired_top_tube_axis_distance

        top_tube_height_local = min(desired_top_tube_center_y, head_tube_top_y_val - top_tube_half_vertical)

        seat_cluster_x_local = bb_x - (top_tube_height_local - bb_y) / math.tan(seat_tube_angle_rad)
        top_tube_rear_x_local = seat_cluster_x_local
        top_tube_rear_y_local = top_tube_height_local
        seat_tube_visible_top_x_local = seat_cluster_x_local
        seat_tube_visible_top_y_local = top_tube_height_local
        seat_tube_visible_length_local = math.hypot(seat_tube_visible_top_x_local - bb_x,
                                                    seat_tube_visible_top_y_local - bb_y)

        head_tube_length_local = (axis_point_target_y - head_tube_top_y_val) / head_axis_y - fork_axis_projection
        if head_tube_length_local <= 0:
                raise ValueError("Derived head tube length is non-positive; check the input targets and fork dimensions.")

        head_tube_bottom_x_local = head_tube_top_x_val + head_axis_x * head_tube_length_local
        head_tube_bottom_y_local = head_tube_top_y_val + head_axis_y * head_tube_length_local

        axis_point_x_local = head_tube_bottom_x_local + head_axis_x * fork_axis_projection
        axis_point_y_local = head_tube_bottom_y_local + head_axis_y * fork_axis_projection
        front_wheel_x_local = axis_point_x_local + axis_perp_x * FORK_OFFSET
        front_wheel_y_local = axis_point_y_local + axis_perp_y * FORK_OFFSET
        wheelbase_local = front_wheel_x_local - rear_wheel_x
        front_center_local = front_wheel_x_local - bb_x

        s_to_top_tube_local = (top_tube_height_local - head_tube_top_y_val) / head_axis_y
        top_tube_axis_distance_local = s_to_top_tube_local
        top_tube_front_x_local = head_tube_top_x_val + head_axis_x * top_tube_axis_distance_local
        top_tube_front_y_local = head_tube_top_y_val + head_axis_y * top_tube_axis_distance_local
        top_tube_length_local = top_tube_front_x_local - top_tube_rear_x_local

        top_tube_upper_surface_axis_distance_local = max(0.0, top_tube_axis_distance_local - top_tube_half_axial)
        top_tube_lower_surface_axis_distance_local = top_tube_axis_distance_local + top_tube_half_axial

        down_tube_upper_surface_axis_distance_local = top_tube_lower_surface_axis_distance_local + HEAD_TUBE_BETWEEN_TUBES
        down_tube_axis_distance_local = down_tube_upper_surface_axis_distance_local + down_tube_half_axial
        down_tube_lower_surface_axis_distance_local = down_tube_axis_distance_local + down_tube_half_axial

        head_tube_lower_clearance_physical_local = head_tube_length_local - down_tube_lower_surface_axis_distance_local
        head_tube_lower_clearance_shortfall_local = max(0.0, HEAD_TUBE_LOWER_CLEARANCE_MIN - head_tube_lower_clearance_physical_local)
        head_tube_lower_clearance_center_local = head_tube_length_local - down_tube_axis_distance_local

        down_tube_weld_x_local = head_tube_top_x_val + head_axis_x * down_tube_axis_distance_local
        down_tube_weld_y_local = head_tube_top_y_val + head_axis_y * down_tube_axis_distance_local

        top_extension_actual_local = top_tube_upper_surface_axis_distance_local
        top_tube_band_axis_local = top_tube_lower_surface_axis_distance_local - top_tube_upper_surface_axis_distance_local
        down_tube_band_axis_local = down_tube_lower_surface_axis_distance_local - down_tube_upper_surface_axis_distance_local

        t_ground_local = (ground_y - head_tube_bottom_y_local) / head_axis_y
        steering_axis_ground_x_local = head_tube_bottom_x_local + head_axis_x * t_ground_local
        trail_local = steering_axis_ground_x_local - front_wheel_x_local

        down_tube_length_local = math.sqrt((down_tube_weld_x_local - bb_x)**2 + (down_tube_weld_y_local - bb_y)**2)

        seat_stay_drop_height_local = top_tube_height_local - SEAT_STAY_DROP
        if seat_stay_drop_height_local <= bb_y:
                raise ValueError("Seat stay drop exceeds seat tube length; adjust SEAT_STAY_DROP or geometry targets.")
        seat_stay_front_x_local = bb_x - (seat_stay_drop_height_local - bb_y) / math.tan(seat_tube_angle_rad)
        seat_stay_front_y_local = seat_stay_drop_height_local
        seat_stay_rear_x_local = rear_wheel_x
        seat_stay_rear_y_local = rear_wheel_y
        seat_stay_length_local = math.sqrt((seat_stay_front_x_local - seat_stay_rear_x_local)**2 +
                                           (seat_stay_front_y_local - seat_stay_rear_y_local)**2)

        # Calculate where the rear wheel outer radius intersects the seat stay
        # Vector from rear axle to seat stay front
        stay_dx = seat_stay_front_x_local - seat_stay_rear_x_local
        stay_dy = seat_stay_front_y_local - seat_stay_rear_y_local
        stay_length = math.hypot(stay_dx, stay_dy)
        # Unit vector along seat stay (from rear to front)
        stay_unit_x = stay_dx / stay_length if stay_length > 0 else 0
        stay_unit_y = stay_dy / stay_length if stay_length > 0 else 0
        # The intersection point is where the seat stay is WHEEL_RADIUS from the rear axle
        # Distance along seat stay from rear axle to wheel intersection
        seat_stay_wheel_intersection_dist = WHEEL_RADIUS
        seat_stay_wheel_intersect_x_local = seat_stay_rear_x_local + stay_unit_x * seat_stay_wheel_intersection_dist
        seat_stay_wheel_intersect_y_local = seat_stay_rear_y_local + stay_unit_y * seat_stay_wheel_intersection_dist
        # Length from seat tube to wheel intersection
        seat_stay_to_wheel_length_local = seat_stay_length_local - seat_stay_wheel_intersection_dist
        # Length from wheel intersection to rear axle (should equal WHEEL_RADIUS along the stay)
        seat_stay_from_wheel_length_local = seat_stay_wheel_intersection_dist

        final_stack_local = head_tube_top_y_val - bb_y
        final_reach_local = head_tube_top_x_val - bb_x
        standover_height_local = top_tube_height_local - ground_y
        bb_height_local = WHEEL_RADIUS - BB_DROP

        return {
                'head_tube_top_x': head_tube_top_x_val,
                'head_tube_top_y': head_tube_top_y_val,
                'HEAD_TUBE_LENGTH': head_tube_length_local,
                'head_tube_bottom_x': head_tube_bottom_x_local,
                'head_tube_bottom_y': head_tube_bottom_y_local,
                'axis_point_x': axis_point_x_local,
                'axis_point_y': axis_point_y_local,
                'front_wheel_x': front_wheel_x_local,
                'front_wheel_y': front_wheel_y_local,
                'wheelbase': wheelbase_local,
                'front_center': front_center_local,
                'top_tube_height': top_tube_height_local,
                'seat_cluster_x': seat_cluster_x_local,
                'top_tube_rear_x': top_tube_rear_x_local,
                'top_tube_rear_y': top_tube_rear_y_local,
                'seat_tube_visible_top_x': seat_tube_visible_top_x_local,
                'seat_tube_visible_top_y': seat_tube_visible_top_y_local,
                'seat_tube_visible_length': seat_tube_visible_length_local,
                'top_tube_front_x': top_tube_front_x_local,
                'top_tube_front_y': top_tube_front_y_local,
                'top_tube_length': top_tube_length_local,
                'top_tube_axis_distance': top_tube_axis_distance_local,
                'top_tube_upper_surface_axis_distance': top_tube_upper_surface_axis_distance_local,
                'top_tube_lower_surface_axis_distance': top_tube_lower_surface_axis_distance_local,
                'down_tube_upper_surface_axis_distance': down_tube_upper_surface_axis_distance_local,
                'down_tube_axis_distance': down_tube_axis_distance_local,
                'down_tube_lower_surface_axis_distance': down_tube_lower_surface_axis_distance_local,
                'head_tube_lower_clearance_physical': head_tube_lower_clearance_physical_local,
                'head_tube_lower_clearance_shortfall': head_tube_lower_clearance_shortfall_local,
                'head_tube_lower_clearance_center': head_tube_lower_clearance_center_local,
                'down_tube_weld_x': down_tube_weld_x_local,
                'down_tube_weld_y': down_tube_weld_y_local,
                'top_extension_actual': top_extension_actual_local,
                'top_tube_band_axis': top_tube_band_axis_local,
                'down_tube_band_axis': down_tube_band_axis_local,
                'trail': trail_local,
                'steering_axis_ground_x': steering_axis_ground_x_local,
                'down_tube_length': down_tube_length_local,
                'seat_stay_front_x': seat_stay_front_x_local,
                'seat_stay_front_y': seat_stay_front_y_local,
                'seat_stay_rear_x': seat_stay_rear_x_local,
                'seat_stay_rear_y': seat_stay_rear_y_local,
                'seat_stay_length': seat_stay_length_local,
                'seat_stay_wheel_intersect_x': seat_stay_wheel_intersect_x_local,
                'seat_stay_wheel_intersect_y': seat_stay_wheel_intersect_y_local,
                'seat_stay_to_wheel_length': seat_stay_to_wheel_length_local,
                'seat_stay_from_wheel_length': seat_stay_from_wheel_length_local,
                'standover_height': standover_height_local,
                'final_stack': final_stack_local,
                'final_reach': final_reach_local,
                'bb_height': bb_height_local
        }


front_end = compute_front_end(head_tube_top_x, head_tube_top_y)

head_tube_extension_axis = 0.0
head_tube_extension_iterations = 0
MAX_HEAD_TUBE_EXTENSION_ITERATIONS = 6
SHORTFALL_TOLERANCE = 1e-4
while front_end['head_tube_lower_clearance_shortfall'] > SHORTFALL_TOLERANCE and head_tube_extension_iterations < MAX_HEAD_TUBE_EXTENSION_ITERATIONS:
        shortfall = front_end['head_tube_lower_clearance_shortfall']
        head_tube_extension_axis += shortfall
        new_head_tube_length = front_end['HEAD_TUBE_LENGTH'] + shortfall
        head_tube_bottom_x = front_end['head_tube_bottom_x']
        head_tube_bottom_y = front_end['head_tube_bottom_y']
        head_tube_top_x = head_tube_bottom_x - head_axis_x * new_head_tube_length
        head_tube_top_y = head_tube_bottom_y - head_axis_y * new_head_tube_length
        front_end = compute_front_end(head_tube_top_x, head_tube_top_y)
        head_tube_extension_iterations += 1

if front_end['head_tube_lower_clearance_shortfall'] > SHORTFALL_TOLERANCE:
        raise ValueError("Unable to satisfy head tube clearance even after iterative extension attempts.")

extension_vertical = head_axis_vertical_component * head_tube_extension_axis
extension_horizontal = head_axis_x * head_tube_extension_axis

# unwrap the final geometry values for clarity downstream
HEAD_TUBE_LENGTH = front_end['HEAD_TUBE_LENGTH']
head_tube_top_x = front_end['head_tube_top_x']
head_tube_top_y = front_end['head_tube_top_y']
head_tube_bottom_x = front_end['head_tube_bottom_x']
head_tube_bottom_y = front_end['head_tube_bottom_y']
axis_point_x = front_end['axis_point_x']
axis_point_y = front_end['axis_point_y']
front_wheel_x = front_end['front_wheel_x']
front_wheel_y = front_end['front_wheel_y']
wheelbase = front_end['wheelbase']
front_center = front_end['front_center']
top_tube_height = front_end['top_tube_height']
seat_cluster_x = front_end['seat_cluster_x']
top_tube_rear_x = front_end['top_tube_rear_x']
top_tube_rear_y = front_end['top_tube_rear_y']
seat_tube_visible_top_x = front_end['seat_tube_visible_top_x']
seat_tube_visible_top_y = front_end['seat_tube_visible_top_y']
seat_tube_visible_length = front_end['seat_tube_visible_length']
top_tube_front_x = front_end['top_tube_front_x']
top_tube_front_y = front_end['top_tube_front_y']
top_tube_length = front_end['top_tube_length']
top_tube_axis_distance = front_end['top_tube_axis_distance']
top_tube_upper_surface_axis_distance = front_end['top_tube_upper_surface_axis_distance']
top_tube_lower_surface_axis_distance = front_end['top_tube_lower_surface_axis_distance']
down_tube_upper_surface_axis_distance = front_end['down_tube_upper_surface_axis_distance']
down_tube_axis_distance = front_end['down_tube_axis_distance']
down_tube_lower_surface_axis_distance = front_end['down_tube_lower_surface_axis_distance']
head_tube_lower_clearance_physical = front_end['head_tube_lower_clearance_physical']
head_tube_lower_clearance_shortfall = front_end['head_tube_lower_clearance_shortfall']
head_tube_lower_clearance_center = front_end['head_tube_lower_clearance_center']
down_tube_weld_x = front_end['down_tube_weld_x']
down_tube_weld_y = front_end['down_tube_weld_y']
top_extension_actual = front_end['top_extension_actual']
top_tube_band_axis = front_end['top_tube_band_axis']
down_tube_band_axis = front_end['down_tube_band_axis']
trail = front_end['trail']
steering_axis_ground_x = front_end['steering_axis_ground_x']
down_tube_length = front_end['down_tube_length']
seat_stay_front_x = front_end['seat_stay_front_x']
seat_stay_front_y = front_end['seat_stay_front_y']
seat_stay_rear_x = front_end['seat_stay_rear_x']
seat_stay_rear_y = front_end['seat_stay_rear_y']
seat_stay_length = front_end['seat_stay_length']
seat_stay_wheel_intersect_x = front_end['seat_stay_wheel_intersect_x']
seat_stay_wheel_intersect_y = front_end['seat_stay_wheel_intersect_y']
seat_stay_to_wheel_length = front_end['seat_stay_to_wheel_length']
seat_stay_from_wheel_length = front_end['seat_stay_from_wheel_length']
standover_height = front_end['standover_height']
final_stack = front_end['final_stack']
final_reach = front_end['final_reach']
bb_height = front_end['bb_height']

# --- Apply Compact Geometry: Lower Top Tube Connection at Seat Tube ---
# This creates a sloped top tube (compact geometry) by keeping the head tube connection
# the same but lowering where the top tube meets the seat tube.
if TOP_TUBE_SEAT_TUBE_DROP > 0:
        # Store original top tube rear position for reference (before the drop)
        top_tube_rear_y_original = top_tube_rear_y
        top_tube_rear_x_original = top_tube_rear_x
        # Lower the top tube rear connection point on the seat tube
        top_tube_rear_y = top_tube_rear_y - TOP_TUBE_SEAT_TUBE_DROP
        # Recalculate the seat tube x-position at this new (lower) height
        top_tube_rear_x = bb_x - (top_tube_rear_y - bb_y) / math.tan(seat_tube_angle_rad)
        seat_cluster_x = top_tube_rear_x
        # Recalculate top tube length (now it's sloped, so use actual distance)
        top_tube_length_actual = math.hypot(top_tube_front_x - top_tube_rear_x, top_tube_front_y - top_tube_rear_y)
        top_tube_length_horizontal = top_tube_front_x - top_tube_rear_x
        # Update standover height (at the lower rear point)
        standover_height = top_tube_rear_y - ground_y

# --- Output Results ---
print("\n" + "="*30 + " GEOMETRY RESULTS " + "="*30)
print(f"INPUTS:")
print(f"  - Target Stack: {TARGET_STACK} mm")
print(f"  - Target Reach: {TARGET_REACH} mm")
print(f"  - Seat Tube Angle: {SEAT_TUBE_ANGLE}°")
print(f"  - Head Tube Angle: {HEAD_TUBE_ANGLE}°")
print(f"  - Chainstay Length: {CHAINSTAY_LENGTH} mm")
print("\nOUTPUTS:")
print(f"  - Wheelbase: {wheelbase:.1f} mm")
print(f"  - Front Center: {front_center:.1f} mm")
print(f"  - Head Tube Length: {HEAD_TUBE_LENGTH:.1f} mm")
if head_tube_extension_axis > 0:
        print(f"      (extended +{head_tube_extension_axis:.1f} mm along axis → +{extension_vertical:.1f} mm stack, -{extension_horizontal:.1f} mm reach)")
clearance_note = "" if head_tube_lower_clearance_shortfall <= SHORTFALL_TOLERANCE else f" (short {head_tube_lower_clearance_shortfall:.1f} mm)"
print(f"      Segments (top→bottom, axial): upper extension {top_extension_actual:.1f} mm (target {HEAD_TUBE_UPPER_EXTENSION:.1f} mm) | top tube band {top_tube_band_axis:.1f} mm | clear gap {HEAD_TUBE_BETWEEN_TUBES:.1f} mm | down tube band {down_tube_band_axis:.1f} mm | crown clearance {head_tube_lower_clearance_physical:.1f} mm phys ({head_tube_lower_clearance_center:.1f} mm center, min {HEAD_TUBE_LOWER_CLEARANCE_MIN} mm){clearance_note}")
print(f"  - Seat Tube Length (effective): {seat_tube_visible_length:.1f} mm")
print(f"  - Top Tube Length (effective, horizontal): {top_tube_length:.1f} mm")
if TOP_TUBE_SEAT_TUBE_DROP > 0:
        print(f"      (compact geometry: top tube lowered {TOP_TUBE_SEAT_TUBE_DROP:.0f} mm at seat tube, sloped length {top_tube_length_actual:.1f} mm)")
print(f"  - Down Tube Length: {down_tube_length:.1f} mm")
print(f"  - Chainstay Length (effective): {chainstay_length_effective:.1f} mm (target {CHAINSTAY_LENGTH} mm, +{chainstay_extension:.1f} mm)")
print(f"  - Seat Stay Length: {seat_stay_length:.1f} mm (with {SEAT_STAY_DROP}mm drop)")
print(f"  - Trail: {trail:.1f} mm (High trail = stable)")
print(f"  - BB Height: {bb_height:.1f} mm")
print(f"  - Standover Height (mid top tube): {standover_height:.1f} mm")
print(f"  - Final Stack: {final_stack:.1f} mm")
print(f"  - Final Reach: {final_reach:.1f} mm")
print(f"  - Stack-to-Reach Ratio: {final_stack/final_reach:.3f}")
if head_tube_extension_axis > 0:
        print(f"      (Δ vs target: stack +{final_stack - TARGET_STACK:.1f} mm, reach {final_reach - TARGET_REACH:+.1f} mm)")
if head_tube_lower_clearance_shortfall > SHORTFALL_TOLERANCE:
        print(f"⚠️  Crown clearance shortfall: need {HEAD_TUBE_LOWER_CLEARANCE_MIN:.1f} mm, currently {head_tube_lower_clearance_physical:.1f} mm. Consider adding stack or reducing HEAD_TUBE_BETWEEN_TUBES.")
print("\nTube Profiles Used (side view):")
print(f"  - Seat Stays: {SEAT_STAY_OD:.1f} mm OD")
print(f"  - Top Tube: {TOP_TUBE_VISIBLE_WIDTH:.1f} mm visible width (flattened {TOP_TUBE_VISIBLE_WIDTH:.0f} x {TOP_TUBE_LATERAL:.0f} mm)")
print(f"  - Down Tube: {DOWN_TUBE_VISIBLE_WIDTH:.1f} mm visible width (flattened 52 x {DOWN_TUBE_LATERAL:.0f} mm)")
print(f"  - Chainstays: {CHAINSTAY_OD:.1f} mm OD")
print(f"  - Seat Tube: {SEAT_TUBE_OD:.1f} mm OD to house a {SEAT_POST_DIAMETER:.1f} mm seatpost")
print(f"  - Head Tube: {HEAD_TUBE_OD:.1f} mm OD (untapered depiction for 1.5\" to 1-1/8\")")
print("=" * 85)


# --- Visualization ---

def draw_tube(ax, start, end, width, color, alpha=0.35, edgecolor=None, zorder=2):
        """Render a tube as a rectangle (or flat oval) with the given outer width."""
        x1, y1 = start
        x2, y2 = end
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        if length == 0:
                return None
        nx = -dy / length
        ny = dx / length
        half = width / 2
        offsets = [half, -half]
        points = []
        for h in offsets:
                points.append((x1 + nx * h, y1 + ny * h))
        for h in reversed(offsets):
                points.append((x2 + nx * h, y2 + ny * h))
        polygon = Polygon(points, closed=True, facecolor=color, edgecolor=edgecolor or color, alpha=alpha, zorder=zorder)
        ax.add_patch(polygon)
        return polygon


def annotate_tube(ax, start, end, text, offset=12, text_kwargs=None):
        """Annotate a tube segment with its length, offset perpendicular to the tube."""
        x1, y1 = start
        x2, y2 = end
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        if length == 0:
                return
        # Unit normal vector for perpendicular offset
        nx = -dy / length
        ny = dx / length
        label_x = mid_x + nx * offset
        label_y = mid_y + ny * offset
        final_kwargs = {
                'fontsize': 10,
                'color': 'dimgray',
                'ha': 'center',
                'va': 'center'
        }
        if text_kwargs:
                final_kwargs.update(text_kwargs)
        ax.text(label_x, label_y, text, **final_kwargs)


def annotate_head_tube_segment(ax, start, end, text, label_x, text_kwargs=None):
        """Place a head tube dimension label at a fixed x-coordinate to avoid overlap."""
        x1, y1 = start
        x2, y2 = end
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        final_kwargs = {
                'fontsize': 9,
                'color': '#0F3557',
                'ha': 'left',
                'va': 'center'
        }
        if text_kwargs:
                final_kwargs.update(text_kwargs)
        connector_end_x = label_x - 6
        ax.plot([mid_x, connector_end_x], [mid_y, mid_y], linestyle='--', linewidth=0.8,
                color=final_kwargs['color'], alpha=0.7)
        ax.text(label_x, mid_y, text, **final_kwargs)


def head_tube_point(distance_along_axis):
        """Return the coordinate of a point along the head tube axis measured from the top."""
        return (
                head_tube_top_x + head_axis_x * distance_along_axis,
                head_tube_top_y + head_axis_y * distance_along_axis
        )


fig, ax = plt.subplots(figsize=(20, 12))
ax.set_aspect('equal')

# Ground line
ax.axhline(y=ground_y, color='saddlebrown', linewidth=3, alpha=0.5, label='Ground')
# Axle-to-axle line and bottom bracket drop indicator
ax.axhline(y=rear_wheel_y, color='gray', linewidth=1.8, linestyle=':', alpha=0.5, label='Axle Line')
ax.plot([bb_x, bb_x], [rear_wheel_y, bb_y], color='firebrick', linestyle='--', linewidth=2, alpha=0.9)
ax.text(bb_x + 18, bb_y + 12, f'BB Drop: {BB_DROP:.0f} mm', color='firebrick', fontsize=11,
        va='bottom', ha='left', bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.75, ec='none'))

# Wheels and Chainring
rear_wheel = Circle((rear_wheel_x, rear_wheel_y), WHEEL_RADIUS, fill=False, edgecolor='black', linewidth=3, alpha=0.8)
front_wheel = Circle((front_wheel_x, front_wheel_y), WHEEL_RADIUS, fill=False, edgecolor='black', linewidth=3, alpha=0.8)
chainring = Circle((bb_x, bb_y), CHAINRING_RADIUS, fill=False, edgecolor='gray', linewidth=2, linestyle='--', alpha=0.6)
ax.add_patch(rear_wheel)
ax.add_patch(front_wheel)
ax.add_patch(chainring)
ax.annotate(f"{WHEEL_DIAMETER:.0f} mm Ø", (front_wheel_x, front_wheel_y + WHEEL_RADIUS),
            xytext=(HEAD_TUBE_LABEL_X_OFFSET - 10, 40), textcoords='offset points', fontsize=11, color='black', ha='left', va='bottom',
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5, shrinkA=5, shrinkB=5))
crank_end_x = bb_x
crank_end_y = bb_y - CRANK_ARM_LENGTH
ax.plot([bb_x, crank_end_x], [bb_y, crank_end_y], color='slategray', linewidth=5, alpha=0.6, solid_capstyle='round', label='Crank Arm')
ax.plot(crank_end_x, crank_end_y, marker='o', color='slategray', markersize=6, alpha=0.8)
ax.text(bb_x + 22, (bb_y + crank_end_y) / 2, f'Crank: {CRANK_ARM_LENGTH:.0f} mm', color='dimgray', fontsize=10,
        rotation=90, va='center', ha='left')

# Frame Tubes
tube_color = '#005A9C' # A nice titanium blue/grey
tube_fill = '#7CA6C7'
draw_tube(ax, (bb_x, bb_y), (seat_tube_visible_top_x, seat_tube_visible_top_y), SEAT_TUBE_OD, tube_fill, alpha=0.45, edgecolor=tube_color, zorder=1.5)
draw_tube(ax, (bb_x, bb_y), (down_tube_weld_x, down_tube_weld_y), DOWN_TUBE_VISIBLE_WIDTH, tube_fill, alpha=0.45, edgecolor=tube_color, zorder=1.5)
draw_tube(ax, (seat_cluster_x, top_tube_rear_y), (top_tube_front_x, top_tube_front_y), TOP_TUBE_VISIBLE_WIDTH, tube_fill, alpha=0.45, edgecolor=tube_color, zorder=1.5)
draw_tube(ax, (head_tube_bottom_x, head_tube_bottom_y), (head_tube_top_x, head_tube_top_y), HEAD_TUBE_OD, tube_fill, alpha=0.45, edgecolor=tube_color, zorder=1.5)
draw_tube(ax, (bb_x, bb_y), (rear_wheel_x, rear_wheel_y), CHAINSTAY_OD, tube_fill, alpha=0.45, edgecolor=tube_color, zorder=1.5)
draw_tube(ax, (seat_stay_front_x, seat_stay_front_y), (seat_stay_rear_x, seat_stay_rear_y), SEAT_STAY_OD, tube_fill, alpha=0.45, edgecolor=tube_color, zorder=1.5)

ax.plot([bb_x, seat_tube_visible_top_x], [bb_y, seat_tube_visible_top_y], color=tube_color, linewidth=8, solid_capstyle='round', label='Seat Tube')

# Compact geometry drop annotation
if TOP_TUBE_SEAT_TUBE_DROP > 0:
        # Draw a faint dotted line showing where the original (undropped) top tube would be
        ax.plot([top_tube_rear_x_original, top_tube_front_x], [top_tube_rear_y_original, top_tube_front_y],
                color='gray', linewidth=2, linestyle=':', alpha=0.5, zorder=1)
        # Draw a small dimension line showing the drop at the seat tube
        drop_line_x = top_tube_rear_x - 30  # offset to the left of the seat tube
        drop_top_y = top_tube_rear_y_original  # original (higher) connection point
        drop_bottom_y = top_tube_rear_y  # new (lower) connection point
        # Vertical dimension line
        ax.plot([drop_line_x, drop_line_x], [drop_top_y, drop_bottom_y], color='darkorange', linewidth=2, zorder=5)
        # Horizontal ticks
        ax.plot([drop_line_x - 5, drop_line_x + 5], [drop_top_y, drop_top_y], color='darkorange', linewidth=2, zorder=5)
        ax.plot([drop_line_x - 5, drop_line_x + 5], [drop_bottom_y, drop_bottom_y], color='darkorange', linewidth=2, zorder=5)
        # Connect to the actual points on the seat tube
        ax.plot([drop_line_x + 5, top_tube_rear_x_original], [drop_top_y, drop_top_y], color='darkorange', linewidth=1, linestyle=':', alpha=0.7, zorder=4)
        ax.plot([drop_line_x + 5, top_tube_rear_x], [drop_bottom_y, drop_bottom_y], color='darkorange', linewidth=1, linestyle=':', alpha=0.7, zorder=4)
        # Label
        ax.text(drop_line_x - 8, (drop_top_y + drop_bottom_y) / 2, f'{TOP_TUBE_SEAT_TUBE_DROP:.0f} mm\ndrop',
                fontsize=10, color='darkorange', ha='right', va='center', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85, ec='darkorange'))
ax.plot([bb_x, down_tube_weld_x], [bb_y, down_tube_weld_y], color=tube_color, linewidth=8, solid_capstyle='round', label='Down Tube')
ax.plot([seat_cluster_x, top_tube_front_x], [top_tube_rear_y, top_tube_front_y], color=tube_color, linewidth=8, solid_capstyle='round', label='Top Tube')
ax.plot([head_tube_bottom_x, head_tube_top_x], [head_tube_bottom_y, head_tube_top_y], color=tube_color, linewidth=10, solid_capstyle='round', label='Head Tube')
ax.plot([bb_x, rear_wheel_x], [bb_y, rear_wheel_y], color=tube_color, linewidth=6, solid_capstyle='round', label='Chainstay')
ax.plot([seat_stay_front_x, seat_stay_rear_x], [seat_stay_front_y, seat_stay_rear_y], color=tube_color, linewidth=6, solid_capstyle='round', label='Seat Stay')

# Fork
fork_color = '#333333'
ax.plot([head_tube_bottom_x, axis_point_x], [head_tube_bottom_y, axis_point_y], color=fork_color, linewidth=6, label='Fork Steerer', alpha=0.9)
ax.plot([axis_point_x, front_wheel_x], [axis_point_y, front_wheel_y], color=fork_color, linewidth=6, alpha=0.9)
ax.plot([head_tube_bottom_x, steering_axis_ground_x], [head_tube_bottom_y, ground_y], color='dimgray', linestyle='--', linewidth=2, alpha=0.8, label='Steering Axis')

# Tube length annotations
annotate_tube(ax, (bb_x, bb_y), (seat_tube_visible_top_x, seat_tube_visible_top_y), f"{seat_tube_visible_length:.0f} mm", offset=36)
annotate_tube(ax, (bb_x, bb_y), (down_tube_weld_x, down_tube_weld_y), f"{down_tube_length:.0f} mm", offset=18)
# Top tube length annotation - show both effective and actual if compact geometry
if TOP_TUBE_SEAT_TUBE_DROP > 0:
        # Effective top tube length (above the faint gray dotted line)
        annotate_tube(ax, (top_tube_rear_x_original, top_tube_rear_y_original), (top_tube_front_x, top_tube_front_y), 
                      f"{top_tube_length:.0f} mm (eff)", offset=20)
        # Actual sloped top tube length (on the actual tube)
        annotate_tube(ax, (seat_cluster_x, top_tube_rear_y), (top_tube_front_x, top_tube_front_y), 
                      f"{top_tube_length_actual:.0f} mm", offset=-18)
else:
        annotate_tube(ax, (seat_cluster_x, top_tube_rear_y), (top_tube_front_x, top_tube_front_y), f"{top_tube_length:.0f} mm")
annotate_tube(ax, (head_tube_bottom_x, head_tube_bottom_y), (head_tube_top_x, head_tube_top_y), f"{HEAD_TUBE_LENGTH:.0f} mm", offset=10)
annotate_tube(ax, (bb_x, bb_y), (rear_wheel_x, rear_wheel_y), f"{chainstay_length_effective:.0f} mm", offset=20)
# Seat stay segment labels (offset left to avoid tube obscuring)
annotate_tube(ax, (seat_stay_front_x, seat_stay_front_y), (seat_stay_wheel_intersect_x, seat_stay_wheel_intersect_y), f"{seat_stay_to_wheel_length:.0f} mm", offset=-24)
annotate_tube(ax, (seat_stay_wheel_intersect_x, seat_stay_wheel_intersect_y), (seat_stay_rear_x, seat_stay_rear_y), f"{seat_stay_from_wheel_length:.0f} mm", offset=-22)
# Full seat stay length label (offset right)
annotate_tube(ax, (seat_stay_front_x, seat_stay_front_y), (seat_stay_rear_x, seat_stay_rear_y), f"{seat_stay_length:.0f} mm", offset=40)
# Mark the wheel intersection point on the seat stay
ax.plot(seat_stay_wheel_intersect_x, seat_stay_wheel_intersect_y, 'o', color='darkorange', markersize=6, zorder=5)
annotate_tube(ax, (head_tube_bottom_x, head_tube_bottom_y), (axis_point_x, axis_point_y), f"{FORK_AXLE_TO_CROWN:.0f} mm", offset=10)

head_tube_top_point = head_tube_point(0.0)
top_tube_upper_point = head_tube_point(top_tube_upper_surface_axis_distance)
top_tube_lower_point = head_tube_point(top_tube_lower_surface_axis_distance)
down_tube_upper_point = head_tube_point(down_tube_upper_surface_axis_distance)
down_tube_lower_point = head_tube_point(down_tube_lower_surface_axis_distance)
head_tube_bottom_point = head_tube_point(HEAD_TUBE_LENGTH)

segment_annotation_kwargs = {'fontsize':9, 'color':'#0F3557'}
head_tube_label_x = head_tube_top_x + HEAD_TUBE_LABEL_X_OFFSET
annotate_head_tube_segment(ax, head_tube_top_point, top_tube_upper_point,
                           f"{top_extension_actual:.0f} mm axial upper",
                           label_x=head_tube_label_x,
                           text_kwargs=segment_annotation_kwargs)
annotate_head_tube_segment(ax, top_tube_upper_point, top_tube_lower_point,
                           f"{top_tube_band_axis:.0f} mm axial top tube",
                           label_x=head_tube_label_x,
                           text_kwargs=segment_annotation_kwargs)
annotate_head_tube_segment(ax, top_tube_lower_point, down_tube_upper_point,
                           f"{HEAD_TUBE_BETWEEN_TUBES:.0f} mm axial clear",
                           label_x=head_tube_label_x,
                           text_kwargs=segment_annotation_kwargs)
annotate_head_tube_segment(ax, down_tube_upper_point, down_tube_lower_point,
                           f"{down_tube_band_axis:.0f} mm axial down tube",
                           label_x=head_tube_label_x,
                           text_kwargs=segment_annotation_kwargs)
annotate_head_tube_segment(ax, down_tube_lower_point, head_tube_bottom_point,
                           f"{head_tube_lower_clearance_physical:.0f} mm axial crown",
                           label_x=head_tube_label_x,
                           text_kwargs=segment_annotation_kwargs)

# Angle annotations
seat_tube_label_x = min(bb_x, seat_tube_visible_top_x) + 10
seat_tube_label_y = max((bb_y + rear_wheel_y) / 2 + 20,
                        bb_y + (seat_tube_visible_top_y - bb_y) * 0.4) - 40
ax.text(seat_tube_label_x, seat_tube_label_y,
        f"{seat_tube_angle_final:.1f}° Seat Tube",
        fontsize=11, color='#0F3557', ha='right', va='center',
        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85, ec='#0F3557'))

head_tube_label_x = front_wheel_x - 100
head_tube_label_y = front_wheel_y + 20
ax.text(head_tube_label_x, head_tube_label_y,
        f"{head_tube_angle_final:.1f}° Head Tube",
        fontsize=11, color='#0F3557', ha='right', va='center',
        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85, ec='#0F3557'))

# Key points
ax.plot(bb_x, bb_y, 'o', color='red', markersize=10, label='Bottom Bracket')
ax.plot(rear_wheel_x, rear_wheel_y, 'o', color='black', markersize=8)
ax.plot(front_wheel_x, front_wheel_y, 'o', color='black', markersize=8)

# Annotations
ax.annotate('BB', (bb_x, bb_y), xytext=(-15, -30), textcoords="offset points", fontsize=12, ha='center')
ax.annotate(f'Final Stack: {final_stack:.0f}\nFinal Reach: {final_reach:.0f}', (head_tube_top_x, head_tube_top_y),
            xytext=(40, 15), textcoords="offset points", ha='left', fontsize=12,
            bbox=dict(boxstyle='round,pad=0.5', fc='lightblue', alpha=0.8))

# Wheelbase dimension line
ax.annotate('', xy=(front_wheel_x, ground_y - 50), xytext=(rear_wheel_x, ground_y - 50),
            arrowprops=dict(arrowstyle='<->', lw=1.5, color='black'))
ax.text((rear_wheel_x + front_wheel_x)/2, ground_y - 75, f'Wheelbase: {wheelbase:.0f} mm',
        ha='center', va='center', fontsize=12, fontweight='bold')

# Trail annotation
front_contact_x = front_wheel_x
ax.plot([front_contact_x, steering_axis_ground_x], [ground_y, ground_y], color='darkorange', linestyle='--', lw=2, label=f'Trail: {trail:.1f}mm')
ax.plot([head_tube_bottom_x, steering_axis_ground_x], [head_tube_bottom_y, ground_y], color='darkorange', linestyle=':', lw=2)
ax.legend(loc='lower right')

# Plot styling
ax.grid(True, linestyle='--', alpha=0.4)
ax.set_xlabel('Distance (mm)', fontsize=12)
ax.set_ylabel('Height (mm)', fontsize=12)
ax.set_title(f'Titanium Frame Geometry (Size {FRAME_SIZE}cm)', fontsize=16, fontweight='bold', pad=20)

# Set limits for a nice view
x_min = rear_wheel_x - WHEEL_RADIUS - 100
x_max = front_wheel_x + WHEEL_RADIUS + 100
y_min = ground_y - 100
y_max = max(seat_tube_top_y, head_tube_top_y) + 100
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

# Scale bar (200 mm)
scale_bar_length = 200  # mm
scale_bar_x = x_min + 50
scale_bar_y = y_min + 40
ax.plot([scale_bar_x, scale_bar_x + scale_bar_length], [scale_bar_y, scale_bar_y], 
        color='black', linewidth=4, solid_capstyle='butt')
ax.plot([scale_bar_x, scale_bar_x], [scale_bar_y - 8, scale_bar_y + 8], color='black', linewidth=2)
ax.plot([scale_bar_x + scale_bar_length, scale_bar_x + scale_bar_length], [scale_bar_y - 8, scale_bar_y + 8], color='black', linewidth=2)
ax.text(scale_bar_x + scale_bar_length / 2, scale_bar_y + 15, '200 mm', 
        fontsize=10, ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('/Users/colincatlin/Documents/aero_duct/titanium_frame_geometry_v4.png', dpi=300)
print("\n✅ Visualization saved as 'titanium_frame_geometry_v4.png'")
backend = plt.get_backend().lower()
if 'agg' in backend:
        plt.close(fig)
else:
        plt.show()
