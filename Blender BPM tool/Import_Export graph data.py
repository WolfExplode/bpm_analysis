import bpy
import csv
import os
import datetime
from mathutils import Vector

# ============================================================================ #
# CONFIGURATION
# ============================================================================ #

# Import scaling factors
IMPORT_X_SCALE = 1 / 60.0  # 1m = 1 minute (60 seconds)
IMPORT_Y_SCALE = 1 / 100.0 # 1m = 100 BPM

# Export scaling factors
EXPORT_X_SCALE = 60.0  # X-axis: multiply by 60 for time in seconds

# ============================================================================ #
# CORE FUNCTIONS - IMPORT
# ============================================================================ #

def read_csv_points(file_path):
    """Read CSV and return list of (x, y, z) coordinates.
    Supports (1) two columns: time, BPM e.g. Time (s), BPM (Pass 3), or
    (2) HTML labels export: time_sec,...,bpm_smoothed (uses time_sec + bpm_smoothed; skips empty BPM cells).
    """
    points = []
    try:
        with open(file_path, 'r', newline='') as csvfile:
            csvreader = csv.reader(csvfile)
            header = next(csvreader, None)
            if not header:
                return []
            lowered = [h.strip().lower() for h in header]

            if "time_sec" in lowered and "bpm_smoothed" in lowered:
                i_t = lowered.index("time_sec")
                i_bpm = lowered.index("bpm_smoothed")
                for row in csvreader:
                    if len(row) <= max(i_t, i_bpm):
                        continue
                    try:
                        time_sec = float(row[i_t].strip())
                        b_cell = row[i_bpm].strip()
                        if not b_cell:
                            continue
                        value = float(b_cell)
                        x = time_sec * IMPORT_X_SCALE
                        y = value * IMPORT_Y_SCALE
                        points.append((x, y, 0.0))
                    except ValueError:
                        print(f"Skipping invalid row: {row}")
            else:
                for row in csvreader:
                    if len(row) != 2:
                        continue
                    try:
                        time_sec = float(row[0])
                        value = float(row[1])
                        x = time_sec * IMPORT_X_SCALE
                        y = value * IMPORT_Y_SCALE
                        points.append((x, y, 0.0))
                    except ValueError:
                        print(f"Skipping invalid row: {row}")

    except Exception as e:
        print(f"Error reading CSV: {e}")
        return []

    return points

def create_curve_from_points(point_list, curve_name):
    """Create a Blender curve object from list of 3D points."""
    if not point_list:
        return None
    
    # Remove existing curve with same name
    if curve_name in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects[curve_name])
    
    # Create curve data
    curve_data = bpy.data.curves.new(name=curve_name, type='CURVE')
    curve_data.dimensions = '3D'
    
    # Create object
    curve_object = bpy.data.objects.new(curve_name, curve_data)
    bpy.context.collection.objects.link(curve_object)
    
    # Create poly spline (direct point mapping, no smoothing)
    spline = curve_data.splines.new(type='POLY')
    spline.points.add(len(point_list) - 1)
    
    # Set point coordinates (x, y, z, weight)
    for i, (x, y, z) in enumerate(point_list):
        spline.points[i].co = (x, y, z, 1.0)
    
    return curve_object

# ============================================================================ #
# CORE FUNCTIONS - EXPORT
# ============================================================================ #

def get_points_from_mesh(mesh_data):
    """Get list of (x_blender, y_blender) from mesh vertices, sorted by x."""
    if not hasattr(mesh_data, 'vertices'):
        raise TypeError("Expected mesh data (has .vertices); got %s. Curves must be converted to mesh first." % type(mesh_data).__name__)
    points = [(v.co[0], v.co[1]) for v in mesh_data.vertices]
    points.sort(key=lambda p: p[0])
    return points

def interpolate_y_at_x_seconds(points_sec, x_sec):
    """
    Linear interpolation: return y at given x_sec from sorted list of (x_sec, y).
    Uses first/last y for x before/after data range.
    """
    if not points_sec:
        return 0.0
    if x_sec <= points_sec[0][0]:
        return points_sec[0][1]
    if x_sec >= points_sec[-1][0]:
        return points_sec[-1][1]
    for i in range(len(points_sec) - 1):
        x0, y0 = points_sec[i]
        x1, y1 = points_sec[i + 1]
        if x0 <= x_sec <= x1:
            if x1 == x0:
                return y0
            t = (x_sec - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
    return points_sec[-1][1]

def _get_series_for_export(mesh_data, original_object_name):
    """
    Return (points_sec, y_header, format_type) for one object.
    format_type: 'bpm'|'spo2'|'hrv'|'pcg'|'velocity_ms'|'velocity_kmh'|'default'
    """
    y_header = "y_scaled"
    scale_factor_y = 1.0
    if "BPM" in original_object_name:
        y_header = "BPM"
        scale_factor_y = 100.0
        format_type = "bpm"
    elif "Velocity" in original_object_name:
        if "km/h" in original_object_name:
            y_header = "km/h"
            format_type = "velocity_kmh"
        else:
            y_header = "Speed m/s"
            format_type = "velocity_ms"
        scale_factor_y = 3.6
    elif "SpO2" in original_object_name:
        y_header = "Arterial SpO₂"
        scale_factor_y = 100.0
        format_type = "spo2"
    elif "HRV" in original_object_name:
        y_header = "rMSSD(ms)"
        scale_factor_y = 1000.0
        format_type = "hrv"
    elif "PCG" in original_object_name:
        y_header = "RMS Amplitude(mV)"
        scale_factor_y = 1.0
        format_type = "pcg"
    elif "Respiratory Rate" in original_object_name:
        y_header = "Respiratory Rate(brpm)"
        scale_factor_y = 100.0  # 0.01 Blender unit = 1 brpm
        format_type = "respiratory"
    else:
        format_type = "default"

    is_spo2 = "SpO2" in original_object_name
    points_blender = get_points_from_mesh(mesh_data)
    points_sec = []
    for x_b, y_b in points_blender:
        time_sec = max(0.0, x_b * EXPORT_X_SCALE)
        y_val = max(0.0, y_b * scale_factor_y)
        if is_spo2:
            y_val = min(100.0, y_val)
        points_sec.append((time_sec, y_val))
    return (points_sec, y_header, format_type)

def _format_export_value(y, format_type):
    """Format a single value for CSV based on series type."""
    if format_type == "bpm":
        return round(y)
    if format_type == "respiratory":
        return round(y)  # brpm
    if format_type in ("hrv", "spo2"):
        return float("{:.1f}".format(y))
    if format_type == "pcg":
        return float("{:.2f}".format(y))
    if format_type == "velocity_kmh":
        return float("{:.2f}".format(y * 3.6))
    return float("{:.2f}".format(y))

def export_mesh_data_multi(series_list, data_file_path, step_sec):
    """
    Export multiple series into one CSV with shared time axis.
    series_list: list of (points_sec, y_header, format_type) from _get_series_for_export.
    """
    if not series_list:
        return
    end_sec = max(int(pts[-1][0]) for pts, _, _ in series_list if pts)
    headers = ["time(s)"] + [h for _, h, _ in series_list]
    rows = []
    for t in range(0, end_sec + 1, step_sec):
        row = [t]
        for points_sec, _, format_type in series_list:
            y = interpolate_y_at_x_seconds(points_sec, float(t))
            if format_type == "spo2":
                y = min(100.0, y)
            if format_type in ("velocity_ms", "velocity_kmh"):
                y = max(0.0, y)
            row.append(_format_export_value(y, format_type))
        rows.append(row)
    with open(data_file_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
    print(f"Exported multi-series to {data_file_path} ({len(rows)} rows, {len(series_list)} series).")

def export_mesh_data(mesh_data, original_object_name, data_file_path, step_sec=1):
    """
    Export graph to CSV with time in seconds.
    Samples the graph every step_sec seconds (0, step_sec, 2*step_sec, ...).
    step_sec comes from the UI "Export resolution scale" (1.0 = 1s, 0.1 = 10s).
    """
    # Determine Y-Axis Header and Scaling based on object name
    y_header = "y_scaled"
    scale_factor_y = 1.0

    if "BPM" in original_object_name:
        y_header = "BPM"
        scale_factor_y = 100.0
    elif "Velocity" in original_object_name:
        # Unit chosen by name: "km/h" or "m/s" (default m/s)
        if "km/h" in original_object_name:
            y_header = "km/h"
            scale_factor_y = 3.6  # graph -> m/s, then we export km/h = m/s * 3.6 in row
        else:
            y_header = "Speed m/s"
            scale_factor_y = 3.6
    elif "SpO2" in original_object_name:
        y_header = "Arterial SpO₂"
        scale_factor_y = 100.0
    elif "HRV" in original_object_name:
        y_header = "rMSSD(ms)"
        scale_factor_y = 1000.0  # 0.1 Blender unit = 100 ms
    elif "PCG" in original_object_name:
        y_header = "RMS Amplitude (mV)"
        scale_factor_y = 1.0
    elif "Respiratory Rate" in original_object_name:
        y_header = "Respiratory Rate(brpm)"
        scale_factor_y = 100.0  # 0.01 Blender unit = 1 brpm

    is_spo2 = "SpO2" in original_object_name
    is_velocity = "Velocity" in original_object_name
    is_velocity_kmh = is_velocity and "km/h" in original_object_name
    is_bpm = "BPM" in original_object_name
    is_hrv = "HRV" in original_object_name
    is_pcg = "PCG" in original_object_name
    is_respiratory = "Respiratory Rate" in original_object_name

    # Get (x_blender, y_blender) and convert to (time_sec, y_scaled)
    points_blender = get_points_from_mesh(mesh_data)
    points_sec = []
    for x_b, y_b in points_blender:
        time_sec = x_b * EXPORT_X_SCALE
        y_val = y_b * scale_factor_y
        time_sec = max(0.0, time_sec)
        y_val = max(0.0, y_val)
        if is_spo2:
            y_val = min(100.0, y_val)  # SpO2 ceiling 100%
        # Velocity: already floored at 0 above
        points_sec.append((time_sec, y_val))

    if not points_sec:
        print("No vertices to export.")
        return

    max_sec = points_sec[-1][0]
    # Sample at every integer second from 0 up to and including floor(max_sec)
    start_sec = 0
    end_sec = int(max_sec)

    print(f"\nExporting '{original_object_name}' to: {data_file_path}")
    print(f"Time range: 0s – {max_sec:.2f}s")
    print(f"Sampling every {step_sec}s: {start_sec} to {end_sec}")
    print(f"Y-Axis: {y_header} (scale {scale_factor_y})")

    # For Velocity, add km/h column (1 m/s = 3.6 km/h)
    rows = []
    for t in range(start_sec, end_sec + 1, step_sec):
        y = interpolate_y_at_x_seconds(points_sec, float(t))
        if is_spo2:
            y = min(100.0, y)
        if is_velocity:
            y = max(0.0, y)
        if is_bpm:
            y_fmt = round(y)  # BPM: nearest whole number
        elif is_respiratory:
            y_fmt = round(y)  # brpm
        elif is_hrv or is_spo2:
            y_fmt = float("{:.1f}".format(y))  # rMSSD, SpO2: 1 decimal
        elif is_pcg:
            y_fmt = float("{:.2f}".format(y))  # RMS Amplitude (mV): 2 decimals
        else:
            y_fmt = float("{:.2f}".format(y))
        if is_velocity_kmh:
            y_fmt = float("{:.2f}".format(y * 3.6))  # export km/h
        if is_velocity:
            rows.append((t, y_fmt))  # m/s or km/h
        else:
            rows.append((t, y_fmt))

    with open(data_file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        if is_velocity:
            writer.writerow(['time(s)', y_header])
        else:
            writer.writerow(['time(s)', y_header])
        writer.writerows(rows)

    print(f"Successfully exported {len(rows)} rows (one per {step_sec}s).")

def get_export_file_path(object_name, ext=".csv", timestamp_override=None):
    """Generate export file path based on blend file location and timestamp.
    If timestamp_override is provided (e.g. collection name like '2025-09-22T212300-0500'),
    it is used instead of the current datetime."""
    if not bpy.data.filepath:
        raise Exception("The current Blender file has not been saved. Please save before exporting.")
    
    blend_dir = os.path.dirname(bpy.data.filepath)
    timestamp = timestamp_override if timestamp_override else datetime.datetime.now().strftime("%Y-%m-%dT%H%M%S%z")
    filename = f"{object_name}_{timestamp}{ext}"
    return os.path.join(blend_dir, filename)

# BPM scale for ΔHR: same as BPM graph (1 Blender unit = 100 BPM on Y, 1 unit = 60s on X)
BPM_EXPORT_Y_SCALE = 100.0

def get_segments_from_curve(curve_obj, depsgraph):
    """
    Get segments from a curve using spline structure: each spline defines one or
    more segments. For each spline, pairs (points[0], points[1]), (points[2], points[3]), ...
    form segments. So one spline with 2 points = 1 segment; 4 points = 2 segments.
    Returns list of ((x_a, y_a), (x_b, y_b)) in world space.
    """
    eval_obj = curve_obj.evaluated_get(depsgraph)
    curve_data = eval_obj.data
    segments = []
    for spline in curve_data.splines:
        pts = spline.points
        n = len(pts)
        for i in range(0, n - 1, 2):
            co_a = Vector((pts[i].co[0], pts[i].co[1], pts[i].co[2]))
            co_b = Vector((pts[i + 1].co[0], pts[i + 1].co[1], pts[i + 1].co[2]))
            world_a = eval_obj.matrix_world @ co_a
            world_b = eval_obj.matrix_world @ co_b
            segments.append(((world_a.x, world_a.y), (world_b.x, world_b.y)))
    return segments

def _segments_from_flat_points(points):
    """Build segments from flat point list: pairs (0,1), (2,3), ... (for mesh ΔHR)."""
    n = len(points)
    if n % 2 != 0:
        n = n - 1
    return [(points[i], points[i + 1]) for i in range(0, n, 2)]

def export_dhr_markdown(segments, original_object_name, data_file_path):
    """
    Export ΔHR as markdown. segments: list of ((x_a, y_a), (x_b, y_b)) in Blender world space.
    Segments are sorted by start time so output is chronological.
    Writes one line per segment, e.g.:
    91.6bpm@14.9s to 185.6bpm@68.6s. Δ94.0bpm in 53.8s, +1.75bpm/s
    """
    if not segments:
        print("ΔHR export: no segments.")
        return

    # Sort by start time (x_a = time axis) so export is chronological
    segments = sorted(segments, key=lambda seg: seg[0][0])

    lines = []
    for (x_a, y_a), (x_b, y_b) in segments:
        time_a = x_a * EXPORT_X_SCALE
        time_b = x_b * EXPORT_X_SCALE
        bpm_a = y_a * BPM_EXPORT_Y_SCALE
        bpm_b = y_b * BPM_EXPORT_Y_SCALE
        delta_bpm = bpm_b - bpm_a
        delta_t = time_b - time_a
        slope = (delta_bpm / delta_t) if abs(delta_t) > 1e-9 else 0.0
        line = (
            f"{bpm_a:.1f}bpm@{time_a:.1f}s to {bpm_b:.1f}bpm@{time_b:.1f}s. "
            f"Δ{delta_bpm:.1f}bpm in {abs(delta_t):.1f}s, {slope:+.2f}bpm/s"
        )
        lines.append(line)

    with open(data_file_path, 'w', newline='', encoding='utf-8') as f:
        f.write("\n".join(lines))

    print(f"Exported ΔHR '{original_object_name}' to: {data_file_path} ({len(lines)} segments)")

# ============================================================================ #
# OPERATORS
# ============================================================================ #

class BPM_ExportSettings(bpy.types.PropertyGroup):
    """Settings for CSV export (time resolution)."""
    export_resolution_scale: bpy.props.FloatProperty(
        name="Export resolution scale",
        description="1.0 = one point per second; 0.1 = one point per 10 seconds",
        default=1.0,
        min=0.01,
        max=1.0,
        step=0.1,
        precision=2,
    )

class BPM_OT_Import_Curve(bpy.types.Operator):
    """Import BPM data from CSV and create a curve"""
    bl_idname = "bpm.import_curve"
    bl_label = "Import CSV as Curve"
    bl_description = "Select CSV file and create a curve with proper scaling"
    bl_options = {'REGISTER', 'UNDO'}
    
    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    filename_ext = ".csv"
    filter_glob: bpy.props.StringProperty(default="*.csv", options={'HIDDEN'})
    
    curve_name: bpy.props.StringProperty(
        name="Curve Name",
        description="Name for the created curve object",
        default="BPM_Curve"
    )
    
    def execute(self, context):
        if not self.filepath:
            self.report({'ERROR'}, "No file selected")
            return {'CANCELLED'}
        
        points = read_csv_points(self.filepath)
        
        if not points:
            self.report({'ERROR'}, "Failed to read CSV or no valid data")
            return {'CANCELLED'}
        
        # Use filename as curve name if default is unchanged
        if self.curve_name == "BPM_Curve":
            self.curve_name = os.path.splitext(os.path.basename(self.filepath))[0]
        
        curve = create_curve_from_points(points, self.curve_name)
        
        if curve:
            try:
                if context.area and context.area.type == 'VIEW_3D':
                    bpy.ops.object.select_all(action='DESELECT')
                    curve.select_set(True)
                    context.view_layer.objects.active = curve
                    bpy.ops.view3d.view_selected(use_all_regions=False)
            except RuntimeError:
                pass
            
            self.report({'INFO'}, f"Created '{self.curve_name}' with {len(points)} points")
            return {'FINISHED'}
        else:
            self.report({'ERROR'}, "Failed to create curve")
            return {'CANCELLED'}
    
    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}
    
    def draw(self, context):
        layout = self.layout
        layout.prop(self, "curve_name")

class BPM_OT_Export_Curve(bpy.types.Operator):
    """Export selected curve or mesh to CSV (single or multiple objects in one file).
    Curves are duplicated and converted to mesh for sampling so that geometry node
    modifiers (and other modifiers) are applied in the exported data; the duplicate
    is removed after export."""
    bl_idname = "bpm.export_curve"
    bl_label = "Export Selected Object"
    bl_description = "Export active mesh or curve to CSV; multiple selected → one CSV with shared time axis"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        selected = [o for o in context.selected_objects if o.type in {'MESH', 'CURVE'}]
        if not selected:
            self.report({'ERROR'}, "No mesh or curve selected")
            return {'CANCELLED'}

        # ΔHR exports as markdown; exclude from CSV multi-export
        csv_candidates = [o for o in selected if "ΔHR" not in o.name]
        dhr_only = [o for o in selected if "ΔHR" in o.name]

        try:
            depsgraph = context.evaluated_depsgraph_get()
            resolution_scale = context.scene.bpm_export_settings.export_resolution_scale
            step_sec = max(1, round(1.0 / resolution_scale))

            # Single object: existing behavior (including ΔHR → markdown)
            if len(selected) == 1:
                obj = selected[0]
                return self._export_single(context, obj, depsgraph, step_sec)

            # Multiple objects: one CSV with shared time axis (skip ΔHR in this file)
            if not csv_candidates:
                self.report({'ERROR'}, "Select at least one non-ΔHR mesh/curve for CSV export.")
                return {'CANCELLED'}

            return self._export_multi(context, csv_candidates, depsgraph, step_sec)

        except Exception as e:
            self.report({'ERROR'}, f"Export failed: {str(e)}")
            return {'CANCELLED'}

    def _export_single(self, context, obj, depsgraph, step_sec):
        """Export one object (CSV or ΔHR markdown)."""
        object_name = obj.name.replace(" ", "_")
        is_dhr = "ΔHR" in obj.name
        # Use collection name as timestamp override (e.g. collection named '2025-09-22T212300-0500')
        timestamp_override = obj.users_collection[0].name if obj.users_collection else None
        data_file_path = get_export_file_path(object_name, ext=".md" if is_dhr else ".csv", timestamp_override=timestamp_override)

        if is_dhr and obj.type == "CURVE":
            segments = get_segments_from_curve(obj, depsgraph)
            if segments:
                export_dhr_markdown(segments, obj.name, data_file_path)
                self.report({'INFO'}, f"Exported: {os.path.basename(data_file_path)}")
                return {'FINISHED'}
            self.report({'ERROR'}, "ΔHR curve has no segments (each spline needs at least 2 points).")
            return {'CANCELLED'}

        if is_dhr and obj.type == "MESH":
            points = [(v.co[0], v.co[1]) for v in obj.data.vertices]
            segments = _segments_from_flat_points(points)
            if segments:
                export_dhr_markdown(segments, obj.name, data_file_path)
                self.report({'INFO'}, f"Exported: {os.path.basename(data_file_path)}")
                return {'FINISHED'}
            self.report({'ERROR'}, "ΔHR mesh has no vertices.")
            return {'CANCELLED'}

        mesh_to_export = None
        temp_object = None
        eval_obj = None

        if obj.type == "MESH":
            eval_obj = obj.evaluated_get(depsgraph)
            mesh_to_export = eval_obj.to_mesh()
        elif obj.type == "CURVE":
            # Duplicate the curve so we can convert it to mesh for vertex sampling without
            # modifying the original. Conversion uses the object's evaluated state, so
            # geometry node modifiers (and other modifiers) are applied in the exported data.
            # The duplicate is removed in the finally block below.
            bpy.ops.object.select_all(action='DESELECT')
            obj.select_set(True)
            context.view_layer.objects.active = obj
            bpy.ops.object.duplicate(linked=False)
            temp_object = context.object
            bpy.ops.object.convert(target="MESH")
            mesh_to_export = temp_object.data

        if not mesh_to_export:
            if temp_object:
                bpy.data.objects.remove(temp_object, do_unlink=True)
            self.report({'ERROR'}, "Failed to get mesh data")
            return {'CANCELLED'}

        try:
            export_mesh_data(mesh_to_export, obj.name, data_file_path, step_sec=step_sec)
            self.report({'INFO'}, f"Exported: {os.path.basename(data_file_path)}")
            return {'FINISHED'}
        finally:
            if obj.type == "MESH" and eval_obj is not None:
                eval_obj.to_mesh_clear()
            if temp_object:
                context.view_layer.objects.active = obj
                bpy.data.objects.remove(temp_object, do_unlink=True)

    def _export_multi(self, context, objects, depsgraph, step_sec):
        """Export multiple objects into one CSV with shared time axis."""
        name_part = "_".join(o.name.replace(" ", "_") for o in objects[:3])
        if len(objects) > 3:
            name_part += f"_and_{len(objects) - 3}_more"
        # Use first object's collection name as timestamp override
        timestamp_override = objects[0].users_collection[0].name if objects and objects[0].users_collection else None
        data_file_path = get_export_file_path(name_part, ext=".csv", timestamp_override=timestamp_override)

        mesh_list = []
        temp_objects = []
        eval_meshes = []

        for obj in objects:
            if obj.type == "MESH":
                eval_obj = obj.evaluated_get(depsgraph)
                mesh_data = eval_obj.to_mesh()
                mesh_list.append((mesh_data, obj.name))
                eval_meshes.append(eval_obj)
            else:
                # Duplicate curve so we can convert to mesh with modifiers (e.g. geometry nodes)
                # applied; duplicate is removed in the finally block below.
                bpy.ops.object.select_all(action='DESELECT')
                obj.select_set(True)
                context.view_layer.objects.active = obj
                bpy.ops.object.duplicate(linked=False)
                temp = context.object
                bpy.ops.object.convert(target="MESH")
                mesh_list.append((temp.data, obj.name))
                temp_objects.append(temp)

        try:
            series_list = [_get_series_for_export(mesh_data, name) for mesh_data, name in mesh_list]
            # Drop any series with no points
            series_list = [s for s in series_list if s[0]]
            if not series_list:
                self.report({'ERROR'}, "No valid series to export.")
                return {'CANCELLED'}
            export_mesh_data_multi(series_list, data_file_path, step_sec)
            self.report({'INFO'}, f"Exported {len(series_list)} series: {os.path.basename(data_file_path)}")
            return {'FINISHED'}
        finally:
            for eval_obj in eval_meshes:
                eval_obj.to_mesh_clear()
            if objects:
                context.view_layer.objects.active = objects[0]
            for temp in temp_objects:
                bpy.data.objects.remove(temp, do_unlink=True)

# ============================================================================ #
# UI PANEL
# ============================================================================ #

class BPM_PT_Panel(bpy.types.Panel):
    """Panel for BPM curve import and export tools"""
    bl_label = "BPM Curve Tools"
    bl_idname = "BPM_PT_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Animation"
    
    def draw(self, context):
        layout = self.layout
        
        # Import Section
        box = layout.box()
        
        # Import scaling info
        box.label(text="Scaling:", icon="INFO")
        box.label(text="  X: 1m = 1 minute")
        box.label(text="  Y: 1m = 100 BPM")
        
        layout.operator(BPM_OT_Import_Curve.bl_idname, icon="FILE_FOLDER")
        
        layout.separator()
        
        # Selection status and export button
        selected = [o for o in context.selected_objects if o.type in {'MESH', 'CURVE'}]
        if selected:
            col = layout.column(align=True)
            if len(selected) == 1:
                col.label(text=f"Selected: {selected[0].name}", icon="OBJECT_DATA")
            else:
                col.label(text=f"{len(selected)} objects selected (export to one CSV)", icon="OBJECT_DATA")
            col.prop(context.scene.bpm_export_settings, "export_resolution_scale")
            col.operator(BPM_OT_Export_Curve.bl_idname, icon="FILE_TICK")
        elif context.object:
            col = layout.column(align=True)
            col.label(text="Select a mesh or curve", icon="ERROR")
        else:
            layout.label(text="No object selected", icon="ERROR")

# ============================================================================ #
# REGISTRATION
# ============================================================================ #

classes = (
    BPM_ExportSettings,
    BPM_OT_Import_Curve,
    BPM_OT_Export_Curve,
    BPM_PT_Panel
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.bpm_export_settings = bpy.props.PointerProperty(type=BPM_ExportSettings)

def unregister():
    try:
        del bpy.types.Scene.bpm_export_settings
    except AttributeError:
        pass
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

if __name__ == "__main__":
    try:
        unregister()
    except RuntimeError:
        pass
    register()