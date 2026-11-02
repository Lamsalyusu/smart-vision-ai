//! Raw WebAssembly geometry kernels.
//!
//! The module intentionally avoids `wasm-bindgen` so it can be instantiated
//! with plain `WebAssembly.instantiate(bytes, {})`. Memory is owned by the
//! caller: allocate with `alloc_f32`, write coordinates, call a kernel, then
//! release both buffers with `dealloc_f32`.

use std::mem;
use std::slice;

/// Allocate space for `length` `f32` values (xyz triples).
#[no_mangle]
pub extern "C" fn alloc_f32(length: usize) -> *mut f32 {
    if length == 0 {
        return std::ptr::null_mut();
    }
    let mut buffer = Vec::<f32>::with_capacity(length);
    let pointer = buffer.as_mut_ptr();
    mem::forget(buffer);
    pointer
}

/// Release a buffer allocated by [`alloc_f32`].
///
/// # Safety
/// `pointer` must come from `alloc_f32(length)` and must not be used afterwards.
#[no_mangle]
pub unsafe extern "C" fn dealloc_f32(pointer: *mut f32, length: usize) {
    if pointer.is_null() || length == 0 {
        return;
    }
    drop(Vec::from_raw_parts(pointer, 0, length));
}

/// Write the axis-aligned bounds `[min_x, min_y, min_z, max_x, max_y, max_z]`.
///
/// Returns `1` on success and `0` for empty input or non-finite coordinates.
///
/// # Safety
/// `points` must point to `count * 3` valid `f32` values and `output` to six.
#[no_mangle]
pub unsafe extern "C" fn point_cloud_bounds(
    points: *const f32,
    count: usize,
    output: *mut f32,
) -> u32 {
    if points.is_null() || output.is_null() || count == 0 {
        return 0;
    }
    let data = slice::from_raw_parts(points, count.saturating_mul(3));
    let mut minimum = [f32::INFINITY; 3];
    let mut maximum = [f32::NEG_INFINITY; 3];
    let mut finite_points = 0usize;

    for index in (0..data.len()).step_by(3) {
        let x = data[index];
        let y = data[index + 1];
        let z = data[index + 2];
        if !x.is_finite() || !y.is_finite() || !z.is_finite() {
            continue;
        }
        finite_points += 1;
        for (axis, value) in [x, y, z].iter().enumerate() {
            if *value < minimum[axis] {
                minimum[axis] = *value;
            }
            if *value > maximum[axis] {
                maximum[axis] = *value;
            }
        }
    }

    if finite_points == 0 {
        return 0;
    }

    let result = slice::from_raw_parts_mut(output, 6);
    result[0..3].copy_from_slice(&minimum);
    result[3..6].copy_from_slice(&maximum);
    1
}

/// Translate and uniformly scale a point cloud so it is centered at the origin
/// with the requested radius. Returns `0` when the input has no finite bounds.
///
/// # Safety
/// `points` must point to `count * 3` valid `f32` values.
#[no_mangle]
pub unsafe extern "C" fn normalize_point_cloud(
    points: *mut f32,
    count: usize,
    target_radius: f32,
) -> u32 {
    if points.is_null() || count == 0 || !target_radius.is_finite() || target_radius <= 0.0 {
        return 0;
    }
    let mut bounds = [0.0f32; 6];
    let ok = point_cloud_bounds(points, count, bounds.as_mut_ptr());
    if ok == 0 {
        return 0;
    }
    let center = [
        (bounds[0] + bounds[3]) * 0.5,
        (bounds[1] + bounds[4]) * 0.5,
        (bounds[2] + bounds[5]) * 0.5,
    ];
    let diagonal = ((bounds[3] - bounds[0]).powi(2)
        + (bounds[4] - bounds[1]).powi(2)
        + (bounds[5] - bounds[2]).powi(2))
    .sqrt();
    let scale = if diagonal > f32::EPSILON {
        target_radius / (diagonal * 0.5)
    } else {
        1.0
    };
    let data = slice::from_raw_parts_mut(points, count.saturating_mul(3));
    for index in (0..data.len()).step_by(3) {
        data[index] = (data[index] - center[0]) * scale;
        data[index + 1] = (data[index + 1] - center[1]) * scale;
        data[index + 2] = (data[index + 2] - center[2]) * scale;
    }
    1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounds_and_normalization_are_correct() {
        let mut points = vec![-1.0f32, -2.0, -3.0, 1.0, 2.0, 3.0];
        let mut bounds = [0.0f32; 6];
        let ok = unsafe { point_cloud_bounds(points.as_ptr(), 2, bounds.as_mut_ptr()) };
        assert_eq!(ok, 1);
        assert_eq!(bounds, [-1.0, -2.0, -3.0, 1.0, 2.0, 3.0]);

        let ok = unsafe { normalize_point_cloud(points.as_mut_ptr(), 2, 1.0) };
        assert_eq!(ok, 1);
        for coordinate in &points {
            assert!(coordinate.abs() <= 1.0001);
        }
    }

    #[test]
    fn rejects_empty_or_non_finite_input() {
        let mut output = [0.0f32; 6];
        let empty = unsafe { point_cloud_bounds(std::ptr::null(), 0, output.as_mut_ptr()) };
        assert_eq!(empty, 0);

        let nan_points = vec![f32::NAN, 0.0, 0.0];
        let invalid = unsafe { point_cloud_bounds(nan_points.as_ptr(), 1, output.as_mut_ptr()) };
        assert_eq!(invalid, 0);
    }
}
