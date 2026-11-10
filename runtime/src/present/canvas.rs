//! The swapchain canvas: the guest frame, kept 1:1, padded with black bars to the client area's
//! aspect ratio. DXGI's stretch scaling then scales the canvas uniformly, so the frame keeps its
//! aspect ratio while D3D12 only copies and presents (no draw).

use super::d3d12::footprint::MAX_DIMENSION;
use crate::{Error, Result};

/// A canvas and the place of the guest image inside it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Canvas {
    pub width: u32,
    pub height: u32,
    pub image_x: u32,
    pub image_y: u32,
    pub image_width: u32,
    pub image_height: u32,
}

impl Canvas {
    /// Fits an `image_width` x `image_height` frame to a `client_width` x `client_height` client
    /// area. Bars go on the axis the client area has relatively more of; the image is centred
    /// (odd remainders put the extra pixel after it) and canvas edges are clamped to 16384.
    pub fn fit(
        image_width: u32,
        image_height: u32,
        client_width: u32,
        client_height: u32,
    ) -> Result<Self> {
        let outside = |edge: u32| edge == 0 || edge > MAX_DIMENSION;
        if outside(image_width) || outside(image_height) {
            return Err(Error::invalid(format!(
                "present: frame {image_width}x{image_height} is outside 1..={MAX_DIMENSION}"
            )));
        }
        if client_width == 0 || client_height == 0 {
            return Err(Error::invalid(format!(
                "present: client area {client_width}x{client_height} is empty"
            )));
        }
        let [image_w, image_h, client_w, client_h] =
            [image_width, image_height, client_width, client_height].map(u64::from);
        let rounded_ratio = |a: u64, b: u64, c: u64| (a * b + c / 2) / c;
        let (width, height) = if image_w * client_h >= client_w * image_h {
            (image_w, rounded_ratio(image_w, client_h, client_w))
        } else {
            (rounded_ratio(image_h, client_w, client_h), image_h)
        };
        let max = u64::from(MAX_DIMENSION);
        let width = width.clamp(image_w, max) as u32;
        let height = height.clamp(image_h, max) as u32;
        Ok(Self {
            width,
            height,
            image_x: (width - image_width) / 2,
            image_y: (height - image_height) / 2,
            image_width,
            image_height,
        })
    }

    /// A canvas that is exactly the image: no window, or a client area of the same shape.
    pub fn exact(width: u32, height: u32) -> Result<Self> {
        Self::fit(width, height, width, height)
    }

    pub fn has_bars(&self) -> bool {
        self.width != self.image_width || self.height != self.image_height
    }

    /// The rectangle a `width` x `height` plane covers when it is fit to the canvas on its own,
    /// as each VideoOut plane is fit to the window: scaled uniformly until one axis fills the
    /// canvas, the other rounded to nearest, and centred like the image. A plane the size of the
    /// image covers exactly the image.
    pub fn plane_rect(&self, width: u32, height: u32) -> Result<PlaneRect> {
        if width == 0 || height == 0 || width > MAX_DIMENSION || height > MAX_DIMENSION {
            return Err(Error::invalid(format!(
                "present: plane {width}x{height} is outside 1..={MAX_DIMENSION}"
            )));
        }
        let [plane_w, plane_h, canvas_w, canvas_h] =
            [width, height, self.width, self.height].map(u64::from);
        let rounded_ratio = |a: u64, b: u64, c: u64| (a * b + c / 2) / c;
        let (rect_w, rect_h) = if plane_w * canvas_h >= canvas_w * plane_h {
            (
                canvas_w,
                rounded_ratio(plane_h, canvas_w, plane_w).clamp(1, canvas_h),
            )
        } else {
            (
                rounded_ratio(plane_w, canvas_h, plane_h).clamp(1, canvas_w),
                canvas_h,
            )
        };
        let (rect_w, rect_h) = (rect_w as u32, rect_h as u32);
        Ok(PlaneRect {
            x: (self.width - rect_w) / 2,
            y: (self.height - rect_h) / 2,
            width: rect_w,
            height: rect_h,
        })
    }

    /// Whether `rect` lies inside the image, leaving the bars alone.
    pub fn image_contains(&self, rect: &PlaneRect) -> bool {
        rect.x >= self.image_x
            && rect.y >= self.image_y
            && rect.x + rect.width <= self.image_x + self.image_width
            && rect.y + rect.height <= self.image_y + self.image_height
    }
}

/// A rectangle of canvas pixels.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PlaneRect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}
