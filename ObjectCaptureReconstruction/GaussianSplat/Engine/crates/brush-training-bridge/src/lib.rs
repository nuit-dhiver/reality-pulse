mod ffi;

pub use ffi::*;

use burn_cubecl::cubecl::config::{GlobalConfig, streaming::StreamingConfig};

pub(crate) fn startup() {
    GlobalConfig::set(GlobalConfig {
        streaming: StreamingConfig {
            max_streams: 1,
            ..Default::default()
        },
        ..Default::default()
    });
}