use std::ffi::{CStr, CString, c_char, c_void};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};

use anyhow::{Context, anyhow};
use brush_process::config::TrainStreamConfig;
use brush_process::message::{ProcessMessage, TrainMessage};
use brush_process::{burn_init_setup, create_process};
use brush_vfs::DataSource;
use tokio::sync::OnceCell;
use tokio_stream::StreamExt;

use crate::startup;

static SETUP: OnceCell<()> = OnceCell::const_new();
static LAST_ERROR: OnceLock<Mutex<Option<CString>>> = OnceLock::new();
static CANCEL_REQUESTED: AtomicBool = AtomicBool::new(false);
static VERSION: &[u8] = b"0.1.0\0";

fn error_slot() -> &'static Mutex<Option<CString>> {
    LAST_ERROR.get_or_init(|| Mutex::new(None))
}

fn sanitize_for_c(message: &str) -> CString {
    let sanitized = message.replace('\0', " ");
    CString::new(sanitized).unwrap_or_else(|_| CString::new("Unknown error").unwrap())
}

fn set_last_error(message: impl AsRef<str>) {
    let value = sanitize_for_c(message.as_ref());
    if let Ok(mut slot) = error_slot().lock() {
        *slot = Some(value);
    }
}

fn clear_last_error_internal() {
    if let Ok(mut slot) = error_slot().lock() {
        *slot = None;
    }
}

fn reset_cancel_internal() {
    CANCEL_REQUESTED.store(false, Ordering::Relaxed);
}

fn cancellation_requested() -> bool {
    CANCEL_REQUESTED.load(Ordering::Relaxed)
}

#[repr(u32)]
#[derive(Clone, Copy)]
pub enum BrushTrainingExitCode {
    Success = 0,
    Error = 1,
}

#[repr(u32)]
#[derive(Clone, Copy)]
pub enum BrushTrainingEventKind {
    ProcessStarted = 0,
    LoadingStarted = 1,
    ConfigResolved = 2,
    DatasetLoaded = 3,
    SplatsUpdated = 4,
    TrainStep = 5,
    RefineStep = 6,
    EvalResult = 7,
    LoadingFinished = 8,
    Warning = 9,
    Done = 10,
    Error = 11,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct BrushTrainingRunConfig {
    pub dataset_path: *const c_char,
    pub output_path: *const c_char,
    pub output_name: *const c_char,
    pub total_train_steps: u32,
    pub refine_every: u32,
    pub max_resolution: u32,
    pub export_every: u32,
    pub eval_every: u32,
    pub seed: u64,
    pub sh_degree: u32,
    pub max_splats: u32,
    pub lod_levels: u32,
    pub lod_refine_steps: u32,
    pub lod_decimation_keep: u32,
    pub lod_image_scale: u32,
    pub lpips_loss_weight: f32,
    pub rerun_enabled: u8,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct BrushTrainingProgress {
    pub kind: BrushTrainingEventKind,
    pub iter: u32,
    pub total_iters: u32,
    pub elapsed_millis: u64,
    pub current_lod: u32,
    pub total_lods: u32,
    pub splat_count: u32,
    pub sh_degree: u32,
    pub train_view_count: u32,
    pub eval_view_count: u32,
    pub avg_psnr: f32,
    pub avg_ssim: f32,
    pub message: *const c_char,
}

pub type BrushTrainingProgressCallback =
    extern "C" fn(progress: *const BrushTrainingProgress, user_data: *mut c_void);

struct ResolvedRun {
    dataset_path: String,
    config: TrainStreamConfig,
}

fn optional_string(ptr: *const c_char) -> Option<String> {
    if ptr.is_null() {
        return None;
    }

    // SAFETY: Caller provides a valid C string pointer for non-null fields.
    let value = unsafe { CStr::from_ptr(ptr).to_string_lossy().into_owned() };
    if value.is_empty() {
        None
    } else {
        Some(value)
    }
}

fn required_string(ptr: *const c_char, field_name: &str) -> anyhow::Result<String> {
    optional_string(ptr).ok_or_else(|| anyhow!("{field_name} must be a non-empty C string"))
}

fn resolve_config(raw: &BrushTrainingRunConfig) -> anyhow::Result<ResolvedRun> {
    let dataset_path = required_string(raw.dataset_path, "dataset_path")?;
    let mut config = TrainStreamConfig::default();

    if let Some(output_path) = optional_string(raw.output_path) {
        config.process_config.export_path = output_path;
    }
    if let Some(output_name) = optional_string(raw.output_name) {
        config.process_config.export_name = output_name;
    }
    if raw.total_train_steps != 0 {
        config.train_config.total_train_iters = raw.total_train_steps;
    }
    if raw.refine_every != 0 {
        config.train_config.refine_every = raw.refine_every;
    }
    if raw.max_resolution != 0 {
        config.load_config.max_resolution = raw.max_resolution;
    }
    if raw.export_every != 0 {
        config.process_config.export_every = raw.export_every;
    }
    if raw.eval_every != 0 {
        config.process_config.eval_every = raw.eval_every;
    }
    if raw.seed != 0 {
        config.process_config.seed = raw.seed;
    }
    if raw.sh_degree != 0 {
        config.model_config.sh_degree = raw.sh_degree;
    }
    if raw.max_splats != 0 {
        config.train_config.max_splats = raw.max_splats;
    }
    if raw.lod_levels != 0 {
        config.train_config.lod_levels = raw.lod_levels;
    }
    if raw.lod_refine_steps != 0 {
        config.train_config.lod_refine_steps = raw.lod_refine_steps;
    }
    if raw.lod_decimation_keep != 0 {
        config.train_config.lod_decimation_keep = raw.lod_decimation_keep;
    }
    if raw.lod_image_scale != 0 {
        config.train_config.lod_image_scale = raw.lod_image_scale;
    }
    if raw.lpips_loss_weight != 0.0 {
        config.train_config.lpips_loss_weight = raw.lpips_loss_weight;
    }

    config.process_config.eval_save_to_disk = true;
    config.rerun_config.rerun_enabled = raw.rerun_enabled != 0;

    Ok(ResolvedRun {
        dataset_path,
        config,
    })
}

fn emit_progress(
    callback: Option<BrushTrainingProgressCallback>,
    user_data: *mut c_void,
    mut progress: BrushTrainingProgress,
    message: Option<&str>,
) {
    let Some(callback) = callback else {
        return;
    };

    let c_message = message.map(sanitize_for_c);
    progress.message = c_message.as_ref().map_or(std::ptr::null(), |value| value.as_ptr());
    callback(&progress as *const BrushTrainingProgress, user_data);
}

#[unsafe(no_mangle)]
pub extern "C" fn brush_training_last_error_message() -> *const c_char {
    if let Ok(slot) = error_slot().lock() {
        return slot.as_ref().map_or(std::ptr::null(), |value| value.as_ptr());
    }
    std::ptr::null()
}

#[unsafe(no_mangle)]
pub extern "C" fn brush_training_clear_last_error() {
    clear_last_error_internal();
}

#[unsafe(no_mangle)]
pub extern "C" fn brush_training_bridge_version() -> *const c_char {
    VERSION.as_ptr() as *const c_char
}

#[unsafe(no_mangle)]
pub extern "C" fn brush_training_request_cancel() {
    CANCEL_REQUESTED.store(true, Ordering::Relaxed);
}

#[unsafe(no_mangle)]
pub extern "C" fn brush_training_reset_cancel() {
    reset_cancel_internal();
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn brush_training_run(
    config: *const BrushTrainingRunConfig,
    callback: Option<BrushTrainingProgressCallback>,
    user_data: *mut c_void,
) -> BrushTrainingExitCode {
    clear_last_error_internal();
    reset_cancel_internal();

    if config.is_null() {
        set_last_error("config must not be null");
        emit_progress(
            callback,
            user_data,
            BrushTrainingProgress {
                kind: BrushTrainingEventKind::Error,
                iter: 0,
                total_iters: 0,
                elapsed_millis: 0,
                current_lod: 0,
                total_lods: 0,
                splat_count: 0,
                sh_degree: 0,
                train_view_count: 0,
                eval_view_count: 0,
                avg_psnr: 0.0,
                avg_ssim: 0.0,
                message: std::ptr::null(),
            },
            Some("config must not be null"),
        );
        return BrushTrainingExitCode::Error;
    }

    // SAFETY: Caller guarantees `config` points to a valid `BrushTrainingRunConfig`.
    let resolved = match resolve_config(unsafe { &*config }).context("Failed to resolve training config") {
        Ok(resolved) => resolved,
        Err(error) => {
            let message = error.to_string();
            set_last_error(&message);
            emit_progress(
                callback,
                user_data,
                BrushTrainingProgress {
                    kind: BrushTrainingEventKind::Error,
                    iter: 0,
                    total_iters: 0,
                    elapsed_millis: 0,
                    current_lod: 0,
                    total_lods: 0,
                    splat_count: 0,
                    sh_degree: 0,
                    train_view_count: 0,
                    eval_view_count: 0,
                    avg_psnr: 0.0,
                    avg_ssim: 0.0,
                    message: std::ptr::null(),
                },
                Some(&message),
            );
            return BrushTrainingExitCode::Error;
        }
    };

    let total_iters = resolved.config.train_config.total_iters();
    let process_config = resolved.config.clone();
    let source = DataSource::Path(resolved.dataset_path.clone());
    let mut process = create_process(source, async move |_| process_config);

    emit_progress(
        callback,
        user_data,
        BrushTrainingProgress {
            kind: BrushTrainingEventKind::ProcessStarted,
            iter: 0,
            total_iters,
            elapsed_millis: 0,
            current_lod: 0,
            total_lods: resolved.config.train_config.lod_levels,
            splat_count: 0,
            sh_degree: resolved.config.model_config.sh_degree,
            train_view_count: 0,
            eval_view_count: 0,
            avg_psnr: 0.0,
            avg_ssim: 0.0,
            message: std::ptr::null(),
        },
        None,
    );

    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("Failed to create tokio runtime")
        .block_on(async {
            SETUP
                .get_or_init(|| async {
                    startup();
                    burn_init_setup().await;
                })
                .await;

            emit_progress(
                callback,
                user_data,
                BrushTrainingProgress {
                    kind: BrushTrainingEventKind::ConfigResolved,
                    iter: 0,
                    total_iters,
                    elapsed_millis: 0,
                    current_lod: 0,
                    total_lods: resolved.config.train_config.lod_levels,
                    splat_count: 0,
                    sh_degree: resolved.config.model_config.sh_degree,
                    train_view_count: 0,
                    eval_view_count: 0,
                    avg_psnr: 0.0,
                    avg_ssim: 0.0,
                    message: std::ptr::null(),
                },
                None,
            );

            while let Some(message_result) = process.stream.next().await {
                match message_result {
                    Ok(message) => match message {
                        ProcessMessage::NewProcess => {}
                        ProcessMessage::StartLoading { name, .. } => emit_progress(
                            callback,
                            user_data,
                            BrushTrainingProgress {
                                kind: BrushTrainingEventKind::LoadingStarted,
                                iter: 0,
                                total_iters,
                                elapsed_millis: 0,
                                current_lod: 0,
                                total_lods: resolved.config.train_config.lod_levels,
                                splat_count: 0,
                                sh_degree: resolved.config.model_config.sh_degree,
                                train_view_count: 0,
                                eval_view_count: 0,
                                avg_psnr: 0.0,
                                avg_ssim: 0.0,
                                message: std::ptr::null(),
                            },
                            Some(&name),
                        ),
                        ProcessMessage::SplatsUpdated {
                            num_splats,
                            sh_degree,
                            ..
                        } => emit_progress(
                            callback,
                            user_data,
                            BrushTrainingProgress {
                                kind: BrushTrainingEventKind::SplatsUpdated,
                                iter: 0,
                                total_iters,
                                elapsed_millis: 0,
                                current_lod: 0,
                                total_lods: resolved.config.train_config.lod_levels,
                                splat_count: num_splats,
                                sh_degree,
                                train_view_count: 0,
                                eval_view_count: 0,
                                avg_psnr: 0.0,
                                avg_ssim: 0.0,
                                message: std::ptr::null(),
                            },
                            None,
                        ),
                        ProcessMessage::TrainMessage(train_message) => match train_message {
                            TrainMessage::TrainConfig { config } => emit_progress(
                                callback,
                                user_data,
                                BrushTrainingProgress {
                                    kind: BrushTrainingEventKind::ConfigResolved,
                                    iter: 0,
                                    total_iters: config.train_config.total_iters(),
                                    elapsed_millis: 0,
                                    current_lod: 0,
                                    total_lods: config.train_config.lod_levels,
                                    splat_count: 0,
                                    sh_degree: config.model_config.sh_degree,
                                    train_view_count: 0,
                                    eval_view_count: 0,
                                    avg_psnr: 0.0,
                                    avg_ssim: 0.0,
                                    message: std::ptr::null(),
                                },
                                None,
                            ),
                            TrainMessage::Dataset { dataset } => emit_progress(
                                callback,
                                user_data,
                                BrushTrainingProgress {
                                    kind: BrushTrainingEventKind::DatasetLoaded,
                                    iter: 0,
                                    total_iters,
                                    elapsed_millis: 0,
                                    current_lod: 0,
                                    total_lods: resolved.config.train_config.lod_levels,
                                    splat_count: 0,
                                    sh_degree: resolved.config.model_config.sh_degree,
                                    train_view_count: dataset.train.views.len() as u32,
                                    eval_view_count: dataset
                                        .eval
                                        .as_ref()
                                        .map_or(0, |scene| scene.views.len() as u32),
                                    avg_psnr: 0.0,
                                    avg_ssim: 0.0,
                                    message: std::ptr::null(),
                                },
                                None,
                            ),
                            TrainMessage::TrainStep {
                                iter,
                                total_elapsed,
                                lod_progress,
                            } => {
                                let (current_lod, total_lods) = lod_progress
                                    .map(|(current, total)| (current, total))
                                    .unwrap_or((0, resolved.config.train_config.lod_levels));
                                emit_progress(
                                    callback,
                                    user_data,
                                    BrushTrainingProgress {
                                        kind: BrushTrainingEventKind::TrainStep,
                                        iter,
                                        total_iters,
                                        elapsed_millis: total_elapsed.as_millis() as u64,
                                        current_lod,
                                        total_lods,
                                        splat_count: 0,
                                        sh_degree: resolved.config.model_config.sh_degree,
                                        train_view_count: 0,
                                        eval_view_count: 0,
                                        avg_psnr: 0.0,
                                        avg_ssim: 0.0,
                                        message: std::ptr::null(),
                                    },
                                    None,
                                );
                            }
                            TrainMessage::RefineStep {
                                cur_splat_count,
                                iter,
                            } => emit_progress(
                                callback,
                                user_data,
                                BrushTrainingProgress {
                                    kind: BrushTrainingEventKind::RefineStep,
                                    iter,
                                    total_iters,
                                    elapsed_millis: 0,
                                    current_lod: 0,
                                    total_lods: resolved.config.train_config.lod_levels,
                                    splat_count: cur_splat_count,
                                    sh_degree: resolved.config.model_config.sh_degree,
                                    train_view_count: 0,
                                    eval_view_count: 0,
                                    avg_psnr: 0.0,
                                    avg_ssim: 0.0,
                                    message: std::ptr::null(),
                                },
                                None,
                            ),
                            TrainMessage::EvalResult {
                                iter,
                                avg_psnr,
                                avg_ssim,
                            } => emit_progress(
                                callback,
                                user_data,
                                BrushTrainingProgress {
                                    kind: BrushTrainingEventKind::EvalResult,
                                    iter,
                                    total_iters,
                                    elapsed_millis: 0,
                                    current_lod: 0,
                                    total_lods: resolved.config.train_config.lod_levels,
                                    splat_count: 0,
                                    sh_degree: resolved.config.model_config.sh_degree,
                                    train_view_count: 0,
                                    eval_view_count: 0,
                                    avg_psnr,
                                    avg_ssim,
                                    message: std::ptr::null(),
                                },
                                None,
                            ),
                            TrainMessage::DoneTraining => emit_progress(
                                callback,
                                user_data,
                                BrushTrainingProgress {
                                    kind: BrushTrainingEventKind::Done,
                                    iter: total_iters,
                                    total_iters,
                                    elapsed_millis: 0,
                                    current_lod: resolved.config.train_config.lod_levels,
                                    total_lods: resolved.config.train_config.lod_levels,
                                    splat_count: 0,
                                    sh_degree: resolved.config.model_config.sh_degree,
                                    train_view_count: 0,
                                    eval_view_count: 0,
                                    avg_psnr: 0.0,
                                    avg_ssim: 0.0,
                                    message: std::ptr::null(),
                                },
                                None,
                            ),
                        },
                        ProcessMessage::Warning { error } => emit_progress(
                            callback,
                            user_data,
                            BrushTrainingProgress {
                                kind: BrushTrainingEventKind::Warning,
                                iter: 0,
                                total_iters,
                                elapsed_millis: 0,
                                current_lod: 0,
                                total_lods: resolved.config.train_config.lod_levels,
                                splat_count: 0,
                                sh_degree: resolved.config.model_config.sh_degree,
                                train_view_count: 0,
                                eval_view_count: 0,
                                avg_psnr: 0.0,
                                avg_ssim: 0.0,
                                message: std::ptr::null(),
                            },
                            Some(&error.to_string()),
                        ),
                        ProcessMessage::DoneLoading => emit_progress(
                            callback,
                            user_data,
                            BrushTrainingProgress {
                                kind: BrushTrainingEventKind::LoadingFinished,
                                iter: 0,
                                total_iters,
                                elapsed_millis: 0,
                                current_lod: 0,
                                total_lods: resolved.config.train_config.lod_levels,
                                splat_count: 0,
                                sh_degree: resolved.config.model_config.sh_degree,
                                train_view_count: 0,
                                eval_view_count: 0,
                                avg_psnr: 0.0,
                                avg_ssim: 0.0,
                                message: std::ptr::null(),
                            },
                            None,
                        ),
                    },
                    Err(error) => {
                        let message = error.to_string();
                        set_last_error(&message);
                        emit_progress(
                            callback,
                            user_data,
                            BrushTrainingProgress {
                                kind: BrushTrainingEventKind::Error,
                                iter: 0,
                                total_iters,
                                elapsed_millis: 0,
                                current_lod: 0,
                                total_lods: resolved.config.train_config.lod_levels,
                                splat_count: 0,
                                sh_degree: resolved.config.model_config.sh_degree,
                                train_view_count: 0,
                                eval_view_count: 0,
                                avg_psnr: 0.0,
                                avg_ssim: 0.0,
                                message: std::ptr::null(),
                            },
                            Some(&message),
                        );
                        return BrushTrainingExitCode::Error;
                    }
                }

                if cancellation_requested() {
                    let message = "Training cancelled";
                    set_last_error(message);
                    emit_progress(
                        callback,
                        user_data,
                        BrushTrainingProgress {
                            kind: BrushTrainingEventKind::Error,
                            iter: 0,
                            total_iters,
                            elapsed_millis: 0,
                            current_lod: 0,
                            total_lods: resolved.config.train_config.lod_levels,
                            splat_count: 0,
                            sh_degree: resolved.config.model_config.sh_degree,
                            train_view_count: 0,
                            eval_view_count: 0,
                            avg_psnr: 0.0,
                            avg_ssim: 0.0,
                            message: std::ptr::null(),
                        },
                        Some(message),
                    );
                    return BrushTrainingExitCode::Error;
                }
            }

            BrushTrainingExitCode::Success
        })
}