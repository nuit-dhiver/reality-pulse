#ifndef BRUSH_TRAINING_BRIDGE_H
#define BRUSH_TRAINING_BRIDGE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum BrushTrainingExitCode {
    BRUSH_TRAINING_EXIT_SUCCESS = 0,
    BRUSH_TRAINING_EXIT_ERROR = 1,
} BrushTrainingExitCode;

typedef enum BrushTrainingEventKind {
    BRUSH_TRAINING_EVENT_PROCESS_STARTED = 0,
    BRUSH_TRAINING_EVENT_LOADING_STARTED = 1,
    BRUSH_TRAINING_EVENT_CONFIG_RESOLVED = 2,
    BRUSH_TRAINING_EVENT_DATASET_LOADED = 3,
    BRUSH_TRAINING_EVENT_SPLATS_UPDATED = 4,
    BRUSH_TRAINING_EVENT_TRAIN_STEP = 5,
    BRUSH_TRAINING_EVENT_REFINE_STEP = 6,
    BRUSH_TRAINING_EVENT_EVAL_RESULT = 7,
    BRUSH_TRAINING_EVENT_LOADING_FINISHED = 8,
    BRUSH_TRAINING_EVENT_WARNING = 9,
    BRUSH_TRAINING_EVENT_DONE = 10,
    BRUSH_TRAINING_EVENT_ERROR = 11,
} BrushTrainingEventKind;

typedef struct BrushTrainingRunConfig {
    const char *dataset_path;
    const char *output_path;
    const char *output_name;
    uint32_t total_train_steps;
    uint32_t refine_every;
    uint32_t max_resolution;
    uint32_t export_every;
    uint32_t eval_every;
    uint64_t seed;
    uint32_t sh_degree;
    uint32_t max_splats;
    uint32_t lod_levels;
    uint32_t lod_refine_steps;
    uint32_t lod_decimation_keep;
    uint32_t lod_image_scale;
    float lpips_loss_weight;
    uint8_t rerun_enabled;
} BrushTrainingRunConfig;

typedef struct BrushTrainingProgress {
    BrushTrainingEventKind kind;
    uint32_t iter;
    uint32_t total_iters;
    uint64_t elapsed_millis;
    uint32_t current_lod;
    uint32_t total_lods;
    uint32_t splat_count;
    uint32_t sh_degree;
    uint32_t train_view_count;
    uint32_t eval_view_count;
    float avg_psnr;
    float avg_ssim;
    const char *message;
} BrushTrainingProgress;

typedef void (*BrushTrainingProgressCallback)(const BrushTrainingProgress *progress, void *user_data);

BrushTrainingExitCode brush_training_run(
    const BrushTrainingRunConfig *config,
    BrushTrainingProgressCallback callback,
    void *user_data
);

const char *brush_training_last_error_message(void);
void brush_training_clear_last_error(void);
const char *brush_training_bridge_version(void);
void brush_training_request_cancel(void);
void brush_training_reset_cancel(void);

#ifdef __cplusplus
}
#endif

#endif