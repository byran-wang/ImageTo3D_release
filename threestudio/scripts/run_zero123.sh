NAME="cute_bird_processed"

# Phase 1 - 64x64
python launch.py --config configs/zero123.yaml --train data.image_path=./load/images/${NAME}_rgba.png use_timestamp=False name=${NAME} tag=Phase1 # system.freq.guidance_eval=0 system.loggers.wandb.enable=false system.loggers.wandb.project="zero123" system.loggers.wandb.name=${NAME}_Phase1

# Phase 1.5 - 512 refine
python launch.py --config configs/zero123-geometry.yaml --train data.image_path=./load/images/${NAME}_rgba.png system.geometry_convert_from=./outputs/${NAME}/Phase1/ckpts/last.ckpt use_timestamp=False name=${NAME} tag=Phase1p5 # system.freq.guidance_eval=0 system.loggers.wandb.enable=false system.loggers.wandb.project="zero123" system.loggers.wandb.name=${NAME}_Phase1p5
# export mesh
python launch.py --config outputs/cute_bird_processed/Phase1p5/configs/parsed.yaml --export resume=./outputs/cute_bird_processed/Phase1p5/ckpts/last.ckpt system.exporter_type=mesh-exporter

# Phase 2 - dreamfusion
proxychains4 python launch.py --config configs/experimental/imagecondition_zero123nerf.yaml --train data.image_path=./load/images/${NAME}_rgba.png system.prompt_processor.prompt="A 3D model of a cute red bird" system.weights="outputs/${NAME}/Phase1/ckpts/last.ckpt" use_timestamp=False  name=${NAME} tag=Phase2 # system.freq.guidance_eval=0 system.loggers.wandb.enable=false system.loggers.wandb.project="zero123" system.loggers.wandb.name=${NAME}_Phase2

# Phase 2 - SDF + dreamfusion
proxychains4 python launch.py --config configs/experimental/imagecondition_zero123nerf_refine.yaml --train data.image_path=./load/images/${NAME}_rgba.png system.prompt_processor.prompt="A 3D model of a cute red bird" name=${NAME} use_timestamp=False system.geometry_convert_from="outputs/${NAME}/Phase1/ckpts/last.ckpt" tag=Phase2_refine # system.freq.guidance_eval=0 system.loggers.wandb.enable=false system.loggers.wandb.project="zero123" system.loggers.wandb.name=${NAME}_Phase2_refine
