graph = "coco_headless"
work_dir = (
    "mymodels/checkpoints/fs26/action_recognition/ProtoGCN/v257.10_classes.2s-15frames.jm"
)

classes = [
    "cho_tay_vao_tui_quan",
    "cho_tay_vao_tui_ao_tui_xach",
    "cho_tay_vao_gio_xe_hang",
    "cho_tay_vao_ke",
    "rut_tay_ve",
    "dung_yen",
    "ngoi_yen",
    "dung_day",
    "ngoi_xuong",
    "di",
]

train_data = {
    "cho_tay_vao_tui_quan": [
        "mydata/processed/fs26/action_recognition/skeleton/action.for_skeleton.17_classes.2s_15frames/cho_tay_vao_tui_quan/train.x6.txt",
    ],
    "cho_tay_vao_tui_ao_tui_xach": [
        "mydata/processed/fs26/action_recognition/skeleton/action.for_skeleton.17_classes.2s_15frames/cho_tay_vao_tui_ao/train.txt",
        "mydata/processed/fs26/action_recognition/skeleton/action.for_skeleton.17_classes.2s_15frames/cho_tay_vao_tui_xach/train.txt",
    ],
    "cho_tay_vao_gio_xe_hang": [
        "mydata/processed/fs26/action_recognition/skeleton/action.for_skeleton.17_classes.2s_15frames/cho_tay_vao_gio_hang/train.txt",
        "mydata/processed/fs26/action_recognition/skeleton/action.for_skeleton.17_classes.2s_15frames/cho_tay_vao_xe_hang/train.txt",
    ],
    "cho_tay_vao_ke": [
        "mydata/processed/fs26/action_recognition/skeleton/action.for_skeleton.17_classes.2s_15frames/cho_tay_vao_ke/train.sample_3000.txt",
    ],
    "rut_tay_ve": [
        "mydata/processed/fs26/action_recognition/skeleton/action.for_skeleton.17_classes.2s_15frames/rut_tay_khoi_tui_quan/train.txt",
        "mydata/processed/fs26/action_recognition/skeleton/action.for_skeleton.17_classes.2s_15frames/rut_tay_khoi_tui_ao/train.txt",
        "mydata/processed/fs26/action_recognition/skeleton/action.for_skeleton.17_classes.2s_15frames/rut_tay_khoi_tui_xach/train.txt",
        "mydata/processed/fs26/action_recognition/skeleton/action.for_skeleton.17_classes.2s_15frames/rut_tay_khoi_gio_hang/train.txt",
        "mydata/processed/fs26/action_recognition/skeleton/action.for_skeleton.17_classes.2s_15frames/rut_tay_khoi_xe_hang/train.txt",
        "mydata/processed/fs26/action_recognition/skeleton/action.for_skeleton.17_classes.2s_15frames/rut_tay_khoi_ke/train.sample_1500.txt",
    ],
    "dung_yen": [
        "mydata/processed/fs26/action_recognition/skeleton/action.for_skeleton.17_classes.2s_15frames/dung_yen/train.sample_3000.txt",
    ],
    "ngoi_yen": [
        "mydata/processed/fs26/action_recognition/skeleton/action.for_skeleton.17_classes.2s_15frames/ngoi_yen/train.sample_3000.txt",
    ],
    "dung_day": [
        "mydata/processed/fs26/action_recognition/skeleton/action.for_skeleton.17_classes.2s_15frames/dung_day/train.x28.txt",
    ],
    "ngoi_xuong": [
        "mydata/processed/fs26/action_recognition/skeleton/action.for_skeleton.17_classes.2s_15frames/ngoi_xuong/train.x70.txt",
    ],
    "di": [
        "mydata/processed/fs26/action_recognition/skeleton/action.for_skeleton.17_classes.2s_15frames/di/train.sample_3000.txt",
    ],
}

val_data = {
    "cho_tay_vao_tui_quan": [
        "mydata/processed/fs26/action_recognition/skeleton/action.for_skeleton.17_classes.2s_15frames/cho_tay_vao_tui_quan/val.txt",
    ],
    "cho_tay_vao_tui_ao_tui_xach": [
        "mydata/processed/fs26/action_recognition/skeleton/action.for_skeleton.17_classes.2s_15frames/cho_tay_vao_tui_ao/val.txt",
        "mydata/processed/fs26/action_recognition/skeleton/action.for_skeleton.17_classes.2s_15frames/cho_tay_vao_tui_xach/val.txt",
    ],
    "cho_tay_vao_gio_xe_hang": [
        "mydata/processed/fs26/action_recognition/skeleton/action.for_skeleton.17_classes.2s_15frames/cho_tay_vao_gio_hang/val.txt",
        "mydata/processed/fs26/action_recognition/skeleton/action.for_skeleton.17_classes.2s_15frames/cho_tay_vao_xe_hang/val.txt",
    ],
    "cho_tay_vao_ke": [
        "mydata/processed/fs26/action_recognition/skeleton/action.for_skeleton.17_classes.2s_15frames/cho_tay_vao_ke/val.txt",
    ],
    "rut_tay_ve": [
        "mydata/processed/fs26/action_recognition/skeleton/action.for_skeleton.17_classes.2s_15frames/rut_tay_khoi_tui_quan/val.txt",
        "mydata/processed/fs26/action_recognition/skeleton/action.for_skeleton.17_classes.2s_15frames/rut_tay_khoi_tui_ao/val.txt",
        "mydata/processed/fs26/action_recognition/skeleton/action.for_skeleton.17_classes.2s_15frames/rut_tay_khoi_tui_xach/val.txt",
        "mydata/processed/fs26/action_recognition/skeleton/action.for_skeleton.17_classes.2s_15frames/rut_tay_khoi_gio_hang/val.txt",
        "mydata/processed/fs26/action_recognition/skeleton/action.for_skeleton.17_classes.2s_15frames/rut_tay_khoi_xe_hang/val.txt",
        "mydata/processed/fs26/action_recognition/skeleton/action.for_skeleton.17_classes.2s_15frames/rut_tay_khoi_ke/val.txt",
    ],
    "dung_yen": [
        "mydata/processed/fs26/action_recognition/skeleton/action.for_skeleton.17_classes.2s_15frames/dung_yen/val.txt",
    ],
    "ngoi_yen": [
        "mydata/processed/fs26/action_recognition/skeleton/action.for_skeleton.17_classes.2s_15frames/ngoi_yen/val.txt",
    ],
    "dung_day": [
        "mydata/processed/fs26/action_recognition/skeleton/action.for_skeleton.17_classes.2s_15frames/dung_day/val.txt",
    ],
    "ngoi_xuong": [
        "mydata/processed/fs26/action_recognition/skeleton/action.for_skeleton.17_classes.2s_15frames/ngoi_xuong/val.txt",
    ],
    "di": [
        "mydata/processed/fs26/action_recognition/skeleton/action.for_skeleton.17_classes.2s_15frames/di/val.txt",
    ],
}

model = dict(
    type="RecognizerGCN",
    backbone=dict(
        type="ProtoGCN",
        num_prototype=100,
        in_channels=2,
        num_person=1,
        tcn_ms_cfg=[(3, 1), (3, 2), (3, 3), (3, 4), ("max", 3), "1x1"],
        graph_cfg=dict(
            layout=graph, mode="random", num_filter=8, init_off=0.04, init_std=0.02
        ),
    ),
    cls_head=dict(
        type="SimpleHead",
        joint_cfg="coco_headless",
        num_classes=10,
        in_channels=384,
        weight=0.2,
    ),
)

dataset_type = "PoseDatasetFromTxt"
train_pipeline = [
    dict(type="NormalizeByMinMaxKeypoints"),
    dict(type="Normalize_01_to_neg11"),
    dict(type="RandomRot", theta=0.2),
    dict(type="SelectKeypoints", indexes=[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]),
    dict(type="GenSkeFeat", dataset="coco_headless", feats=['jm']),
    dict(type="FormatGCNInput_v2"),
]
val_pipeline = [
    dict(type="NormalizeByMinMaxKeypoints"),
    dict(type="Normalize_01_to_neg11"),
    dict(type="SelectKeypoints", indexes=[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]),
    dict(type="GenSkeFeat", dataset="coco_headless", feats=['jm']),
    dict(type="FormatGCNInput_v2"),
]
test_pipeline = [
    dict(type="NormalizeByMinMaxKeypoints"),
    dict(type="Normalize_01_to_neg11"),
    dict(type="SelectKeypoints", indexes=[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]),
    dict(type="GenSkeFeat", dataset="coco_headless", feats=['jm']),
    dict(type="FormatGCNInput_v2"),
]

data = dict(
    videos_per_gpu=128,
    workers_per_gpu=4,
    test_dataloader=dict(videos_per_gpu=128),
    train=dict(
        type=dataset_type,
        classes=classes,
        data_dict=train_data,
        pipeline=train_pipeline,
        shuffle=True,
    ),
    val=dict(
        type=dataset_type,
        classes=classes,
        data_dict=val_data,
        pipeline=val_pipeline,
        test_mode=True,
    ),
    test=dict(
        type=dataset_type,
        classes=classes,
        data_dict=val_data,
        pipeline=test_pipeline,
        test_mode=True,
    ),
)

optimizer = dict(
    type="SGD", lr=6.25e-2, momentum=0.9, weight_decay=0.0005, nesterov=True
)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy="CosineAnnealing",
    min_lr=6.25e-6,
    by_epoch=False,
    warmup="linear",
    warmup_iters=2,
    warmup_ratio=0.01,
    warmup_by_epoch=True,
)
total_epochs = 60
checkpoint_config = dict(interval=-1)
evaluation = dict(
    interval=1,
    metrics=["macro_f1"],
    greater_keys=["macro_f1"],
    class_map=None,
    class_weights=None,
)
log_config = dict(interval=50, hooks=[dict(type="TextLoggerHook")])
enable_deterministic_hook = False
