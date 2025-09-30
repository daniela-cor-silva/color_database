from ultralytics import YOLO

# load yolo
model = YOLO("yolo11n.pt")

train_results = model.train(
    data='/data/classes.yaml', # args file
    epochs=300,
    imgsz=800,
    batch=16,
    patience=100,
    optimizer='auto',
    lr0=0.001,
    augment=True,
    name='yolo_trained',
    save_period=10,
    seed=42,
    device=0,
    verbose=True
)

metrics = model.val() # perform validation
