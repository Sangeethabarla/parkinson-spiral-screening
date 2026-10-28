import json
import tensorflow as tf
from src.common import ROOT, load_config, set_seed
from src.data import get_split, make_dataset
from src.model import build_model

def main():
    cfg=load_config(); set_seed(cfg["seed"])
    train_df=get_split("train"); val_df=get_split("validation")
    train_ds=make_dataset(train_df, True); val_ds=make_dataset(val_df, False)
    model, backbone=build_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(cfg["phase1_learning_rate"]), loss="binary_crossentropy", metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy"),tf.keras.metrics.AUC(name="auc"),tf.keras.metrics.Precision(name="precision"),tf.keras.metrics.Recall(name="recall")])
    art=ROOT/"artifacts"; art.mkdir(exist_ok=True)
    checkpoint=art/"best_model.keras"
    callbacks=[tf.keras.callbacks.ModelCheckpoint(checkpoint, monitor="val_auc", mode="max", save_best_only=True, verbose=1),tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=5, restore_best_weights=True),tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=.3, patience=2, min_lr=1e-7, verbose=1)]
    hist1=model.fit(train_ds,validation_data=val_ds,epochs=cfg["phase1_epochs"],callbacks=callbacks)
    # Fine-tune only last configured layers; keep BatchNorm layers frozen for small datasets.
    backbone.trainable=True
    n=max(1,int(cfg["fine_tune_last_layers"]))
    for layer in backbone.layers[:-n]: layer.trainable=False
    for layer in backbone.layers:
        if isinstance(layer,tf.keras.layers.BatchNormalization): layer.trainable=False
    model.compile(optimizer=tf.keras.optimizers.Adam(cfg["phase2_learning_rate"]),loss="binary_crossentropy",metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy"),tf.keras.metrics.AUC(name="auc"),tf.keras.metrics.Precision(name="precision"),tf.keras.metrics.Recall(name="recall")])
    model.fit(train_ds,validation_data=val_ds,epochs=cfg["phase2_epochs"],callbacks=callbacks)
    # Reload best validation-AUC checkpoint to ensure final artifact is best monitored checkpoint.
    best=tf.keras.models.load_model(checkpoint)
    best.save(checkpoint)
    print(f"Best model saved to {checkpoint}")

if __name__=="__main__": main()
