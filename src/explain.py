import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from src.common import ROOT, load_image, resolve_path

def find_last_conv(model):
    candidates=[l for l in model.submodules if isinstance(l,tf.keras.layers.Conv2D)]
    if not candidates: raise ValueError("Could not find Conv2D layer for Grad-CAM.")
    return candidates[-1].name

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--image",required=True); args=parser.parse_args()
    model=tf.keras.models.load_model(ROOT/"artifacts"/"best_model.keras")
    image=load_image(args.image); batch=tf.expand_dims(image,0)
    # EfficientNetV2-S nested backbone convolutional layer. A nested model is used to obtain activations.
    backbone=next((l for l in model.layers if isinstance(l,tf.keras.Model) and "efficientnetv2" in l.name.lower()),None)
    if backbone is None: raise RuntimeError("EfficientNetV2 backbone not found in saved model.")
    convs=[l for l in backbone.layers if isinstance(l,tf.keras.layers.Conv2D)]
    if not convs: raise RuntimeError("No convolutional layer found.")
    feature_model=tf.keras.Model(backbone.input,[backbone.get_layer(convs[-1].name).output,backbone.output])
    with tf.GradientTape() as tape:
        features, pooled=feature_model(batch,training=False)
        # Reuse classifier head on backbone output to keep graph connected.
        x=pooled
        for name in ["global_average_pooling","classifier_dropout","feature_dense","mc_dropout","pd_probability"]:
            x=model.get_layer(name)(x,training=False)
        score=x[:,0]
    grads=tape.gradient(score,features)
    weights=tf.reduce_mean(grads,axis=(1,2),keepdims=True)
    heat=tf.reduce_sum(weights*features,axis=-1)[0]
    heat=tf.maximum(heat,0); heat=heat/(tf.reduce_max(heat)+1e-8); heat=tf.image.resize(heat[...,None],(224,224))[...,0].numpy()
    raw=image.numpy().mean(axis=-1); overlay=np.stack([raw/255,raw/255,raw/255],axis=-1)
    cmap=plt.get_cmap("jet")(heat)[...,:3]; blended=np.clip(.55*overlay+.45*cmap,0,1)
    out=ROOT/"reports"/"gradcam_explanation.png"; out.parent.mkdir(exist_ok=True)
    plt.figure(figsize=(10,4)); plt.subplot(1,2,1); plt.imshow(raw,cmap="gray"); plt.title("Input spiral"); plt.axis("off"); plt.subplot(1,2,2); plt.imshow(blended); plt.title("Grad-CAM (exploratory)"); plt.axis("off"); plt.tight_layout(); plt.savefig(out,dpi=160); plt.close()
    p=float(model(batch,training=False).numpy().reshape(-1)[0]); print(f"PD probability: {p:.4f}; saved {out}")
if __name__=="__main__": main()
