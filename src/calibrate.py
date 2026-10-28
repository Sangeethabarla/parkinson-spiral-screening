"""Fit a scalar temperature on validation predictions only."""
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from src.common import ROOT

def main():
    path=ROOT/"reports"/"validation_predictions.csv"
    if not path.exists(): raise FileNotFoundError("Run python -m src.evaluate first.")
    df=pd.read_csv(path); y=df.label.to_numpy(); p=np.clip(df.pd_probability.to_numpy(),1e-6,1-1e-6); z=logit(p)
    def nll(temp):
        q=np.clip(expit(z/temp),1e-7,1-1e-7)
        return -np.mean(y*np.log(q)+(1-y)*np.log(1-q))
    opt=minimize_scalar(nll,bounds=(.05,10),method="bounded")
    out={"temperature":float(opt.x),"validation_nll_before":float(nll(1.0)),"validation_nll_after":float(opt.fun),"fit_split":"validation"}
    (ROOT/"artifacts").mkdir(exist_ok=True)
    (ROOT/"artifacts"/"temperature.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
    print(json.dumps(out,indent=2))
if __name__=="__main__": main()
