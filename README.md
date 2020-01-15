# HomologyDR_Tearing

This is a demo for implementation of homology-preserving dimension reduction(DR) algorithm. The implementation is described in ["Homology-Preserving Dimensionality Reduction via Manifold Landmarking and Tearing"](https://arxiv.org/pdf/1806.08460.pdf).

<center><img src="octa.png" width="512"></center>
<center><img src="tearing.png" width="512"></center>

# How to use

Tested with both Python 2.7 and Ubuntu 16.04.

### What to expect

You can see results of:
- DR with datasets: Swiss-roll-with-a-hole and Fishing-Net in "LandmarkingDR_synthetic_dataset.py"
- DR with datasets: Octa and Airfoil1 in "LandmarkingDR_Realword_dataset.py"
- Tearing with datasets: Cylinder-3 and Cylinder-5 in "Tearing.py"

### Run

    $ virtualenv HomologyDR-demo-env
    $ source HomologyDR-demo-env/bin/activate
    (HomologyDR-demo-env) $ pip install -r requirements.txt
    (HomologyDR-demo-env) $ cd scripts
    (HomologyDR-demo-env) $ python LandmarkingDR_Realword_dataset.py
    (HomologyDR-demo-env) $ python LandmarkingDR_synthetic_dataset.py
    (HomologyDR-demo-env) $ python Tearing.py

### Notes

- If you want to run stage 4 in "LandmarkingDR*.py", please install [Ripser](https://github.com/Ripser/ripser) and [Hera](https://bitbucket.org/grey_narn/hera).
- We modified mapper algorithm based on [KeplerMapper](https://github.com/MLWave/kepler-mapper), including adding more filter functions. You can see the modification in [/src/km.py](https://github.com/LynneYan/HomologyDR_Tearing/blob/master/src/km.py)


# License

Standard MIT disclaimer applies, see LICENSE for full text.
