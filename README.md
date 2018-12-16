# HomologyDR_Tearing

This is a demo for implementation of homology-preserving dimension reduction algorithm. The implementation is described in "Homology-Preserving Dimensionality Reduction via Manifold Landmarking and Tearing".

<center><img src="octa.png" width="512"></center>
<center><img src="tearing.png" width="512"></center>

    $ virtualenv HomologyDR-demo-env
    $ source HomologyDR-demo-env/bin/activate
    (HomologyDR-demo-env) $ pip install -r requirements.txt
    (HomologyDR-demo-env) $ cd scripts
    (HomologyDR-demo-env) $ python LandmarkingDR_Realword_dataset.py
    (HomologyDR-demo-env) $ python LandmarkingDR_synthetic_dataset.py
    (HomologyDR-demo-env) $ python Tearing.py
