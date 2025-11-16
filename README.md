# Project Name

### Project Description:
#### Summary - Image Dehazing System: Improve visibility in aerial or drone images by removing haze using dark-channel prior, histogram equalization, and guided filtering. This enhances image clarity for applications like environmental monitoring, urban planning, and disaster assessment.

#### Course concepts used - 
1. -Morphological Processing
2. -Color Image Processing
3. -Smoothing & Restoration
   
#### Additional concepts used -
1. -Dark Channel Prior (DCP)
2. -Transmission map estimation & refinement
3. -Guided Image Filtering
4. -HSV-based CLAHE for enhanced visibility
   
#### Dataset - 
Link and/or Explanation if generated

#### Novelty - 
1. -Introduced variance-based adaptive patch size for improved dark channel estimation
2. -Implemented the entire dehazing pipeline in Python from scratch
3. -Added fixed vs adaptive transmission map comparison with SSIM analysis
4. -Applied CLAHE-based enhancement for improved contrast and visibility
5. -Designed the complete flowchart and evaluation pipeline for the method
   
### Contributors:
1. SHASHWAT (PES1UG23EC283)
2. RISHIKA YADAV (PES1UG23EC916)
3. C S R VEDVIKAS(PES1UG23EC072)

### Steps:
1. Clone Repository
```git clone https://github.com/Digital-Image-Processing-PES-ECE/project-name.git ```

2. Install Dependencies
pip install opencv-contrib-python
pip install numpy
pip install matplotlib
pip install scikit-image


3. Run the Code
python haze_and_smog_removal.py

### Outputs:
* Important intermediate steps
* Final output images 

### References:
1. -Kaiming He et al., “Single Image Haze Removal Using Dark Channel Prior,” IEEE CVPR
2. -Gonzalez & Woods, Digital Image Processing, Pearson
3. -OpenCV Documentation – https://opencv.org
4. -Guided Filter: K. He, J. Sun, X. Tang, “Guided Image Filtering,” ECCV
5. -Scikit-Image Library Documentation

   
### Limitations and Future Work:
Poor performance in low-texture or bright areas
Cannot fully recover details lost due to extremely dense haze

Future work:
Use fusion-based or multi-scale Retinex methods for smoother results
Extend the pipeline with deep-learning based dehazing models
