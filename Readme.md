A deep learning–based framework for hyperspectral band reduction using 1D-CNNs and Integrated Gradients, enabling low-cost multispectral sensor design for tomato leaf disease detection using VIS–NIR hyperspectral imagery.

PROJECT EXPLANATION
-------------------

Project Title:
Deep Learning–Based Hyperspectral Band Reduction for Low-Cost Tomato Disease Detection


1. Introduction
---------------

Agricultural crop diseases are a major cause of yield loss worldwide, especially in high-value crops such as tomatoes. Early detection of plant stress and disease is crucial for timely intervention, reduced pesticide usage, and improved crop productivity.

Traditional imaging systems such as RGB cameras fail to detect early-stage diseases because visible symptoms often appear much later than the underlying physiological changes. Hyperspectral Imaging (HSI), which captures reflectance information across hundreds of narrow spectral bands, can detect these subtle biochemical and structural changes in plant leaves. However, hyperspectral cameras are expensive, bulky, and computationally intensive, making them impractical for real-world agricultural deployment.

This project addresses the challenge of retaining the diagnostic power of hyperspectral imaging while drastically reducing hardware cost and system complexity.


2. Real-Life Problem Being Solved
---------------------------------

Real-world agricultural monitoring faces the following challenges:

- Hyperspectral cameras capture 200–300 spectral bands, leading to very high hardware costs.
- Large data volumes make real-time processing difficult.
- Most spectral bands are redundant or weakly informative.
- Farmers and field operators require affordable, portable, and fast sensing solutions.
- Existing disease detection systems lack interpretability and hardware guidance.

As a result, despite the effectiveness of hyperspectral imaging in research labs, it has limited adoption in real agricultural environments.

This project solves the real-life problem of:
"How to design a low-cost plant disease detection system without sacrificing the benefits of hyperspectral imaging."


3. Objective of the Project
---------------------------

The primary objectives of this project are:

- To identify the most informative spectral bands for tomato leaf disease detection.
- To reduce hyperspectral dimensionality from 300 bands to a small subset.
- To maintain high classification accuracy using deep learning.
- To provide an interpretable, data-driven framework for multispectral sensor design.
- To enable affordable deployment of plant disease monitoring systems.


4. Dataset and Preprocessing
----------------------------

The project uses VIS–NIR hyperspectral images of tomato leaves stored in BIL format. Each image cube has dimensions:

- Spatial resolution: 500 × 900 pixels
- Spectral resolution: 300 wavelength bands (approximately 400–1050 nm)

Each pixel represents a reflectance spectrum. Pixel-level spectral data is extracted from the hyperspectral cubes. Background pixels (soil, shadows, non-leaf regions) are removed using a reflectance threshold.

The dataset is balanced by sampling equal numbers of healthy and unhealthy pixel spectra. Spectral normalization is applied to ensure stable and efficient neural network training.


5. Baseline Deep Learning Model
-------------------------------

A 1D Convolutional Neural Network (1D-CNN) is used as the baseline classifier. The model operates purely in the spectral domain, treating each pixel spectrum as a one-dimensional signal.

Key reasons for using a 1D-CNN:
- Efficient learning of local spectral patterns.
- Ability to capture red-edge and NIR transitions.
- Lower computational cost compared to 2D or 3D CNNs.
- Ideal for pixel-level hyperspectral analysis.

The baseline model trained on all 300 bands achieved a classification accuracy of approximately 97.4%, confirming that hyperspectral signatures contain strong discriminative information for tomato leaf health.


6. Interpretability Using Integrated Gradients
----------------------------------------------

Instead of treating the CNN as a black box, the project applies Integrated Gradients (IG), an explainable AI technique, to interpret the model’s decisions.

Integrated Gradients:
- Computes the contribution of each spectral band to the model’s prediction.
- Produces an importance score for every wavelength.
- Identifies which bands are critical for classification.

By averaging IG scores across thousands of samples, a stable band-importance ranking is obtained. The most important bands are found to lie primarily in the red-edge and near-infrared regions, which are known to be sensitive to plant stress and chlorophyll concentration.


7. Band Reduction and Performance Analysis
------------------------------------------

Using the IG-based ranking, reduced-band models are trained using only the top-k most important bands (k = 10, 20, 30, 40, 60).

Key observations:
- 10 bands achieve ~87% accuracy.
- 40 bands achieve ~91% accuracy.
- 60 bands achieve ~92% accuracy.
- 60 bands preserve over 94% of the baseline performance while reducing dimensionality by 80%.

This demonstrates that most hyperspectral bands are redundant and that a compact multispectral configuration can deliver strong performance.


8. Correlation and Redundancy Analysis
--------------------------------------

A correlation heatmap of the top-60 selected bands shows:
- Strong clustering in the NIR region (high redundancy).
- Lower correlation in the red-edge region (high uniqueness).

This indicates that further band reduction or channel grouping is possible, enabling even simpler sensor designs.


9. Real-Life Impact and Applications
------------------------------------

This project has significant real-world impact:

- Enables the design of low-cost multispectral cameras instead of expensive hyperspectral systems.
- Reduces sensor cost by an estimated 80–90%.
- Makes early plant disease detection affordable and scalable.
- Supports deployment on drones, handheld devices, and edge systems.
- Reduces pesticide usage through early diagnosis.
- Improves crop yield and sustainability.
- Provides interpretable, trustworthy AI for agricultural decision-making.

Potential applications include:
- Precision agriculture
- Smart farming systems
- UAV-based crop monitoring
- Automated greenhouse management
- Agricultural IoT platforms


10. Key Contributions
---------------------

- End-to-end hyperspectral band reduction pipeline.
- High-accuracy 1D-CNN spectral classification.
- Explainable band selection using Integrated Gradients.
- Data-driven guidance for multispectral sensor design.
- Practical bridge between hyperspectral research and real-world deployment.


11. Conclusion
--------------

This project demonstrates that deep learning combined with explainable AI can significantly reduce hyperspectral dimensionality while maintaining strong disease detection performance. By identifying a small, optimal subset of spectral bands, the work provides a practical and scalable pathway toward affordable plant disease monitoring systems, bringing hyperspectral intelligence from the laboratory to the field.


Author:
Shubham

Guided by:
Dr. Deepika Kukreja
