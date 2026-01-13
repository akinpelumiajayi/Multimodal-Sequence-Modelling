This project investigates multimodal next-step prediction in visual narratives, where the input consists of K paired image–text story steps, and the goal is to generate the (K+1)-th image and its associated text description. 
This project multimodal sequential prediction architecture gives the first three image–text story steps (I_1,T_1),(I_2,T_2),(I_3,T_3),. 
The first stage of my architecture is the visual encoder, extracts visual features using ResNet50 which was pertained(Transfer learning) where the last layer were used to train it from the scratch and textual features using BERT as tokenizer, fuses them into a shared embedding space, and models temporal evolution using a Transformer Encoder to obtain story conditioning tokens z_pred. 
The second stage where I use these tokens to generate the next story element (I_4,T_4)by predicting the next text using a conditional autoregressive Transformer decoder with GPT2 as tokenizer and generate the next image through conditional latent diffusion using a UNet operating in Stable Diffusion VAE latent space.


Ablation Evaluation
[ablate] idx=0 steps=10 str=0.020 cfg=1.00 | CLIP(pp)=0.2440 CLIP(pg)=0.1939
[ablate] idx=0 steps=10 str=0.020 cfg=3.00 | CLIP(pp)=0.2035 CLIP(pg)=0.1831
[ablate] idx=0 steps=10 str=0.020 cfg=5.00 | CLIP(pp)=0.2752 CLIP(pg)=0.1958
[ablate] idx=0 steps=10 str=0.020 cfg=7.00 | CLIP(pp)=0.2262 CLIP(pg)=0.1982
[ablate] idx=0 steps=10 str=0.050 cfg=1.00 | CLIP(pp)=0.2365 CLIP(pg)=0.2085
[ablate] idx=0 steps=10 str=0.050 cfg=3.00 | CLIP(pp)=0.2340 CLIP(pg)=0.1831
[ablate] idx=0 steps=10 str=0.050 cfg=5.00 | CLIP(pp)=0.2768 CLIP(pg)=0.2037
[ablate] idx=0 steps=10 str=0.050 cfg=7.00 | CLIP(pp)=0.2398 CLIP(pg)=0.2013
[ablate] idx=0 steps=10 str=0.100 cfg=1.00 | CLIP(pp)=0.1930 CLIP(pg)=0.1873
[ablate] idx=0 steps=10 str=0.100 cfg=3.00 | CLIP(pp)=0.2429 CLIP(pg)=0.1987
[ablate] idx=0 steps=10 str=0.100 cfg=5.00 | CLIP(pp)=0.2390 CLIP(pg)=0.1773
[ablate] idx=0 steps=10 str=0.100 cfg=7.00 | CLIP(pp)=0.2769 CLIP(pg)=0.2021
[ablate] idx=0 steps=20 str=0.020 cfg=1.00 | CLIP(pp)=0.2409 CLIP(pg)=0.1780
[ablate] idx=0 steps=20 str=0.020 cfg=3.00 | CLIP(pp)=0.1951 CLIP(pg)=0.1621
[ablate] idx=0 steps=20 str=0.020 cfg=5.00 | CLIP(pp)=0.2579 CLIP(pg)=0.1764
[ablate] idx=0 steps=20 str=0.020 cfg=7.00 | CLIP(pp)=0.2493 CLIP(pg)=0.1970

## Quick Links
- **[Experiments Notebook](experiment_notebook.ipynb)** – Full experimental workflow and implementation  
- **[Ablations Results](results/)** – All result are in CSV file in the result folder
- **[Model Architecture](src/)** – Model.py, train.py, Utils.py, metrics.py, ablate.py, visualize.py 

The result can be accessed and verified by running; python main.py
The visualization result can be accessed by typing; python visualize.py on the terminal

Result Analysis;
<img width="867" height="770" alt="image" src="https://github.com/user-attachments/assets/dec778ce-430e-4c64-b918-84bada20c0f0" /> ---this is a 10 epochs training that brought the credible reconstruction result. The model worked, even at 5 epochs the noise was minimise. I would have prefer to training more but an epochs last for minimise of 1hour 48minutes. There are lot of bottleneck during training more of memory issues. And the predicted image and the target images has good relationship from the result and if well observed the predicted texts has good coherence as specify by the result and can seen from the notebook added. 

HOW TO IMPROVE THE MODEL FOR BETTER RESULT:
1) The training epoch has to be minimize of 30 epochs
2)The cached dataset has to be minimise of 10,000k above.
1

 #For Reproduction:
1. Install dependencies  
   ```bash
   pip install -r requirements.txt
   And always ensure the accelerate is run only once to avoid runtime issue.
