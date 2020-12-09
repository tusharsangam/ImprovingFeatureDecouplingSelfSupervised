This work tries to improve upon the [Self Supervised Learning By Rotation Feature Decoupling]((https://openaccess.thecvf.com/content_CVPR_2019/papers/Feng_Self-Supervised_Representation_Learning_by_Rotation_Feature_Decoupling_CVPR_2019_paper.pdf)) paper from Feng.et.al. by using the Supervised Finetuning on fraction of data & the contrastive loss inspired from [SimCLR](https://arxiv.org/abs/2002.05709).
All the coding was done in Jupyter Notebooks & Pytorch.  
Read the Project Report for detailed experiments.

**cite** contains the reference paper [paper](https://openaccess.thecvf.com/content_CVPR_2019/html/Feng_Self-Supervised_Representation_Learning_by_Rotation_Feature_Decoupling_CVPR_2019_paper.html).

    @InProceedings{Feng_2019_CVPR,
		author = {Feng, Zeyu and Xu, Chang and Tao, Dacheng},
		title = {Self-Supervised Representation Learning by Rotation Feature Decoupling},
		booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
		month = {June},
		year = {2019}
	}

** [Github Link](https://github.com/philiptheother/FeatureDecoupling) ** of the author from where base model & loss functions of the code is borrowed. 

**Requirements** - Pytorch, Numpy, Matplotlib, tqdm, Jupyter, tensorboard, Pytorch 1.1 environment  

**Special Note** - Author provided **Noise Contrastive Estimation loss** works in initial versions of Pytorch only, since I don't use that loss in my improvements I haven't tried porting it to latest version. Thus running pretraining on basemethod requires **pytorch 1.1**  

There are four models that needs testing for every experiment  

1.  Base Model
2.  Contrastive Model (Improved)
3.  Base Model Fine Tuned (20% data)
4.  Contrastive Model Fine Tuned (20% data)

**Code Factoring** - Following is the folder arrangement & thier meaning

*   "architectures" contains all the models, thier altered versions & custom losses
*   "DatasetPreProcessingScripts" converts datasets into ["ImageFolder"](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder) format of Pytorch Dataset, also creates smaller subset of 20% split for fine tuning
*   "Linear-ImageNet-exp" contains all the four models on Linear Classification as well as a supervised model trained with 20% labels.
*   "NonLinear-ImageNet-exp" contains all the four models on Non Linear Classification
*   "TransferLearning-exp" contains four models on following datasets using Transfer Learning Technique
    *   CIFAR10
    *   MiniPlaces
*   "Pretraining" contains pretraining for two main models (Base & Contrastive)
*   "CustomDataset.py" imitates author's dataset object without the added complexities, I modified the exisiting dataset object in pytorch
*   "GetDataLoaders.py" returns the dataloaders for given training type "unsupervised" or supervised, "get_short_dataloaders" return the loaders for 20% subset.  
    Any Changes in dataset directory should be directly modified in this file.

**Formats** even though all is done in Jupyter Notebooks, thier **python & html exports** are availed in each folder.  
**Running code** to run code, dataset root folder for instance "tiny-image-net" or "cifar10" & the corresponding Jypyter Notebook or .py file should be present at the root directory level. Thus to run any experiment simply copy the script file from inside of the folders to root project directory (where this readme file exsists).  

All the training, loss calculation & optimization steps are given at one place in corresponding notebooks in a easy way without using complicated Software Programming practices like in the Author's github.  
Pretrained weights will be available through google drive [here](https://drive.google.com/drive/u/0/folders/1X3yYKj54L0z-Ws-Pm5-1RLTVzpHoYq93). Make a new folder "weights" in the same directory level as main script file.  

Access Jupyter HTML Outputs

*   Pretraining on ImageNet
    *   [Base Method](./Pretraining/htmlexports/pretraining-basemethod.html)
    *   [Contrastive Method](./Pretraining/htmlexports/pretraining-ConstrastiveLoss.html)
*   Linear Classification on Image-Net
    *   [Base Method (Top-1 - 26.580)](./Linear-ImageNet-exp/htmlexports/Imagenet-Classification-basemethod.html)
    *   [Contrastive Method (Top-1 - 27.5450)](./Linear-ImageNet-exp/htmlexports/Imagenet-Classification-contrastive.html)
    *   [Base Method Fine Tuned (Top-1 - 33.5650)](./Linear-ImageNet-exp/htmlexports/Imagenet-Classification-finetuning-basemethod.html)
    *   [Contrastive Method Fine Tuned (Top-1 - 34.900)](./Linear-ImageNet-exp/htmlexports/Imagenet-Classification-finetuning-Contrastive.html)
    *   [Supervised Cross Entropy on 20% labels (Top-1 - 18.4250)](./Linear-ImageNet-exp/htmlexports/Imagenet-Classification-supervised-training-on-percent-of-data.html)
*   Non-Linear Classification on Image-Net
    *   [Base Method (Top-1 - 37.220)](./NonLinear-ImageNet-exp/htmlexports/Imagenet-Classification-basemethod-NonLinearClassifier.html)
    *   [Contrastive Method (Top-1 - 38.1950)](./NonLinear-ImageNet-exp/htmlexports/Imagenet-Classification-contrastive-NonLinear.html)
    *   [Base Method Fine Tuned (Top-1 - 40.2500)](./NonLinear-ImageNet-exp/htmlexports/Imagenet-Classification-basemethod-finetuned-NonLinearClassifier.html)
    *   [Contrastive Method Fine Tuned (Top-1 - 40.9950)](./NonLinear-ImageNet-exp/htmlexports/Imagenet-Classification-contrastive-finetuned-NonLinear.html)
*   Transfer Learning
    *   CIFAR10
        *   [Base Method (Top-1 - 65.600)](./TransferLearning-exp/CIFAR10/htmlexports/CIFAR10-Classification-basemethod.html)
        *   [Contrastive Method (Top-1 - 66.4600)](./TransferLearning-exp/CIFAR10/htmlexports/CIFAR10-Classification-constrastive.html)
        *   [Base Method Fine Tuned (Top-1 - 67.5500)](./TransferLearning-exp/CIFAR10/htmlexports/CIFAR10-Classification-basemethod-finetuned.html)
        *   [Contrastive Method Fine Tuned (Top-1 - 67.7900)](./TransferLearning-exp/CIFAR10/htmlexports/CIFAR10-Classification-contrastive-finetuned.html)
    *   Places
        *   [Base Method (Top-1 - 26.3100)](./TransferLearning-exp/Places/htmlexports/Places-Classification-basemethod.html)
        *   [Contrastive Method (Top-1 - 26.5400)](./TransferLearning-exp/Places/htmlexports/Places-Classification-Contrastive.html)
        *   [Base Method Fine Tuned (Top-1 - 27.2500)](./TransferLearning-exp/Places/htmlexports/Places-Classification-basemethod-finetuned.html)
        *   [Contrastive Method Fine Tuned (Top-1 - 27.6200)](./TransferLearning-exp/Places/htmlexports/Places-Classification-Contrastive-finetuned.html)
*   SimCLR Failure
    *   [Pretraining](./SimCLR-failure/htmlexports/pretraining-SimCLR.html)
    *   [Classification](./SimCLR-failure/htmlexports/Imagenet-Classification-SimCLR.html)
*   PU Rotation Probabilities Experiement
    *   [Pretrained](./PURotationProbability-exp/htmlexports/Pre-trained-RotNetPUProbs.html)
    *   [Predicting](./PURotationProbability-exp/htmlexports/RotNetPuProbs for tinyImagenet.html)