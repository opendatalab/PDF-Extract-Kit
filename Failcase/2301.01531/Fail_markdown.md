# MoBYv2AL: Self-supervised Active Learning Stor Image Classification  

Razvan Caramalau1  r.caramalau18@imperial.ac.uk Binod Bhattarai? b.bhattarai@ucl.ac.uk Danail Stoyanov2 danail.stoyanov@ucl.ac.uk Jae-Kyun Kim1,3 tR.kim@imperial.ac.uk  

1 Imperial College London, UK 2 University College London, UK 3 School of Computing, KAIST, Daejeon, South Korea  

# Abstract  

Active learning(AL) has recently gained popularity for deep learning(DL) models This is due to efficient and informative sampling, especially when the learner requires large-scale labelled datasets. Commonly, the sampling and training happen in stages while more batches are added. One main bottleneck in this strategy is the narrow repre. sentation learned by the model that affects the overall AL selection  

We present  $M o B Y\nu2A L$  , a novel self-supervised active learning framework for im- age classification. Our contribution lies in lifting MoBY - one of the most successful self-supervised learning algorithms to the AL pipeline. Thus, we add the downstream task-aware objective function and optimize it jointly with contrastive loss. Further, we derive a data-distribution selection function from labelling the new examples. Finally, we test and study our pipeline robustness and performance for image classification tasks. We successfully achieved state-of-the-art results when compared to recent AL methods. Code available: https: //github. com/razvancaramalau/MoBYv2AL  

# 1Introduction  

Active Learning (AL) [1, 3, 6, 13, 21, 22, 32, 42] has recently gained more popularity in the research community. The goal of AL is to sample the most informative and diverse examples from a large pool of unlabelled data to query their labels. The existing AL meth. ods can be grouped into two based on the selection criteria. The first group is uncertainty. based algorithms [12, 15, 42] that select the challenging and informative examples. Whereas representative-based algorithms select the most diverse examples from the data set. To select diverse examples, existing methods first project the images into a feature space followed by applying sampling techniques such as CoreSet [28]. Our work falls in the latter category.  

Prominent works on representative-based methods for AL in the past few years have tack. led a wide range of architectures to learn the image representations such as Convolutional Neural Network [42], Graph Convolutional Neural Networks [6], Bayesian Network [5], Variational Auto-Encoders [21, 32], and too few to mention. These works have proven that the learned features of the images have directly influenced the performance of the pipeline However, these methods suffer from cold-start problem. As we know, in the early selection stage, we have limited annotated examples, and the above-mentioned architectures are hard to train with the small number of training examples. Thus, the features extracted from such models get biased from the beginning and continue to become sub-optimal in the subsequent selection stages. This problem is commonly known as cold-start problem. To address such a problem, recent works in AL have explored self-supervised learning methods [3, 13, 17, 20]  

Self-supervised learning methods [4, 7, 16, 19, 41] have made tremendous progress in generating discriminative representations of the images. Some methods have even come close to supervised methods in generalization [10, 16, 41]. One of the earliest works in this direction [13] employed consistency loss between the input image and its geometrically aug. mented versions along with the objective of downstream tasks. However, this method limits augmentation methods in the primitive form. Similarly, J. Bengar et al. [3] introduced con. trastive learning in AL, but the self-supervised method and end-task objective are optimised in multi-stage form. This makes the model sub-optimal, affecting the features' representa- tiveness during selection. Simple random labelling overpasses any AL criteria. Thus, the existing works in this direction show explicit limitations.  

To address the issues of those methods, we introduce contrastive learning as MoBYv2 (from its predecessor MoBY [41]) in our AL pipeline, MoBYv2AL, and jointly train the learner. We choose MoBY SSL because it addresses the computational complexities and shortcomings of other previous methods, such as SimCLR [8] or BYOL[16]. MoBY has two branches (as shown in Figure 1). One updates with gradient (query encoder) and another with momentum (key encoder). The parameters of the momentum encoder are updated in slow-moving averages with the query one. Moreover, the memory bank of keys from the mo- mentum encoder keeps long dependencies with several mini-batches. Apart from minimising a contrastive loss, another advantage consists in the asymmetric structure of BYOL that cap. tures distances from mean representation. The AL process of MoBYv2AL culminates with the concept-aware selection function, CoreSet.  

We state our contributions and achievements with the following: a task-aware self-supervised method jointly trained with the learner - MoBYv2;  

a quantitative evaluation with MoBYv2AL on multiple image classification bench. marks such as: CIFAR-10/100[24], SVHN[14] and FashionMNIST [40]  

state-of-the-art performance over the existing AL baselines  

#  Related Works  

Recent Advances in Active Learning. Recent advancement in AL are either uncertainty. oriented [5, 12, 15, 30, 32, 42] or data representative ness [1, 28, 35]; and some of them are the mixture of both [2, 6, 13, 21]  

Under the pool-based setting [29], deep active learning has been initially tackled with un- certainty estimation. For classification tasks, this was addressed from the maximum entropy of the posterior or through Bayesian approximation with Monte Carlo (MC) Dropout[5, 12, 15]. Concurrently, methods that used latent representations to sample have outperformed the ones that explored uncertainty. From these works, we recognise CoreSet [28] as the most revised and competitive baseline. However, more recently, a new trend shifted the AL acquisition process to parameterised modules. The first work, Learning Loss [42] opti mises a predictor for the loss of the learner. Still tracking uncertainty, VAAL [32] deploys a dedicated variational auto-encoder (VAE) to adversarial distinguish between labelled and unlabelled images. CoreGCN[6] and CDAL [1], on the other hand, proposed to improve data representative ness with graph convolutional networks and categorical contextual diver sity, respectively. We test these methods in the experiments section and we further detail them in the Supplementary. Given the shared selection criteria with CoreSet, our MoBYv2 AL framework falls in the representative ness-based category.  

Self-supervised Learning (SSL). For the past years, a new pillar, SSL, has arisen in unsu. pervised environments with linked goals to AL. Learning generalised concepts from large. scale data is critical for further expansion to various vision applications. We can divide the SSL in two approaches: consistency-based [4, 7, 33, 34] and contrastive energy-based [8, 10, 16, 19, 25, 41]. Consistency regularisation looks to preserve the class of unlabelled data even after a series of augmentations. For example, both MixMatch [4] and DINO [7] sharpen the averaged pseudo-labelled predictions. Conversely, contrastive learning generally demands pairs of positive and negative examples while optimising the similarity/contrast between them. Dual networks are usually deployed to evaluate these losses either within the batch (as in SimCLR [8]) or within a dictionary of keys (for methods like MoCo[19] MoBY[41]). Because contrastive learning is foundational to our proposal, we revise these techniques in Sec. 3.  

AL with self-supervision. In the beginning, SSL and AL evolved in parallel. Only re. cently, these fields have merged to further progress data sampling. Although SSL brings better visual constructs, there is still the question of which labelled information to allocate By leveraging the unlabelled data behaviour, CSAL [13] firstly integrated MixMatch in the AL training and selection. We follow a similar strategy, but our end-to-end training learns contrastive representations. Despite this, CSAL is included in the SSL-based experiment as it is directly comparable. Two new works tackle contrastive learning either in acquiring language samples, CAL [26], or by adapting the sequential SSL SimSiam [9] in [3]. CAL is task-dependent on natural language processing. In [3], the multi-stage AL selection has no effect against random sampling. To this extent, we omit these works in our analysis.  

# 3Methodology  

In this section, we explain our pipeline in detail. First, we introduce deep active learning for image classification in general, followed by our contributions.  

Standard AL requires an online environment where the task learner selects and optimises simultaneously. We consider a large unlabelled pool of data  ${\bf D}_{U}$   from which we uniformly random sample and label an initial subset  $\mathbf{S}_{L}^{0}<<\mathbf{D}_{U}$  .Let  $(\mathbf{x}^{L},\mathbf{y}^{L})\in\mathbf{S}_{$   be the available images and their corresponding classes. Commonly, we deploy a learner by a DL model comprising of a feature encoder f and a class discriminator g. The objective loss for the learner is the categorical cross-entropy defined as  $\begin{array}{r}{\mathcal{L}_{c l a s s i f i c a t i o n$  

Following the AL objective, we decide upon the exploration-exploitation trade-off in conjunction with our classification performance. Thus, we set up the exploitation rate through a budget  $b$  across  $\mathbf{D}_{U}/\mathbf{\DeltaS}_{L}^{0}$  guided by a selection criteria. Consequently, we label the new sampled subset  $\mathbf{S}_{L}^{1}$  and re-train our learner. The exploration factor is expressed by the num-  $\mathbf{S}_{L}^{0\dots N}$  

![](custom_collection/markdown/images_per_pdf/xaw/0a18b2c4ca9040a77d1ac89138feed71/2592df1af26eceb4be77a128450ea17460248f8e7b1d268bb5f9491778abe74f.jpg)  

Figure 1: SSL-AL training framework under the proposed MoBYv2AL configuration. The query feature encoder plays two roles: to map the features to the task discriminator for classi fication; to capture contrastive visual representation with the asymmetry of the query and key modules. For unlabelled data, the blue lines show the back-propagation of contrastive loss and its exponential moving average (dashed). The green lines also include the cross-entropy loss during training when the annotation is available. Once training ends, the unlabelled samples pass through the learner for AL selection.  

limit the exploration cycles, in our proposal, we primarily focus on exploitation.  

Contrastive Semi-supervised learning framework. We tackle the contrastive unsupervised learning approach compared to previous semi-supervised AL techniques [13, 17, 20] that rely on consistency measurement. Here, we briefly re-introduce the key aspects of the previous SSL techniques. These are constituent to our MoBYv2AL proposal.  

The goal of self-supervised learning aligns with the AL problem, where there is plenty of unlabelled data and a costly annotation procedure. However, the former tends to learn generalised visual representation in aid of the objective task. For contrastive learning, the main approach to obtaining these representations is by analysing the similarity (dissimilarity) within the data space. From the most successful works [7, 8, 16, 19, 41], we can broadly form the contrastive learning process of these main parts: data augmentation with or without dual encoder, feature-vector projections, and similarity approximation by a dedicated loss function.  

We design the self-supervision framework according to MoBY [41]. This method com. bines two innovative prior works MoCo[19] and ByOL[16] on visual transformers [11, 38] MoCo[19] pioneers contrastive learning by addressing the similarity between an image and a specific dictionary of samples. To deploy the loss, positive examples are required through data augmentation of the input query together with the other negative keys from the dictio. nary. The self-supervision training pipeline consists of two feature encoders and two MLP projectors for mapping the query and the keys. Consequently, the keys are permuted in a large memory bank, while the positive examples are inferred through the online encoder. The gradient over the dictionary of keys needs a slower update. Thus, a gradual momentum update is implemented.  

BYOL[16], on the other hand, has a different approach for contrastive self-supervision.  

It simplifies MoCo by relying only on positive examples. In this way, the memory bank can be discarded. The InfoNCE[27] loss is also replaced with a l2 loss given the new setting. The contrastive learning strategy of BYOL is indirectly obtained through batch normalisation. To achieve this, further modifications are proposed. Thus, the architecture of the dual encoders is asymmetric in regard to MoCo, and BYOL adds a prediction module to the projector of the online encoder. Following only positive examples, the inputs to the two networks are strong. augmented versions of the same image. Finally, BYOL preserves the common mode from the data and inherits contrastive learning when passing a slow exponential moving average from the online to the momentum encoder. We intuitively explore the contrastive learning strategies from both MoCo and BYOL and align the self-supervision with MoBY[41]. We further present the combined pipeline depicted in 1.  

From a design perspective, we adopt the asymmetric dual encoders from BYOL as shown in Fig. 1. The top branch in Figure 1 culminates with a discriminator  ${\bf g}_{q}$  to match the outputs from the bottom. Despite this, both branches consist of the same feature extractor archi tecture followed by an MLP projector  $(\mathbf{f}_{q}^{\prime},\mathbf{f}_{k}^{\prime}$  for query and key, respectively). Distinctively from MoBY, we tackle convolutional neural networks (CNNs) as feature encoders. More. over, we reduce the MLP projectors and the query discriminator to a single layer with batch normalisation and ReLU activation.  

The asymmetric pipeline helps to mimic the contrastive learning principle of BYOL However, to include the concepts from MoCo, we minimise our objective with the InfoNCE loss. In this case, we will also need to keep the memory bank for the queue of keys. We define the contrastive loss as a sum of InfoNCE from two augmented versions of a query  $\{q,q^{\prime}\}$  and of a different key  $\{k,k^{\prime}\}$  

$$
\mathcal{L}_{c o n t r a s t i v e}=-\log\frac{e x p(
$$  

where  $m$  is the size of the memory bank and  $\tau$  is the adjusting temperature [39]. During training, the online query encoder branch is updated by gradient while the key encoder takes the slow-moving average with momentum. We ensure with this combined design the preser vation of both MoCo and BYOL representation concepts. On the one hand, the asymmetric structure indirectly finds discrepancies from the average image with moving average and batch normalisation. On the other hand, the contrastive loss with the queue of different key. maintains the direct distinctiveness between the images.  

The standard SSL techniques MoCo, BYOL and MoBY demand the supervision stage where the pre-trained models are fine-tuned for the task objective. Such multi-stage pipelines seem ineffective in AL [3]. In this paper, we extended the SSL pipeline of MoBY to minimise both the self-supervised objective and downstream task objective jointly.  

Joint Objective. A final step to clarify before presenting the joint training procedure is data augmentation. MoBY derives the augmentation strategy from BYOL, where the inputs suffer strong transformations. In our proposal, we choose an alternation between strong and weak augmentation, similarly to MoCov2[10]. This change boosted the performance of its predecessor [19]. We also observed in our experiments that using only strong augmentations can affect the optimisation of the task-aware branch. The weak augmentations comprise horizontal flips and random crops. In addition, the strong transformation includes colou. jitter (on brightness, contrast, saturation, hue), Gaussian blur, grayscale conversion and pixel inversion (solarise). From equation 1,  $\{\boldsymbol{q},\boldsymbol{k}\}$   can be referred as the weak transformations of query and key, and  $\{q^{'},k^{'}\}$  their corresponding stronger versions.  

With all these elements in place, we can change the learner from the existing AL frame. work with the modified MoBY and train jointly the pipeline. Starting from the first cycle, we consider the available labelled samples  $(\mathbf{x}^{L},\mathbf{y}^{L})\in\mathbf{S}_{$  and the remaining unlabelled  $\mathbf{x}^{U}\in\mathbf{D}_{U}$  as queries and keys. A strong augmentation is marked as  $\{\widetilde{\mathbf{X}}_{q}^{L},\widetilde{\mathbf{X}}_$  , while a weak is rep- resented with  $\big\{\bar{\mathbf{x}}_{q}^{L},\bar{\mathbf{x}$  . When training, we alternate between batches of labelled and unla- belled data with every inference. Therefore, we back-propagate only the contrastive loss for the unlabelled to 1. In this context, given the pipeline from Figure 1 for this contrastive loss  $\mathcal{L}_{c o n t r a s t i v e}^{U}(q,q^{'$   $\{\check{q^{'}},k^{'}\}$  

$$
\begin{array}{r}{\{q,q^{'}\}=\mathbf{g}_{q}
$$  

$\mathcal{L}_{c o n t r a s t i v e}^{L}$   $\mathcal{L}_{c l a s s i f i c a t i o n}$  the task discriminator. Once computed, we back-propagate both the contrastive and the classification loss. Therefore, the combined loss, adjusted by a scaling factor  $\lambda_{c}$  , can be expressed as:  

$$
\mathcal{L}_{c o m b i n e d}^{L}=\mathcal{L}_{c l
$$  

While the contrastive loss is computed continuously regarding the classification loss, we de. cide to reduce its influence over the gradients with  $\lambda_{c}=0.5$  . Finally, it is worth mentioning that the exponential moving average and the queue of keys are updated on the bottom branch for both labelled and unlabelled samples.  

Unlabelled samples selection. We emphasise that our proposal minimises the self-supervised loss inspired by MoBy. With this, the end-task objective jointly enriches the visual repre- sentations of the data compared to the standard AL strategy. AL selection methods that rely on the learner's data distribution will perform better. CoreSet [28] has been proven to be effective in such scenario. To this extent, we primarily choose this selection function with MoBYv2AL. Briefly, CoreSet aims to find a subset of data points where a constant radius bounds the loss difference with the entire data space. This technique is approximated with k-Centre Greedy algorithm [37] in the euclidean space of our feature encoder outputs  $\ensuremath{\mathbf{f}_{q}}(\ensuremath{\mathbf{x}})$  - A thorough visual selection of different AL selection approaches together with CoreSet in presented in the Supplementary.  

# 4Experiments  

Datasets.For the quantitative evaluation, we put forward four well-known image classifica. tion datasets: CIFAR-10, CIFAR-100 [24], SVHN[14] and FashionMNIST[40]  

Models. We mentioned in 3 that we use different CNNs for feature encoders. To show that MoBYv2AL is robust to architectural changes, we opt for VGG-16 [31] in the CIFAR 10/100 quantitative experiments and for ResNet-18 [18] in SVHN and FashionMNIST  

Training settings. We train at every selection stage for 200 epochs, and we keep the batch size at 128. The dictionary size for the keys m is set up as in MoBY at 4096. We noticed in our experiments that the contrastive and cross-entropy loss converge together after 200 epochs. The learning rate starts at 0.01, and it follows a schedule for the queue encoder and  

![](custom_collection/markdown/images_per_pdf/xaw/0a18b2c4ca9040a77d1ac89138feed71/3d880378ad4242578266b2007237b6d914ebf755b9ab81fe22d6b79d5b96ace0.jpg)  
Figure 2: Evaluations on CIFAR-10 (left), CIFAR-100 (right) [Zoom in for better view]  

![](custom_collection/markdown/images_per_pdf/xaw/0a18b2c4ca9040a77d1ac89138feed71/84866fc3b13cf6ad7022b81f0fa6d213fc8e549b75f2d54f360a9ba8b905a84c.jpg)  
Figure 3: Evaluations on SVHN (left), FashionMNIST (right) [Zoom in for better view]  

task discriminator that decreases ten times at 120 and 160 epochs. However, we keep the momentum scheduler update in the key bottom branch (gradual momentum increment from 0.99). In the contrastive loss, for both queues, we fix the temperature parameter to 0.2.  

AL settings. We followed the AL settings of VAAL[32], CDAL[1] and CoreGCN[6]. Fol more details, please see Supplementary.  

Baselines. We compared our method MoBYv2AL with a wide range of methods in active learning such as: MC Dropout [15], DBAL [12], Learning Loss[42], VAAL[32], , Learning Loss [42], CoreGCN[6] and CDAL[1]  

# 4.1Quantitative experiments  

CIFAR10/100. To maintain a fair comparison, in Figure 2, we report the performance charts obtained by CDAL[1] and VAAL[32]. All methods use VGG-16 for the feature encoder. MoBYv2AL has a considerable advantage with the proposed SSL framework in the CIFAR- 10/100 experiments from the first selection stage. In both scenarios, we gain  $20\%$   testing accuracy over standard learning (  $62\%$  and  $28\%$  on CIFAR-10/100). This justifies the impor- tance of the joint training framework from MoBYv2AL.  

Our pipeline's more refined visual representations direct helpful information to the Core. Set selection method. Thus, we notice a gradual increase in Figure 2, where after 7 cy. cles, with  $40\%$  labelled data, MoBYv2AL achieves  $89.6\%$   mean accuracy on CIFAR-10 and  $63.1\%$   on CIFAR-100. Another observation in the CIFAR-10 experiment is that the AI performance saturates faster than in CIFAR-100. This effect occurs due to a large initial labelled pool in relation to the complexity of the task. MoBYv2 exploits more contrastive information, and it limits the exploratory potential in the next stages.  

SVHN/FashionMNIST. We can deduct, from Figure 3 as well, that MoBYv2AL balances the exploration-exploitation trade-off when the initial labelled set is relatively low to the number of classes. The dark dashed line displays the supervised baseline training on the entire labelled set. While on CIFAR-10/100 and FashionMNIST, MoBYv2AL reaches com- parable performance, by the end of the cycles, on SVHN, it surpasses after the sixth one  $(95\%)$  . Here, we emphasise the relevance of the strong/weak augmentations in enriching the discrete data distribution. Furthermore, grayscale data (as in FashionMNIST) can also ben- efit from the proposed AL framework. In Figure 3, we keep the same results of the previous baselines from CoreGCN[6]. Even under these settings, we outperform the state-of-the-arts with a noticeable consistent margin: for SVHN and FashionMNIST a gap of at least  $2\%$  1  $3\%$  -  

![](custom_collection/markdown/images_per_pdf/xaw/0a18b2c4ca9040a77d1ac89138feed71/ce31a18eecd9fd0825a33c41854ae806f03ac84fbf5b928b4ed8374efb9d0387.jpg)  

Comparison with other SSL-AL. MoBYv2 leverages unlabelled data for contrastive learn. ing in the AL framework. Previously, we chose this amount of data equal to the avail able labelled samples. Therefore, at every AL cycle, this size increases with the newly selected data. Another recent SSL-AL baseline CSAL[13], however, deployed the consis. tency measurements from MixMatch[4] on the entire unlabelled data. We could identify that MoBYv2AL over-exploits as CSAL the captured representation under these conditions We further compare the 2 methods on CIFAR-100 in Table 1 and adjust the feature encoder to WideResNet-28[43]. In this experiment, MoBYv2AL maintains the initial performance gain.  

Imbalanced dataset experiment. Apart from SVHN, all the previous experiments have a uniform distribution over the classes. This rarely occurs during real-world acquisition scenarios. Therefore, as in CoreGCN, we simulate an imbalanced CIFAR-10 unlabelled set. Each of the ten classes has originally 5000 training examples. We decide to reduce 5 of the classes to 500 images (resulting in a pool of 27500). The learner contains a ResNet-18 encoder, and it is trained with an initial set of 1000 labelled examples. We apply MoBYv2AL together with the other baselines from CoreGCN[6] for 7 cycles. Figure 4(left) presents the ability of MoBYv2AL to outperform the previous methods even in possible real-world environments. Investigation of long-tail distributions is still part of our future work  

![](custom_collection/markdown/images_per_pdf/xaw/0a18b2c4ca9040a77d1ac89138feed71/b66c1d8addc7a7283b2d4a0e20d31d19cfe619426714c22ef4f4c80bed5dfdda.jpg)  

Figure 4: CIFAR-10 imbalanced dataset experiment(left); Mitigating the distribution shift with MoBYv2AL(right) [Zoom in for better view]  

# 4.2Distribution shift discussion  

In deep AL, the cyclical process of re-training the learner with the new labelled data may result in optimising to different local minima. Therefore, the exploration and exploitation of the AL method will be affected by this distribution shift at every stage. During experiments. this is commonly shown through jaggy curves (especially for uncertainty-based methods like MC Dropout[15], DBAL[12] or UncertainGCN[6]). To address this known issue [23], we analyse MoBYv2AL performance on the entire CIFAR-10 training set when providing 1000 and 2000 samples.  

The dark blue bars of each class in Figure 4 (right) level the corresponding classification accuracy with the first 1000 random samples. Tracking the performance on the entire set challenges the learner to prefer certain classes. We continue to select with MoBYv2AL another set of images. Consequently, the resulted accuracy is displayed by the cyan bar. We can clearly observe that the minima shifted in a different direction where only some classes improved at the expense of the others. To mitigate this shift, we investigated what impact the unlabelled samples have in our end-to-end training. These samples play a key role in building up the dictionary of keys. Our insight is that the CoreSet selection on MoBYv2 data representation targets primarily high contrastive samples. We can control this effect by customising the unlabelled set deployed in training our learners. To this extent, we propose to use the unlabelled data with the lowest contrastive loss. In Figure 4 (right), we displayed on green bars the performance with this mechanism. From an initial 1000 set accuracy (dark blue) we get an effective linear increase for all the 10 classes. This effect is consistent throughout all the previous quantitative experiments as well.  

# 4.3 SSL modules variation and ablation study  

We continue to motivate the proposed design of MoBYv2AL with a set of ablation experi ments and by varying its SSL module. On the left side of Table 2, we swap in the end-to-enc training pipeline the original version of MoBY [41] and the preceding SSL state-of-the-arts MoCov2 [10] and BYOL [16]. Apart from MoBY, the learner did not converge on any selec. tion cycle with the other SSL modules. Thus, the setup of large batches and specific training conditions (low learning rates, cosine scheduler) and learners can hardly adapt to this semi supervision configuration. For MoBYv2AL, the weak-augmented inferences to the learner stabilise its performance in regard to the original version. Furthermore, our method distances by  $4\%$  class accuracy with each AL cycle. One can argue that our SSL framework comprises  

![](custom_collection/markdown/images_per_pdf/xaw/0a18b2c4ca9040a77d1ac89138feed71/89b2e3928f0d09b010ebd4125e5f4c45b6a6011d62b5cd4029dfad80bb50c6f1.jpg)  

Table 2: Variation of SSL pipeline (left) and ablation study of MoBYv2AL (right). Average testing performance (5 trials) on CIFAR-10 for 3 AL cycles with ResNet-18 encoder  

several building blocks, and its implementation can deter developers. While we value the significant dominance of MoBYv2 in AL selection, we still motivate the relevance of each part in Table 2 (right). In the ablation evaluation, we successfully remove the queue Discrim. inator and the MLP projectors. As a result, we detect a continuous accuracy drop. Projecting larger features and simulating the asymmetry brings the advantage of contrastive learning in MoBYv2. Moreover, strong augmentations also play a crucial role in the SSL pipeline.  

# 4.4SSL results and multi-stage AL  

MoBYv2 SSL for AL strategy is designed in a joint manner with the end task. Despite this. the recent work [3] that proposes contrastive learning with SimSiam[9] adopts multi-stage learning for the learner. The pipeline proposed fails to sample better than random in the AL paradigm. In Table 3(left), we experiment with MoBYv2 the multi-stage training (with unsupervised contrastive learning and second task fine-tuning) for CIFAR-10. We observe that the performance suffers in context to the end-task, where limited labelled examples are used. Similarly to [3], we also notice a minor improvement when adding more selected data with CoreSet. To this extent, we decided to use the entire training set during fine-tuning. We re-iterated the same experiment for SSL cross-validation with MoCov2[10], DINO[7] ang BYOL[16].  

# Limitations and Conclusions  

Although we can adapt MoBYv2AL to other applications, we expect further research on the effects of the augmentations and the momentum encoder. Another limiting factor should be analysed at the first AL selection stage, where developers may tune the exploration exploitation ratio to avoid saturation.  

We have presented an SSL-based AL framework for image classification. The main contributions lie in the task-aware contrastive learning pipeline. MoBYv2AL retains the higher visual concepts and aligns them with the downstream task. The joint training is efficient and modular, allowing diverse backbones and sampling functions. We conduct quantitative experiments and demonstrate the state-of-the-art on four datasets. Our method shows robustness even in simulated class-imbalanced data pools  

# 6 Acknowledgements  

This work is in part sponsored by KAIA grant (22CTAP-C163793-02, MOLIT), NST grant (CRC 21011, MSIT), KOCCA grant (R2022020028, MCST) and the Samsung Display cor- poration. BB and DS are funded in whole, or in part, by the Wellcome/EPSRC Centre for Interventional and Surgical Sciences (WEISS) [203145/Z/16/Z]; the Engineering and Phys. ical Sciences Research Council (EPSRC) [EP/P027938/1, EP/R004080/1, EP/P012841/1] and the Royal Academy of Engineering Chair in Emerging Technologies Scheme; and En. doMapper project by Horizon 2020 FET (GA 863146)  

# References  

[1] Sharat Agarwal, Himanshu Arora, Saket Anand, and Chetan Arora. Contextual diver sity for active learning. In ECCV, 2020.  

[2] William H Beluch Bcai, Andreas Nurnberger, and Jan M Kohler Bcai. The power of ensembles for active learning in image classification. In CVPR, 2018.

 [3] Javad Zolfaghari Bengar, Joost van de Weijer, Bartlomiej Twardowski, and Bogdan Raducanu. Reducing label effort: Self-supervised meets active learning. In ICCVW pages 1631-1639, 2021.

 [4] David Berthelot, Nicholas Carlini, Ian Goodfellow, Avital Oliver, Nicolas Papernot and Colin Raffel. Mixmatch: A holistic approach to semi-supervised learning. In NeurIPS, 2019.

 [5] Razvan Caramalau, Binod Bhattarai, and Tae-Kyun Kim. Active learning for bayesian 3d hand pose estimation. In WACV, 2021.

 [6] Razvan Caramalau, Binod Bhattarai, and Tae-Kyun Kim. Sequential graph convolu tional network for active learning. In CVPR, 2021.

 [7] Mathilde Caron, Hugo Touvron, Ishan Misra, Herve Jegou, Julien Mairal, Piotr Bo janowski, and Armand Joulin. Emerging properties in self-supervised vision trans- formers. In ICCV, 2021.

 [8] Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey E. Hinton. A simple framework for contrastive learning of visual representations. In ICML, 2020.

 [9] Xinlei Chen and Kaiming He. Exploring simple siamese representation learning. In CVPR, 2021.

 [10] Xinlei Chen, Haoqi Fan, Ross Girshick, and Kaiming He. Improved baselines with momentum contrastive learning. arXiv preprint arXiv:2003.04297, 2020.

 [11] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Syl. vain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Trans. formers for image recognition at scale, 2020.

 [12] Yarin Gal and Zoubin Ghahramani. Dropout as a Bayesian Approximation: Represent ing Model Uncertainty in Deep Learning. In ICML, 2016.

 [13] Mingfei Gao, Zizhao Zhang, Guo Yu, Sercan Arik, Larry Davis, and Tomas Pfister Consistency-based semi-supervised active learning: Towards minimizing labeling cost In ECCV, 2020.

 [14] Ian J Goodfellow, Yaroslav Bulatov, Julian Ibarz, Sacha Arnoud, and Vinay Shet Multi-digit Number Recognition from Street View Imagery using Deep Convolutional Neural Networks, 2013. 1312.6082v4.

 [15] Marc Gorriz, Axel Carlier, Emmanuel Faure, and Xavier Giro-i-Nieto. Cost-effective active learning for melanoma segmentation. CoRR, abs/1711.09168, 2017.

 [16] Jean-Bastien Grill, Florian Strub, Florent Altche, Corentin Tallec, Pierre Richemond Elena Buchatskaya, Carl Doersch, Bernardo Avila Pires, Zhaohan Guo, Mohammad Gheshlaghi Azar, Bilal Piot, koray kavukcuoglu, Remi Munos, and Michal Valko. Bootstrap your own latent - a new approach to self-supervised learning. In NeurIPS, 2020.  

[17] Jiannan Guo, Haochen Shi, Yangyang Kang, Kun Kuang, Siliang Tang, Zhuoren Jiang Changlong Sun, Fei Wu, and Yueting Zhuang. Semi-supervised active learning for semi-supervised models: Exploit adversarial examples with graph-based virtual labels In ICCV, 2021.

 [18] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In CVPR, 2016.

 [19] Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross B. Girshick. Momentum contrast for unsupervised visual representation learning. In CVPR, 2020.

 [20] Siyu Huang, Tianyang Wang, Haoyi Xiong, Jun Huan, and Dejing Dou. Semi- supervised active learning with temporal output discrepancy. In ICCV, 2021

 [21] Kwanyoung Kim, Dongwon Park, Kwang In Kim, and Se Young Chun. Task-aware variational adversarial active learning. In CVPR, pages 8166-8175, 2021.

 [22] Seong Tae Kim, Farrukh Mushtaq, and Nassir Navab. Confident Coreset for Active Learning in Medical Image Analysis, 2020. 2004.02200v1.

 [23] Andreas Kirsch, Tom Rainforth, and Yarin Gal. Test distribution-aware active learning. A principled approach against distribution shift and outliers. 2021.

 [24] Alex Krizhevsky. Learning multiple layers of features from tiny images. University of Toronto, 05 2012.

 [25] Junnan Li, Caiming Xiong, and Steven C.H. Hoi. Semi-supervised learning with con trastive graph regularization. In ICCV, 2021.

 [26] Katerina Margatina, Giorgos Vernikos, Loic Barrault, and Nikolaos Aletras. Active learning by acquiring contrastive examples. In EMNLP, 2021.

 [27] Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with con. trastive predictive coding. arXiv.org perpetual, non-exclusive license, 2018.

 [28] Ozan Sener and Silvio Savarese. Active Learning for Convolutional Neural Networks: A Core-set approach. In ICLR, 2018.

 [29] Burr Settles. Active learning literature survey. Computer Sciences Technical Report 1648, University of Wisconsin-Madison, 2009.

 [30] H. S. Seung, M. Opper, and H. Sompolinsky. Query by committee. In Proceedings of the Fifth Annual Workshop on Computational Learning Theory, COLT '92, page 287-294, New York, NY, USA, 1992. Association for Computing Machinery. ISBN 089791497X. doi: 10.1145/130385.130417.

 [31] Karen Simonyan and Andrew Zisserman. Very Deep Convolutional Network for Large scale image recognition. In ICLR, 2015.

 [32] Samarth Sinha, Sayna Ebrahimi, and Trevor Darrell. Variational Adversarial Active Learning. In ICCV, 2019.  

[33] Kihyuk Sohn, David Berthelot, Chun-Liang Li, Zizhao Zhang, Nicholas Carlini Ekin D. Cubuk, Alex Kurakin, Han Zhang, and Colin Raffel. Fixmatch: Simplifying semi-supervised learning with consistency and confidence. In NeurIPs2020, 2020.

 [34] Antti Tarvainen and Harri Valpola. Mean teachers are better role models: Weight averaged consistency targets improve semi-supervised deep learning results. In NeurIPS, 2017.

 [35] Ivor W. Tsang, James T. Kwok, and Pak-Ming Cheung. Core vector machines: Fast svm training on very large data sets. JMLR, 2005.

 [36] Laurens van der Maaten and Geoffrey Hinton. Visualizing data using t-sne, 2008 JMLR.

 [37] Gert Wolf. Facility location: concepts, models, algorithms and case studies. In Contri butions to Management Science, 2011.

 [38] Bichen Wu, Chenfeng Xu, Xiaoliang Dai, Alvin Wan, Peizhao Zhang, Zhicheng Yan Masayoshi Tomizuka, Joseph Gonzalez, Kurt Keutzer, and Peter Vajda. Visual trans. formers: Token-based image representation and processing for computer vision, 2020

 [39] Zhirong Wu, Yuanjun Xiong, X Yu Stella, and Dahua Lin. Unsupervised feature learn ing via non-parametric instance discrimination. In CVPR, 2018.

 [40] Han Xiao, Kashif Rasul, and Roland Vollgraf. Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms, 2017. 1708.07747v2.

 [41] Zhenda Xie, Yutong Lin, Zhuliang Yao, Zheng Zhang, Qi Dai, Yue Cao, and Han Hu Self-supervised learning with swin transformers. arXiv preprint arXiv:2105.04553 2021.

 [42] Donggeun Yoo and In So Kweon. Learning Loss for Active Learning. In CVPR, 2019

 [43] Sergey Zagoruyko and Nikos Komodakis. Wide residual networks. In BMvC, 2016  

# A Detailed settings for the AL experiments on MoBYv2AL  

Datasets.For the quantitative evaluation, we put forward four well-known image classifica. tion datasets: CIFAR-10, CIFAR-100 [24], SVHN[14] and FashionMNIST[40]. CIFAR-10 and CIFAR-100 contain the same 50000 training examples but with different labelling sys. tems (10 and 100 classes). SVHN and FashionMNIST are separated into ten classes each as CIFAR-10. However, both datasets are larger, with 73257 coloured street numbers and 60000 grayscale images for FashionMNIST. Although CIFAR-10/100 and FashionMNIST have class-balanced data, this is not the case for SVHN. From another perspective, deploying grayscale images from FashionMNIST challenges our contrastive learning approach, previ- ously customised to RGB data.  

Models. We mentioned in the Methodology that we use different CNNs for feature en. coders. To show that MoBYv2 is robust to architectural changes, we opt for VGG-16 [31] in the CIFAR-10/100 quantitative experiments and for ResNet-18 [18] in SVHN and FashionMNIST. Moreover, for the SSL comparison with CSAL we align the encoder with WideResNet-28[43]  

AL settings. Under the exploration-exploitation trade-off, we characterise the budget to se. lect as an exploiting factor while the exploration is captured in the number selection cycles. The initial random-sampled labelled dataset varies between the CIFAR-10/100 experiments and SVHN/FashionMNIST. For CIFAR-10/100, we consider  $10\%$  (5000) of the entire train- ing set as labelled and the rest as unlabelled data. The budget is limited to  $5\%$  (2500) samples for selection, and we repeat this cycle seven times. In the second set of experiments, we test our method in a more restrictive environment with a starting set of 1000 labelled and a sim. ilar fixed budget. Despite this, we expanded the exploration to 10 cycles reaching 10000 labelled data. As a performance measurement, we evaluate the average of 5 trials testing accuracy in the AL framework  

# B Selection function analysis  

![](custom_collection/markdown/images_per_pdf/xaw/0a18b2c4ca9040a77d1ac89138feed71/0d5a2535ddecfcdf583d4ef6c6775c9940fc62b583264ade99498184f003e856.jpg)  

Figure B.1: Quantitative evaluation with different selection functions for CIFAR-10 (left) CIFAR-100 (right) [Zoom in for better view]  

Our proposed pipeline, MoBYv2AL, can easily adapt to multiple selection methods Here, we quantitatively motivate the choice of CoreSet from section 3. Therefore, we re.  

evaluate MoBYv2AL on CIFAR-10/100 benchmarks in Figure B.1. We vary the selection of the new budget between random, maximum class entropy and CoreSet. Intuitively, we alsc analyse the effect of selecting unlabelled examples with high contrastive loss.  

In both benchmarks, sampling with random or max entropy benefits the less MoBYv2AL pipeline. On the other hand, a representative ness-oriented method like CoreSet suits our hypothesis better. When sampling with high contrastive loss, we detected repetitive examples from some specific classes. This can be explained by higher contextual variance in that category. Specifically, on CIFAR-10, animal classes (cat, deer, dog), with stronger patterns. were more preferred than the vehicle ones (car, truck, ship).  

For a better visual analysis, we have simulated a toy-set experiment with the first five classes from SVHN. Here, we take t-SNE[36] representations of the MoBYv2AL query en coder outputs of unlabelled data. In Figure B.2, the samples marked with crosses construct the new labelled set. The selection behaviour of the Max Entropy and CoreSet can be inter.  

![](custom_collection/markdown/images_per_pdf/xaw/0a18b2c4ca9040a77d1ac89138feed71/948aacb1756ceb440c5b1badac26143fdab85c885d6dfcfa42cc6429159ac1b6.jpg)  
Figure B.2: Qualitative AL selection analysis on MoBYv2. t-SNE representations at the first selection stage for 5 classes of SVHN. [Zoom in for better view]  

preted as expected: on the left side, the uncertainty-based technique tracks the most class variant images; CoreSet, on the right side, samples both in and out-of-distribution according to the Euclidean space.  