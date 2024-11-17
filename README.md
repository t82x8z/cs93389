java c
Machine Learning Practical 2024/25: Coursework 2 
Released: Monday 11 November 2024 
Submission due: 12:00 Friday 22 November 2024 
1 Introduction The aim of this coursework is to explore the classification of images using convolutional neural networks (CNNs) on a different dataset,CIFAR100 (pronounced as “see far 100”). CIFAR100 consists of 60,000 32 × 32 colour 	images in 100 classes, with 600 images per class. The first part of the coursework concerns with debugging a “broken” neural network and quantitatively analyzing the problem behind this broken network. The second part involves exploring two solutions in literature for fixing this “broken” neural network, and then subsequently implementing them to improve the performance and training of this network.In order to support your experiments, we have acquired Google Cloud Platform. credits which allow the use of the Google Compute Engine infrastructure. You will need this to run all the tasks of the coursework (which will be carried out in PyTorch).  Each student enrolled on the MLP course will receive a $50 Google Cloud credit coupon which is enough to carry out the experiments required for this coursework. You will receive an email which will give you the URL you will need to access in order to request your Google Cloud Platform. coupon. Note that there is no ability to provide you with further credits, and you need to be careful to not unnecessarily leave processes running on the cloud that will unproductively consume your allowance. 
As in the previous coursework, you will need to submit your python code and a report. The detailed submission instructions are given in Section 8.2– please follow these instructions carefully.
2 Github branch mlp2024-25/coursework2The provided code and setup information for this coursework is available on the course Github repository on a branch mlp2024-25/coursework2. To create a local working copy of this branch in your local repository you need to do the following.
1.  Make sure all modified files on the branch you are currently have been committed (see notes/getting- started-in-a-lab.mdif you are unsure how to do this).
2.  Fetch changes to the upstream origin repository by running git  fetch  origin
3.  Checkout a new local branch from the fetched branch using
git  checkout  -b  coursework2  origin/mlp2024-25/coursework2
4.  Register the MLP python package using python  setup.py  developYou will now have a new branch in your local repository with everything you need to carry out the coursework.Before  continuing,   remember  to  run  bash  install .sh  to  install  some  additional  dependencies  re- quired.    This  assumes  that  you  have  already  installed  your  environment  as  explained  in Lab1,  under notes/environment-set-up.md. Note this is only necessary if you are not using the Google Cloud VM, which comes with the dependencies pre-installed.
This branch includes the following additions to your setup:
• For the PyTorch Framework:
– A Jupyter notebook, Coursework_2_Pytorch_Introduction .ipynb, which introducesPytorch, and provides further resources on tutorials, documentation and debugging.
– A  directory  pytorch_mlp_framework/,   which  includes  tooling  and  ready  to  run  scripts to  enable  straightforward  experimentation  on  GPU.  Documentation  on  this  is  included  in notes/pytorch-experiment-framework.md
• For the report:
– A directory called report which contains the LaTeX template and style. files for your report. You should copy all these files into the directory which will contain your report.
3 Tasks 
This coursework has three tasks and objectives:
• Diagnose a deep CNN that cannot be properly trained due to optimization issues.  Discuss what the problem is, why it happens in a deep CNN and quantitatively analyze the problem.
• Describe the solutions, Batch Normalization (BN) and Residual Connection in (RC) detail and explain how these methods address the aforementioned problem.
• Implement BN and BN+RC, run the respective experiments and then present and discuss these and other results provided in the Template.Carrying out larger CNN experiments using the MLP numpy framework is computationally inefficient because (1) it runs on CPU and not on GPU, and (2) a default implementation of a convolutional layer is unlikely to be computationally efficient. For these reasons in this coursework, which involves training CNNs, we will use GPU computing (on the Google Compute Engine) and the efficient PyTorch framework.
4 Task 1: Identifying training problems of a deep CNN 
Time budget: This section should take about 20% of the time you have allocated for your coursework. 
This part of the coursework involves debugging and tuning deep CNNs using PyTorch and the Google Compute Engine.Identifying and resolving any optimization-related problems that might prevent your model from fitting the training set are critical skills when working with deep neural networks. Being able to workaround this problem is key to building high performance deep neural networks.
4.1 Introducing our broken CNN 
Note: This section’s objective is to test your knowledge, understanding and research skill. It’s not meant to be an implementation-oriented section. The solutions only require you to add at most 4-5 lines of code. Using the Pytorch-based research framework that can be found in the pytorch_mlp_framework folder, we have built, trained and evaluated 2 deep CNNs. One consisting of a total of 7 convolutional layers + one fully connected, and another consisting of 37 convolutional layers + one fully connected layer. Figure 1 illustrates the



Figure 1:  Training and validation plots of Healthy (VGG_08) vs Broken (VGG_38) in terms of error and accuracy.training/val loss performance of the two models. One can clearly see that the 37 layer CNN (VGG_38) was unable to minimize its loss, unlike the healthy 7 layer CNN (VGG_08) which converges to a low error. Given that we know that extra layers means more abstraction power, parameters and capacity, one would expect the deeper model to be doing better at learning than the shallow one, however this is simply not the case.Identifying the problem. One of our two networks suffers from the Vanishing Gradients problem as you can  see in Figure 1. You can reproduce the figures by running the notebooks/Plot_Results .ipynb notebook. This file takes as input the collected metrics in folders VGG_08 and VGG_38.Quantitative analysis of the problem. Question 1 on the template asks you to identify the problem and support your position using arguments based on quantitative observations. For instance, Figure 2 shows how the gradient flows in the healthy network, VGG_08 across layers. These are the gradients with respect to the weights of the model. Visualize and discuss how gradient flows for the broken network affects/does not affect the loss curves, training and convergence of the VGG_38.Implementation details. The curves shown in Figure 1 can be reproduced by running the bash scripts to  train  each  model  from   scratch  with  the  default   settings  given  in  run_vgg_08_default .sh  and run_vgg_38_default .sh.For reproducing Figure 2 and visualizing the gradient flows for the broken network, implement the function plot_grad_flows() within pytorch_mlp_framework/experiment_builder.py. This function takes as input the model parameters during training, accumulates the absolute mean of the gradients in all_grads and the layer names in layers. The matplotlib function plt plots gradient values for each layer and the function plot_grad_flows() returns this final plot.
Template Questions related to Task 1 are: 
• Question Figure 3 
• Question 1 (20 Marks) 
5 Task 2: Proposed Solutions 
Time budget : This section should take about 20% of the time you have allocated for your coursework. 
There exist several methods that can improve the training performance of the broken 37 layer CNN introduced in Section 4.1. In the template, we ask you to consider some evidence from the Residual Connections paper [He 

Figure 2: Gradient Flow in each layer for the VGG_08 network.et al.,2016] and put it into perspective. Later in the text, we explain Residual Connections [He et al.,2016], and Batch Normalization [Ioffe and Szegedy,2015] in detail, using relevant equations, while also explaining how it can help mitigate the Vanishing Gradient problem.
Template Questions related to Task 2 are: 
• Question 2 (20 Marks) 
6 Task 3: Solution and Experiments 
Time budget : This section should take about 50% of the time you have allocated for your coursework. Solution Overview. You will need to incorporate Batch Normalization, and Batch Normalization with Residual Connections to VGG_38 model to improve its training performance. You will then run one experiment instance for each setup and, on the Template, fill in the respective Table, and generate and add Figures.  Finally, you will present and discuss all relevant results (including those already filled in on the Template), while also making suggestions for further experiments you could have run, and alternative implementations for Residual Connections.Experiments. The techniques used in these solutions are described in Ioffe and Szegedy [2015], He et al. [2016]. We have written the MLP Pytorch framework in a way that allows you to implement said solutions with  minimal effort. Each solution will require at approximately 8-10 lines of code.
Important Components to Note 
1.  The classes ConvolutionalProcessingBlock and ConvolutionalDimensionalityReductionBlock located within pytorch_mlp_framework/model_architectures .py. The former class implements a basic cascade of 2 convolutional layers, each followed by a Leaky ReLU activation function, while the latter implements a basic cascade of 2 convolutional layers, with an average pooling layer in the middle, which effectively halves the height and width dimensions of the tensor volume passing through it. The former is used as the basic building block of the model (repeated num_blocks_per_stage  times), while the latter is used only as a dimensionality reduction layer, usually once after each network stage. A network stage is considered a cascade of convolutional layers, followed by a dimensionality reduction function such as average pooling. 
2.  Lines 43-50 in pytorch_mlp_framework/train_evaluate_image_classification_system.py showcase how one can create an if-else structure that can read arguments from the command line in order to choose which processing and dim_reduction blocks will be used.
3.  Lines 17-27 in the same file showcase how some simple data-augmentation strategies can be applied to the system (useful in improving the generalization of the model.)
Implementing a proposed solution 
1.  Copy the classes ConvolutionalProcessingBlock and ConvolutionalDimensionalityReductionBlock located within pytorch_mlp_framework/model_architectures .py, paste them at the end of the file and provide a name referring to your proposed solution.
2.  Modify the two blocks to implement your proposed solution. Make sure your code is clear and easily readable as you will be marked on it. When used, BN is applied after each convolutional layer, before the Leaky ReLU non-linearity. Similarly, the skip connections are applied from before the convolution layer to before the final activation function of the block as per Figure 2 of He et al. [2016]. Note that adding residual connections between the feature maps before and after downsampling requires special treatment, as there is a dimensional mismatch between them. Therefore in the coursework, we do not use residual connections in the down-sampling blocks. However, please note that batch normalization should still be implemented for these blocks.
3.  To create a new block ensure that you rewrite the build_module() and forward() methods.
4. Add a new clause in the if-else structure in lines 43-50 of pytorch_mlp_framework/train_evaluate _image_classification_system.py to add a choice for a configuration using your new blocks.
5. Write small unit-tests for your blocks to ensure that the layers can fprop some data without throwing errors related to Pytorch.
6.  Once tested, you can use the argument parser to easily run experiments with your new modules. Hint: Have a look at the block_type argument under pytorch_mlp_framework/arg_extractor.py, to get an idea on how to easily pass module names as an argument.  Additionally, we ask you to run an experiment with a different learning rate. The parser can come in handy for this.
Using Google Compute Engine代 写Machine Learning Practical 2024/25: Coursework 2Python
代做程序编程语言 
1. You will receive an email containing the URL you will need to access in order to request a Google Cloud Platform. coupon, and information about how to do this.
2.  In themlp_compute_engines branch of the GitHub, notes/google_cloud_setup.md gives the in- structions you should follow to set up a Google Compute Engine instance to carry out this coursework.
3.  The PyTorch experimental framework that is used for this coursework is described in notes/pytorch-experiment-framework.md and in
notebooks/Coursework 2 Pytorch_Introduction .ipynb
Template Questions related to Task 3 are: 
• Question Figure 4 
• Question Figure 5 
• Question Table 1 
• Question 3 
• Question 4 
Your implementations of VGG38 BN and VGG38 BN+RC will also be marked as part of this Task. (50 Marks) 
7 Report 
Time budget : This section should take about 10% of the time you have allocated for your coursework. 
The report template is divided into sections which roughly corresponds to each task in the coursework specs. In addition to the mark distribution for tasks, remaining sections of the report will be graded as follows:
Conclusion: 10 Marks The directory coursework2/report contains a template for your report (mlp-cw2-template.tex) and  a  file  for  filling  in  your  answers  to  the  questions  (mlp-cw2-questions.tex);  the  generated  pdf  file  (mlp-cw2-template.pdf) is also provided, and you should read this file carefully as it contains some useful  information on the relevant theory, and about the required structure and content.  The template is written in  LaTeX, and uses the file (mlp-cw2-questions.tex) to fill in the answers to the questions on the template. You should fill in your answers in (mlp-cw2-questions.tex) and compile (mlp-cw2-template.tex) to  produce (mlp-cw2-template.pdf) which you will need to submit .You should copy the files in the report directory to the directory containing the LaTeX file of your report, as pdflatex will need to access these files when building the pdf document from the LaTeX source file.  The coursework 1 spec (see Learn) outlineshow to create a pdf file from a LaTeX source file.
As discussed in the coursework 1 spec, all figures should ideally be included in your report file as vector graphics files, rather than raster files as this will make sure all detail in the plot is visible.
If you make use of any any books, articles, web pages or other resources you should appropriately cite these in your report. You do not need to cite material from the course lecture slides or lab notebooks.
Template Questions related to the Report are: 
• Question 5 
(10 Marks) 
8 Mechanics 
Marks: This assignment will be assessed out of 100 marks and forms 40% of your final grade for the course.
Academic conduct: Assessed work is subject to University regulations on academic conduct:
http://web.inf.ed.ac.uk/infweb/admin/policies/academic-misconduct Extension requests: The deadline and late policy for this assignment are specified on Learn in the “Coursework Planner” . Guidance on late submissions is at https://web.inf.ed.ac.uk/node/4533Do not email any course staff directly about extension requests; you must follow the instructions on the web page.Submission: You can submit more than once up until the submission deadline. All submissions are timestamped automatically.  Identically named files will overwrite earlier submitted versions, so we will mark the latest submission that comes in before the deadline.If you submit anything before the deadline, you may not resubmit after the deadline. (This policy allows us to begin marking submissions immediately after the deadline, without having to worry that some may need to be re-marked).If you do not submit anything before the deadline, you may submit exactly once after the deadline, and a late penalty will be applied to this submission unless you have received an approved extension. Please be aware that late submissions may receive lower priority for marking, and marks may not be returned within the same timeframe. as for on-time submissions.Submissions are only accepted via Learn, per policy. If you are encountering technical difficulties you need to be in contact with Computing Support in good time to resolve them, and/or your Student Advisor if you need Adjustments for reasons beyond your control. Warning: Unfortunately the submission system on Learn will technically allow you to submit late even if you submitted before the deadline (i.e. it does not enforce the above policy). Don’t do this! We will mark the version that we retrieve just after the deadline.
http://web.inf.ed.ac.uk/infweb/student-services/ito/admin/coursework-projects/late-coursework-extension-requests 
Late submission penalty: Following the University guidelines, late coursework submitted without an authorised extension will be recorded as late and the following penalties will apply: 5 percentage points will be deducted for every calendar day or part thereof it is late, up to a maximum of 7 calendar days. After this time a mark of zero will be recorded. 
8.1 Backing up your work It is strongly recommended you use some method for backing up your work.  Those working in their AFS homespace on DICE will have their work automatically backed up as part of the routine backup of all user homespaces. If you are working on a personal computer you should have your own backup method in place (e.g. saving additional copies to an external drive, syncing to a cloud service or pushing commits to your local Git repository to a private repository on Github). Loss of work through failure to back up does not constitute a good reason for late submission. 
You may additionally wish to keep your coursework under version control in your local Git repository on the coursework2 branch.If you make regular commits of your work on the coursework this will allow you to better keep track of the changes you have made and if necessary revert to previous versions of files and/or restore accidentally deleted work.  This is not however required and you should note that keeping your work under version control is a distinct issue from backing up to guard against hard drive failure. If you are working on a personal computer you should still keep an additional back up of your work as described above.
8.2 Submission 
Your coursework submission should be done online on the Learncourse webpage.
Your submission should include one zip file sxxxxxxx .zip that should contain a folder named sxxxxxxx. This folder should have -
• your completed report as a PDF file renamed as sxxxxxxx_report.pdf, using the provided template
• your local version of the Pytorch experiment framework (mlp/pytorch_experiment_scripts), includ- ing any changes you’ve made to existing files and any newly created files. In particular, this should contain  the model_architectures.py python script. with your implementation of BN and RC.
•  a copy of your Pytorch experiment directories (mlp/results), including only the  .csv files for your training, validation and test statistics. Please do not include model weights.
Please do not submit anything else (e.g. log files).
You should copy all of the files to a single directory, sxxxxxxx, e.g. Your folder directory structure should look like this -
sxxxxxxx
·  sxxxxxxx_report .pdf ·  
mlp
·  pytorch_experiment_scripts/ ·  
results/
You can use this command on Linux machines to zip all the files together -
zip  -r  sxxxxxxx .zip  sxxxxxxx
Replace sxxxxxxx with your student id.
Once you have  successfully created the  .zip file,  you need to login to your Learn Machine  Learning Practical  (2024-2025)[YR] webpage and submit the file.
• From the main Learn page, find the item named Assessment and click on it.
• Click on CW 2.
• A page will appear where you will need to browse and upload your .zip file that you created previously in Attach  Files (click on the paperclip icon) and then click Submit.
You can amend an existing submission by attaching a different .zip file using the Attach Files option and then Submit again.
Note that we will only mark the last uploaded coursework in case you amend your files. Thus it is your responsibility to make sure that correct files are uploaded. 
9 Marking Guidelines 
This document (Section 3 in particular) and the template report (mlp-cw2-template.pdf) provide a description of what you are expected to do in this assignment, and how the report should be written and structured.
Assignments will be marked using the scale defined by the University Common Marking Scheme:
Numeric mark    Equivalent letter grade     Approximate meaning
< 40                    F                                        fail
40-49                   D                                       poor
50-59                   C                                       acceptable
60-69                   B                                       good
70-79                  A3                                    very good/distinction
80-100                A1, A2                             excellent/outstanding/high distinction
Please note the University specifications for marks above 50:
A1 90-100 Often faultless. The work is well beyond what is expected for the level of study.
A2 80-89 A truly professional piece of scholarship, often with an absence of errors.
As ‘A3’ but shows (depending upon the item of assessment): significant personal insight / creativity / originality and / or extra depth and academic maturity in the elements of assessment.
A3 70-79 
Knowledge: Comprehensive range of up-to-date material handled in a professional way.
Understanding/handling of key concepts: Shows a command of the subject and current theory.
Focus on the subject: Clear and analytical; fully explores the subject.
Critical analysis and discussion: Shows evidence of serious thought in critically evaluating and integrating the evidenced and ideas. Deals confidently with the complexities and subtleties of the arguments. Shows elements of personal insight / creativity / originality.
Structure: Clear and coherent showing logical, ordered thought.
Presentation: Clear and professional with few, relatively minor flaws. Accurate referencing. Figures and tables well constructed and accurate. Good standard of spelling and grammar.
B 60-69 
Knowledge: Very good range of up-to-date material, perhaps with some gaps, handled in a competent way.
Understanding and handling of key concepts: Shows a firm grasp of the subject and current theory but there may be gaps.
Focus on the subject: Clear focus on the subject with no or only trivial deviation.
Critical analysis and discussion: Shows initiative, the ability to think clearly, critically evaluate ideas, to bring different ideas together, and to draw sound conclusions.
Structure: Clear and coherent showing logical, ordered thought.  Additionally for code: re-usability may be somewhat limited. No unused variables or dead code.
Presentation: Clear and well presented with few, relatively minor flaws.  For writing: Accurate referencing; using the correct referencing system.  Figures and tables well-constructed and accurate.  Good standard of spelling and grammar. Alternatively for code: well-documented, readable code.
C 50-59 
Knowledge: Sound but limited. Inaccuracies, if any, are minor.
Understanding and handling of key concepts: Understands the subject but does not have a firm grasp and depth of understanding of all the key concepts.
Focus on the subject: Addresses the subject with relatively little irrelevant material.
Critical analysis and discussion: Limited critical analysis and evaluation of sources of evidence.
Structure: Reasonably clear and coherent, generally structuring ideas and information or code in a logical way. Additionally for code: Few or no unused variables or dead code.Presentation: Generally well presented but there may be some flaws, for example in figures, tables, referencing technique and standard of English. Alternatively for code: generally well-documented, readable code, but with some weaknesses.
References 
Kaiming He, Xiangyu Zhang, ShaoqingRen, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 770–778, 2016.
Sergey Ioffe and Christian Szegedy.  Batch normalization: Accelerating deep network training by reducing internal covariate shift. In ICML, pages 448–456, 2015. URL http://proceedings.mlr.press/v37/ioffe15.html. 

         
加QQ：99515681  WX：codinghelp  Email: 99515681@qq.com
