# Imitation-Learning-in-bipedal-robots

- publication
  
- **Requirements**
   - _Programming languages:_
      - Python
         - **packages**
             - PyBullet
             - Numpy
             - PyTorch
             - OpenCV
             - Matplotlib
             - Pandas
             - Gym


- Models used
   - **Lightweight OpenPose**

- How to Run?
   - clone the Repository
   - download the pretrained model for lightweight openpose [here](https://download.01.org/opencv/openvino_training_extensions/models/human_pose_estimation/checkpoint_iter_370000.pth).
   - run Part_1/imitation_p_1.py to create data folder and save the data in folder.
   - For feed forward network run Part_2/FNN_network.py
   - For Convolutional neural network run Part_2/CNN_network.py
   - both files will plot loss values plot for training

- Series of Experiments
  - Simple movements are tested only where limbs movement are not fast.
  - Cases
      - Adult Male normal walking
      - Adult Female normal walking
      - Adult Male long stride walking
      - Adult Female long stride walking

- imitation_p_1.py will generate joint data angles from the video and save them in a .txt file for each joint, file hierarchy for joint data will be
      
- In codes Section the code is available with name stochastic_model.jl. It will generate six main plots simulation_confidence_interval 2.5.jpeg, simulation_confidence_interval 3.5.jpeg, simulation_exposed_infected 2.5.jpeg, simulation_exposed_infected 3.5.jpeg, simulation_SEIR 2.5.jpeg, simulation_SEIR 3.5.jpeg when you applied the control.You can run the code without control by making the control_applied parameter=false.
- simulation_SEIR 2.5.jpeg, simulation_SEIR 3.5.jpeg are the main plots to visualize when you run without control.
 
> Thankyou all the contributors for making this project successful.
- Contributors
  - Dr Sinnu Susan Thomas(IIITMK)
  - Dr Edilson F. Arruda (University Of Southampton)
  - Rodrigue E.A. Alexandre (Federal University)

<a href="https://github.com/remarkablemark">
  <img src="https://github.com/Tarun-Sharma9168/Optimal_Control_And_Decision_Making/blob/main/contri_images/sinnu_mam.png" width="150" height="150">
</a>
<a href="https://github.com/remarkablemark">
  <img src="https://github.com/Tarun-Sharma9168/Optimal_Control_And_Decision_Making/blob/main/contri_images/edilson_arruda.jpg" width="150" height="150">
</a>
<a href="https://github.com/remarkablemark">
  <img src="https://github.com/Tarun-Sharma9168/Optimal_Control_And_Decision_Making/blob/main/contri_images/foto_rodrigo.jpeg" width="150" height="150">
</a>
