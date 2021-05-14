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

- imitation_p_1.py will generate joint data angles from the video and save them in a .txt file for each joint, running FNN_network.py and CNN_Network.py will train the model and plot its loss, by running the function data_plot() in Part_1/UTILS.py will generate a plot which compare the original data and smoothed data from savitzky_golay filter and save the figure as smoothed_data.png. 
                    
> Thankyou Dr. Sinnu Susan Thomas(IIITMK) for making this project successful.
