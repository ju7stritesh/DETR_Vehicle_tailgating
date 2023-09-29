## Vehicle Tailgating solution using Deep Sort and DETR

https://github.com/facebookresearch/detr

### We use DETR to detect vehicles and Deepsort to ReID the vehicle (One can use a better detection model like YOLO, CNNs or transformer based models).

### The results produced from the same can be viewed here
[tailgating.avi](VideoResults%2Ftailgating.avi)

### detr_deep_sort_vehicle.py - has the logic behind detecting and tracking vehicles. It also gives an option to draw Region of Interest

### inference_tailgating.py is the inference file where it accepts the video path in the event() function.

### Paramerters needed for the understanding of the environment:

### Steps to run the algorithm:
````
1) Run inference_tailgating.py
2) First frame of the video will pop up
3) Select a region by dragging the mouse pointer from top-let to bottom-right
4) Press 'q' to continue with the algorithm
5) We detect the vehicles tracking their IDs
6) The thought behind tailgating is to understand how long a certain vehicle was in the ROI or behind another vehicle when the gate was opened
7) The console statements shows the final output
````
It might be a little overwhelming at first to understand but the data analysis will generate the results for tailgating

Some important parameters and their thresholds
- ROI_OVERLAP_THRESHOLD - Overlap with ROI, default 0.8
- stop_second_threshold = Stoppage time for vehicle in the Region of Interest, default 4 seconds
- Similarity - Similarity between two vehicles, default 0.7
- max_dist_covered - Max distnace covered by the same vehicle, default 800
- min_dist - Minimum distance a vehicle travels inside ROI, default 144
- All the parameters are part of initilization within Tailgating class in inference_tailgating.py, this can be changed per camera basis when calling the event() function

author: ju7stritesh@gmail.com
