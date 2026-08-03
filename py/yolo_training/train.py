from ultralytics import YOLO

#Desginate True if creating a new model, False otherwise
New_model = True

#load a model
if New_model:
    model= YOLO("yolov8n.yaml") #New model using v8 nano version
       
else:
    model = YOLO("/Users/willicon/Desktop/dronemodeling_Logan/runs/detect/train2/weights/last.pt")              #path to last.pt of currentl model
          


#train the model
results = model.train(              #Determined by new_model
                     data= "training/ultralytics_model/config.yaml" #'/Users/willicon/Desktop/dronemodeling_Logan/py/ultralytics_model/config.yaml'  #Configuation file with training and validation images desginated, as well as the classes
                    ,epochs =500
                    ,device =[-1,-1]) #,device="mps")                             #How many times the training will run. 
#The [-1,-1] should train the model on the two most idel GPUs?
