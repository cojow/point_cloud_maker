from ultralytics import YOLO

name = "provo_oakhills_wasatch_seed15"
model = YOLO("/Users/willicon/Desktop/dronemodeling_Logan/train2_provo/weights/best.pt") 

results = model.val(data= "py/yolo_model/config_val.yaml" #Uses validation congfig.
                    ,name = f"{name}_validation"
                    ,device="mps") #optimized to run on a mac processor