# Point Cloud Maker

Turns photo sets (taken from drone images) into georeferenced point
clouds. Individual buildings are extracted from those point clouds with
measurements (area, volume, height) and a cropped photo.

[DJI Flight Planner](https://github.com/cojow/flight_planner): For those using DJI drones, this is an free open-source flight planner desinged to be used with this code. 

## Running the pipeline
### 1. Install dependecies and setting up account
While the code can be run locally, it requires a large amount of processing power. To circumvent this, we will use the BYU supercomputer resources. The following will walk you though setting up your account and directories. Skip to the end of this section to just install the dependicies.  

>**Note**: Whenever "username" appears in a file path or terminal input,
replace it will the username you created when setting up your account.


#### Clone this repository
Simple as that. Save it somewhere you're remember.

#### Account Setup (For BYU Supercomputer)
Go to [`BYU Research Computing`](https://rc.byu.edu/account/create/) and follow the steps for making an account. **gmac** is the user that you’ll be under.  

Once you’ve created an account, open the terminal and run the following 
command (Replace “username” with your username):
``` 
ssh username@ssh.rc.byu.edu 
 
```
> **Note:** You will be required to set up a two factor authentication when
> you first log in. Read about it [`here`](https://rc.byu.edu/wiki/?id=Two-Factor+Authentication).
> Since BYU uses DUO, this will be the easiest to use. But you can use whatever authenticatino app you want.

You will be prompted to enter your password and authentication code. You will then be placed in the `/home/username` directory. This is where you will store and run all the files.  

#### Setup directories
Using the command `mkdir folder_name`, make the following file paths in the supercomputer terminal in the order given below. This will keep your workspace organized: 

>/home/username/point_cloud 

>/home/username/point_cloud/py 

>/home/username/point_cloud/scjobs 

>/home/username/point_cloud/data 

>/home/username/point_cloud/data/log_out

(i.e. `mkdir /home/username/point_cloud`)

#### Dependecies and files
A major part of the code's backbone is built upon the point cloud densifier code from [Opendronemap](https://github.com/opendronemap/ODM). In order to run the pipeline, you'll need a container engine:
- **If running locally**: [Docker](https://www.docker.com/) (uses the
  `opendronemap/odm:latest` image, pulled automatically on first run) (Not suggested due to the amounts of computational power needed.)
  ```
  docker://opendronemap/odm:latest
  ```
- **Using the BYU supercomputer**: [Apptainer](https://apptainer.org/).
Apptainer may work locally as well, this has not be verified. Docker for sure does not work on the BYU supercomputer.
  ```
  apptainer pull odm.sif 
  ```

> **Note:** [`py/auto_reconstruct.py`](py/auto_reconstruct.py) hardcodes the
> path to `odm.sif` (`APPTAINER_IMAGE`) for the Apptainer/Linux case, since
> that path lives outside this repo and differs per machine/account. When
> you download the repository, update that path at the
> top of the file before running.

Upload the `py` and `scjobs` folders to their respective folders in the supercomputer. Upload `requirements.txt` to `home/username`.

>**Note**: See the Appendix at the bottom of the README to see how to upload files to the supercomputer.

Once the above is done,
run the following command in the supercomputer 
to install the required packages:

```
pip install -r requirements.txt
```

You are now ready to run the pipeline using the BYU supercomputer! 🥳


### 2. Reconstruction: Photos to Point Cloud
>**Note**: 
```
python py/auto_reconstruct.py data/<project_name>
```

Put your drone images in `data/<project_name>/`. This will:
- Sort images into a `images/` subfolder
- Run the OpenSfM reconstruction pipeline (feature detection, matching,
  reconstruction, undistortion, depthmap densification)
- Produce `data/<project_name>/scene_dense.ply`

### 2. Extraction — point cloud to buildings

```
python py/ex_building_elev.py data/<project_name> --ground-cell-size 2.0 --ground-opening-span 20.0
```

- `ground-cell-size` size of each ground cell when creating the synthetic ground used to identify buildings. 
- `--ground-opening-span` How dense the surface created is. 
- The Defauts for each of theses should be acceptable.

This produces, in the folder `data/<project_name>/analysis_<project_name>_v1_4/` the following:
- `measurements.csv` — area (sqft), volume (cuft), height (ft) per building
- `location_lookup.csv` — UTM + lat/lon per building
- `individual_houses/` — per-building `.ply` point clouds
- `best_images/` and `best_images_cropped/` — the nearest source photo for
  each building, and a version cropped to just that building

## Known issues / in progress

- Best images and Best cropped images are still a work in progress. Code works well if the picture is looking at the house from plan view, but not as well from the front of the house.
- House segmenting still has problems. 
- Synthetic floors aren't perfect yet. 


## Appendix

### Uploading To and Downloading From The Supercomputer
To upload a folder/file to the supercomputer, you'll run a command composed of the following:

```
scp -r /path/from/local/computer username@ssh.rc.byu.edu:/home/path/to/output/location
```

- `scp -r` tells the terminal you want to access a remote network, and that you want to transfer a folder/file to or from that network. 

- `/Users/username/Desktop/normandy`, the first path, is the file/folder that you want to send. This file path will send the `normandy` folder from my local desktop. 

- `username@ssh.rc.byu.edu:/home/username/point_cloud/data`, the second path, is the location where you want to send the file. This file path will send the file/folder to the BYU supercomputer in the `point_cloud/data` folder.  

>**Note**: When uploading, always place the local path first, then the supercomputer path. 
>Linux, what the BYU computer runs, is case sensative. So keep that in mind.

To download a folder/file from the supercomputer, switch the second and third argument. 
Place the supercomputer path first, then the local path, like this: 

```
scp -r willicon@ssh.rc.byu.edu:/home/willicon/point_cloud/data /Users/willicon/Desktop/normandy/scene_dense.ply
``` 

Whenever you upload or download, you will have to input your password and authentication code. It’s annoying, but there is no way around it.  


### Creating and Runnig Jobs on the BYU Supercomputer
The BYU supercomputer uses `Slurm` commands to run it's jobs. Their [`documentation`](https://rc.byu.edu/wiki/?id=Slurm) and [`videos`](https://www.youtube.com/watch?v=i1r9BxHBG0I&list=PL326A5EB4E3B16FED) are old and somewhat outdated, but worth a look through. Below I have reproduced the minimum you need to run the pipeline. 

#### nano
This command lets you edit files that are on the supercomputer.

nano /home/username/point_cloud/scjobs/auto_reconstruct.sh

#### sbatch
This command runs the job commands on the supercomputer.

sbatch /home/username/point_cloud/scjobs/auto_reconstruct.sh


### Useful Supercomputer Commands
**Change File directory**
```
cd path/to/folder
```
**See all files/folders in the current directory**
```
ls 
``` 
**Moving a file** 
```
mv /path/to/current/location /path/to/new/destination/ 
```
**Submit your job** 
```
Sbatch [name of .sh file where job script is saved] 
``` 
**Editing a file** 

Use control + O to save the file, control + X to exit 
```
Nano path/to/file 
```
**Check on status of the job** 
```
Scontrol show job [job number] 
 ```
**Check the status of all jobs** 
```
squeue -u willicon 
 ```
**Canceling jobs** 
```
scancel [job ID(s)] 
```
**Following output for code**  
```
tail -f (reconstruct or house)_(job number).out 
```
>(ex.) tail -f reconstruct_1234567.out 

**Removing Files** 
```
rm filename1 filename2 etc 
```
**Removing Folders** 
```
rm -r foldername 
```

