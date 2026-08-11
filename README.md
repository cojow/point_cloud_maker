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

A major part of the code's backbone is built upon the point cloud densifier code from [Opendronemap](https://github.com/opendronemap/ODM). In order to run the pipeline, you'll need a container engine:
- **If running locally**: [Docker](https://www.docker.com/) (uses the
  `opendronemap/odm:latest` image, pulled automatically on first run) (Not suggested due to the amounts of computational power needed.)
  ```
  docker://opendronemap/odm:latest
  ```
- **Using the BYU supercomputer**: [Apptainer](https://apptainer.org/).
Apptainer may work locally as well, this has not be verified. Docker for sure does not work on the BYU supercomputer.

  Apptainer isn't on your `PATH` by default - load it first, every time you start a new shell session:
  ```
  module load apptainer
  ```
  Pull the image, then convert it into a sandbox (a plain, uncompressed directory). This is required, not optional - some compute nodes on the BYU supercomputer can't mount a `.sif` file directly (fails with a `squashfuse_ll ... fuse: device not found` error), but a sandbox sidesteps that since it doesn't need FUSE to build or to run against, either way:
  ```
  apptainer pull odm.sif docker://opendronemap/odm:latest
  apptainer build --sandbox odm_sandbox odm.sif
  ```
  >**Note**: A sandbox is a full, uncompressed copy of the image, so it takes noticeably more disk space than the `.sif` - This shouldn't be a problem, however.

> **Note:** [`py/auto_reconstruct.py`](py/auto_reconstruct.py) hardcodes the
> path to the sandbox (`APPTAINER_IMAGE`) for the Apptainer/Linux case, since
> that path lives outside this repo and differs per machine/account. When
> you download the repository, update that path at the
> top of the file to point at wherever you built `odm_sandbox` above.

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
>**Notes**: Replace "username" with your username when it shows up in the instructions. These instructions are written for running on the BYU supercomputer. See the apendix for how to run files on the supercomputer. 

Open the terminal and upload your photos folder to the supercomputer.
Input your username and authentication code and then wait for the files to finish uploading.
In the terminal, run “ssh username@ssh.rc.byu.edu”, and enter your password and authentication code. Now that you are in the supercomputer terminal, if you have already set up the file paths, run `nano /home/username/point_cloud/scjobs/auto_reconstruct.sh`. This will edit the script that runs the code to create the point cloud.  



Edit the last part of the line second from the bottom (ex. /home/willicon/point_cloud/data/normandy) so that it reflects the path to your folder with the images. Press control + O and enter to save the file, and control + X to exit.  




Run `cd /home/username/point_cloud/data/log_out`. This changes your file path to the folder that will hold all the files, showing the terminal output. 
Run “=`sbatch /home/username/point_cloud/scjobs/auto_reconstruct.sh`. This will submit your job. It will say “Submitted batch job” and then the project number. If you want to follow the code output, use `tail -f reconstruct_#jobnumber.out``=. Run this in the same location you ran the sbatch line.  

Once the process is running, you can edit the scjobs file to run another job.  

```
python py/auto_reconstruct.py data/<project_name>
```
Put your drone images in `data/<project_name>/`. This will:
- Sort images into a `images/` subfolder
- Run the OpenSfM reconstruction pipeline (feature detection, matching,
  reconstruction, undistortion, depthmap densification)
- Produce `data/<project_name>/scene_dense.ply`

### 3. Extraction — point cloud to buildings
>**Notes**: Since the process is similar to reconstuction, the following instructions are condensed. 

In the supercomputer, run `cd /home/username/point_cloud/data/log_out` so that the outputs are in the correct folder. 

Run `nano /home/username/point_cloud/scjobs/extract_buildings.sh` 




Edit the last part of the line second from the bottom (ex. /home/willicon/point_cloud/data/normandy) so that it reflects the path to your folder with the images. Press control + O and enter to save the file, and control + X to exit.  




Run `sbatch /home/username/point_cloud/scjobs/extract_buildings.sh` to submit your job. Wait for it to finish.  

The .out file is in the form of “house_jobnumber.out 

While it runs, you can edit the .sh file to run other jobs. 


 


```
python py/building_extractor.py data/<project_name>
```

All the tunable settings live in [`py/config.yml`](py/config.yml) now, not command-line flags - `nano py/config.yml` before submitting a job to change any of them (see the comments in that file for what each one does and, where relevant, what a real run showed about picking a good value). The defaults there should be acceptable for a first run. Briefly:
- `ground_model.cell_size` / `ground_model.opening_span` - resolution and reach of the synthetic ground surface used to identify buildings. `opening_span` must be wider than a normal building's footprint or roofs get misread as ground.
- `ground_model.relevel` - only for point clouds that did NOT go through the updated `auto_reconstruct.py`.
- `vehicle_rejection.reject_small_structures` (off by default) drops `Possible_Vehicle` candidates
  (small/short enough they might be a parked car) from the output entirely,
  instead of just flagging them for review. **Warning:** no test found so far
  can tell a parked car apart from a small legitimate structure (shed, small
  garage) by geometry alone - turning this on WILL also remove some real small
  buildings. Only use it if you'd rather lose those than manually discard cars
  from `measurements.csv`. Either way, check the remaining rows: `Needs_Review`
  flags likely vegetation contamination, `Possible_Vehicle` flags rows that
  might still be a car (always present unless `reject_small_structures` is
  used, in which case anything it would have flagged is gone instead).
- `imagery.nadir_pitch_threshold` - splits source photos into the nadir/oblique
  views described below. **Not universal** - confirmed the right value depends
  on how the site was actually flown (see that view's own section for a real
  example where the default is wrong).
- `performance.ransac_max_fit_points` - a speed/quality knob for messy sites
  with oversized or multi-building candidates; the default is validated but
  untested below the low tens of thousands.

If you want a different config for a specific project instead of editing
`py/config.yml` in place, save your own copy anywhere and pass
`--config /path/to/your.yml`.

This produces, in the folder `data/<project_name>/analysis_<project_name>_v1_4/` the following:
- `measurements.csv` — one row per building: area (sqft), volume (cuft),
  height (ft), UTM + lat/lon location, source image references, and two
  review flags (`Needs_Review` for likely vegetation contamination,
  `Possible_Vehicle` for candidates small/short enough that they might be a
  parked car rather than a building - neither flag deletes anything, both
  just mark rows worth a manual look). Source photos come in two views,
  each with its own three columns following the same `{View}_Best_Image` /
  `{View}_Original_Image` / `{View}_Image_Precision_Cropped` pattern:
  - `Nadir_*` — straight-down (DJI gimbal pitch at or steeper than
    `imagery.nadir_pitch_threshold` in `config.yml`, default -75°)
  - `Oblique_*` — the shallower of the site's two survey passes, if it has
    one - not a true side-on elevation shot, but steep enough to still show
    real wall/facade detail a nadir shot can't. **The default -75° threshold
    is not universal** - confirmed on the Blockall dataset (two passes,
    splitting cleanly around -75°) but confirmed WRONG on the fir dataset,
    whose steepest pass only reaches -72° - with the default threshold every
    single fir image gets classified as oblique and none as nadir. Check the
    "Indexed N nadir + M oblique" line in a run's log against what you
    actually expect before trusting either view on a new site, and adjust
    `imagery.nadir_pitch_threshold` in the config if it looks wrong.

  `{View}_Image_Precision_Cropped` says whether that view's `Best_Image`/
  `Original_Image` is a verified precision crop (projected from the
  building's real 3D footprint into a registered photo) or just the
  closest-by-GPS fallback photo, uncropped and unverified - only trust the
  image as accurate when this is `True`. Oblique has no fallback at all: if
  precision cropping fails for that view, `Oblique_Best_Image` is `N/A` and
  there's no oblique photo for that building.
- `individual_houses/` — per-building `.ply` point clouds
- `best_images/` and `best_images_cropped/` — source photos and crops for
  each building, one pair per view (`{house_ID}_nadir.jpg` /
  `{house_ID}_oblique.jpg`) - a cropped file only exists when that view's
  `Image_Precision_Cropped` is `True`

## Known issues / in progress

- Best images and Best cropped images are still a work in progress. Code works well if the picture is looking at the house from plan view, but not as well from the front of the house.
- House segmenting still has problems. 
- Synthetic floors aren't perfect yet, but they are better.


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

```
nano /home/username/point_cloud/scjobs/auto_reconstruct.sh
```

#### sbatch
This command runs the job commands on the supercomputer.

```
sbatch /home/username/point_cloud/scjobs/auto_reconstruct.sh
```

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

