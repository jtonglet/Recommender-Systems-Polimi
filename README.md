# Recommender  Systems Challenge 2021 @ Polimi


[![Kaggle](https://img.shields.io/badge/open-kaggle-blue)](https://www.kaggle.com/c/recommender-system-2021-challenge-polimi)

This repository contains the files and datasets used for a recommender system competition hosted on Kaggle. 







## Cython
<p align="justify">
Some of the models use Cython implementations. As written in the original repository you have to <b>compile all Cython algorithms</b>. 
In order to compile you must first have installed: gcc and python3 dev. Under Linux those can be installed with the following commands:
</p>

```
sudo apt install gcc 
sudo apt-get install python3-dev
```
  
<p align="justify">
If you are using Windows as operating system, the installation procedure is a bit more complex. 
You may refer to <a href="https://github.com/cython/cython/wiki/InstallingOnWindows">the official guide</a>.
</p>

<p align="justify">
Now you can compile all Cython algorithms by running the following command. 
The script will compile within the current active environment. The code has been developed for Linux and Windows platforms. 
During the compilation <b>you may see some warnings</b>. 
</p>
  
```
python run_compile_all_cython.py
```

## Visualization
To see a plot of MAP@10 for the best model and the hybrids composing it on various user groups, you can run the following command:
```
python HybridFinalParall.py
```
<p align="justify">
Note that the script tries to train in parallel as many recommenders as possible, and this may cause problems on machines with less than 16GB of RAM.
</p>
  
## Final grades
* 26 on 27 points
* MAP@10 : 0.48549
