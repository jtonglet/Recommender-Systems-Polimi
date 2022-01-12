# Recommender  Systems Challenge 2021 @ Polimi
<p align="center">
  <img width="50%" src="https://images.unsplash.com/photo-1560169897-fc0cdbdfa4d5?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=872&q=80" alt="header" />
</p>


[![Kaggle](https://img.shields.io/badge/closed-kaggle-blue)](https://www.kaggle.com/c/recommender-system-2021-challenge-polimi)

This repository contains the files and datasets used for the kaggle competition of the recommender systems course at Politecnico di Milano. The competition took place from November 2021 till January 2022. 


## Overview
<p align="justify">

The application domain of the competition was TV show recommendation. Starting from a sparse User Rating Matrix with implicit feedback and an Item Content Matrix with information on the genre, subgenre and diffusion channels of each tv show, we had to recommend 10 relevant TV shows for each user. The quality of the recommender was evaluated with  the **MAP@10** metric.
  
</p>
  
## Best Model
<p align="justify">
  
The final model used for the best submission is an 2-level hybrid recommender which averages the prediction of six different base recommenders. Please refer to the presentation [slide deck](https://github.com/jtonglet/Recommender-Systems-Polimi/blob/main/RecSys_2021.pdf) for more details.

</p>




## Recommenders
<p align="justify">
This repo contains the implementation of the following base recommenders : 
</p>

- User based Collaborative Filtering
- Item Content Based Filtering
- P3alpha and RP3Beta Graph Based models
- Pure SVD 
- iALS 
- Slim BPR 
- [EASE<sup>R</sup> Recommender](https://dl.acm.org/doi/pdf/10.1145/3308558.3313710)

The recommenders come from the course [repository](https://github.com/MaurizioFD/RecSys_Course_AT_PoliMi).


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

  
## Final grades
* 26 out of  27 points
* MAP@10 : 0.48549
